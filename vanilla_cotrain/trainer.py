"""Vanilla Co-Training trainer for text-only humanitarian tweet classification.

Implements the classic Blum & Mitchell (1998) co-training algorithm:
two models teach each other by exchanging their most confident predictions
on the unlabeled pool. Adapted from the user's cotrain_crisisMMD repo
(text_only path), stripped of all image/multimodal code.

The two models use the same architecture (BERTweet by default) but are
trained on different data splits (D_l1 and D_l2), so diversity comes from
the data partition rather than architectural differences.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from lg_cotrain.data_loading import (
    CLASS_LABELS,
    TweetDataset,
    build_label_encoder,
    detect_event_classes,
    load_tsv,
    split_labeled_set,
)
from lg_cotrain.evaluate import compute_ece, compute_metrics, ensemble_predict
from lg_cotrain.model import BertClassifier
from lg_cotrain.utils import EarlyStopping, get_device, set_seed, setup_logging

from .config import VanillaCoTrainConfig

logger = logging.getLogger("vanilla_cotrain")


class VanillaCoTrainer:
    """Classic co-training: two models, iterative mutual labeling, no LLM."""

    def __init__(self, config: VanillaCoTrainConfig):
        self.config = config
        self.device = get_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def _make_dataset(self, texts, labels):
        return TweetDataset(
            texts=list(texts),
            labels=list(labels),
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )

    def _encode_labels(self, labels):
        return [self.label2id[l] for l in labels]

    def _create_model(self):
        return BertClassifier(
            self.config.model_name, self.config.num_labels
        ).to(self.device)

    def _select_top_k_per_class(self, probs, pred_labels):
        """Select top-k most confident samples per predicted class.

        Args:
            probs: softmax probabilities, shape (n_samples, n_classes)
            pred_labels: predicted class ids, shape (n_samples,)

        Returns:
            np.ndarray of selected indices into the input arrays.
        """
        k = self.config.samples_per_class
        selected = []
        confidences = probs.max(axis=1)

        for cls_id in range(self.config.num_labels):
            cls_mask = pred_labels == cls_id
            cls_indices = np.where(cls_mask)[0]
            if len(cls_indices) == 0:
                continue
            cls_confs = confidences[cls_indices]
            top_k_local = min(k, len(cls_indices))
            top_k_idx = np.argsort(cls_confs)[-top_k_local:]
            selected.extend(cls_indices[top_k_idx])

        return np.array(selected, dtype=int)

    def _predict_proba(self, model, loader):
        """Run inference and return (probs, true_labels) as numpy arrays."""
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                probs = model.predict_proba(input_ids, attention_mask)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch["labels"].numpy())
        return np.concatenate(all_probs), np.concatenate(all_labels)

    def _train_model(self, model, loader, num_epochs, scheduler=None):
        """Train a model for a fixed number of epochs."""
        cfg = self.config
        opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        if scheduler is None and num_epochs > 0:
            total_steps = num_epochs * len(loader)
            warmup_steps = int(total_steps * cfg.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                opt,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
                n_batches += 1

        return model

    def run(self):
        """Run the full vanilla co-training pipeline.

        Returns:
            dict: metrics from the final evaluation on the test set.
        """
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(cfg.output_dir)
        set_seed(cfg.seed_set)

        logger.info(
            f"Starting Vanilla Co-Training: event={cfg.event}, "
            f"budget={cfg.budget}, seed_set={cfg.seed_set}"
        )
        logger.info(f"Using device: {self.device}, model: {cfg.model_name}")

        # --- Load data ---
        df_labeled = load_tsv(cfg.labeled_path)
        df_unlabeled = load_tsv(cfg.unlabeled_path)
        df_dev = load_tsv(cfg.dev_path)
        df_test = load_tsv(cfg.test_path)

        # Detect classes and build label encoder
        classes = detect_event_classes(df_labeled, df_unlabeled, df_dev, df_test)
        cfg.num_labels = len(classes)
        logger.info(f"Detected {cfg.num_labels} classes for event {cfg.event}: {classes}")

        self.label2id, self.id2label = build_label_encoder(classes)

        # Split labeled set
        df_l1, df_l2 = split_labeled_set(df_labeled, seed=cfg.seed_set)
        logger.info(f"D_l1: {len(df_l1)}, D_l2: {len(df_l2)}, Unlabeled: {len(df_unlabeled)}")

        # Build dev and test datasets (fixed throughout)
        dev_labels = self._encode_labels(df_dev["class_label"])
        test_labels = self._encode_labels(df_test["class_label"])
        ds_dev = self._make_dataset(df_dev["tweet_text"], dev_labels)
        ds_test = self._make_dataset(df_test["tweet_text"], test_labels)
        loader_dev = DataLoader(ds_dev, batch_size=cfg.batch_size, shuffle=False)
        loader_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

        # Initialize labeled sets for each model (will grow during co-training)
        labeled_a = df_l1.copy()
        labeled_b = df_l2.copy()
        unlabeled_pool = df_unlabeled.copy()

        # --- Iterative co-training loop ---
        logger.info(f"=== Iterative Co-Training: {cfg.num_iterations} iterations ===")
        start_time = time.time()

        for iteration in range(1, cfg.num_iterations + 1):
            if len(unlabeled_pool) == 0:
                logger.info(f"Iteration {iteration}: unlabeled pool exhausted. Stopping.")
                break

            # Create fresh models each iteration
            model_a = self._create_model()
            model_b = self._create_model()

            # Build training datasets from current labeled sets
            labels_a = self._encode_labels(labeled_a["class_label"])
            labels_b = self._encode_labels(labeled_b["class_label"])
            ds_a = self._make_dataset(labeled_a["tweet_text"], labels_a)
            ds_b = self._make_dataset(labeled_b["tweet_text"], labels_b)
            loader_a = DataLoader(ds_a, batch_size=cfg.batch_size, shuffle=True)
            loader_b = DataLoader(ds_b, batch_size=cfg.batch_size, shuffle=True)

            # Train both models
            model_a = self._train_model(model_a, loader_a, cfg.train_epochs)
            model_b = self._train_model(model_b, loader_b, cfg.train_epochs)

            # Predict on unlabeled pool
            unlabeled_labels_dummy = [0] * len(unlabeled_pool)
            ds_unlabeled = self._make_dataset(
                unlabeled_pool["tweet_text"], unlabeled_labels_dummy
            )
            loader_unlabeled = DataLoader(
                ds_unlabeled, batch_size=cfg.batch_size, shuffle=False
            )

            probs_a, _ = self._predict_proba(model_a, loader_unlabeled)
            probs_b, _ = self._predict_proba(model_b, loader_unlabeled)
            preds_a = probs_a.argmax(axis=1)
            preds_b = probs_b.argmax(axis=1)

            # Select top-k per class from each model
            selected_by_a = self._select_top_k_per_class(probs_a, preds_a)
            selected_by_b = self._select_top_k_per_class(probs_b, preds_b)

            if len(selected_by_a) == 0 and len(selected_by_b) == 0:
                logger.info(f"Iteration {iteration}: no samples selected. Stopping.")
                break

            # Exchange pseudo-labels:
            # Model A's confident predictions -> added to labeled_B (and vice versa)
            if len(selected_by_a) > 0:
                new_for_b = unlabeled_pool.iloc[selected_by_a].copy()
                new_for_b["class_label"] = [
                    self.id2label[p] for p in preds_a[selected_by_a]
                ]
                labeled_b = pd.concat([labeled_b, new_for_b], ignore_index=True)

            if len(selected_by_b) > 0:
                new_for_a = unlabeled_pool.iloc[selected_by_b].copy()
                new_for_a["class_label"] = [
                    self.id2label[p] for p in preds_b[selected_by_b]
                ]
                labeled_a = pd.concat([labeled_a, new_for_a], ignore_index=True)

            # Remove selected samples from unlabeled pool
            all_selected = sorted(set(selected_by_a.tolist() + selected_by_b.tolist()))
            unlabeled_pool = unlabeled_pool.drop(
                unlabeled_pool.index[all_selected]
            ).reset_index(drop=True)

            # Evaluate ensemble on dev set
            dev_preds, dev_true, _ = ensemble_predict(
                model_a, model_b, loader_dev, self.device
            )
            dev_metrics = compute_metrics(dev_true, dev_preds)

            logger.info(
                f"Iteration {iteration}/{cfg.num_iterations}: "
                f"selected_a={len(selected_by_a)}, selected_b={len(selected_by_b)}, "
                f"labeled_a={len(labeled_a)}, labeled_b={len(labeled_b)}, "
                f"unlabeled_remaining={len(unlabeled_pool)}, "
                f"dev_macro_f1={dev_metrics['macro_f1']:.4f}, "
                f"dev_err={dev_metrics['error_rate']:.2f}%"
            )

            # Clean up GPU memory
            del model_a, model_b
            torch.cuda.empty_cache()

        iter_elapsed = time.time() - start_time
        logger.info(
            f"Co-training iterations done in {iter_elapsed:.0f}s. "
            f"Final labeled_a={len(labeled_a)}, labeled_b={len(labeled_b)}"
        )

        # --- Fine-tuning phase ---
        logger.info(f"=== Fine-Tuning (max {cfg.finetune_max_epochs} epochs, patience={cfg.finetune_patience}) ===")

        model_a = self._create_model()
        model_b = self._create_model()

        labels_a = self._encode_labels(labeled_a["class_label"])
        labels_b = self._encode_labels(labeled_b["class_label"])
        ds_a = self._make_dataset(labeled_a["tweet_text"], labels_a)
        ds_b = self._make_dataset(labeled_b["tweet_text"], labels_b)
        loader_a = DataLoader(ds_a, batch_size=cfg.batch_size, shuffle=True)
        loader_b = DataLoader(ds_b, batch_size=cfg.batch_size, shuffle=True)

        opt_a = AdamW(model_a.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        opt_b = AdamW(model_b.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        total_steps = cfg.finetune_max_epochs * max(len(loader_a), len(loader_b))
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        sched_a = get_linear_schedule_with_warmup(
            opt_a, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        sched_b = get_linear_schedule_with_warmup(
            opt_b, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        es_a = EarlyStopping(patience=cfg.finetune_patience)
        es_b = EarlyStopping(patience=cfg.finetune_patience)

        for epoch in range(1, cfg.finetune_max_epochs + 1):
            # Train model A
            model_a.train()
            for batch in loader_a:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = model_a(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                opt_a.zero_grad()
                loss.backward()
                opt_a.step()
                sched_a.step()

            # Train model B
            model_b.train()
            for batch in loader_b:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = model_b(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                opt_b.zero_grad()
                loss.backward()
                opt_b.step()
                sched_b.step()

            # Evaluate ensemble on dev
            dev_preds, dev_true, _ = ensemble_predict(
                model_a, model_b, loader_dev, self.device
            )
            dev_metrics = compute_metrics(dev_true, dev_preds)
            dev_f1 = dev_metrics["macro_f1"]

            stop_a = es_a.step(dev_f1, model_a)
            stop_b = es_b.step(dev_f1, model_b)

            logger.info(
                f"Fine-tune epoch {epoch}: dev_macro_f1={dev_f1:.4f}, "
                f"dev_err={dev_metrics['error_rate']:.2f}%, "
                f"es_counter_a={es_a.counter}, es_counter_b={es_b.counter}"
            )

            if stop_a and stop_b:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best models
        es_a.restore_best(model_a)
        es_b.restore_best(model_b)

        # --- Final evaluation ---
        logger.info("=== Final Evaluation ===")

        test_preds, test_true, test_probs = ensemble_predict(
            model_a, model_b, loader_test, self.device
        )
        test_metrics = compute_metrics(test_true, test_preds)
        test_ece = compute_ece(test_true, test_probs)

        dev_preds, dev_true, dev_probs = ensemble_predict(
            model_a, model_b, loader_dev, self.device
        )
        dev_metrics = compute_metrics(dev_true, dev_preds)
        dev_ece = compute_ece(dev_true, dev_probs)

        results = {
            "event": cfg.event,
            "budget": cfg.budget,
            "seed_set": cfg.seed_set,
            "test_error_rate": test_metrics["error_rate"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_ece": test_ece,
            "test_per_class_f1": test_metrics["per_class_f1"],
            "dev_error_rate": dev_metrics["error_rate"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_ece": dev_ece,
        }

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {metrics_path}")
        logger.info(
            f"Test error rate: {test_metrics['error_rate']:.2f}%, "
            f"Test macro-F1: {test_metrics['macro_f1']:.4f}, "
            f"Test ECE: {test_ece:.4f}"
        )

        return results
