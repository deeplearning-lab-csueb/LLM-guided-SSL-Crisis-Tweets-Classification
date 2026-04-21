"""Supervised-only fine-tuning trainer for tweet classification.

Trains a single BERTweet model on labeled data only -- no pseudo-labels,
no co-training. Serves as a baseline to measure the contribution of
LG-CoTrain's semi-supervised components.
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
    TweetDataset,
    build_label_encoder,
    detect_event_classes,
    load_tsv,
)
from lg_cotrain.evaluate import compute_ece, compute_metrics
from lg_cotrain.model import BertClassifier
from lg_cotrain.utils import EarlyStopping, get_device, set_seed, setup_logging

from .config import SupervisedBaselineConfig

logger = logging.getLogger("supervised_baseline")


class SupervisedTrainer:
    """Fine-tune a single model on labeled data with early stopping."""

    def __init__(self, config: SupervisedBaselineConfig):
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

    def _predict(self, model, loader):
        """Run inference. Returns (preds, true_labels, probs) as numpy arrays."""
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                probs = model.predict_proba(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch["labels"].numpy())
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        all_preds = all_probs.argmax(axis=1)
        return all_preds, all_labels, all_probs

    def _save_predictions(self, df, preds, probs, output_dir, filename):
        """Save per-sample predictions to a TSV file."""
        out_df = df.copy().reset_index(drop=True)
        out_df["predicted_label"] = [self.id2label[p] for p in preds]
        out_df["confidence"] = probs.max(axis=1).round(4)

        for class_id in range(probs.shape[1]):
            label_name = self.id2label[class_id]
            out_df[f"prob_{label_name}"] = probs[:, class_id].round(6)

        out_path = Path(output_dir) / filename
        out_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
        logger.info(f"Predictions saved to {out_path} ({len(out_df)} samples)")

    def run(self):
        """Run supervised fine-tuning and return comprehensive metrics dict."""
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(cfg.output_dir)
        set_seed(cfg.seed_set)

        logger.info(
            f"Starting Supervised Baseline: event={cfg.event}, "
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
        self.label2id, self.id2label = build_label_encoder(classes)
        logger.info(f"Detected {cfg.num_labels} classes for event {cfg.event}: {classes}")
        logger.info(f"Labeled: {len(df_labeled)}, Dev: {len(df_dev)}, Test: {len(df_test)}")

        # Build datasets
        ds_train = self._make_dataset(
            df_labeled["tweet_text"], self._encode_labels(df_labeled["class_label"])
        )
        ds_dev = self._make_dataset(
            df_dev["tweet_text"], self._encode_labels(df_dev["class_label"])
        )
        ds_test = self._make_dataset(
            df_test["tweet_text"], self._encode_labels(df_test["class_label"])
        )

        loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
        loader_dev = DataLoader(ds_dev, batch_size=cfg.batch_size, shuffle=False)
        loader_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

        # --- Training ---
        model = BertClassifier(cfg.model_name, cfg.num_labels).to(self.device)
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_steps = cfg.max_epochs * len(loader_train)
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
        es = EarlyStopping(patience=cfg.patience)

        start_time = time.time()
        epochs_trained = 0
        best_epoch = 0

        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader_train:
                optimizer.zero_grad()
                logits = model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                loss = F.cross_entropy(logits, batch["labels"].to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Evaluate on dev
            dev_preds, dev_labels, _ = self._predict(model, loader_dev)
            dev_metrics = compute_metrics(dev_labels, dev_preds)
            dev_f1 = dev_metrics["macro_f1"]

            epochs_trained = epoch
            if es.step(dev_f1, model):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if dev_f1 >= es.best_score:
                best_epoch = epoch

            logger.info(
                f"Epoch {epoch}/{cfg.max_epochs}: "
                f"loss={epoch_loss/n_batches:.4f}, "
                f"dev_macro_f1={dev_f1:.4f}, "
                f"dev_err={dev_metrics['error_rate']:.2f}%, "
                f"es_counter={es.counter}"
            )

        es.restore_best(model)
        train_elapsed = time.time() - start_time
        # best_epoch is the epoch where best dev F1 was achieved
        best_epoch = epochs_trained - cfg.patience if es.counter >= cfg.patience else epochs_trained
        best_epoch = max(1, best_epoch)

        logger.info(
            f"Training done in {train_elapsed:.0f}s. "
            f"Epochs: {epochs_trained}, Best epoch: {best_epoch}"
        )

        # --- Final evaluation ---
        logger.info("=== Final Evaluation ===")

        test_preds, test_labels, test_probs = self._predict(model, loader_test)
        test_metrics = compute_metrics(test_labels, test_preds)
        test_ece = compute_ece(test_labels, test_probs)

        dev_preds, dev_labels, dev_probs = self._predict(model, loader_dev)
        dev_metrics = compute_metrics(dev_labels, dev_preds)
        dev_ece = compute_ece(dev_labels, dev_probs)

        results = {
            # Identifiers
            "event": cfg.event,
            "budget": cfg.budget,
            "seed_set": cfg.seed_set,
            "method": "supervised",
            "model_name": cfg.model_name,
            # Test metrics
            "test_error_rate": test_metrics["error_rate"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_ece": test_ece,
            "test_per_class_f1": test_metrics["per_class_f1"],
            # Dev metrics
            "dev_error_rate": dev_metrics["error_rate"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_ece": dev_ece,
            "dev_per_class_f1": dev_metrics["per_class_f1"],
            # Training metadata
            "epochs_trained": epochs_trained,
            "best_epoch": best_epoch,
            "best_dev_macro_f1": float(es.best_score),
            "training_time_seconds": round(train_elapsed, 1),
            # Hyperparameters (for reproducibility)
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio,
            "max_seq_length": cfg.max_seq_length,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            # Class info
            "num_labels": cfg.num_labels,
            "class_labels": classes,
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

        # Save per-sample predictions
        self._save_predictions(
            df_test, test_preds, test_probs, cfg.output_dir, "test_predictions.tsv"
        )
        self._save_predictions(
            df_dev, dev_preds, dev_probs, cfg.output_dir, "dev_predictions.tsv"
        )

        return results
