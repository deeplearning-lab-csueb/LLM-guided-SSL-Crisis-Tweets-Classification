# bert_ft.py — Original WandB-based supervised fine-tuning script.
#
# NOTE: This file is the original training script used during experimentation
# with WandB sweeps for hyperparameter search. It is kept for transparency and
# reproducibility of the WandB-logged runs. For the clean, Optuna-based
# supervised baseline used by the rest of this codebase (e.g., NB27), see
# trainer.py and run_experiment.py in this directory instead.
#
# Example:
#   python bert_ft.py ^
#     --dataset_path data\crisismmd2inf ^
#     --raw_format tsvdir ^
#     --output_dir outputs\bert_supervised_min ^
#     --text_col tweet_text --label_col label --id_col tweet_id

import argparse, os, json, itertools, time, random, sys, tempfile
from typing import List, Dict, Optional

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../utils'))
try:
    from run_humaid import get_paths
except ImportError:
    print("Warning: Could not import get_paths from run_humaid. Make sure ../utils exists.")
    get_paths = None

import wandb


import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ----------------------------- Metrics -----------------------------
def expected_calibration_error(probs, labels, bins=10):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def fit_temperature_scaling(logits, labels):
    """Fit optimal temperature to minimize NLL on dev set (Guo et al., 2017).
    Divides logits by scalar T before softmax: T>1 softens, T<1 sharpens."""
    from scipy.optimize import minimize_scalar

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    def nll(t):
        scaled = logits_t / t
        log_probs = torch.nn.functional.log_softmax(scaled, dim=-1)
        return -log_probs[range(len(labels_t)), labels_t].mean().item()

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return result.x


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    ece = expected_calibration_error(probs, labels)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "ece": ece}


# ----------------------------- Utils -----------------------------
def load_raw_dataset(dataset_path: str,
                     raw_format: str = "auto",
                     tsv_train: str = "train.tsv",
                     tsv_dev:   str = "dev.tsv",
                     tsv_test:  str = "test.tsv",
                     tsv_delim: str = "\t",
                     event_filter: Optional[str] = None) -> DatasetDict:
    """Load dataset from HF dir or TSV folder."""
    def _is_dir_with_tsvs(p):
        return os.path.isdir(p) and any(
            os.path.exists(os.path.join(p, name)) for name in [tsv_train, tsv_dev, tsv_test]
        )

    def _load_hf(p):
        d = load_from_disk(p)
        if isinstance(d, Dataset):
            return DatasetDict({"train": d})
        return d

    def _load_tsvdir(p):
        files = {}
        fp_train = os.path.join(p, tsv_train)
        fp_dev   = os.path.join(p, tsv_dev)
        fp_test  = os.path.join(p, tsv_test)
        if os.path.exists(fp_train): files["train"] = fp_train
        if os.path.exists(fp_dev):   files["dev"]   = fp_dev
        if os.path.exists(fp_test):  files["test"]  = fp_test
        assert "dev" in files and "test" in files, "TSV folder must contain at least dev/test."
        
        # If event filter is active, read via pandas, filter, and convert
        if event_filter:
            dd = {}
            for split, fp in files.items():
                df = pd.read_csv(fp, sep=tsv_delim)
                if "event" in df.columns:
                    df = df[df["event"] == event_filter]
                else:
                    print(f"Warning: --event {event_filter} passed but 'event' col missing in {fp}")
                dd[split] = Dataset.from_pandas(df)
            return DatasetDict(dd)
        else:
            ds = load_dataset("csv", data_files=files, delimiter=tsv_delim)
            return DatasetDict({k: v for k, v in ds.items()})

    if raw_format == "hf":
        return _load_hf(dataset_path)
    if raw_format == "tsvdir":
        return _load_tsvdir(dataset_path)

    try:
        return _load_hf(dataset_path)
    except Exception:
        if _is_dir_with_tsvs(dataset_path):
            return _load_tsvdir(dataset_path)
        raise ValueError(f"Could not load dataset from '{dataset_path}'.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_labels(ds: Dataset, label_col: str) -> List[str]:
    labs = sorted(set(ds[label_col]))
    return [str(x) for x in labs]


def build_label_maps(label_order: Optional[List[str]], train_like: Dataset, label_col: str):
    if label_order is None:
        label_order = infer_labels(train_like, label_col)
    label2id = {l: i for i, l in enumerate(label_order)}
    id2label = {i: l for l, i in label2id.items()}
    return label_order, label2id, id2label


# ---------- CHANGE #2 ONLY: enforce safe max_length truncation ----------
def tokenize_with_labels(ds: Dataset, tok, text_col: str, label_col: str,
                         label2id: Dict[str, int], max_length: int) -> Dataset:
    def add_label(ex):
        return {"labels": label2id[str(ex[label_col])]}
    ds = ds.map(add_label)
    ds = ds.map(lambda b: tok(b[text_col], truncation=True, max_length=max_length), batched=True)

    # Always keep input_ids, attention_mask, labels, and token_type_ids (if any)
    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop)


def load_train_dataset(train_path: Optional[str], train_hf: Optional[str], split: str, event_filter: Optional[str] = None) -> Dataset:
    if train_path:
        ext = os.path.splitext(train_path)[-1].lower()
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            if event_filter:
                df = pd.read_csv(train_path, sep=sep)
                if "event" in df.columns:
                    df = df[df["event"] == event_filter]
                elif "10lb" not in train_path and "5lb" not in train_path and "25lb" not in train_path and "50lb" not in train_path:
                     # Only warn if not in a lbcl partitioned folder where filtering might be implicit? 
                     # Actually, we verified 10lb/1/labeled.tsv HAS event column, so we should allow filtering.
                     print(f"Warning: --event {event_filter} passed but 'event' col missing in {train_path}")
                return Dataset.from_pandas(df)
            else:
                return load_dataset("csv", data_files=train_path, split="train", delimiter=sep)
        elif ext in (".json", ".jsonl"):
            # Filtering for json not implemented yet but unlikely for this task
            return load_dataset("json", data_files=train_path, split="train")
        else:
            try:
                dsd = load_from_disk(train_path)
                ds = dsd[split] if isinstance(dsd, DatasetDict) else dsd
                # Filter if HF dataset has event column
                if event_filter and "event" in ds.column_names:
                     return ds.filter(lambda x: x["event"] == event_filter)
                return ds
            except Exception as e:
                raise ValueError(f"Could not load --train_path='{train_path}': {e}")
    if train_hf:
        dsd = load_from_disk(train_hf)
        ds = dsd[split] if isinstance(dsd, DatasetDict) else dsd
        if event_filter and "event" in ds.column_names:
                return ds.filter(lambda x: x["event"] == event_filter)
        return ds
    raise ValueError("Provide --train_path or --train_hf (or rely on dataset_path train split).")


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path")
    ap.add_argument("--output_dir")
    ap.add_argument("--train_path")
    ap.add_argument("--train_hf")
    ap.add_argument("--train_split_name", default="train")
    ap.add_argument("--dev_split_name", default="dev")
    ap.add_argument("--test_split_name", default="test")
    ap.add_argument("--raw_format", choices=["auto","hf","tsvdir"], default="auto")
    ap.add_argument("--tsv_train", default="train.tsv")
    ap.add_argument("--tsv_dev", default="dev.tsv")
    ap.add_argument("--tsv_test", default="test.tsv")
    ap.add_argument("--tsv_delim", default="\t")

    ap.add_argument("--text_col", default="tweet_text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--id_col", default="tweet_id")

    ap.add_argument("--label_order", nargs="*")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--lrs", nargs="*", type=float, default=[2e-5, 3e-5])
    ap.add_argument("--epochs_list", nargs="*", type=int, default=[3, 5], dest="epochs_list")
    ap.add_argument("--batch_sizes_list", nargs="*", type=int, default=[16, 32], dest="batch_sizes_list")
    ap.add_argument("--selection_metric", choices=["f1", "accuracy"], default="f1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_cpu", action="store_true")
    ap.add_argument("--max_train_samples", type=int)

    # CHANGE #2: allow overriding max_length
    ap.add_argument("--max_length", type=int, default=0,
                    help="If >0, cap sequence length to this; else auto = min(512, model_max-2).")

    # HPO Single Args
    ap.add_argument("--learning_rate", type=float, help="Learning rate (WandB sweep)")
    ap.add_argument("--batch_size", type=int, help="Batch size (WandB sweep)")
    ap.add_argument("--epochs", type=int, help="Epochs (WandB sweep)")

    # HPO / HumAID Auto args
    ap.add_argument("--event", type=str, help="HumAID event name for auto-path resolution")
    ap.add_argument("--lbcl", type=int, help="Labels per class for auto-path resolution")
    ap.add_argument("--set_num", type=int, help="Set number for auto-path resolution")
    ap.add_argument("--project_name", type=str, default="humaid_supervised_hpo", help="WandB project name")

    args = ap.parse_args()
    
    # Initialize WandB
    if args.event and args.lbcl and args.set_num:
        run_name = f"{args.event}_{args.lbcl}_{args.set_num}_{int(time.time())}"
        # Auto-mode: use specific run name config
        wandb.init(project=args.project_name, name=run_name, config=args, reinit=True)
        # Allow wandb sweep to override these, OR command line args
        # Prioritize: 1. CLI arg 2. WandB config 3. First item of list default
        
        lr_val = args.learning_rate if args.learning_rate else wandb.config.get("learning_rate", args.lrs[0])
        ep_val = args.epochs if args.epochs else wandb.config.get("epochs", args.epochs_list[0])
        bs_val = args.batch_size if args.batch_size else wandb.config.get("batch_size", args.batch_sizes_list[0])
        
        args.model_name = wandb.config.get("model_name", args.model_name)
        args.max_length = wandb.config.get("max_length", args.max_length)
        
        # Override lrs/epochs/batch_sizes lists to single values for compatibility with downstream loop code
        args.lrs = [float(lr_val)]
        args.epochs_list = [int(ep_val)]
        args.batch_sizes_list = [int(bs_val)]

        print(f"✅ Auto-Mode: Event={args.event}, LBCL={args.lbcl}, Set={args.set_num}")
        print(f"   Hyperparams: LR={args.lrs[0]}, BS={args.batch_sizes_list[0]}, Eps={args.epochs_list[0]}")

        # We want to filter manually, so we can just point to the ROOT folders
        # path to joined root for dev/test
        args.dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/humaid/joined"))
        
        # path to train labeled (contains all events?)
        # User said "see tree output".
        # The tree output shows: anh_4o_mini/sep/10lb/1/labeled.tsv
        # so we should construct this path.
        
        base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/humaid"))
        # try anh_4o_mini first as per user tree
        train_base = os.path.join(base_data, "anh_4o_mini", "sep", f"{args.lbcl}lb", str(args.set_num), "labeled.tsv")
        if not os.path.exists(train_base):
                # Fallback to anh_4o if exists (old path)
                train_base = os.path.join(base_data, "anh_4o", "sep", f"{args.lbcl}lb", str(args.set_num), "labeled.tsv")
        
        args.train_path = train_base
        args.output_dir = os.path.join(os.path.dirname(__file__), f"outputs/sup_{args.event}_{args.lbcl}lb_set{args.set_num}")
            
        args.label_col = "class_label"
        
        print(f"   Dataset Path: {args.dataset_path}")
        print(f"   Train Path: {args.train_path}")

    if not args.dataset_path or not args.output_dir:
        raise ValueError("Must provide either --dataset_path and --output_dir OR (--event, --lbcl, --set_num) for auto-resolution.")

    # Force output_dir to be a temporary directory to save NOTHING locally
    temp_dir_obj = tempfile.TemporaryDirectory()
    original_output_dir = args.output_dir
    args.output_dir = temp_dir_obj.name
    print(f"   [System] Output Dir redirected to Temp Dir: {args.output_dir} (was {original_output_dir})")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # GPU check
    if not torch.cuda.is_available() and not args.allow_cpu:
        print("No GPU detected. Use --allow_cpu to run on CPU.")
        return

    # Load data
    dsd_raw = load_raw_dataset(
        args.dataset_path,
        raw_format=args.raw_format,
        tsv_train=args.tsv_train, tsv_dev=args.tsv_dev, tsv_test=args.tsv_test,
        tsv_delim=args.tsv_delim,
        event_filter=args.event # Pass event filter
    )
    assert "dev" in dsd_raw and "test" in dsd_raw, "need dev/test splits"
    dev_raw = dsd_raw[args.dev_split_name]
    test_raw = dsd_raw[args.test_split_name]

    if args.train_path or args.train_hf:
        train_raw = load_train_dataset(args.train_path, args.train_hf, args.train_split_name, event_filter=args.event)
    else:
        assert args.train_split_name in dsd_raw
        train_raw = dsd_raw[args.train_split_name]

    if args.max_train_samples and len(train_raw) > args.max_train_samples:
        train_raw = train_raw.shuffle(seed=args.seed).select(range(args.max_train_samples))

    label_order, label2id, id2label = build_label_maps(args.label_order, dev_raw, args.label_col)
    with open(os.path.join(args.output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label_order": label_order, "label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
                  f, ensure_ascii=False, indent=2)

    # Tokenizer & tokenization
    tok = AutoTokenizer.from_pretrained(
        args.model_name, 
        use_fast=True,
        normalization=True,
    )
    _tmp_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model_max = int(getattr(_tmp_model.config, "max_position_embeddings", 512))
    del _tmp_model
    safe_max_length = int(args.max_length) if args.max_length > 0 else min(512, model_max - 2)

    train_ds = tokenize_with_labels(train_raw, tok, args.text_col, args.label_col, label2id, safe_max_length)
    dev_ds   = tokenize_with_labels(dev_raw, tok, args.text_col, args.label_col, label2id, safe_max_length)
    test_ds  = tokenize_with_labels(test_raw, tok, args.text_col, args.label_col, label2id, safe_max_length)
    collator = DataCollatorWithPadding(tokenizer=tok)

    test_ids = test_raw[args.id_col] if args.id_col in test_raw.column_names else list(range(len(test_raw)))
    test_gold_names = [str(x) for x in test_raw[args.label_col]]
    test_gold_ids = [label2id[name] for name in test_gold_names]

    # Grid search
    best_cfg, best_score = None, -1.0
    summary_rows = []
    # If using Auto Mode, these lists have exactly 1 item each
    for lr, ep, bs in itertools.product(args.lrs, args.epochs_list, args.batch_sizes_list):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
        )
        targs = TrainingArguments(
            output_dir=f"{args.output_dir}/trial_lr{lr}_ep{ep}_bs{bs}",
            eval_strategy="epoch", save_strategy="no",
            learning_rate=lr, num_train_epochs=ep,
            per_device_train_batch_size=bs, per_device_eval_batch_size=max(8, bs),
            report_to=[], seed=args.seed, load_best_model_at_end=False,
            lr_scheduler_type="constant",
        )
        trainer = Trainer(
            model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
            tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics
        )
        trainer.train()
        eval_out = trainer.evaluate()
        metric_key = f"eval_{args.selection_metric}"
        score = float(eval_out.get(metric_key, float("nan")))
        print(f"[dev] lr={lr} ep={ep} bs={bs} -> {args.selection_metric}={score:.4f}")
        summary_rows.append({"lr": lr, "epochs": ep, "batch_size": bs, **eval_out})
        if score > best_score:
            best_score, best_cfg = score, (lr, ep, bs)
        
        if wandb.run:
            wandb.log({
                "lr": lr, "epochs": ep, "batch_size": bs,
                **eval_out,
                "best_score_so_far": best_score
            })

        # trainer.save_model(f"{args.output_dir}/trial_lr{lr}_ep{ep}_bs{bs}")

    with open(os.path.join(args.output_dir, "grid_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    cfg_str = f"[best] cfg={best_cfg} {args.selection_metric}={best_score:.4f} seed={args.seed}"
    print(cfg_str)
    with open(os.path.join(args.output_dir, "best_cfg.txt"), "w", encoding="utf-8") as f:
        f.write(cfg_str + "\n")

    # Retrain best & predict
    lr, ep, bs = best_cfg
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    targs = TrainingArguments(
        output_dir=f"{args.output_dir}/best", eval_strategy="no", save_strategy="no",
        learning_rate=lr, num_train_epochs=ep,
        per_device_train_batch_size=bs, per_device_eval_batch_size=max(8, bs),
        report_to=[], seed=args.seed, lr_scheduler_type="constant",
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, 
        tokenizer=tok, data_collator=collator
    )
    trainer.train()
    # trainer.save_model(f"{args.output_dir}/best")

    test_pred_logits = trainer.predict(test_ds).predictions
    test_pred_ids = np.argmax(test_pred_logits, axis=-1)
    test_pred_names = [str(id2label[int(p)]) for p in test_pred_ids]

    # Calculate and log Test Metrics
    test_probs = torch.nn.functional.softmax(torch.tensor(test_pred_logits), dim=-1).numpy()
    test_acc = accuracy_score(test_gold_ids, test_pred_ids)
    _, _, test_f1, _ = precision_recall_fscore_support(
        test_gold_ids, test_pred_ids, average="macro", zero_division=0
    )
    # Per-class F1 breakdown
    _, _, per_class_f1, per_class_support = precision_recall_fscore_support(
        test_gold_ids, test_pred_ids, average=None, zero_division=0
    )
    per_class_f1_dict = {}
    for cls_id, (f1_val, sup_val) in enumerate(zip(per_class_f1, per_class_support)):
        cls_name = id2label[cls_id]
        per_class_f1_dict[cls_name] = {"f1": float(f1_val), "support": int(sup_val)}
    print("[per-class F1]")
    for cls_name, vals in sorted(per_class_f1_dict.items()):
        print(f"  {cls_name}: f1={vals['f1']:.4f} (n={vals['support']})")

    test_ece = expected_calibration_error(test_probs, np.array(test_gold_ids))

    # ---- Post-hoc Temperature Scaling (Guo et al., 2017) ----
    dev_pred_logits = trainer.predict(dev_ds).predictions
    dev_gold_names = [str(x) for x in dev_raw[args.label_col]]
    dev_gold_ids_list = [label2id[name] for name in dev_gold_names]
    optimal_temp = fit_temperature_scaling(dev_pred_logits, dev_gold_ids_list)
    test_probs_calibrated = torch.nn.functional.softmax(
        torch.tensor(test_pred_logits) / optimal_temp, dim=-1
    ).numpy()
    test_ece_calibrated = expected_calibration_error(test_probs_calibrated, np.array(test_gold_ids))
    print(f"[calibration] Optimal temperature: {optimal_temp:.4f}")

    print(f"[test] accuracy={test_acc:.4f} macro_f1={test_f1:.4f} ece_raw={test_ece:.4f} ece_calibrated={test_ece_calibrated:.4f}")
    
    test_metrics = {
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "test_ece": test_ece,
        "test_ece_calibrated": test_ece_calibrated,
        "optimal_temperature": optimal_temp,
    }
    # Add per-class F1 to metrics
    for cls_name, vals in per_class_f1_dict.items():
        test_metrics[f"test_f1_{cls_name}"] = vals["f1"]
        test_metrics[f"test_support_{cls_name}"] = vals["support"]
    
    # Log to WandB if active
    if wandb.run:
        wandb.log(test_metrics)
        
    # Save to JSON
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    csv_int = os.path.join(args.output_dir, "pred_bert_int.csv")
    with open(csv_int, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["id", "gold", "pred"])
        for _id, g, p in zip(test_ids, test_gold_ids, test_pred_ids):
            w.writerow([_id, int(g), int(p)])

    csv_str = os.path.join(args.output_dir, "pred_bert_str.csv")
    with open(csv_str, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["id", "gold_name", "pred_name"])
        for _id, gname, pname in zip(test_ids, test_gold_names, test_pred_names):
            w.writerow([_id, gname, pname])

    print(f"Wrote: {csv_int}")
    print(f"Wrote: {csv_str}")

    if wandb.run:
        artifact_name = f"preds-{wandb.run.id}"
        artifact = wandb.Artifact(name=artifact_name, type="predictions")
        artifact.add_file(csv_int)
        artifact.add_file(csv_str)
        wandb.log_artifact(artifact)
        print("Logged prediction artifact to WandB.")


if __name__ == "__main__":
    main()
