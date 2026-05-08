#!/usr/bin/env python
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

from datasets import load_dataset, concatenate_datasets, Dataset

def vmatch_separator():
    p = argparse.ArgumentParser(
        description="Separate labeled data to have n rows per class (randomized). "
                    "The rest is pseudo-labeled data. Save both using save_to_disk"
    )
    p.add_argument("--n", type=int, required=True, help="Number of rows to keep labeled PER CLASS")
    p.add_argument("--union_path", required=True, help="Path to the union of the labeled and pseudo-labeled set (jsonl)")
    p.add_argument("--labeled_path", help="Output path of labeled data (dir). Defaults to ./labeled")
    p.add_argument("--unlabeled_path", help="Output path of unlabeled data (dir). Defaults to ./unlabeled")
    p.add_argument("--gold_col", default="label", help="Column with gold labels (default: 'label')")
    p.add_argument("--pred_col", default="label_ds2", help="Column with pseudo-labels/predictions (default: 'label_ds2')")
    p.add_argument("--seed", type=int, default=int(time.time()),
                   help="RNG seed for shuffling (default: current time)")
    args = p.parse_args()

    n = args.n
    union_path = args.union_path
    labeled_path = args.labeled_path or "./labeled"
    unlabeled_path = args.unlabeled_path or "./unlabeled"
    gold_col = args.gold_col
    pred_col = args.pred_col
    seed = args.seed

    # 0) Basic checks
    if n <= 0:
        print(f"[error] --n must be positive (got {n})", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(union_path):
        print(f"[error] union_path not found: {union_path}", file=sys.stderr)
        sys.exit(2)

    # 1) Load the union dataset (JSONL) as a single split
    #    Accepts .jsonl or .json; if it's a directory with multiple shards, pass the file path(s).
    try:
        ds = load_dataset("json", data_files=union_path, split="train")
    except Exception as e:
        print(f"[error] Failed to load JSONL at {union_path}: {e}", file=sys.stderr)
        sys.exit(2)

    # 2) Validate required columns
    missing = [c for c in (gold_col, pred_col) if c not in ds.column_names]
    if missing:
        print(f"[error] Missing required columns in dataset: {missing}\n"
              f"Available columns: {ds.column_names}", file=sys.stderr)
        sys.exit(2)

    total = len(ds)
    if total == 0:
        print("[error] Empty dataset.", file=sys.stderr)
        sys.exit(2)

    # 3) Compute class counts on gold labels
    #    We keep exactly up to n per class for the labeled split.
    #    If a class has < n examples, we keep all of them and warn.
    gold_counts = Counter(ds[gold_col])
    classes = sorted(gold_counts.keys())

    print(f"[info] Loaded union dataset: rows={total}")
    print(f"[info] Gold label column: '{gold_col}' | Pred column: '{pred_col}'")
    print(f"[info] Classes ({len(classes)}): {classes}")
    for k in classes:
        print(f"  - {k}: {gold_counts[k]}")

    # 4) Build labeled/unlabeled via per-class shuffle & take
    labeled_parts = []
    unlabeled_parts = []

    for cls in classes:
        # filter to this class
        sub = ds.filter(lambda ex, c=cls: ex[gold_col] == c)
        # shuffle deterministically
        sub = sub.shuffle(seed=seed)

        take_k = min(n, len(sub))
        if take_k < n:
            print(f"[warn] Class '{cls}' has only {len(sub)} examples (< n={n}); taking all of them.",
                  file=sys.stderr)

        labeled_parts.append(sub.select(range(take_k)))
        if take_k < len(sub):
            unlabeled_parts.append(sub.select(range(take_k, len(sub))))

    # 5) Concatenate parts (handle edge cases with empty splits)
    if labeled_parts:
        labeled = labeled_parts[0] if len(labeled_parts) == 1 else concatenate_datasets(labeled_parts)
    else:
        labeled = Dataset.from_dict({c: [] for c in ds.column_names})  # empty

    if unlabeled_parts:
        unlabeled = unlabeled_parts[0] if len(unlabeled_parts) == 1 else concatenate_datasets(unlabeled_parts)
    else:
        unlabeled = Dataset.from_dict({c: [] for c in ds.column_names})  # empty

    # 6) Drop the pred_col from labeled and update the remaining label column
    labeled = labeled.remove_columns([pred_col])
    if gold_col != "label":
        labeled = labeled.rename_column(gold_col, "label")

    # 7) Drop the gold_col from unlabeled and update the remaining label column
    unlabeled = unlabeled.remove_columns([gold_col])
    if pred_col != "label":
        unlabeled = unlabeled.rename_column(pred_col, "label")  # standardize to 'label'

    # 8) Print distributions for sanity
    def dist(d):
        cnt = Counter(d["label"]) if len(d) else Counter()
        return {k: cnt.get(k, 0) for k in classes}

    print("\n[info] Labeled split stats:")
    print(f"  rows={len(labeled)}  per-class={dist(labeled)}")
    print("[info] Unlabeled split stats:")
    print(f"  rows={len(unlabeled)}  (gold kept for reference; downstream can ignore)")

    # 9) Save with save_to_disk (Arrow directory format)
    for out_dir in (labeled_path, unlabeled_path):
        os.makedirs(out_dir, exist_ok=True)

    labeled.save_to_disk(labeled_path)
    unlabeled.save_to_disk(unlabeled_path)

    print(f"\n[write] Labeled saved:   {labeled_path} (rows={len(labeled)})")
    print(f"[write] Unlabeled saved: {unlabeled_path} (rows={len(unlabeled)})")

    # 10) Extra: quick recap of how many per class were requested vs. kept
    kept_per_class = defaultdict(int)
    if len(labeled):
        for v in labeled["label"]:
            kept_per_class[v] += 1
    print("\n[recap] Requested per-class n =", n)
    for cls in classes:
        print(f"  - {cls}: kept {kept_per_class[cls]} (available {gold_counts[cls]})")

if __name__ == "__main__":
    vmatch_separator()
