import argparse
import os
import time
from datasets import load_from_disk, DatasetDict, concatenate_datasets

def hf_separator():
    p = argparse.ArgumentParser(description="Separate labeled data to have n rows per class (randomized).")
    p.add_argument("--n", type=int, required=True, help="Number of rows to keep labeled PER CLASS")
    p.add_argument("--raw_path", required=True, help="Path of the dataset saved via save_to_disk()")
    p.add_argument("--labeled_path", help="Output path of labeled data (dir). Defaults to ./[raw_path]_labeled_jsonl")
    p.add_argument("--unlabeled_path", help="Output path of unlabeled data (dir). Defaults to ./[raw_path]_unlabeled_jsonl")
    p.add_argument("--label_col", default="label", help="Column with class labels (default: 'label')")
    p.add_argument("--keep", nargs="+", help="Column(s) to keep before saving UNLABELED set")
    p.add_argument("--seed", type=int, default=int(time.time()), help="RNG seed for shuffling (default: current time)")
    args = p.parse_args()

    print(f"Current seed: {args.seed}")

    raw = load_from_disk(args.raw_path)
    if not isinstance(raw, DatasetDict):
        raw = DatasetDict({"train": raw})

    labeled_splits = {}
    unlabeled_splits = {}

    for split, ds in raw.items():
        if args.label_col not in ds.column_names:
            raise ValueError(f"Label column '{args.label_col}' not found in split '{split}'. Available: {ds.column_names}")

        labeled_rows = []
        unlabeled_rows = []

        # Stratify per class with randomization
        classes = set(ds[args.label_col])
        for lbl in classes:
            class_ds = ds.filter(lambda x: x[args.label_col] == lbl)
            class_ds = class_ds.shuffle(seed=args.seed)

            take = min(args.n, len(class_ds))
            if take > 0:
                labeled_rows.append(class_ds.select(range(take)))
            if len(class_ds) > take:
                unlabeled_rows.append(class_ds.select(range(take, len(class_ds))))

        # Build labeled/unlabeled datasets for this split
        if labeled_rows:
            labeled_split = concatenate_datasets(labeled_rows).shuffle(seed=args.seed)
            labeled_splits[split] = labeled_split

        if unlabeled_rows:
            unlabeled_split = concatenate_datasets(unlabeled_rows).shuffle(seed=args.seed)
            # Apply --keep ONLY to unlabeled (drop all other columns)
            if args.keep:
                keep_cols = [c for c in unlabeled_split.column_names if c in set(args.keep)]
                if keep_cols:
                    try:
                        unlabeled_split = unlabeled_split.select_columns(keep_cols)
                    except Exception:
                        # Fallback for older datasets versions
                        drop_cols = [c for c in unlabeled_split.column_names if c not in keep_cols]
                        unlabeled_split = unlabeled_split.remove_columns(drop_cols)
            unlabeled_splits[split] = unlabeled_split

    # Default output dirs
    labeled_path = args.labeled_path or (args.raw_path.rstrip("/").rstrip("\\") + "_labeled")
    unlabeled_path = args.unlabeled_path or (args.raw_path.rstrip("/").rstrip("\\") + "_unlabeled")
    os.makedirs(labeled_path, exist_ok=True)
    os.makedirs(unlabeled_path, exist_ok=True)

    # Save labeled split as arrow, unlabeled split as JSONL (readable)
    labeled_ds = DatasetDict(labeled_splits)
    labeled_ds.save_to_disk(labeled_path)

    for split, d in unlabeled_splits.items():
        out_fp = os.path.join(unlabeled_path, f"{split}.jsonl")
        d.to_json(out_fp, lines=True, force_ascii=False)
        print(f"[unlabeled] wrote {split}: {out_fp} (rows={len(d)})")

if __name__ == "__main__":
    hf_separator()
