from datasets import load_dataset, load_from_disk, Dataset
import argparse
import sys
import os

def load_hf_dataset(spec: str) -> Dataset:
    """
    Load a single-split Hugging Face Dataset from:
      - a local json/jsonl file
      - a dataset saved with `save_to_disk`
      - a hub dataset name
    Always returns a Dataset (not a DatasetDict).
    """
    # Local json/jsonl
    if spec.lower().endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=spec, split="train")

    # Saved-to-disk dataset
    if os.path.isdir(spec) and os.path.exists(os.path.join(spec, "dataset_info.json")):
        # load_from_disk returns a Dataset or DatasetDict; ensure Dataset
        d = load_from_disk(spec)
        if isinstance(d, Dataset):
            return d
        return d["train"] if "train" in d else list(d.values())[0]

    # Hub dataset name or path
    d = load_dataset(spec, split="train")
    return d

def join_datasets(ds1: Dataset, ds2: Dataset, on_col: str = "tweet_text", max_warnings: int = 50) -> Dataset:
    if on_col not in ds1.column_names or on_col not in ds2.column_names:
        raise ValueError(f"Join column '{on_col}' must exist in both datasets.")

    n1, n2 = len(ds1), len(ds2)
    if n1 != n2:
        print(f"[warn] Length mismatch: ds1={n1}, ds2={n2}", file=sys.stderr)

    # Iterate row-by-row but limit prints to avoid terminal freezes
    rows = []
    mismatches = 0
    limit = min(n1, n2)

    for i in range(limit):
        row1, row2 = ds1[i], ds2[i]
        if row1.get(on_col) != row2.get(on_col):
            mismatches += 1
            if mismatches <= max_warnings:
                print(
                    f"[warn] Row {i}: mismatch in '{on_col}'\n"
                    f"  ds1: {row1.get(on_col)}\n"
                    f"  ds2: {row2.get(on_col)}",
                    file=sys.stderr,
                )

        # merge dicts (suffix ds2 keys to avoid collisions except the join key)
        merged = dict(row1)
        for k, v in row2.items():
            if k == on_col:
                continue
            if k in merged:
                merged[f"{k}_ds2"] = v
            else:
                merged[k] = v
        rows.append(merged)

    if mismatches > max_warnings:
        print(f"[warn] ...and {mismatches - max_warnings} more mismatches suppressed.", file=sys.stderr)

    if n1 != n2:
        print(f"[info] Truncated to {limit} rows due to length mismatch.", file=sys.stderr)

    return Dataset.from_list(rows)

def main():
    parser = argparse.ArgumentParser(description="Join two HuggingFace datasets on a column and save the result.")
    parser.add_argument("--dataset1", required=True, help="Path/name of the first dataset (json/jsonl, saved dir, or hub id)")
    parser.add_argument("--dataset2", required=True, help="Path/name of the second dataset (json/jsonl, saved dir, or hub id)")
    parser.add_argument("--output_path", required=True, help="Path to save the csv")
    parser.add_argument("--on_col", default="tweet_text", help="Column to join on (default: tweet_text)")
    parser.add_argument("--id_col", default="tweet_id", help="Column name for the id (default: tweet_id)")
    parser.add_argument("--gold_col", default="label", help="Column name for the gold label (default: label)")
    parser.add_argument("--pred_col", default="label_ds2", help="Column name for the predicted label (default: pred_label)")
    parser.add_argument("--max_warnings", type=int, default=50, help="Max mismatch warnings to print (default: 50)")
    args = parser.parse_args()

    ds1 = load_hf_dataset(args.dataset1)
    ds2 = load_hf_dataset(args.dataset2)

    print(f"[info] Loaded ds1: rows={len(ds1)}, cols={ds1.column_names}", file=sys.stderr)
    print(f"[info] Loaded ds2: rows={len(ds2)}, cols={ds2.column_names}", file=sys.stderr)
    print(f"[info] Joining on column: '{args.on_col}'", file=sys.stderr)

    joined = join_datasets(ds1, ds2, on_col=args.on_col, max_warnings=args.max_warnings)
    joined.to_json(args.output_path + ".jsonl")

    joined = joined.rename_column(args.id_col, "id")
    joined = joined.rename_column(args.gold_col, "gold")
    joined = joined.rename_column(args.pred_col, "pred")

    LABEL_ORDER = ["not_informative", "informative"]
    label2id = {l:i for i, l in enumerate(LABEL_ORDER)}
    id2label = {i:l for l, i in label2id.items()}

    # convert gold/pred string labels to numeric ids
    def to_ids(batch):
        return {
            "gold": [label2id[str(v).lower()] if v in label2id else v for v in batch["gold"]],
            "pred": [label2id[str(v).lower()] if v in label2id else v for v in batch["pred"]],
        }

    joined = joined.map(to_ids, batched=True)

    ds = joined.select_columns(["id", "gold", "pred"])
    ds.to_csv(args.output_path, index=False)
    print(f"[ok] Saved {len(ds)} rows to {args.output_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
