"""
Take a dataset, then extract the ids, inputs, and labels into a new dataset (jsonl).
Then, make a new jsonl with only the inputs needed, keeping rows aligned.
"""

import argparse
from datasets import load_dataset, load_from_disk, Dataset
from datasets.features import ClassLabel

def resolve_dataset(source: str, split: str):
    # If it's a path previously saved with .save_to_disk, use load_from_disk
    try:
        return load_from_disk(source)
    except Exception:
        # Otherwise assume a HF hub dataset id
        return load_dataset(source, split=split)

def main():
    ap = argparse.ArgumentParser(description="Extract ids, inputs, labels to JSONL; plus inputs-only JSONL (aligned). Uses HF datasets only.")
    ap.add_argument("--dataset", required=True, help="HF dataset id (e.g. 'tweetnlp/tweet_topic_multi') OR path from .save_to_disk")
    ap.add_argument("--split", default="train", help="Split name, e.g. 'train', 'validation', 'test'")
    ap.add_argument("--id_col", default="tweet_id", help="Column name for the id")
    ap.add_argument("--input_col", default="tweet_text", help="Column name for the input text")
    ap.add_argument("--label_col", default="label", help="Column name for the label")
    ap.add_argument("--out_full", default="full.jsonl", help="Output JSONL with id,input,label")
    ap.add_argument("--out_inputs", default="inputs_only.jsonl", help="Output JSONL with input only (aligned)")
    args = ap.parse_args()

    ds = resolve_dataset(args.dataset, args.split)

    # If we loaded a DatasetDict from disk, select the split
    if not isinstance(ds, Dataset):
        if args.split not in ds:
            raise ValueError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")
        ds = ds[args.split]

    # Sanity checks
    for c in (args.id_col, args.input_col, args.label_col):
        if c not in ds.column_names:
            raise ValueError(f"Column '{c}' not in dataset columns {ds.column_names}")
    
    # Remove duplicates based on id_col
    df = ds.to_pandas().drop_duplicates(args.id_col)
    ds = Dataset.from_pandas(df)

    # If label is a ClassLabel, convert to its string form; otherwise keep as-is
    feat = ds.features.get(args.label_col)
    if isinstance(feat, ClassLabel):
        def label_to_str(example):
            idx = example[args.label_col]
            example[args.label_col] = feat.int2str(idx) if isinstance(idx, int) else idx
            return example
        ds_for_full = ds.map(label_to_str, remove_columns=[])
    else:
        ds_for_full = ds


    # Build the two views (keep original order; do not shuffle)
    cols_full = [args.id_col, args.input_col, args.label_col]
    full_view = ds_for_full.remove_columns([c for c in ds_for_full.column_names if c not in cols_full])

    inputs_only = ds.remove_columns([c for c in ds.column_names if c != args.input_col])

    # Persist as JSONL
    full_view.to_json(args.out_full, lines=True, force_ascii=False)
    inputs_only.to_json(args.out_inputs, lines=True, force_ascii=False)

    # Quick confirmation
    assert len(full_view) == len(inputs_only), "Row counts differ — alignment broken."
    print(f"[ok] wrote {args.out_full} (rows={len(full_view)})")
    print(f"[ok] wrote {args.out_inputs} (rows={len(inputs_only)})")

if __name__ == "__main__":
    main()
