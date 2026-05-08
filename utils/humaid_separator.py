# Make labeled/unlabeled TSVs per set from a global union, using pre-decided labeled files
# Pattern searched in each event folder: labeled_<lbcl>_<set>.tsv
#
# Example:
#   python make_sets_from_union.py \
#       --root_dir data/events \
#       --union_path data/union.tsv \
#       --out_root data/splits_out \
#       --lbcl 10 \
#       --sets 1 2 3 \
#       --join_col tweet_id

import argparse, os, sys, glob
import pandas as pd

def collect_labeled_keys(root_dir: str, lbcl: str, set_num: int, join_col: str) -> tuple[set, list[str]]:
    """
    Look under all immediate subfolders of root_dir for files:
      labeled_<lbcl>_<set_num>.tsv
    Return the set of join keys and a few example missing columns if join_col absent.
    """
    keys = set()
    missing_col_examples = []
    # All event dirs (any subdir)
    event_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, d))]

    pattern_suffix = f"labeled_{lbcl}_set{set_num}.tsv"
    for ed in sorted(event_dirs):
        cand = os.path.join(ed, pattern_suffix)
        if not os.path.exists(cand):
            continue
        try:
            df = pd.read_csv(cand, sep="\t", dtype=str, keep_default_na=False)
        except Exception as e:
            print(f"[warn] Failed to read {cand}: {e}", file=sys.stderr)
            continue
        if join_col not in df.columns:
            missing_col_examples.append(cand)
            continue
        # Collect keys from this labeled file
        keys.update(df[join_col].dropna().tolist())

    return keys, missing_col_examples

def main():
    ap = argparse.ArgumentParser(description="Materialize labeled/unlabeled TSVs per set from a global union using labeled_<lbcl>_<set>.tsv files found under event folders.")
    ap.add_argument("--root_dir", required=True, help="Root folder containing per-event subfolders with labeled_<lbcl>_<set>.tsv files.")
    ap.add_argument("--union_path", required=True, help="Path to the global union TSV (all events combined).")
    ap.add_argument("--out_root", required=True, help="Where to write outputs. One subfolder per set will be created.")
    ap.add_argument("--lbcl", type=int, nargs="+", default=[5, 10, 25, 50], help="The lb/cl token used in labeled file names (e.g., 10 → labeled_10_<set>.tsv).")
    ap.add_argument("--sets", type=int, nargs="+", default=[1, 2, 3], help="Set numbers to process (default: 1 2 3).")
    ap.add_argument("--join_col", default="tweet_id", help="Key column used to identify rows (default: tweet_id).")
    args = ap.parse_args()

    # Load union once
    if not os.path.exists(args.union_path):
        print(f"[error] union_path not found: {args.union_path}", file=sys.stderr)
        sys.exit(1)

    union_df = pd.read_csv(args.union_path, sep="\t", dtype=str, keep_default_na=False)
    if args.join_col not in union_df.columns:
        print(f"[error] join_col '{args.join_col}' not in union columns: {union_df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    # Deduplicate union on the join key
    before = len(union_df)
    union_df = union_df.drop_duplicates(subset=[args.join_col], keep="first")
    if len(union_df) != before:
        print(f"[info] Dropped {before - len(union_df)} duplicate union rows on '{args.join_col}'", file=sys.stderr)

    union_keys = set(union_df[args.join_col].tolist())

    for lb in args.lbcl:
        lb_dir = os.path.join(args.out_root, f"{lb}lb")
        os.makedirs(lb_dir, exist_ok=True)
        for s in args.sets:
            labeled_keys, missing_col_files = collect_labeled_keys(args.root_dir, lb, s, args.join_col)

            if missing_col_files:
                print(f"[warn] The following labeled files for set {s} do not contain '{args.join_col}':", file=sys.stderr)
                for fp in missing_col_files[:5]:
                    print(f"       - {fp}", file=sys.stderr)
                if len(missing_col_files) > 5:
                    print(f"       (+ {len(missing_col_files)-5} more)", file=sys.stderr)

            if not labeled_keys:
                print(f"[skip] No labeled files found for lbcl={lb}, set={s}.", file=sys.stderr)
                continue

            # Warn about labeled keys not found in union
            not_in_union = labeled_keys - union_keys
            if not_in_union:
                print(f"[warn] set {s}: {len(not_in_union)} labeled keys not present in union by '{args.join_col}'. Examples: {list(not_in_union)[:5]}", file=sys.stderr)

            # Build labeled/unlabeled views from union (ensures consistent columns)
            labeled_df = union_df[union_df[args.join_col].isin(labeled_keys)].copy()
            unlabeled_df = union_df[~union_df[args.join_col].isin(labeled_keys)].copy()

            # Drop pred col in labeled set
            if "label" in labeled_df.columns:
                labeled_df = labeled_df.drop(columns=["label"])

            # Drop gold col in unlabeled set 
            if "class_label" in unlabeled_df.columns:
                unlabeled_df = unlabeled_df.drop(columns=["class_label"])
                unlabeled_df = unlabeled_df.rename(columns={"label": "class_label"})

            # Output
            set_dir = os.path.join(lb_dir, str(s))
            os.makedirs(set_dir, exist_ok=True)
            labeled_out = os.path.join(set_dir, "labeled.tsv")
            unlabeled_out = os.path.join(set_dir, "unlabeled.tsv")

            labeled_df.to_csv(labeled_out, sep="\t", index=False)
            unlabeled_df.to_csv(unlabeled_out, sep="\t", index=False)

            print(f"[ok] lbcl={lb}, set {s}: labeled rows = {len(labeled_df)} → {labeled_out}")
            print(f"[ok] lbcl={lb}, set {s}: unlabeled rows = {len(unlabeled_df)} → {unlabeled_out}")

if __name__ == "__main__":
    main()