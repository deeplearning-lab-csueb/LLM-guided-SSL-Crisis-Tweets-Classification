#!/usr/bin/env python3
# join_splits.py
# Usage:
#   python join_splits.py --root /path/to/datasets --out /path/to/outdir
# Options:
#   --add_event       Add an "event" column with the folder name

import argparse
from pathlib import Path
import re
import sys
import pandas as pd

SPLIT_PATTERN = re.compile(r"_(train|dev|test)(?:\.tsv(?:\.gz)?)?$", re.IGNORECASE)

def detect_split(filename: str):
    m = SPLIT_PATTERN.search(filename)
    return m.group(1).lower() if m else None

def main():
    ap = argparse.ArgumentParser(description="Join first 3 TSV files per event folder into massive split-wise TSVs.")
    ap.add_argument("--root", required=True, help="Root directory containing event subfolders")
    ap.add_argument("--out", required=True, help="Output directory where train/dev/test TSVs will be written")
    ap.add_argument("--add_event", action="store_true", help='Add an "event" column with the folder (event) name')
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect dataframes by split
    buckets = {"train": [], "dev": [], "test": []}
    ref_cols_per_split = {"train": None, "dev": None, "test": None}

    # Iterate immediate subdirectories (events)
    events = sorted([p for p in root.iterdir() if p.is_dir()])
    if not events:
        print(f"No event folders found under: {root}", file=sys.stderr)
        sys.exit(1)

    for event_dir in events:
        # First 3 TSV files (alphabetically). Accept .tsv and .tsv.gz
        tsvs = sorted(list(event_dir.glob("*.tsv")) + list(event_dir.glob("*.tsv.gz")))
        if len(tsvs) < 3:
            print(f"[WARN] {event_dir} has fewer than 3 TSVs (found {len(tsvs)}). Skipping.", file=sys.stderr)
            continue

        candidates = tsvs[:3]

        # Map files to splits using filename suffix
        split_map = {}
        for fp in candidates:
            split = detect_split(fp.name)
            if split not in {"train", "dev", "test"}:
                print(f"[WARN] Could not detect split from filename: {fp.name}. Skipping this file.", file=sys.stderr)
                continue
            if split in split_map:
                print(f"[WARN] Duplicate split '{split}' among first 3 files in {event_dir.name}. Keeping first and skipping {fp.name}.", file=sys.stderr)
                continue
            split_map[split] = fp

        # Read & append
        for split, fp in split_map.items():
            try:
                df = pd.read_csv(fp, sep="\t", dtype=str, engine="python")
            except Exception as e:
                print(f"[WARN] Failed to read {fp}: {e}. Skipping.", file=sys.stderr)
                continue

            if args.add_event:
                df.insert(0, "event", event_dir.name)

            # Column consistency check within each split
            if ref_cols_per_split[split] is None:
                ref_cols_per_split[split] = list(df.columns)
            else:
                if list(df.columns) != ref_cols_per_split[split]:
                    print(
                        f"[ERROR] Column mismatch in split '{split}' for event '{event_dir.name}'.\n"
                        f"  Expected: {ref_cols_per_split[split]}\n"
                        f"  Found:    {list(df.columns)}\n"
                        f"Aborting to avoid corrupt output.",
                        file=sys.stderr,
                    )
                    sys.exit(2)

            buckets[split].append(df)
        # (If a folder is missing a split among the first three files, we just skip that split for this event.)

    for split in ["train", "dev", "test"]:
        if not buckets[split]:
            print(f"[INFO] No data collected for split '{split}'. Skipping write.", file=sys.stderr)
            continue
        big = pd.concat(buckets[split], ignore_index=True)

        # Optional: remove duplicates
        before = len(big)
        big = big.drop_duplicates()
        after = len(big)
        print(f"[INFO] Removed {before - after} duplicate rows from '{split}'")

        out_fp = out_dir / f"{split}.tsv"
        big.to_csv(out_fp, sep="\t", index=False)
        print(f"[OK] Wrote {split}: {out_fp} (rows={len(big)}, cols={len(big.columns)})")

if __name__ == "__main__":
    main()
