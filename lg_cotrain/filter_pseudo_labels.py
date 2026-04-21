"""Filter self-trained teacher pseudo-labels to keep only the top-p most
confident predictions per class.

Reads the per-cell pseudo-label CSVs from
``data/pseudo-labelled/self-trained/{event}/labeled_{budget}_set{seed}_pseudo.csv``
and writes filtered versions to
``data/pseudo-labelled/self-trained-top-p/{event}/labeled_{budget}_set{seed}_pseudo.csv``.

The filtered CSVs have the same schema as the originals (tweet_id,
tweet_text, predicted_label, confidence) but with fewer rows. The
LG-CoTrain pipeline can consume them via
``--pseudo-label-source self-trained-top-p``.

Usage::

    # Filter with top-50 per class (default)
    python -m lg_cotrain.filter_pseudo_labels

    # Filter with top-100 per class
    python -m lg_cotrain.filter_pseudo_labels --samples-per-class 100

    # Filter only specific budgets/seeds
    python -m lg_cotrain.filter_pseudo_labels --budgets 5 50 --seed-sets 1 2 3

    # Overwrite existing filtered CSVs
    python -m lg_cotrain.filter_pseudo_labels --force
"""

import argparse
import csv
import sys
from pathlib import Path

# All 10 HumAID events
ALL_EVENTS = [
    "california_wildfires_2018",
    "canada_wildfires_2016",
    "cyclone_idai_2019",
    "hurricane_dorian_2019",
    "hurricane_florence_2018",
    "hurricane_harvey_2017",
    "hurricane_irma_2017",
    "hurricane_maria_2017",
    "kaikoura_earthquake_2016",
    "kerala_floods_2018",
]
DEFAULT_BUDGETS = [5, 10, 25, 50]
DEFAULT_SEEDS = [1, 2, 3]


def filter_csv(input_path: str, output_path: str, samples_per_class: int) -> dict:
    """Read a pseudo-label CSV and write a filtered version keeping only the
    top-``samples_per_class`` most confident rows per predicted class.

    Returns a dict with stats: {total_in, total_out, classes_in, classes_out}.
    """
    with open(input_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by predicted_label
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        cls = row["predicted_label"]
        by_class.setdefault(cls, []).append(row)

    # For each class, sort by confidence descending and keep top-p
    kept = []
    for cls in sorted(by_class):
        cls_rows = by_class[cls]
        cls_rows.sort(key=lambda r: float(r["confidence"]), reverse=True)
        kept.extend(cls_rows[:samples_per_class])

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tweet_id", "tweet_text", "predicted_label", "confidence"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)

    return {
        "total_in": len(rows),
        "total_out": len(kept),
        "classes_in": len(by_class),
        "classes_out": len(set(r["predicted_label"] for r in kept)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter self-trained pseudo-labels to top-p per class"
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=50,
        help="Number of most confident predictions to keep per class (default: 50)",
    )
    parser.add_argument(
        "--events", nargs="+", default=ALL_EVENTS,
        help="Events to filter (default: all 10)",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS,
        help="Budgets to filter (default: 5 10 25 50)",
    )
    parser.add_argument(
        "--seed-sets", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Seed sets to filter (default: 1 2 3)",
    )
    parser.add_argument(
        "--data-root", type=str,
        default=str(Path(__file__).parent.parent / "data"),
        help="Data root directory (default: data/)",
    )
    parser.add_argument(
        "--source", type=str, default="self-trained",
        help="Source pseudo-label directory name (default: self-trained)",
    )
    parser.add_argument(
        "--target", type=str, default="self-trained-top-p",
        help="Target pseudo-label directory name (default: self-trained-top-p)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing filtered CSVs (default: skip if exists)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    source_root = data_root / "pseudo-labelled" / args.source
    target_root = data_root / "pseudo-labelled" / args.target

    total = 0
    filtered = 0
    skipped = 0
    missing = 0

    for event in args.events:
        for budget in args.budgets:
            for seed in args.seed_sets:
                fname = f"labeled_{budget}_set{seed}_pseudo.csv"
                src = source_root / event / fname
                dst = target_root / event / fname

                if not src.exists():
                    missing += 1
                    continue

                total += 1

                if dst.exists() and not args.force:
                    skipped += 1
                    continue

                stats = filter_csv(str(src), str(dst), args.samples_per_class)
                filtered += 1
                print(
                    f"  {event}/{fname}: "
                    f"{stats['total_in']} -> {stats['total_out']} rows "
                    f"({stats['classes_out']}/{stats['classes_in']} classes)"
                )

    print()
    print(f"Done. samples_per_class={args.samples_per_class}")
    print(f"  Total cells:    {total}")
    print(f"  Filtered:       {filtered}")
    print(f"  Skipped (exist):{skipped}")
    print(f"  Missing source: {missing}")


if __name__ == "__main__":
    main()
