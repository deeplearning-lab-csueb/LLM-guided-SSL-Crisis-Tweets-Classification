#!/usr/bin/env python3
"""Merge and summarize per-experiment Optuna results.

Two modes:
  1. Summary only:  Scan target directory and regenerate summary_{n}.json
  2. Merge + summary: Copy results from source directories into target, then summarize

Standalone — does not import from lg_cotrain.

Usage:
    # Regenerate summary from whatever exists
    python merge_optuna_results.py --target results/bert-base/optuna/per_experiment --n-trials 10

    # Merge from other PCs, then generate summary
    python merge_optuna_results.py \
        --sources pc2_results/ pc3_results/ \
        --target results/bert-base/optuna/per_experiment \
        --n-trials 10

    # Dry run (show what would be copied, don't copy)
    python merge_optuna_results.py \
        --sources pc2_results/ \
        --target results/bert-base/optuna/per_experiment \
        --n-trials 10 --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path

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

BUDGETS = [5, 10, 25, 50]
SEED_SETS = [1, 2, 3]

SEARCH_SPACE = {
    "lr": "1e-5 to 1e-3 (log-uniform)",
    "batch_size": [8, 16, 32, 64],
    "cotrain_epochs": "5 to 20",
    "finetune_patience": "4 to 10",
    "weight_decay": "0.0 to 0.1",
    "warmup_ratio": "0.0 to 0.3",
}

TOTAL_STUDIES = len(ALL_EVENTS) * len(BUDGETS) * len(SEED_SETS)  # 120


def _study_path(base_dir: Path, event: str, budget: int, seed: int, n_trials: int) -> Path:
    return base_dir / event / f"{budget}_set{seed}" / f"trials_{n_trials}"


def merge_sources(sources: list, target: Path, n_trials: int, dry_run: bool = False) -> dict:
    """Copy best_params.json and study.log from sources into target.

    Returns dict with counts: copied, skipped, conflicts.
    """
    stats = {"copied": 0, "skipped": 0, "conflicts": []}

    for source_dir in sources:
        source = Path(source_dir)
        if not source.exists():
            print(f"  WARNING: source directory does not exist: {source}")
            continue

        for event in ALL_EVENTS:
            for budget in BUDGETS:
                for seed in SEED_SETS:
                    src_trial_dir = _study_path(source, event, budget, seed, n_trials)
                    src_params = src_trial_dir / "best_params.json"
                    if not src_params.exists():
                        continue

                    dst_trial_dir = _study_path(target, event, budget, seed, n_trials)
                    dst_params = dst_trial_dir / "best_params.json"

                    if dst_params.exists():
                        stats["skipped"] += 1
                        continue

                    if dry_run:
                        print(f"  [dry-run] Would copy: {event} b={budget} s={seed}")
                        stats["copied"] += 1
                        continue

                    dst_trial_dir.mkdir(parents=True, exist_ok=True)

                    # Copy best_params.json
                    shutil.copy2(str(src_params), str(dst_params))

                    # Copy study.log if it exists
                    src_log = src_trial_dir / "study.log"
                    if src_log.exists():
                        shutil.copy2(str(src_log), str(dst_trial_dir / "study.log"))

                    stats["copied"] += 1

    return stats


def generate_summary(target: Path, n_trials: int) -> dict:
    """Scan target for best_params.json files and write summary_{n}.json.

    Returns the summary dict.
    """
    studies = []
    completed = 0
    failed = 0
    missing = []

    for event in ALL_EVENTS:
        for budget in BUDGETS:
            for seed in SEED_SETS:
                params_path = _study_path(target, event, budget, seed, n_trials) / "best_params.json"

                if params_path.exists():
                    with open(params_path) as f:
                        data = json.load(f)
                    status = data.get("status", "done")
                    studies.append({
                        "event": event,
                        "budget": budget,
                        "seed_set": seed,
                        "status": status,
                        "best_params": data.get("best_params"),
                        "best_value": data.get("best_value"),
                    })
                    if status == "done":
                        completed += 1
                    else:
                        failed += 1
                else:
                    missing.append((event, budget, seed))

    summary = {
        "total_studies": TOTAL_STUDIES,
        "completed": completed,
        "failed": failed,
        "n_trials_per_study": n_trials,
        "search_space": SEARCH_SPACE,
        "studies": studies,
    }

    summary_path = target / f"summary_{n_trials}.json"
    target.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, missing


def print_report(summary: dict, missing: list, merge_stats: dict = None):
    """Print a human-readable report."""
    total = summary["total_studies"]
    completed = summary["completed"]
    failed = summary["failed"]
    found = len(summary["studies"])
    n_missing = len(missing)

    print(f"\n{'=' * 60}")
    print(f"  Optuna Per-Experiment Summary (trials={summary['n_trials_per_study']})")
    print(f"{'=' * 60}")

    if merge_stats:
        print(f"\n  Merge: {merge_stats['copied']} copied, "
              f"{merge_stats['skipped']} skipped (already exist)")

    print(f"\n  Found:     {found} / {total} studies")
    print(f"  Completed: {completed}")
    print(f"  Failed:    {failed}")
    print(f"  Missing:   {n_missing}")

    if found > 0:
        values = [s["best_value"] for s in summary["studies"]
                  if s.get("best_value") is not None]
        if values:
            print(f"\n  Best dev F1: mean={sum(values)/len(values):.4f}, "
                  f"min={min(values):.4f}, max={max(values):.4f}")

    if missing and n_missing <= 30:
        print(f"\n  Missing studies:")
        for event, budget, seed in missing:
            print(f"    {event} b={budget} s={seed}")
    elif missing:
        # Group by event for compact display
        by_event = {}
        for event, budget, seed in missing:
            by_event.setdefault(event, []).append(f"b{budget}s{seed}")
        print(f"\n  Missing studies ({n_missing}):")
        for event in sorted(by_event):
            print(f"    {event}: {', '.join(by_event[event])}")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge and summarize per-experiment Optuna results. "
        "Without --sources, just regenerates summary from existing results."
    )
    parser.add_argument(
        "--target", required=True,
        help="Target directory (e.g., results/bert-base/optuna/per_experiment)",
    )
    parser.add_argument(
        "--n-trials", type=int, required=True,
        help="Trial count to look for (scans trials_{n}/best_params.json)",
    )
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="Source directories to merge from (optional)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be copied without copying (merge mode only)",
    )
    args = parser.parse_args()

    target = Path(args.target)
    merge_stats = None

    # Step 1: Merge if sources provided
    if args.sources:
        action = "Dry-run merge" if args.dry_run else "Merging"
        print(f"{action} from {len(args.sources)} source(s) into {target}...")
        merge_stats = merge_sources(args.sources, target, args.n_trials, args.dry_run)

    if args.dry_run:
        print("\nDry run — no summary written.")
        if merge_stats:
            print(f"Would copy {merge_stats['copied']} studies, "
                  f"skip {merge_stats['skipped']}.")
        return

    # Step 2: Generate summary
    print(f"Scanning {target} for trials_{args.n_trials}/best_params.json...")
    summary, missing = generate_summary(target, args.n_trials)
    summary_path = target / f"summary_{args.n_trials}.json"
    print(f"Summary written to: {summary_path}")

    # Step 3: Report
    print_report(summary, missing, merge_stats)


if __name__ == "__main__":
    main()
