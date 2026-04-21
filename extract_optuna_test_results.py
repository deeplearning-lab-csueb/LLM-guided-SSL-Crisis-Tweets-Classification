#!/usr/bin/env python
"""Extract best Optuna hyperparameters and their test-set metrics.

Reads per-experiment Optuna results (best_params.json + study.log) and
produces a consolidated JSON with best hyperparameters, dev macro-F1,
and test-set metrics (error rate, macro-F1, ECE) for each experiment.

Test-set metrics are only recorded in study.log files (the objective
function returns dev_macro_f1 only), so this script parses log files
to extract them.

Usage:
    python extract_optuna_test_results.py \\
        --target results/bert-base/optuna/per_experiment --n-trials 10

    python extract_optuna_test_results.py \\
        --target results/bert-base/optuna/per_experiment --n-trials 10 \\
        --output results/bert-base/optuna/per_experiment/best_results.json
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# --- Regex patterns for study.log parsing ---
_RE_TRIAL_START = re.compile(r"--- Trial (\d+)/(\d+) ---")
_RE_TEST_METRICS = re.compile(
    r"Test error rate: ([\d.]+)%, Test macro-F1: ([\d.]+), Test ECE: ([\d.]+)"
)


def parse_study_log(log_path):
    """Parse a study.log file and extract test metrics per trial.

    Args:
        log_path: Path to a study.log file.

    Returns:
        Dict mapping 0-indexed trial number to test metrics dict:
        {trial_num: {"test_error_rate": float, "test_macro_f1": float, "test_ece": float}}
    """
    results = {}
    current_trial = None  # 0-indexed

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = _RE_TRIAL_START.search(line)
            if m:
                current_trial = int(m.group(1)) - 1  # convert to 0-indexed
                continue

            m = _RE_TEST_METRICS.search(line)
            if m and current_trial is not None:
                results[current_trial] = {
                    "test_error_rate": float(m.group(1)),
                    "test_macro_f1": float(m.group(2)),
                    "test_ece": float(m.group(3)),
                }

    return results


def collect_all_test_metrics(experiment_dir):
    """Collect test metrics from all study.log files in an experiment directory.

    Handles continuation chains by parsing all trials_*/study.log files
    found under the experiment directory. Later files take priority on
    conflicts (same trial number in multiple logs).

    Args:
        experiment_dir: Path to an experiment directory
            (e.g. results/bert-base/optuna/per_experiment/california_wildfires_2018/5_set1/)

    Returns:
        Dict mapping 0-indexed trial number to test metrics dict.
    """
    experiment_dir = Path(experiment_dir)
    merged = {}

    # Sort by trial count so later continuations overwrite earlier ones
    log_files = sorted(experiment_dir.glob("trials_*/study.log"))
    for log_path in log_files:
        trial_metrics = parse_study_log(log_path)
        merged.update(trial_metrics)

    return merged


def _find_latest_trials(experiment_dir):
    """Find the highest trials_N subfolder with a best_params.json.

    Returns (n_trials, data_dict) or (None, None) if nothing found.
    """
    experiment_dir = Path(experiment_dir)
    best_n = None
    best_data = None

    for bp_path in experiment_dir.glob("trials_*/best_params.json"):
        folder_name = bp_path.parent.name  # e.g. "trials_10"
        try:
            n = int(folder_name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        with open(bp_path, encoding="utf-8") as f:
            data = json.load(f)
        if best_n is None or n > best_n:
            best_n = n
            best_data = data

    return best_n, best_data


def extract_results(results_dir, n_trials=None):
    """Extract best hyperparameters and test metrics for all experiments.

    Args:
        results_dir: Path to per_experiment results directory.
        n_trials: Trial count to read from. If None, uses the latest
            available for each experiment.

    Returns:
        List of result dicts sorted by (event, budget, seed_set).
    """
    results_dir = Path(results_dir)
    all_results = []

    # Walk: results_dir / {event} / {budget}_set{seed} /
    for event_dir in sorted(results_dir.iterdir()):
        if not event_dir.is_dir():
            continue
        event = event_dir.name

        for exp_dir in sorted(event_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            # Parse directory name: "{budget}_set{seed_set}"
            parts = exp_dir.name.split("_set")
            if len(parts) != 2:
                continue
            try:
                budget = int(parts[0])
                seed_set = int(parts[1])
            except ValueError:
                continue

            # Load best_params.json
            if n_trials is not None:
                bp_path = exp_dir / f"trials_{n_trials}" / "best_params.json"
                if not bp_path.exists():
                    continue
                with open(bp_path, encoding="utf-8") as f:
                    data = json.load(f)
                actual_n_trials = n_trials
            else:
                actual_n_trials, data = _find_latest_trials(exp_dir)
                if data is None:
                    continue

            # Find best trial number (0-indexed) by highest dev_macro_f1
            trials = data.get("trials", [])
            if not trials:
                continue

            best_trial_rec = max(
                (t for t in trials if t.get("state") == "COMPLETE"),
                key=lambda t: t.get("dev_macro_f1", -1),
                default=None,
            )
            if best_trial_rec is None:
                continue

            best_trial_num = best_trial_rec["number"]

            # Collect test metrics from study.log files
            test_metrics_map = collect_all_test_metrics(exp_dir)
            test_metrics = test_metrics_map.get(best_trial_num)

            if test_metrics is None:
                print(
                    f"WARNING: No test metrics found for best trial {best_trial_num} "
                    f"in {event} budget={budget} seed={seed_set}",
                    file=sys.stderr,
                )

            result = {
                "event": event,
                "budget": budget,
                "seed_set": seed_set,
                "n_trials": actual_n_trials,
                "best_trial": best_trial_num,
                "best_params": data.get("best_params", best_trial_rec.get("params")),
                "dev_macro_f1": round(data.get("best_value", best_trial_rec.get("dev_macro_f1", 0)), 6),
                "test_error_rate": test_metrics["test_error_rate"] if test_metrics else None,
                "test_macro_f1": test_metrics["test_macro_f1"] if test_metrics else None,
                "test_ece": test_metrics["test_ece"] if test_metrics else None,
            }
            all_results.append(result)

    return all_results


def save_results(results, output_path):
    """Save results list to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} experiment results to {output_path}")


def print_summary(results):
    """Print a CLI summary of the extracted results."""
    if not results:
        print("No results found.")
        return

    total = len(results)
    with_test = sum(1 for r in results if r["test_macro_f1"] is not None)
    missing_test = total - with_test

    print(f"\n{'='*60}")
    print(f"Optuna Best Results Summary")
    print(f"{'='*60}")
    print(f"Total experiments: {total}")
    print(f"With test metrics: {with_test}")
    if missing_test:
        print(f"Missing test metrics: {missing_test}")

    # Overall stats
    test_f1s = [r["test_macro_f1"] for r in results if r["test_macro_f1"] is not None]
    dev_f1s = [r["dev_macro_f1"] for r in results if r["dev_macro_f1"] is not None]

    if test_f1s:
        print(f"\nTest macro-F1:  mean={statistics.mean(test_f1s):.4f}, "
              f"std={statistics.stdev(test_f1s):.4f}, "
              f"min={min(test_f1s):.4f}, max={max(test_f1s):.4f}")
    if dev_f1s:
        print(f"Dev macro-F1:   mean={statistics.mean(dev_f1s):.4f}, "
              f"std={statistics.stdev(dev_f1s):.4f}, "
              f"min={min(dev_f1s):.4f}, max={max(dev_f1s):.4f}")

    # By event
    events = sorted(set(r["event"] for r in results))
    print(f"\n{'Event':<35} {'Test F1':>8} {'Dev F1':>8} {'Count':>6}")
    print("-" * 60)
    for event in events:
        ev_results = [r for r in results if r["event"] == event and r["test_macro_f1"] is not None]
        if ev_results:
            ev_test = [r["test_macro_f1"] for r in ev_results]
            ev_dev = [r["dev_macro_f1"] for r in ev_results]
            print(f"{event:<35} {statistics.mean(ev_test):>8.4f} "
                  f"{statistics.mean(ev_dev):>8.4f} {len(ev_results):>6}")

    # By budget
    budgets = sorted(set(r["budget"] for r in results))
    print(f"\n{'Budget':<35} {'Test F1':>8} {'Dev F1':>8} {'Count':>6}")
    print("-" * 60)
    for budget in budgets:
        b_results = [r for r in results if r["budget"] == budget and r["test_macro_f1"] is not None]
        if b_results:
            b_test = [r["test_macro_f1"] for r in b_results]
            b_dev = [r["dev_macro_f1"] for r in b_results]
            print(f"{budget:<35} {statistics.mean(b_test):>8.4f} "
                  f"{statistics.mean(b_dev):>8.4f} {len(b_results):>6}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract best Optuna hyperparameters and test-set metrics"
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Path to per_experiment results directory",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Trial count to read from (default: latest available)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="JSON output path (default: {target}/best_results_{n_trials}.json)",
    )
    args = parser.parse_args()

    results = extract_results(args.target, n_trials=args.n_trials)
    print_summary(results)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        n_label = args.n_trials if args.n_trials else "latest"
        output_path = str(Path(args.target) / f"best_results_{n_label}.json")

    save_results(results, output_path)


if __name__ == "__main__":
    main()
