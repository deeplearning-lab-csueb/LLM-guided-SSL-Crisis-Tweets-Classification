#!/usr/bin/env python3
"""Standalone progress checker for per-experiment Optuna studies.

Scans study.log files and reports trial-level progress with ETA.
Read-only — does not modify any files or import from lg_cotrain.

Usage:
    python check_progress.py
    python check_progress.py --results-dir /path/to/results
    python check_progress.py --watch          # refresh every 30s
    python check_progress.py --watch --interval 10
    python check_progress.py --num-gpus 2     # override GPU count for ETA
    python check_progress.py --num-events 2   # running on 2 events (2x4x3=24 studies)
"""

import argparse
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

# Display width — adapts to terminal, minimum 90 for readability
WIDTH = max(shutil.get_terminal_size((100, 24)).columns, 90)


def parse_timestamp(line: str):
    """Extract datetime from a log line."""
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+", line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return None


def parse_study_log(log_path: str) -> dict:
    """Parse a single study.log and return progress info."""
    info = {
        "path": log_path,
        "event": None,
        "budget": None,
        "seed": None,
        "target_trials": 0,
        "completed_trials": 0,
        "trial_durations": [],      # seconds per completed trial
        "best_f1": 0.0,
        "current_phase": None,      # "Phase 1", "Phase 2", "Phase 3", "done"
        "current_trial": 0,
        "current_epoch": None,
        "study_start": None,
        "last_timestamp": None,
        "last_line": "",
        "failed": False,
    }

    trial_start_time = None

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue

                ts = parse_timestamp(line)
                if ts:
                    info["last_timestamp"] = ts
                    if info["study_start"] is None:
                        info["study_start"] = ts

                info["last_line"] = line

                # Study header
                m = re.search(
                    r"Optuna study: (\S+) budget=(\d+) seed=(\d+) \| target=(\d+) trials",
                    line,
                )
                if m:
                    info["event"] = m.group(1)
                    info["budget"] = int(m.group(2))
                    info["seed"] = int(m.group(3))
                    info["target_trials"] = int(m.group(4))

                # Trial start
                m = re.search(r"--- Trial (\d+)/(\d+) ---$", line)
                if m:
                    info["current_trial"] = int(m.group(1))
                    info["target_trials"] = int(m.group(2))
                    trial_start_time = ts

                # Trial done
                m = re.search(
                    r"--- Trial (\d+)/(\d+) done: dev_macro_f1=([\d.]+)",
                    line,
                )
                if m:
                    info["completed_trials"] = int(m.group(1))
                    f1 = float(m.group(3))
                    if f1 > info["best_f1"]:
                        info["best_f1"] = f1
                    if trial_start_time and ts:
                        dur = (ts - trial_start_time).total_seconds()
                        info["trial_durations"].append(dur)
                    trial_start_time = None

                # Phase tracking
                if "=== Phase 1:" in line:
                    info["current_phase"] = "Phase 1"
                    info["current_epoch"] = None
                elif "=== Phase 2:" in line:
                    info["current_phase"] = "Phase 2"
                    info["current_epoch"] = None
                elif "=== Phase 3:" in line:
                    info["current_phase"] = "Phase 3"
                    info["current_epoch"] = None
                elif "=== Final Evaluation ===" in line:
                    info["current_phase"] = "Eval"

                # Epoch tracking
                m = re.search(r"Phase [123] epoch (\d+)/(\d+)", line)
                if m:
                    info["current_epoch"] = f"{m.group(1)}/{m.group(2)}"

                # Error detection
                if "failed:" in line or "ERROR" in line:
                    info["failed"] = True

    except Exception:
        pass

    # If all trials completed, mark phase as done
    if info["completed_trials"] >= info["target_trials"] and info["target_trials"] > 0:
        info["current_phase"] = "done"

    return info


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m"


def detect_gpu_info() -> list:
    """Detect GPU names and memory via nvidia-smi. Returns list of dicts."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "mem_total_mb": int(parts[2]),
                        "mem_used_mb": int(parts[3]),
                        "util_pct": int(parts[4]),
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return gpus


def find_study_logs(results_dir: str) -> list:
    """Find all study.log files under the per-experiment results directory."""
    base = Path(results_dir) / "optuna" / "per_experiment"
    logs = []
    if base.exists():
        for log_path in sorted(base.rglob("study.log")):
            logs.append(str(log_path))
    return logs


def print_progress(results_dir: str, num_gpus_override: int = None,
                    num_events: int = 10, num_budgets: int = 4,
                    num_seeds: int = 3):
    """Parse all study logs and print a progress report."""
    logs = find_study_logs(results_dir)
    w = WIDTH

    if not logs:
        print(f"No study.log files found under {results_dir}/optuna/per_experiment/")
        print("Studies may not have started yet.")
        return

    studies = [parse_study_log(p) for p in logs]

    # Separate into categories
    completed = [s for s in studies if s["current_phase"] == "done"]
    in_progress = [s for s in studies if s["current_phase"] not in ("done", None) and not s["failed"]]
    failed = [s for s in studies if s["failed"]]

    # Aggregate stats
    total_trials_done = sum(s["completed_trials"] for s in studies)
    all_durations = []
    for s in studies:
        all_durations.extend(s["trial_durations"])

    # Compute target trials per study (assume uniform)
    total_studies = num_events * num_budgets * num_seeds
    trials_per_study = studies[0]["target_trials"] if studies else 10
    total_expected_trials = total_studies * trials_per_study
    not_started = total_studies - len(studies)

    # GPU info
    gpus = detect_gpu_info()

    # Header
    now = datetime.now()
    print(f"\n{'=' * w}")
    print(f"  Optuna Per-Experiment Progress  |  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * w}")

    # Device info
    if gpus:
        print(f"\n  Devices:")
        for g in gpus:
            mem_pct = g["mem_used_mb"] / g["mem_total_mb"] * 100 if g["mem_total_mb"] > 0 else 0
            print(f"    cuda:{g['index']}  {g['name']:<30}  "
                  f"mem {g['mem_used_mb']:,}/{g['mem_total_mb']:,} MB ({mem_pct:.0f}%)  "
                  f"util {g['util_pct']}%")
    else:
        print(f"\n  Devices:  (nvidia-smi not available)")

    # Overall summary
    print(f"\n  Studies:  {len(completed)} done, {len(in_progress)} running, "
          f"{len(failed)} failed, {not_started} not started  (of {total_studies})")
    print(f"  Trials:   {total_trials_done} / {total_expected_trials} completed")

    if total_expected_trials > 0:
        pct = total_trials_done / total_expected_trials * 100
        bar_len = 40
        filled = int(bar_len * total_trials_done / total_expected_trials)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"  Progress: [{bar}] {pct:.1f}%")

    # Timing
    if all_durations:
        avg_trial = sum(all_durations) / len(all_durations)
        min_trial = min(all_durations)
        max_trial = max(all_durations)
        print(f"\n  Trial duration:  avg={format_duration(avg_trial)}, "
              f"min={format_duration(min_trial)}, max={format_duration(max_trial)}")

        remaining_trials = total_expected_trials - total_trials_done
        if remaining_trials > 0:
            # Parallelism: use override, or detected GPU count, or concurrent studies
            if num_gpus_override:
                n_parallel = num_gpus_override
            elif gpus:
                n_parallel = len(gpus)
            else:
                n_parallel = max(len(in_progress), 1)
            # Sequential time for remaining trials
            seq_seconds = remaining_trials * avg_trial
            # Parallel estimate: divide by concurrent workers
            par_seconds = seq_seconds / n_parallel
            eta = now + timedelta(seconds=par_seconds)
            print(f"  Remaining:       {remaining_trials} trials "
                  f"(~{format_duration(par_seconds)} with {n_parallel} parallel)")
            print(f"  Estimated done:  {eta.strftime('%Y-%m-%d %H:%M')}")

    # In-progress studies detail
    if in_progress:
        print(f"\n  {'─' * (w - 4)}")
        print(f"  RUNNING ({len(in_progress)} studies):")
        print(f"  {'─' * (w - 4)}")
        for s in in_progress:
            name = f"{s['event']} b={s['budget']} s={s['seed']}"
            trial_str = f"trial {s['current_trial']}/{s['target_trials']}"
            phase_str = s["current_phase"] or "?"
            epoch_str = f" ep {s['current_epoch']}" if s["current_epoch"] else ""

            # Total elapsed time for the study
            elapsed = ""
            if s["last_timestamp"] and s["study_start"]:
                total_elapsed = (s["last_timestamp"] - s["study_start"]).total_seconds()
                elapsed = f" [{format_duration(total_elapsed)} elapsed]"

            print(f"    {name:<45} {trial_str}  {phase_str}{epoch_str}{elapsed}")

    # Completed studies - compact summary
    if completed:
        print(f"\n  {'─' * (w - 4)}")
        print(f"  COMPLETED ({len(completed)} studies):")
        print(f"  {'─' * (w - 4)}")

        # Group by event for readability
        by_event = {}
        for s in completed:
            by_event.setdefault(s["event"], []).append(s)

        for event in sorted(by_event):
            entries = sorted(by_event[event], key=lambda x: (x["budget"], x["seed"]))
            avg_f1 = sum(s["best_f1"] for s in entries) / len(entries)
            print(f"    {event:<40} [{len(entries)}] avg_best_f1={avg_f1:.4f}")

    # Failed studies
    if failed:
        print(f"\n  {'─' * (w - 4)}")
        print(f"  FAILED ({len(failed)} studies):")
        print(f"  {'─' * (w - 4)}")
        for s in failed:
            name = f"{s['event']} b={s['budget']} s={s['seed']}"
            print(f"    {name}")

    print(f"\n{'=' * w}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check progress of per-experiment Optuna studies"
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="Path to results directory (default: ./results)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh progress",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Refresh interval in seconds for --watch (default: 30)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Override GPU count for ETA calculation (default: auto-detect via nvidia-smi)",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=10,
        help="Number of disaster events being tuned (default: 10)",
    )
    parser.add_argument(
        "--num-budgets",
        type=int,
        default=4,
        help="Number of budget levels being tuned (default: 4)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of seed sets being tuned (default: 3)",
    )
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                print_progress(args.results_dir, args.num_gpus,
                               args.num_events, args.num_budgets, args.num_seeds)
                print(f"  (refreshing every {args.interval}s — Ctrl+C to stop)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_progress(args.results_dir, args.num_gpus,
                       args.num_events, args.num_budgets, args.num_seeds)


if __name__ == "__main__":
    main()
