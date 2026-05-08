import argparse
import concurrent.futures
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone

import pandas as pd
import wandb


EVENTS = [
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
TARGET_LBCL = 5
SETS = [1, 2, 3]


EXPECTED_SWEEP_NAMES = {
    f"aum_mixup_{event}_{TARGET_LBCL}lbcl_set{set_num}" for event in EVENTS for set_num in SETS
}

print(f"Targeting {len(EXPECTED_SWEEP_NAMES)} expected 5lb/class sweeps.")

print_lock = threading.Lock()


def parse_iso_datetime(raw_value):
    if not raw_value:
        return None
    if isinstance(raw_value, datetime):
        dt = raw_value
    else:
        txt = str(raw_value).strip()
        if not txt:
            return None
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(txt)
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_run_datetime(run):
    # Prefer the latest meaningful timestamp if available.
    for attr_name in ("updated_at", "heartbeatAt", "created_at"):
        dt = parse_iso_datetime(getattr(run, attr_name, None))
        if dt is not None:
            return dt

    summary_ts = None
    try:
        summary_ts = run.summary.get("_timestamp")
    except Exception:
        summary_ts = None

    if summary_ts is not None:
        try:
            return datetime.fromtimestamp(float(summary_ts), tz=timezone.utc)
        except Exception:
            pass

    return datetime.fromtimestamp(0, tz=timezone.utc)


def parse_metrics_from_log(run):
    """
    Downloads output.log and parses:
    1) Best Dev F1 (fallback only)
    2) Test F1 and ECE from final JSON.
    Returns (dev_f1, test_f1, test_ece).
    """
    download_root = f"temp_logs/{run.id}"
    try:
        try:
            files = run.files()
            has_log = any(f.name == "output.log" for f in files)
            if not has_log:
                return None, None, None
        except Exception:
            return None, None, None

        os.makedirs(download_root, exist_ok=True)
        run.file("output.log").download(replace=True, root=download_root)
        log_path = os.path.join(download_root, "output.log")

        dev_f1 = None
        test_f1 = None
        test_ece = None

        if not os.path.exists(log_path):
            return None, None, None

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        dev_matches = list(re.finditer(r"New best macro validation\s+([\d\.]+)", content))
        if dev_matches:
            dev_f1 = max(float(m.group(1)) for m in dev_matches)

        matches = list(re.finditer(r'\{\s+"Temperature Scaling"', content))
        if matches:
            start_idx = matches[-1].start()
            potential_json = content[start_idx:]

            balance = 0
            end_idx = -1
            for i, char in enumerate(potential_json):
                if char == "{":
                    balance += 1
                elif char == "}":
                    balance -= 1
                    if balance == 0:
                        end_idx = i + 1
                        break

            if end_idx != -1:
                json_str = potential_json[:end_idx]
                try:
                    data = json.loads(json_str)
                    model_data = data.get("Best ST+AumMixup model")
                    if model_data:
                        f1 = model_data.get("F1 after temp scaling") or model_data.get("F1 before temp scaling")
                        ece_raw = model_data.get("ECE after temp scaling") or model_data.get("ECE before temp scaling")

                        if f1 is not None:
                            test_f1 = float(f1)
                        if ece_raw is not None:
                            if "tensor" in str(ece_raw):
                                m = re.search(r"tensor\(([\d\.]+)", str(ece_raw))
                                if m:
                                    test_ece = float(m.group(1))
                            else:
                                try:
                                    test_ece = float(ece_raw)
                                except Exception:
                                    pass
                except Exception:
                    pass

        return dev_f1, test_f1, test_ece
    except Exception:
        return None, None, None
    finally:
        if os.path.exists(download_root):
            try:
                shutil.rmtree(download_root)
            except Exception:
                pass


def prettify_event(event_str):
    if not event_str:
        return "Unknown"
    return event_str.replace("_", " ").title()


def extract_metrics_for_run(run):
    test_f1 = run.summary.get("test_macro-F1")
    test_ece = run.summary.get("test_ece")
    dev = run.summary.get("dev_macro-F1") or run.summary.get("dev_f1")

    p_dev, p_test, p_ece = parse_metrics_from_log(run)
    if dev is None and p_dev is not None:
        dev = p_dev
    if test_f1 is None and p_test is not None:
        test_f1 = p_test
    if test_ece is None and p_ece is not None:
        test_ece = p_ece

    if dev is not None:
        try:
            dev = float(dev)
        except Exception:
            dev = None

    try:
        test_f1 = float(test_f1) if test_f1 is not None else 0.0
    except Exception:
        test_f1 = 0.0

    try:
        test_ece = float(test_ece) if test_ece is not None else 0.0
    except Exception:
        test_ece = 0.0

    return dev, test_f1, test_ece


def process_sweep_recent(sweep):
    m = re.match(r"(?:aum_mixup_)?(.+)_(\d+)lbcl_set(\d+)", sweep.name or "")
    if not m:
        return None

    event, lbcl, set_num = m.groups()
    lbcl = int(lbcl)
    set_num = int(set_num)

    if lbcl != TARGET_LBCL:
        return None

    runs = [r for r in sweep.runs if r.state in ("finished", "crashed", "failed")]
    if not runs:
        return None

    latest_run = max(runs, key=get_run_datetime)
    run_dt = get_run_datetime(latest_run)
    dev_f1, test_f1, test_ece = extract_metrics_for_run(latest_run)

    with print_lock:
        print(
            f"Processed Auto-Sweep {sweep.name}: "
            f"latest run={latest_run.id} at {run_dt.isoformat()}"
        )

    return {
        "source": "sweep",
        "source_id": sweep.id,
        "run_id": latest_run.id,
        "event": event,
        "lbcl": lbcl,
        "set_num": set_num,
        "dev_f1": dev_f1 if dev_f1 is not None else 0.0,
        "test_f1": test_f1,
        "test_ece": test_ece,
        "run_time": run_dt.isoformat(),
        "run_time_sort": run_dt.timestamp(),
    }


def process_direct_run(run):
    m = re.match(r"(.+)_(\d+)lb_set(\d+)_aum_mixup", run.name or "")
    if not m:
        return None

    event, lbcl, set_num = m.groups()
    lbcl = int(lbcl)
    set_num = int(set_num)

    if lbcl != TARGET_LBCL:
        return None
    if run.state not in ("finished", "crashed", "failed"):
        return None

    run_dt = get_run_datetime(run)
    dev_f1, test_f1, test_ece = extract_metrics_for_run(run)

    with print_lock:
        print(f"Processed Direct Run {run.name}: at {run_dt.isoformat()}")

    return {
        "source": "run",
        "source_id": run.id,
        "run_id": run.id,
        "event": event,
        "lbcl": lbcl,
        "set_num": set_num,
        "dev_f1": dev_f1 if dev_f1 is not None else 0.0,
        "test_f1": test_f1,
        "test_ece": test_ece,
        "run_time": run_dt.isoformat(),
        "run_time_sort": run_dt.timestamp(),
    }


def fetch_recent_5lb_results(project_name, entity=None):
    print(f"Fetching data from WandB project: {project_name}...")
    api = wandb.Api(timeout=60)

    try:
        project = api.project(project_name, entity=entity)
        sweeps = list(project.sweeps())

        if entity:
            runs_path = f"{entity}/{project_name}"
            runs = list(api.runs(runs_path))
        else:
            try:
                runs = list(api.runs(project_name))
            except Exception:
                runs_path = f"{project.entity}/{project.name}"
                runs = list(api.runs(runs_path))
    except Exception as e:
        print(f"Error accessing WandB project '{project_name}': {e}")
        return []

    data = []

    valid_sweeps = [
        s
        for s in sweeps
        if s.name in EXPECTED_SWEEP_NAMES
        or (s.name and f"_{TARGET_LBCL}lbcl_set" in s.name and "aum_mixup_" in s.name)
    ]
    valid_runs = [
        r
        for r in runs
        if r.name and re.match(rf".+_{TARGET_LBCL}lb_set\d+_aum_mixup", r.name)
    ]

    print(f"Found {len(sweeps)} sweeps. Processing {len(valid_sweeps)} valid 5lb sweeps...")
    print(f"Found {len(runs)} individual runs. Processing {len(valid_runs)} direct 5lb reruns...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_sweep_recent, s): s for s in valid_sweeps}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    data.append(result)
            except Exception:
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_direct_run, r): r for r in valid_runs}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    data.append(result)
            except Exception:
                pass

    print(f"Collected {len(data)} 5lb candidates before deduplication.")
    return data


def format_csv(data, output_path):
    if not data:
        print("No data to save.")
        return

    df = pd.DataFrame(data)
    df["Event Name"] = df["event"].apply(prettify_event)
    df["Set"] = "Set " + df["set_num"].astype(str)

    # Keep only the most recent run per (event, lbcl, set).
    df = df.sort_values(by="run_time_sort", ascending=False)
    df = df.drop_duplicates(subset=["event", "lbcl", "set_num"], keep="first")

    pivot_f1 = df.pivot_table(index="lbcl", columns=["Event Name", "Set"], values="test_f1")
    pivot_ece = df.pivot_table(index="lbcl", columns=["Event Name", "Set"], values="test_ece")

    all_events = sorted(df["Event Name"].unique())
    all_sets = ["Set 1", "Set 2", "Set 3"]

    final_rows = []

    header_events = ["", "", "Metrics/Event Name", "Average"]
    header_sets = ["", "", "", ""]

    for event in all_events:
        header_events.append(event)
        header_events.extend([""] * (len(all_sets) - 1))
        for set_name in all_sets:
            header_sets.append(set_name)

    # Keep the same structure as the original summary file.
    for lb in sorted(df["lbcl"].unique()):
        row_f1 = ["AUM-ST-Mixup", f"{lb} lb/class", "Macro F1", ""]
        f1_vals = []
        for event in all_events:
            for set_name in all_sets:
                try:
                    val = pivot_f1.loc[lb, (event, set_name)]
                    row_f1.append(f"{val:.4f}")
                    f1_vals.append(val)
                except KeyError:
                    row_f1.append("")
        if f1_vals:
            row_f1[3] = f"{sum(f1_vals) / len(f1_vals):.4f}"
        final_rows.append(row_f1)

        row_ece = ["", "", "ECE", ""]
        ece_vals = []
        for event in all_events:
            for set_name in all_sets:
                try:
                    val = pivot_ece.loc[lb, (event, set_name)]
                    row_ece.append(f"{val:.4f}")
                    ece_vals.append(val)
                except KeyError:
                    row_ece.append("")
        if ece_vals:
            row_ece[3] = f"{sum(ece_vals) / len(ece_vals):.4f}"
        final_rows.append(row_ece)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(header_events) + "\n")
        f.write(",".join(header_sets) + "\n")
        for row in final_rows:
            f.write(",".join(row) + "\n")

    print(f"Saved recent 5lb/class summary CSV to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--entity", required=True, help="WandB entity")
    parser.add_argument(
        "--output",
        default="aum_mixup_recent_5lb_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    data = fetch_recent_5lb_results(args.project, args.entity)
    format_csv(data, args.output)


if __name__ == "__main__":
    main()
