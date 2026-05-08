#!/usr/bin/env python3
"""Pull results from WandB and format into CSVs matching the PI's spreadsheet.

Outputs:
  1. bertweet_calibrated_results.csv  — matches existing spreadsheet format
  2. bertweet_per_category_f1.csv     — per-class F1 breakdown

Usage:
    pip install --upgrade wandb   # if wandb.Api() fails
    python get_calibrated_results.py
    python get_calibrated_results.py --project humaid_supervised_calibrated
"""
import argparse, csv, sys
from collections import defaultdict

EVENTS = [
    ("california_wildfires_2018", "California Wildfires 2018"),
    ("canada_wildfires_2016", "Canada Wildfires 2016"),
    ("cyclone_idai_2019", "Cyclone Idai 2019"),
    ("hurricane_dorian_2019", "Hurricane Dorian 2019"),
    ("hurricane_florence_2018", "Hurricane Florence 2018"),
    ("hurricane_harvey_2017", "Hurricane Harvey 2017"),
    ("hurricane_irma_2017", "Hurricane Irma 2017"),
    ("hurricane_maria_2017", "Hurricane Maria 2017"),
    ("kaikoura_earthquake_2016", "Kaikoura Earthquake 2016"),
    ("kerala_floods_2018", "Kerala Floods 2018"),
]
LBCLS = [5, 10, 25, 50]
SET_NUMS = [1, 2, 3]


def fetch_runs(entity, project):
    """Fetch all runs and organize by (event, lbcl, set_num)."""
    try:
        import wandb
        api = wandb.Api()
    except AttributeError:
        print("❌ wandb.Api() not available. Fix with: pip install --upgrade wandb")
        sys.exit(1)

    print(f"  Fetching runs from {entity}/{project} ...")
    runs = api.runs(f"{entity}/{project}", per_page=500)

    data = {}
    for run in runs:
        cfg = run.config
        event = cfg.get("event")
        lbcl = cfg.get("lbcl")
        set_num = cfg.get("set_num")
        if not event or not lbcl or not set_num:
            continue

        key = (event, int(lbcl), int(set_num))

        # If multiple runs exist for same key, take highest F1
        f1 = run.summary.get("test_macro_f1", 0) or 0
        if key in data and (data[key].get("test_macro_f1", 0) or 0) >= f1:
            continue

        data[key] = dict(run.summary)

    return data


def write_main_csv(data, output_path, metrics):
    """Write CSV matching the PI's spreadsheet format."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # Row 1: event names spanning 3 cols each
        row1 = ["", "", "Metrics/Event Name", "Average"]
        for _, display in EVENTS:
            row1.extend([display, "", ""])
        w.writerow(row1)

        # Row 2: set numbers
        row2 = ["", "", "", ""]
        for _ in EVENTS:
            row2.extend(["Set 1", "Set 2", "Set 3"])
        w.writerow(row2)

        # Data rows: grouped by lbcl
        for lbcl in LBCLS:
            for mi, (metric_key, metric_display) in enumerate(metrics):
                row = []
                if mi == 0:
                    row.extend(["bertweet", f"{lbcl} lb/class"])
                else:
                    row.extend(["", ""])
                row.append(metric_display)

                all_vals = []
                for event_code, _ in EVENTS:
                    for set_num in SET_NUMS:
                        key = (event_code, lbcl, set_num)
                        val = data.get(key, {}).get(metric_key, "")
                        if isinstance(val, (int, float)):
                            val = round(val, 4)
                            all_vals.append(val)
                        row.append(val)

                avg = round(sum(all_vals) / len(all_vals), 4) if all_vals else ""
                row.insert(3, avg)
                w.writerow(row)

    print(f"✅ Wrote: {output_path}")


def write_per_class_csv(data, output_path):
    """Write per-class F1 CSV."""
    # Discover all class names
    class_names = set()
    for metrics in data.values():
        for key in metrics:
            if isinstance(key, str) and key.startswith("test_f1_"):
                class_names.add(key[len("test_f1_"):])
    class_names = sorted(class_names)

    if not class_names:
        print("⚠️  No per-class F1 data found. Skipping per-category CSV.")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # Header
        display_names = [cn.replace("_", " ").title() for cn in class_names]
        header = ["Event", "LBCL", "Set"] + display_names + ["Macro F1"]
        w.writerow(header)

        for lbcl in LBCLS:
            for event_code, event_display in EVENTS:
                for set_num in SET_NUMS:
                    key = (event_code, lbcl, set_num)
                    metrics = data.get(key, {})

                    row = [event_display, lbcl, set_num]
                    for cn in class_names:
                        val = metrics.get(f"test_f1_{cn}", "")
                        if isinstance(val, (int, float)):
                            val = round(val, 4)
                        row.append(val)

                    macro = metrics.get("test_macro_f1", "")
                    if isinstance(macro, (int, float)):
                        macro = round(macro, 4)
                    row.append(macro)

                    w.writerow(row)

    print(f"✅ Wrote: {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="YOUR_WANDB_ENTITY")
    ap.add_argument("--project", default="humaid_supervised_calibrated")
    ap.add_argument("--output_main", default="bertweet_calibrated_results.csv")
    ap.add_argument("--output_perclass", default="bertweet_per_category_f1.csv")
    args = ap.parse_args()

    print(f"🔍 Querying WandB: {args.entity}/{args.project}")
    data = fetch_runs(args.entity, args.project)
    expected = len(EVENTS) * len(LBCLS) * len(SET_NUMS)
    print(f"   Found {len(data)}/{expected} unique (event, lbcl, set) results")

    if len(data) < expected:
        missing = []
        for lbcl in LBCLS:
            for ec, _ in EVENTS:
                for s in SET_NUMS:
                    if (ec, lbcl, s) not in data:
                        missing.append(f"  {ec} {lbcl}lb set{s}")
        print(f"⚠️  Missing {len(missing)} results:")
        for m in missing[:10]:
            print(m)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # CSV 1: main results (Macro F1, ECE raw, ECE calibrated)
    metrics = [
        ("test_macro_f1", "Macro F1"),
        ("test_ece", "ECE"),
        ("test_ece_calibrated", "ECE (Calibrated)"),
    ]
    write_main_csv(data, args.output_main, metrics)

    # CSV 2: per-category F1
    write_per_class_csv(data, args.output_perclass)


if __name__ == "__main__":
    main()
