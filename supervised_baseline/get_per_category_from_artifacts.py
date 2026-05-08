#!/usr/bin/env python3
"""Extract per-category F1 from ORIGINAL WandB prediction artifacts.

Downloads pred_bert_str.csv artifacts from the original humaid_supervised_hpo
project and computes per-class F1 — 100% consistent with the paper's reported F1.

Usage:
    pip install "wandb<0.17"    # if needed
    python get_per_category_from_artifacts.py
"""
import argparse, csv, os, sys, tempfile
from collections import defaultdict

EVENTS_DISPLAY = {
    "california_wildfires_2018": "California Wildfires 2018",
    "canada_wildfires_2016": "Canada Wildfires 2016",
    "cyclone_idai_2019": "Cyclone Idai 2019",
    "hurricane_dorian_2019": "Hurricane Dorian 2019",
    "hurricane_florence_2018": "Hurricane Florence 2018",
    "hurricane_harvey_2017": "Hurricane Harvey 2017",
    "hurricane_irma_2017": "Hurricane Irma 2017",
    "hurricane_maria_2017": "Hurricane Maria 2017",
    "kaikoura_earthquake_2016": "Kaikoura Earthquake 2016",
    "kerala_floods_2018": "Kerala Floods 2018",
}
EVENTS = list(EVENTS_DISPLAY.keys())
LBCLS = [5, 10, 25, 50]
SET_NUMS = [1, 2, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="YOUR_WANDB_ENTITY")
    ap.add_argument("--project", default="humaid_supervised_hpo")
    ap.add_argument("--output", default="bertweet_original_per_category_f1.csv")
    args = ap.parse_args()

    try:
        import wandb
        api = wandb.Api()
    except (AttributeError, ImportError):
        print("❌ wandb.Api() not available. Fix: pip install 'wandb<0.17'")
        sys.exit(1)

    from sklearn.metrics import precision_recall_fscore_support

    print(f"🔍 Fetching runs from {args.entity}/{args.project} ...")
    runs = api.runs(f"{args.entity}/{args.project}", per_page=1000)

    # Step 1: Find best run per (event, lbcl, set) by test_macro_f1
    best_runs = {}
    total = 0
    for run in runs:
        total += 1
        cfg = run.config
        event = cfg.get("event")
        lbcl = cfg.get("lbcl")
        set_num = cfg.get("set_num")
        if not event or not lbcl or not set_num:
            continue

        # Pick the best run based on validation set performance (eval_f1)
        # This properly simulates realistic HPO selection without test-set leakage.
        eval_f1 = run.summary.get("eval_f1", 0) or 0
        key = (event, int(lbcl), int(set_num))

        if key not in best_runs or eval_f1 > best_runs[key]["eval_f1"]:
            best_runs[key] = {
                "eval_f1": eval_f1, 
                "test_macro_f1": run.summary.get("test_macro_f1", 0),
                "run_id": run.id, 
                "run": run
            }

    print(f"  Scanned {total} runs, found {len(best_runs)} unique (event, lbcl, set) combos")

    # Step 2: Download prediction artifacts and compute per-class F1
    results = {}
    all_class_names = set()

    for key, info in sorted(best_runs.items()):
        event, lbcl, set_num = key
        run = info["run"]

        # Find prediction artifact
        try:
            artifacts = list(run.logged_artifacts())
            pred_artifact = None
            for art in artifacts:
                if art.type == "predictions":
                    pred_artifact = art
                    break

            if pred_artifact is None:
                print(f"  ⚠️  No prediction artifact for {event} {lbcl}lb s{set_num}")
                continue

            # Download to temp dir
            with tempfile.TemporaryDirectory() as tmpdir:
                pred_artifact.download(root=tmpdir)

                # Find pred_bert_str.csv
                str_csv = os.path.join(tmpdir, "pred_bert_str.csv")
                if not os.path.exists(str_csv):
                    # Try finding it
                    for f in os.listdir(tmpdir):
                        if "str" in f and f.endswith(".csv"):
                            str_csv = os.path.join(tmpdir, f)
                            break

                if not os.path.exists(str_csv):
                    print(f"  ⚠️  No pred_bert_str.csv for {event} {lbcl}lb s{set_num}")
                    continue

                # Read predictions
                gold_labels = []
                pred_labels = []
                with open(str_csv, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        gold_labels.append(row["gold_name"])
                        pred_labels.append(row["pred_name"])

                # Get unique classes
                classes = sorted(set(gold_labels) | set(pred_labels))
                all_class_names.update(classes)

                # Compute per-class F1
                _, _, f1_scores, supports = precision_recall_fscore_support(
                    gold_labels, pred_labels, labels=classes, average=None, zero_division=0
                )

                per_class = {}
                for cls, f1_val, sup in zip(classes, f1_scores, supports):
                    per_class[cls] = round(float(f1_val), 4)

                macro_f1 = round(sum(f1_scores) / len(f1_scores), 4) if len(f1_scores) > 0 else 0
                results[key] = {"per_class": per_class, "macro_f1": macro_f1}
                print(f"  ✅ {event} {lbcl}lb s{set_num}: macro_f1={macro_f1:.4f} ({len(classes)} classes)")

        except Exception as e:
            print(f"  ❌ Error for {event} {lbcl}lb s{set_num}: {e}")
            continue

    # Step 3: Write CSV
    all_class_names = sorted(all_class_names)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Event", "LBCL", "Set"] + \
                 [cn.replace("_", " ").title() for cn in all_class_names] + \
                 ["Macro F1"]
        w.writerow(header)

        for lbcl in LBCLS:
            for event in EVENTS:
                for set_num in SET_NUMS:
                    key = (event, lbcl, set_num)
                    row = [EVENTS_DISPLAY.get(event, event), lbcl, set_num]

                    if key in results:
                        for cn in all_class_names:
                            row.append(results[key]["per_class"].get(cn, ""))
                        row.append(results[key]["macro_f1"])
                    else:
                        row.extend([""] * (len(all_class_names) + 1))

                    w.writerow(row)

    print(f"\n✅ Wrote: {args.output}")
    print(f"   {len(results)}/{len(EVENTS) * len(LBCLS) * len(SET_NUMS)} experiments recovered")


if __name__ == "__main__":
    main()
