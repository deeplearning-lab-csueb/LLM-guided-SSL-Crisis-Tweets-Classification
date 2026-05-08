#!/usr/bin/env python3
"""Get per-category F1 for the 2 fixed runs (most recent by config match)."""
import os, sys, csv, tempfile

TARGETS = [
    {"event": "hurricane_maria_2017", "lbcl": 25, "set_num": 3},
    {"event": "hurricane_dorian_2019", "lbcl": 50, "set_num": 3},
]

def main():
    import wandb
    from sklearn.metrics import precision_recall_fscore_support
    api = wandb.Api()

    project = "YOUR_WANDB_ENTITY/humaid_supervised_hpo"
    print(f"🔍 Fetching runs from {project} ...")
    runs = list(api.runs(project, per_page=500))
    print(f"   Total runs: {len(runs)}")

    for target in TARGETS:
        ev, lb, sn = target["event"], target["lbcl"], target["set_num"]
        print(f"\n{'='*60}")
        print(f"  {ev}  |  {lb} lb/class  |  Set {sn}")
        print(f"{'='*60}")

        # Find ALL matching runs, pick the most recent one
        matches = []
        for r in runs:
            cfg = r.config
            if (cfg.get("event") == ev and
                int(cfg.get("lbcl", 0)) == lb and
                int(cfg.get("set_num", 0)) == sn):
                matches.append(r)

        if not matches:
            print("  ❌ No matching runs found.")
            continue

        # Sort by creation time, newest first
        matches.sort(key=lambda r: r.created_at, reverse=True)
        run = matches[0]
        print(f"  Run: {run.name} (id: {run.id})")
        print(f"  Created: {run.created_at}")
        print(f"  test_macro_f1: {run.summary.get('test_macro_f1', 'N/A')}")

        # Download prediction artifact
        artifacts = list(run.logged_artifacts())
        pred_art = next((a for a in artifacts if a.type == "predictions"), None)
        if not pred_art:
            print("  ⚠️  No prediction artifact yet (run might still be going).")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_art.download(root=tmpdir)
            str_csv = None
            for f in os.listdir(tmpdir):
                if "str" in f and f.endswith(".csv"):
                    str_csv = os.path.join(tmpdir, f)
                    break
            if not str_csv:
                print("  ⚠️  No pred_bert_str.csv in artifact.")
                continue

            gold, pred = [], []
            with open(str_csv) as f:
                for row in csv.DictReader(f):
                    gold.append(row["gold_name"])
                    pred.append(row["pred_name"])

            classes = sorted(set(gold) | set(pred))
            _, _, f1s, _ = precision_recall_fscore_support(
                gold, pred, labels=classes, average=None, zero_division=0
            )
            f1_dict = {c: round(float(v), 4) for c, v in zip(classes, f1s)}

            # Print copy-pasteable CSV row
            all_classes = [
                "caution_and_advice", "displaced_people_and_evacuations",
                "infrastructure_and_utility_damage", "injured_or_dead_people",
                "missing_or_found_people", "not_humanitarian",
                "other_relevant_information", "requests_or_urgent_needs",
                "rescue_volunteering_or_donation_effort", "sympathy_and_support",
            ]
            macro = round(float(run.summary.get("test_macro_f1", 0)), 4)
            vals = [str(f1_dict.get(c, "")) for c in all_classes] + [str(macro)]

            print(f"\n  📋 Copy-paste this row into your spreadsheet:")
            print(f"  {','.join(vals)}")


if __name__ == "__main__":
    main()
