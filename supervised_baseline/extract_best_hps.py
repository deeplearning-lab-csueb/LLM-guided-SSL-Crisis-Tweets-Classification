#!/usr/bin/env python3
"""Extract best hyperparameters per (event, lbcl, set) from WandB sweeps.

Picks the best HP config (by F1) for each (event, lbcl, set) combo,
then generates rerun_jobs.txt with per-set optimized HPs.

Usage:
    python extract_best_hps.py                     # query WandB API
    python extract_best_hps.py --use_defaults      # skip API, use safe defaults
"""
import argparse, json

EVENTS = [
    "california_wildfires_2018", "canada_wildfires_2016", "cyclone_idai_2019",
    "hurricane_dorian_2019", "hurricane_florence_2018", "hurricane_harvey_2017",
    "hurricane_irma_2017", "hurricane_maria_2017", "kaikoura_earthquake_2016",
    "kerala_floods_2018",
]
LBCLS = [5, 10, 25, 50]
SET_NUMS = [1, 2, 3]

# Reasonable defaults for BERTweet fine-tuning if WandB is unavailable
DEFAULT_HP = {"learning_rate": 2e-5, "epochs": 15, "batch_size": 16}


def extract_from_wandb(entity, project):
    """Query WandB API and return best HP per (event, lbcl, set) by highest F1."""
    import wandb
    api = wandb.Api()

    print(f"  Fetching runs from {entity}/{project} ...")
    runs = api.runs(f"{entity}/{project}", per_page=500)

    best = {}
    total_runs = 0
    for run in runs:
        total_runs += 1
        cfg = run.config
        event = cfg.get("event")
        lbcl = cfg.get("lbcl")
        set_num = cfg.get("set_num")
        if not event or not lbcl or not set_num:
            continue

        f1 = run.summary.get("test_macro_f1", run.summary.get("eval_f1", 0))
        if f1 is None or f1 == 0:
            continue

        key = f"{event}_{lbcl}_s{set_num}"
        if key not in best or f1 > best[key]["f1"]:
            best[key] = {
                "f1": float(f1),
                "learning_rate": float(cfg.get("learning_rate", DEFAULT_HP["learning_rate"])),
                "epochs": int(cfg.get("epochs", DEFAULT_HP["epochs"])),
                "batch_size": int(cfg.get("batch_size", DEFAULT_HP["batch_size"])),
            }

    print(f"  Scanned {total_runs} runs")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="YOUR_WANDB_ENTITY")
    ap.add_argument("--project", default="humaid_supervised_hpo")
    ap.add_argument("--use_defaults", action="store_true",
                    help="Skip WandB; use default HPs (lr=2e-5, ep=15, bs=16)")
    ap.add_argument("--output", default="rerun_jobs.txt")
    ap.add_argument("--hp_json", default="best_hps.json")
    args = ap.parse_args()

    if args.use_defaults:
        print("⚡ Using default HPs for all combos (no WandB query)")
        best = {}
    else:
        print(f"🔍 Querying WandB: {args.entity}/{args.project}")
        best = extract_from_wandb(args.entity, args.project)
        expected = len(EVENTS) * len(LBCLS) * len(SET_NUMS)
        print(f"✅ Found best HPs for {len(best)}/{expected} (event, lbcl, set) combos")

    # Save HP reference JSON
    with open(args.hp_json, "w") as f:
        json.dump(best, f, indent=2)
    print(f"📄 Saved: {args.hp_json}")

    # Generate jobs file: event <TAB> lbcl <TAB> set_num <TAB> lr <TAB> epochs <TAB> bs
    missing = []
    with open(args.output, "w") as f:
        for lbcl in LBCLS:
            for event in EVENTS:
                for set_num in SET_NUMS:
                    key = f"{event}_{lbcl}_s{set_num}"
                    hp = best.get(key, None)
                    if hp is None:
                        missing.append(key)
                        hp = DEFAULT_HP
                    lr = hp.get("learning_rate", DEFAULT_HP["learning_rate"])
                    epochs = hp.get("epochs", DEFAULT_HP["epochs"])
                    bs = hp.get("batch_size", DEFAULT_HP["batch_size"])
                    f.write(f"{event}\t{lbcl}\t{set_num}\t{lr}\t{epochs}\t{bs}\n")

    total = len(LBCLS) * len(EVENTS) * len(SET_NUMS)
    print(f"📋 Wrote {total} jobs to {args.output}")
    if missing:
        print(f"⚠️  {len(missing)} combos used default HPs (no WandB data found):")
        for m in missing[:10]:
            print(f"     {m}")
        if len(missing) > 10:
            print(f"     ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
