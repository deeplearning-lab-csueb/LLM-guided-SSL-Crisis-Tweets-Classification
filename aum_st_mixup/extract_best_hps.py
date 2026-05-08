"""
Pull the best hyperparameter config for each 5lb/cl (event, set) combo
from the WandB project and write a JSON file that the submission script
can consume directly.
"""
import wandb
import json
import re
import os
import argparse

ENTITY = "YOUR_WANDB_ENTITY"
PROJECT = "humaid_aum_mixup_st_hpo"

EVENTS = [
    "california_wildfires_2018", "canada_wildfires_2016", "cyclone_idai_2019",
    "hurricane_dorian_2019", "hurricane_florence_2018", "hurricane_harvey_2017",
    "hurricane_irma_2017", "hurricane_maria_2017", "kaikoura_earthquake_2016",
    "kerala_floods_2018",
]
SETS = [1, 2, 3]

HP_KEYS = [
    "sup_epochs", "unsup_epochs", "sup_batch_size", "unsup_batch_size",
    "alpha", "T", "sample_size", "unsup_size",
    "sample_scheme", "N_base", "label_smoothing", "hidden_dropout_prob",
    "attention_probs_dropout_prob", "dense_dropout",
]

# --- Shared Log Parsing Function ---
def parse_metrics_from_log(run):
    import shutil
    download_root = f"temp_logs_{run.id}"
    try:
        try:
             files = run.files() 
             has_log = any(f.name == "output.log" for f in files)
             if not has_log: return None
        except: return None

        os.makedirs(download_root, exist_ok=True)
        run.file("output.log").download(replace=True, root=download_root)
        log_path = os.path.join(download_root, "output.log")
        
        dev_f1 = None
        if not os.path.exists(log_path): return None

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        dev_matches = list(re.finditer(r"New best macro validation\s+([\d\.]+)", content))
        if dev_matches:
            dev_scores = [float(m.group(1)) for m in dev_matches]
            dev_f1 = max(dev_scores)

        return dev_f1
    except Exception as e: return None
    finally:
        if os.path.exists(download_root):
            try: shutil.rmtree(download_root)
            except: pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="best_5lb_hps.json")
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    print(f"Fetching sweeps for {ENTITY}/{PROJECT}...")
    sweeps = list(api.project(PROJECT, entity=ENTITY).sweeps())

    target_sweeps = {}  # Map (event, set_num) -> list of runs
    
    # Also pull standalone runs in case the 5lb runs were not sweeps
    standalone_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    print("Parsing Sweeps...")
    for s in sweeps:
        if not s.name: continue
        m = re.match(r"(?:aum_mixup_)?(.+)_(\d+)lbcl_set(\d+)", s.name)
        if not m: continue
            
        event, lbcl, set_num = m.groups()
        if int(lbcl) != 5: continue
            
        set_num = int(set_num)
        key = (event, set_num)
        
        all_runs = list(s.runs)
        finished_runs = [r for r in all_runs if r.state in ("finished", "crashed", "failed")]
        
        if key not in target_sweeps or len(finished_runs) > len(target_sweeps[key]):
            target_sweeps[key] = finished_runs
            
    print("Parsing Standalone Runs...")
    # Add standalone runs if they exist and are for 5lbcl
    for r in standalone_runs:
        if not r.name: continue
        m = re.match(r"(.+)_(\d+)lb_set(\d+)_aum_mixup", r.name)
        if not m: continue
        event, lbcl, set_num = m.groups()
        if int(lbcl) != 5: continue
        if r.state not in ("finished", "crashed", "failed"): continue
            
        set_num = int(set_num)
        key = (event, set_num)
        if key not in target_sweeps:
            target_sweeps[key] = []
        target_sweeps[key].append(r)

    print(f"Found {len(target_sweeps)} matching 5lb configs")

    results = []
    missing = []

    for event in EVENTS:
        for set_num in SETS:
            key = (event, set_num)
            if key not in target_sweeps:
                print(f"⚠️  Missing sweep entirely for {event} set{set_num}")
                missing.append(key)
                continue

            runs = target_sweeps[key]
            if not runs:
                print(f"⚠️  No finished runs inside the sweep for {event} set{set_num}")
                missing.append(key)
                continue

            # Find best run by dev F1
            best_run = None
            best_dev = -1.0
            for r in runs:
                dev = r.summary.get("dev_macro-F1") or r.summary.get("dev_f1")
                if dev is None:
                    # Fallback to log parsing if summary is empty
                    dev = parse_metrics_from_log(r)
                    
                if dev is not None and float(dev) > best_dev:
                    best_dev = float(dev)
                    best_run = r

            if best_run is None:
                print(f"⚠️  No run with a measurable dev F1 for {event} set{set_num}")
                missing.append(key)
                continue

            # Extract HP config
            config = best_run.config
            hp = {}
            for k in HP_KEYS:
                if k in config:
                    hp[k] = config[k]

            entry = {
                "event": event,
                "set_num": set_num,
                "lbcl": 5,
                "best_dev_f1": best_dev,
                "source_run_id": best_run.id,
                "source_sweep": getattr(best_run.sweep, "name", best_run.name) if hasattr(best_run, "sweep") and best_run.sweep else best_run.name,
                "hyperparameters": hp,
            }
            results.append(entry)
            print(f"✅ {event} set{set_num}: dev_f1={best_dev:.4f} (run {best_run.id})")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Extracted {len(results)}/30 configs → {args.output}")
    if missing:
        print(f"Missing {len(missing)}: {missing}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
