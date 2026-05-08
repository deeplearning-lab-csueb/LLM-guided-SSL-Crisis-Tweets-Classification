import wandb
import json
import os
import sys

# Constants
PROJECT = "humaid_vmatch_category_match_bayes"
OUTPUT_FILE = "top_sweep_configs.jsonl"
TOP_K = 10
METRIC = "best_dev_f1"
# Entities to try in order. None = default user entity.
ENTITIES_TO_TRY = [None, "YOUR_WANDB_ENTITY", "mining-for-metadata"]

def analyze_sweeps():
    api = wandb.Api()
    sweeps = None
    
    for entity in ENTITIES_TO_TRY:
        print(f"Checking project '{PROJECT}' with entity='{entity}'...")
        try:
            # api.project(...) doesn't validate existence immediately, sweeps() creates iterator
            # We must try to iterate or fetch length to trigger the 404/validation
            candidate_sweeps = api.project(PROJECT, entity=entity).sweeps()
            
            # Force a check by converting to list (if small) or checking length
            # accessing len() forces validation in wandb library
            count = len(candidate_sweeps)
            print(f"  Found project! Sweep count: {count}")
            sweeps = candidate_sweeps
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if not sweeps:
        print("Could not find the project in any of the attempted entities.")
        print("Please check the project name or your WandB permissions.")
        return

    print(f"Analyze top {TOP_K} runs per sweep...")

    all_best_configs = []

    for sweep in sweeps:
        print(f"\nProcessing Sweep: {sweep.name} (ID: {sweep.id})")
        
        # Get finished runs
        runs = sweep.runs
        finished_runs = [r for r in runs if r.state == "finished"]
        
        if not finished_runs:
            print("  No finished runs found.")
            continue

        # Sort by metric
        # Check if metric exists in summary
        valid_runs = []
        for r in finished_runs:
            if METRIC in r.summary:
                valid_runs.append(r)
            elif "val/f1" in r.summary:
                 # Fallback
                 r.summary[METRIC] = r.summary["val/f1"]
                 valid_runs.append(r)
        
        # Sort descending (higher F1 is better)
        valid_runs.sort(key=lambda x: x.summary.get(METRIC, 0), reverse=True)
        
        top_runs = valid_runs[:TOP_K]
        print(f"  Found {len(valid_runs)} valid runs. Extracting top {len(top_runs)}...")

        for i, run in enumerate(top_runs):
            f1 = run.summary.get(METRIC, 0)
            print(f"    Rank {i+1}: F1={f1:.4f} | ID={run.id}")
            
            # Extract relevant config
            # We want hyperparameters, not wandb system info
            filtered_config = {k: v for k, v in run.config.items() if not k.startswith("_") and k != "wandb_version"}
            
            record = {
                "sweep_id": sweep.id,
                "sweep_name": sweep.name,
                "run_id": run.id,
                "rank": i + 1,
                "metric": METRIC,
                "score": f1,
                "config": filtered_config
            }
            all_best_configs.append(record)

    # Save to JSONL
    output_path = os.path.join(os.getcwd(), OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_best_configs:
            f.write(json.dumps(rec) + "\n")

    print(f"\n✅ Analysis Complete. Saved {len(all_best_configs)} top configs to: {output_path}")

if __name__ == "__main__":
    analyze_sweeps()
