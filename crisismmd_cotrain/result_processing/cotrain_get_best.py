import wandb
import pandas as pd
import sys

# Configuration
ENTITY = "YOUR_WANDB_ENTITY"
PROJECT = "lg-cotrain-humaid"
OUTPUT_FILE = "best_plm_id.csv"

def main():
    print(f"Connecting to WandB Project: {ENTITY}/{PROJECT}")
    try:
        api = wandb.Api()
        sweeps = api.project(name=PROJECT, entity=ENTITY).sweeps()
    except Exception as e:
        print(f"Error connecting to WandB: {e}")
        sys.exit(1)

    best_runs_data = []

    print(f"Found {len(sweeps)} sweeps. Processing...")

    for sweep in sweeps:
        try:
            # sweep.best_run() uses the metric/goal defined in the sweep config.
            # initialize_sweeps.py sets metric_name='val_f1', metric_goal='maximize'.
            best_run = sweep.best_run()
            
            if not best_run:
                # print(f"  [Skipping] No runs found for sweep: {sweep.id}")
                continue
                
            # Extract metrics and config
            plm_id = best_run.config.get("plm_id")
            
            # Using .get() for safety, though these should exist given the sweep setup
            event = best_run.config.get("event")
            lbcl = best_run.config.get("lbcl")
            
            if plm_id is None:
                # print(f"  [Skipping] 'avg_test_f1' missing in summary for sweep: {sweep.id}")
                continue
                
            if event and lbcl:
                best_runs_data.append({
                    "event": event,
                    "lbcl": int(lbcl), # Ensure lbcl is numeric for proper sorting
                    "plm_id": plm_id
                })
                # print(f"  + {event} (lbcl={lbcl}): plm_id={plm_id}")

        except Exception as e:
            print(f"  [Error] Processing sweep {sweep.id}: {e}")
            continue

    if not best_runs_data:
        print("No valid data found to process.")
        return

    # Create DataFrame
    df = pd.DataFrame(best_runs_data)

    # Pivot: Index=event, Columns=lbcl, Values=val_f1
    # Sort index (events) alphabetically
    pivot_df = df.pivot(index="event", columns="lbcl", values="plm_id").sort_index()

    # Save to CSV
    pivot_df.to_csv(OUTPUT_FILE)
    print(f"\nSuccess! Saved pivot table to {OUTPUT_FILE}")
    print("\nPreview:")
    print(pivot_df)

if __name__ == "__main__":
    main()
