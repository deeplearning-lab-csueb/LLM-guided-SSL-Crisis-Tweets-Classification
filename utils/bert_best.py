"""
Given a WandB project, find the best run (by eval_f1) for each Sweep (representing a configuration)
and generate a CSV report with Test results (Macro F1, ECE).
"""

import argparse
import sys
import pandas as pd
import wandb

def prettify_event(event_str):
    if not event_str: return "Unknown"
    # standard: "california_wildfires_2018" -> "California Wildfires 2018"
    return event_str.replace("_", " ").title()

def fetch_wandb_sweeps_results(project_name):
    print(f"Fetching sweeps from WandB project: {project_name}...")
    api = wandb.Api()
    
    try:
        sweeps = api.project(project_name).sweeps()
    except Exception as e:
        print(f"Error accessing WandB project '{project_name}': {e}")
        return []

    data = []
    print(f"Found {len(sweeps)} sweeps. Processing...")
    
    for i, sweep in enumerate(sweeps):
        print(f"Processing sweep {i+1}/{len(sweeps)}: {sweep.name}...", end="\r")
        
        # Get all finished runs in this sweep
        runs = sweep.runs
        finished = [r for r in runs if r.state == "finished"]
        
        if not finished:
            continue
            
        # We need to identify Event, LBCL, Set from the sweep or its runs.
        # Assuming all runs in a sweep belong to the same Event/LBCL/Set.
        # We can peek at the first run's config (or the sweep config if available/consistent).
        # Sweep name might also contain info, but config is safer.
        
        sample_run = finished[0]
        cfg = sample_run.config
        event = cfg.get("event")
        lbcl = cfg.get("lbcl")
        set_num = cfg.get("set_num")
        
        if not (event and lbcl and set_num):
            # Try to parse from sweep name if config is missing? 
            # Pattern: event_lbcl_setnum_timestamp
            # But let's rely on config first.
            continue
            
        # Find best run in this sweep
        best_run = None
        best_score = -1.0
        
        for r in finished:
            summary = r.summary
            # Metric: eval_f1. 'best_score_so_far' is what we want.
            s = summary.get("best_score_so_far")
            if s is None:
                s = summary.get("eval_f1", 0.0)
            
            s = float(s)
            
            # Check if this run has test results!
            if summary.get("test_macro_f1") is None:
                continue
                
            if s > best_score:
                best_score = s
                best_run = r
        
        if best_run:
            summ = best_run.summary
            data.append({
                "sweep_id": sweep.id,
                "sweep_name": sweep.name,
                "best_run_id": best_run.id,
                "event": event,
                "lbcl": int(lbcl),
                "set_num": int(set_num),
                "eval_f1": best_score,
                "test_f1": float(summ.get("test_macro_f1", 0.0)),
                "test_ece": float(summ.get("test_ece", 0.0))
            })

    print("\n")
    print(f"Extracted best runs from {len(data)} valid sweeps.")
    return data

def format_csv(data, output_path):
    if not data:
        print("No data to save.")
        return

    df = pd.DataFrame(data)
    
    # 1. Prettify Event Names
    df["Event Name"] = df["event"].apply(prettify_event)
    df["Set"] = "Set " + df["set_num"].astype(str)
    
    # Check for duplicates? (Multiple sweeps for same condition?)
    # If duplicates, pick best eval_f1
    df = df.sort_values(by="eval_f1", ascending=False)
    df = df.drop_duplicates(subset=["event", "lbcl", "set_num"], keep="first")
    
    # 2. Pivot for F1
    pivot_f1 = df.pivot_table(
        index=["lbcl"], 
        columns=["Event Name", "Set"], 
        values="test_f1"
    )
    
    # 3. Pivot for ECE
    pivot_ece = df.pivot_table(
        index=["lbcl"], 
        columns=["Event Name", "Set"], 
        values="test_ece"
    )
    
    all_events = sorted(df["Event Name"].unique())
    all_sets = ["Set 1", "Set 2", "Set 3"] # Fixed set ordering usually desired
    
    # Prepare data rows
    final_rows = []
    
    # Header construction
    header_events = ["", "", "Metrics/Event Name", "Average"]
    header_sets = ["", "", "", ""]
    
    # Ensure all events are columns even if missing data? Only present ones.
    
    for event in all_events:
        header_events.append(event)
        header_events.extend([""] * (len(all_sets) - 1))
        for s in all_sets:
            header_sets.append(s)

    for lb in sorted(df["lbcl"].unique()):
        # F1 Row
        row_f1 = ["bertweet", f"{lb} lb/class", "Macro F1", ""]
        f1_vals = []
        for event in all_events:
            for s in all_sets:
                try:
                    val = pivot_f1.loc[lb, (event, s)]
                    row_f1.append(f"{val:.4f}")
                    f1_vals.append(val)
                except KeyError:
                    row_f1.append("")
        
        # Calculate Average
        if f1_vals:
            avg_f1 = sum(f1_vals) / len(f1_vals)
            row_f1[3] = f"{avg_f1:.4f}"
            
        final_rows.append(row_f1)
        
        # ECE Row
        row_ece = ["", "", "ECE", ""]
        ece_vals = []
        for event in all_events:
            for s in all_sets:
                try:
                    val = pivot_ece.loc[lb, (event, s)]
                    row_ece.append(f"{val:.4f}")
                    ece_vals.append(val)
                except KeyError:
                    row_ece.append("")
        
        # Calculate Average
        if ece_vals:
            avg_ece = sum(ece_vals) / len(ece_vals)
            row_ece[3] = f"{avg_ece:.4f}"
            
        final_rows.append(row_ece)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(header_events) + "\n")
        f.write(",".join(header_sets) + "\n")
        for r in final_rows:
            f.write(",".join(r) + "\n")
            
    print(f"Saved formatted CSV to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="humaid_supervised_hpo", help="WandB project name")
    parser.add_argument("--output", default="bertweet_results.csv", help="Output CSV path")
    args = parser.parse_args()
    
    data = fetch_wandb_sweeps_results(args.project)
    format_csv(data, args.output)

if __name__ == "__main__":
    main()
