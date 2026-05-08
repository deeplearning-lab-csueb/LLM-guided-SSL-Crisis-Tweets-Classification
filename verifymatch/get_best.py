"""
Given a WandB project, find the best run (by dev_macro-F1) for each Sweep,
download its prediction artifact, calculate ECE, and generate a CSV report 
with Test results (Macro F1, ECE).
"""

import argparse
import sys
import pandas as pd
import wandb
import json
import os
import numpy as np
import torch
import shutil
import traceback

# Reuse calibration logic from verifymatch/calibrate.py
# We will copy the necessary functions to avoid importing issues if paths are tricky,
# or we can try to import if the path allows.
# For robustness in a standalone script, copying the small logic helpers is safer.

def compute_ece(confidences, preds, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    ece = 0.0
    confidences = np.array(confidences)
    accuracies = np.array([int(p == y) for p, y in zip(preds, labels)])

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin

    return ece

def calculate_ece_from_preds(preds_path):
    """
    Reads a jsonl predictions file and computes ECE.
    File format expected: {"label": int, "pred": int, "conf": float, ...} per line
    """
    y_true = []
    y_pred = []
    y_conf = []
    
    try:
        with open(preds_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                y_true.append(d['label'])
                y_pred.append(d['pred'])
                y_conf.append(d['conf'])
    except Exception as e:
        print(f"Error reading {preds_path}: {e}")
        return None

    if not y_true:
        return None

    # Use the new compute_ece function
    # Note: compute_ece returns value in 0.0 - 1.0 range usually, but original code returned * 100
    # Original 'calculate_error' returned tuples of * 100.
    # The new code returns raw float.
    # IF the user wants percentages in the CSV, we might need to multiply by 100.
    # However, standard ECE is often 0-1. 
    # Let's check the CSV formatting logic.
    # The CSV formatting logic `row_ece.append(f"{val:.4f}")` handles whatever value comes back.
    # I will stick to returning the raw value from compute_ece as requested ("copy the code").
    
    ece = compute_ece(y_conf, y_pred, y_true, n_bins=10)
    return ece

def prettify_event(event_str):
    if not event_str: return "Unknown"
    return event_str.replace("_", " ").title()

def fetch_wandb_sweeps_results(project_name, entity=None):
    print(f"Fetching sweeps from WandB project: {project_name}...")
    api = wandb.Api()
    
    try:
        path = f"{entity}/{project_name}" if entity else project_name
        sweeps = api.project(project_name, entity=entity).sweeps()
    except Exception as e:
        print(f"Error accessing WandB project '{project_name}': {e}")
        return []

    data = []
    print(f"Found {len(sweeps)} sweeps. Processing...")
    
    for i, sweep in enumerate(sweeps):
        print(f"Processing sweep {i+1}/{len(sweeps)}: {sweep.name}...", end="\r")
        
        runs = sweep.runs
        finished = [r for r in runs if r.state == "finished"]
        
        if not finished:
            continue
            
        # Identify params from first run config
        sample_run = finished[0]
        cfg = sample_run.config
        event = cfg.get("event")
        lbcl = cfg.get("lbcl")
        set_num = cfg.get("set_num")
        
        if not (event and lbcl and set_num):
            continue
            
        # Find best run by dev_macro-F1
        best_run = None
        best_score = -1.0
        
        for r in finished:
            summary = r.summary
            s = summary.get("dev_macro-F1")
            
            # Handle cases where metric might be missing or None
            if s is None: continue
            
            s = float(s)
            
            if s > best_score:
                best_score = s
                best_run = r
        
        if best_run:
            summ = best_run.summary
            test_f1 = float(summ.get("test_macro-F1", 0.0))
            
            # --- ECE Calculation ---
            ece_val = None
            try:
                # [DEBUG NOTE]
                # We previously attempted to use GQL to fetch artifacts, but encountered schema errors.
                # We also attempted `run.logged_artifacts()` which failed due to library bugs.
                # The current strategy is "Collection Lookup": construct the name and find it.
                # ISSUE: This often fails to find the collection, implying a mismatched naming convention 
                # between 'train.py' generation and this reconstruction, specifically regarding the 'seed'.
                
                run_path_parts = best_run.path
                r_entity = run_path_parts[0]
                r_project = run_path_parts[1]
                
                # Fetch seed from summary (reliable) or config (unreliable)
                seed = best_run.summary.get("seed")
                if seed is None:
                    seed = best_run.config.get("seed", 67)
                
                # Verify this string matches exactly what `train.py` produces
                artifact_name = f"preds-HumAID-{event}-lb{lbcl}-set{set_num}-seed{seed}"
                
                pred_art = None
                try:
                    # Direct collection lookup
                    # api.artifact_type(type_name, project).collection(name)
                    art_type = api.artifact_type("predictions", r_project)
                    try:
                        print(f"   [Debug] Looking up collection: {artifact_name}")
                        collection = art_type.collection(artifact_name)
                        print(f"   [Debug] Collection found. Iterating versions...")
                        
                        # Use robust iteration logic from verification script
                        iterator = None
                        if hasattr(collection, 'versions'):
                            iterator = collection.versions()
                        elif hasattr(collection, 'artifacts'):
                             iterator = collection.artifacts()
                        else:
                             iterator = collection

                        producers_found = []
                        for version in iterator:
                            producer = version.logged_by()
                            p_id = producer.id if producer else "None"
                            producers_found.append(p_id)
                            print(f"     -> Version {version.version} logged by {p_id}")
                            
                            if p_id == best_run.id:
                                pred_art = version
                                print(f"     [Debug] MATCH FOUND for run {best_run.id}")
                                break
                        
                        if not pred_art:
                             raise RuntimeError(f"Run {best_run.id} (Seed {seed}) listed as best, but no matching artifact found in {artifact_name}. Found versions logged by: {producers_found}")

                    except Exception as e:
                         # Propagate if it's our RuntimeError, else print
                         if isinstance(e, RuntimeError): raise e
                         print(f"   [Debug] Collection lookup failed or iteration error: {e}") 
                         pass
                            
                except Exception as e:
                    if isinstance(e, RuntimeError): raise e
                    pass

                if pred_art:
                    # Construct valid name/path for download 
                    # pred_art is an Artifact object now
                    
                    # Download to a temporary location
                    download_dir = f"./tmp_artifacts/{best_run.id}"
                    if os.path.exists(download_dir):
                        shutil.rmtree(download_dir)
                    
                    pred_art.download(download_dir)
                    preds_file = os.path.join(download_dir, "preds.jsonl")
                    
                    if os.path.exists(preds_file):
                        # Count rows
                        with open(preds_file, 'r') as f:
                            row_count = sum(1 for _ in f)
                        print(f"\n [Run {best_run.id}: {row_count} rows in {pred_art.name}]")

                        ece_val = calculate_ece_from_preds(preds_file)
                    
                    # Cleanup
                    shutil.rmtree(download_dir)
                else:
                    # If we got here, we failed to find pred_art and didn't raise earlier (e.g. collection lookup crash)
                    # We should also crash here to be safe.
                    raise RuntimeError(f"Failed to retrieve artifact for run {best_run.id}. Searched collection: {artifact_name}")

            except Exception as e:
                if isinstance(e, RuntimeError): raise e
                print(f"\n[Error calculating ECE for run {best_run.id}]")
                traceback.print_exc()
                print(f"Error message: {e}")

            except Exception as e:
                # Duplicate block? Ensuring we catch both if they existed in original
                if isinstance(e, RuntimeError): raise e
                print(f"\n[Error calculating ECE for run {best_run.id}]")
                traceback.print_exc()
                print(f"Error message: {e}")





            data.append({
                "sweep_id": sweep.id,
                "sweep_name": sweep.name,
                "best_run_id": best_run.id,
                "event": event,
                "lbcl": int(lbcl),
                "set_num": int(set_num),
                "dev_f1": best_score,
                "test_f1": test_f1,
                "test_ece": ece_val if ece_val is not None else 0.0
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
    
    # Sort and Deduplicate
    df = df.sort_values(by="dev_f1", ascending=False)
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
    all_sets = ["Set 1", "Set 2", "Set 3"]
    
    final_rows = []
    
    # Header
    header_events = ["", "", "Metrics/Event Name", "Average"]
    header_sets = ["", "", "", ""]
    
    for event in all_events:
        header_events.append(event)
        header_events.extend([""] * (len(all_sets) - 1))
        for s in all_sets:
            header_sets.append(s)

    for lb in sorted(df["lbcl"].unique()):
        # F1 Row
        row_f1 = ["verifymatch", f"{lb} lb/class", "Macro F1", ""]
        f1_vals = []
        for event in all_events:
            for s in all_sets:
                try:
                    val = pivot_f1.loc[lb, (event, s)]
                    row_f1.append(f"{val:.4f}")
                    f1_vals.append(val)
                except KeyError:
                    row_f1.append("")
        
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
                    # Format as percentage to match typical ECE reporting if needed, 
                    # but calibrate.py returns 0..100 already.
                    row_ece.append(f"{val:.4f}")
                    ece_vals.append(val)
                except KeyError:
                    row_ece.append("")
        
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
    # Updated default project name to match the implementation plan
    parser.add_argument("--project", default="humaid_vmatch_category_match_es2", help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity")
    parser.add_argument("--output", default="verifymatch_results.csv", help="Output CSV path")
    args = parser.parse_args()
    
    data = fetch_wandb_sweeps_results(args.project, args.entity)
    format_csv(data, args.output)

if __name__ == "__main__":
    main()
