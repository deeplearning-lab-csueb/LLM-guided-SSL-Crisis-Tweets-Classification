import wandb
import re
import os
import argparse
import numpy as np

def recover_metrics(dry_run=True, project="lg-cotrain-humaid", entity="YOUR_WANDB_ENTITY"):
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return

    print(f"Found {len(runs)} finished runs in {entity}/{project}")
    
    # Regex to capture F1, Accuracy, and ECE from the Test log line
    # Matches format: ... Test ... F1: 0.1234, Test Accuracy: 0.5678, ECE: 0.0912
    test_metrics_pattern = re.compile(r"Test .*F1: ([\d\.]+), .*Test Accuracy: ([\d\.]+), ECE: ([\d\.]+)")
    
    for run in runs:
        # Optional: Check if already updated to avoid re-processing
        if 'avg_test_f1' in run.summary:
            print(f"Skipping run {run.name} ({run.id}) - already has test metrics.")
            continue
            
        print(f"Processing run {run.name} ({run.id})...")
        
        try:
            # Check if output.log exists
            files = run.files()
            has_output_log = any(f.name == 'output.log' for f in files)
            
            if not has_output_log:
                print(f"  No output.log found for run {run.name}")
                continue
                
            # Download output.log to current directory temporarily
            log_file = run.file('output.log').download(replace=True, root=".")
            log_content = log_file.read()
            log_file.close()
            
            # Find all matches
            matches = test_metrics_pattern.findall(log_content)
            
            if not matches:
                print(f"  No 'Test ...' metrics found in log for run {run.name}")
                os.remove('output.log') # Clean up
                continue
                
            print(f"  Found {len(matches)} sets of metrics (Sets found: {len(matches)})")
            
            # We expect 3 matches for sets 1, 2, 3
            # If fewer or more, we process what we have, but warn.
            
            f1_scores = []
            accuracies = []
            eces = []
            
            updates = {}
            
            for i, match in enumerate(matches):
                # match is a tuple: (f1, acc, ece)
                f1 = float(match[0])
                acc = float(match[1])
                ece = float(match[2])
                
                set_num = i + 1
                
                f1_scores.append(f1)
                accuracies.append(acc)
                eces.append(ece)
                
                updates[f"set_{set_num}_test_f1"] = f1
                updates[f"set_{set_num}_test_accuracy"] = acc
                updates[f"set_{set_num}_test_ece"] = ece
            
            if f1_scores:
                updates["avg_test_f1"] = np.mean(f1_scores)
                updates["avg_test_accuracy"] = np.mean(accuracies)
                updates["avg_test_ece"] = np.mean(eces)
            
            if dry_run:
                print(f"  [DRY RUN] Would update run {run.name} with:")
                for k, v in updates.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  Updating run {run.name}...")
                run.summary.update(updates)
                run.update()
                
            # Clean up downloaded file
            if os.path.exists('output.log'):
                os.remove('output.log')
                
        except Exception as e:
            print(f"  Error processing run {run.name}: {e}")
            if os.path.exists('output.log'):
                os.remove('output.log')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Print actions without updating WandB")
    parser.add_argument("--project", type=str, default="lg-cotrain-humaid", help="WandB Project Name")
    parser.add_argument("--entity", type=str, default="YOUR_WANDB_ENTITY", help="WandB Entity/Username")
    
    args = parser.parse_args()
    
    recover_metrics(dry_run=args.dry_run, project=args.project, entity=args.entity)
