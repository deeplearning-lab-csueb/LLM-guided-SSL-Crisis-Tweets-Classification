import wandb
import re
import json
import argparse
import sys
import os

def parse_log_and_update(run):
    """
    Downloads output.log, parses the final metrics JSON, and updates wandb summary.
    """
    print(f"Processing run {run.name} ({run.id})...")
    
    try:
        # Check if log exists
        files = run.files()
        log_file = None
        for f in files:
            if f.name == "output.log":
                log_file = f
                break
        
        if not log_file:
            print("  No output.log found.")
            return False

        # Download to temp file
        log_file.download(replace=True, root=".")
        
        with open("output.log", "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # Look for the JSON block
        # Pattern: Starts with { and contains "Best mixmatch model"
        # We capture everything from that opening brace to the end, then try to parse valid JSON
        # effectively finding the last JSON block.
        
        # Regex to find the start of the JSON block
        # It looks like:
        # {
        #     "Temperature Scaling": ...
        
        # specific header often used
        matches = list(re.finditer(r'\{\s+"Temperature Scaling"', content))
        if not matches:
             print("  No metrics JSON block found in log.")
             return False
             
        # Take the last occurrence
        start_idx = matches[-1].start()
        potential_json = content[start_idx:]
        
        # The log might have trailing newlines or other noise, but the JSON itself closes with }
        # We need to find the matching closing brace.
        # Simple stack approach to find the end of the JSON object
        balance = 0
        end_idx = -1
        for i, char in enumerate(potential_json):
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
                if balance == 0:
                    end_idx = i + 1
                    break
        
        if end_idx == -1:
             print("  Could not find closing brace for JSON block.")
             return False
             
        json_str = potential_json[:end_idx]
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return False
            
        # Extract metrics
        # "Best mixmatch model": {
        #     "F1 before temp scaling": "0.3823319118568042",
        #     "ECE before temp scaling": "tensor(0.4353, device='cuda:0')",
        # }
        
        model_data = data.get("Best mixmatch model")
        if not model_data:
            print("  JSON found but 'Best mixmatch model' key missing (might be UST run?).")
            return False
            
        f1 = model_data.get("F1 before temp scaling")
        ece_raw = model_data.get("ECE before temp scaling")
        
        # Clean up ECE string
        # "tensor(0.4353, device='cuda:0')" -> 0.4353
        ece = 0.0
        if ece_raw and "tensor" in ece_raw:
            # extract number between ( and ,
            m = re.search(r'tensor\(([\d\.]+)', ece_raw)
            if m:
                ece = float(m.group(1))
        elif ece_raw:
             try:
                 ece = float(ece_raw)
             except:
                 pass
                 
        if f1:
            f1 = float(f1)
            
            print(f"  Found Mactrics -> F1: {f1}, ECE: {ece}")
            
            # Update summary
            run.summary["test_macro-F1"] = f1
            run.summary["test_ece"] = ece
            run.summary.update()
            print("  Run summary updated.")
            return True
            
    except Exception as e:
        print(f"  Error processing run: {e}")
        
    finally:
        if os.path.exists("output.log"):
            os.remove("output.log")
            
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--entity", required=True, help="WandB entity")
    parser.add_argument("--dry_run", action="store_true", help="Don't actually update wandb")
    
    args = parser.parse_args()
    
    api = wandb.Api()
    print(f"Fetching runs from {args.entity}/{args.project}...")
    runs = api.runs(f"{args.entity}/{args.project}")
    
    count_updated = 0
    count_skipped = 0
    
    for run in runs:
        if run.state != "finished":
            continue
            
        # Check if already has metrics
        if "test_macro-F1" in run.summary:
            # print(f"Run {run.name} already has metrics. Skipping.")
            count_skipped += 1
            continue
            
        # Only process mixmatch runs if we can distinguish them?
        # The logs will tell us. run_mixmatch.py runs usually have specific names or configs.
        # But parsing "Best mixmatch model" JSON key will strictly filter for us anyway.
        
        success = parse_log_and_update(run)
        if success:
            count_updated += 1
            
    print(f"\nDone. Updated {count_updated} runs. Skipped {count_skipped} (already had metrics or failed).")

if __name__ == "__main__":
    main()
