import wandb
import sys
import numpy as np
import collections

# Configuration
ENTITY = "YOUR_WANDB_ENTITY"
PROJECT = "lg-cotrain-humaid"
OUTPUT_FILE = "cotrain_humaid_bharani_results.csv"

def main():
    print(f"Connecting to WandB Project: {ENTITY}/{PROJECT}")
    try:
        api = wandb.Api()
        sweeps = api.project(name=PROJECT, entity=ENTITY).sweeps()
    except Exception as e:
        print(f"Error connecting to WandB: {e}")
        sys.exit(1)

    if not sweeps:
        print("No sweeps found.")
        return

    # Data structure: data[plm][lbcl][event] = { 'f1': [s1, s2, s3], 'ece': [s1, s2, s3] }
    data = collections.defaultdict(lambda: collections.defaultdict(dict))
    all_events = set()

    print(f"Found {len(sweeps)} sweeps. Processing...")

    count = 0
    for sweep in sweeps:
        try:
            best_run = sweep.best_run()
            if not best_run:
                continue

            config = best_run.config
            summary = best_run.summary

            plm_id = config.get("plm_id")
            event = config.get("event")
            lbcl = config.get("lbcl")

            if not plm_id or not event or not lbcl:
                continue
            
            # Normalize lbcl to int for sorting
            try:
                lbcl = int(lbcl)
            except ValueError:
                continue

            # Extract metrics
            # Keys verified: set_1_test_f1, set_1_test_ece, etc.
            f1_scores = []
            ece_scores = []
            
            missing_metrics = False
            for i in range(1, 4):
                f1_key = f"set_{i}_test_f1"
                ece_key = f"set_{i}_test_ece"
                
                if f1_key not in summary or ece_key not in summary:
                    missing_metrics = True
                    break
                
                f1_scores.append(summary[f1_key])
                ece_scores.append(summary[ece_key])

            if missing_metrics:
                # specific logging could be added here
                continue

            data[plm_id][lbcl][event] = {
                'f1': f1_scores,
                'ece': ece_scores
            }
            all_events.add(event)
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} valid runs...")

        except Exception as e:
            print(f"Error processing sweep {sweep.id}: {e}")
            continue

    if not data:
        print("No valid data collected.")
        return

    sorted_events = sorted(list(all_events))
    
    # Prepare CSV lines
    lines = []
    
    # Header Row 1
    # ,,Metrics/Event Name,,,Average, [Event 1],,, [Event 2],,, ...
    header1 = ["", "", "Metrics/Event Name", "", "", "Average"]
    # Average is now 1 column

    
    for event in sorted_events:
        header1.append(event)
        header1.append("")
        header1.append("")
    
    lines.append(",".join(header1))

    # Header Row 2
    # ,,,,,, , Set 1,Set 2,Set 3, ...
    header2 = ["", "", "", "", "", ""]

    
    for _ in sorted_events:
        header2.extend(["Set 1", "Set 2", "Set 3"])
    
    lines.append(",".join(header2))

    # Iterate PLMs
    for plm in sorted(data.keys()):
        plm_data = data[plm]
        sorted_lbcls = sorted(plm_data.keys())
        
        first_lbcl = True
        
        for lbcl in sorted_lbcls:
            lb_data = plm_data[lbcl]
            
            # Calculate Averages across available events for this lbcl
            # Note: valid_events might differ per lbcl if runs failed, but we use strict sorted_events for columns.
            # We compute average over the events present in lb_data.
            
            # Calculate Grand Average across all events and sets for this lbcl
            all_f1s = []
            all_eces = []
            
            for event in lb_data:
                metrics = lb_data[event]
                all_f1s.extend(metrics['f1'])
                all_eces.extend(metrics['ece'])
            
            avg_f1 = np.mean(all_f1s) if all_f1s else 0.0
            avg_ece = np.mean(all_eces) if all_eces else 0.0
            
            # Construct F1 Row
            # [PLM], [lb] lb/class, Macro F1, [Avg S1], [Avg S2], [Avg S3], [Ev1 S1], ...
            row_f1 = []
            if first_lbcl:
                row_f1.append(plm)
            else:
                row_f1.append("")
            
            row_f1.append(f"{lbcl} lb/class")
            row_f1.append("Macro F1")
            row_f1.append("") # Old
            row_f1.append("") # VM
            
            # Avg Column
            row_f1.append(f"{avg_f1:.4f}")
            
            # Event Columns
            for event in sorted_events:
                if event in lb_data:
                    row_f1.extend([str(x) for x in lb_data[event]['f1']])
                else:
                    row_f1.extend(["", "", ""])
            
            lines.append(",".join(row_f1))
            
            # Construct ECE Row
            # , , ECE, [Avg S1], [Avg S2], [Avg S3], ...
            row_ece = ["", "", "ECE", "", ""] # padding cols
            
            # Avg Column
            row_ece.append(f"{avg_ece:.4f}")
            
            # Event Columns
            for event in sorted_events:
                if event in lb_data:
                    row_ece.extend([str(x) for x in lb_data[event]['ece']])
                else:
                    row_ece.extend(["", "", ""])
            
            lines.append(",".join(row_ece))
            
            first_lbcl = False
        
        # Add separator row between PLMs
        # lines.append("," * (len(header1))) 
        # Actually user example just had new data, but typically a blank line is nice. 
        # User said "Just make a table for each plm."
        # If I strictly follow the "one big table" structure but separated, I'll add a blank line.
        lines.append(",".join([""] * len(header1)))

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))
        
    print(f"Successfully generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
