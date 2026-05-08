#!/usr/bin/env python3
"""
initialize_sweeps.py

Generates WandB sweeps for all Event x LBCL combinations.
Order: Event outer loop, LBCL inner loop.
Ref: verifymatch/make_container.sh
"""

import wandb
import yaml
from generate_sweep import generate_sweep_yaml

# Configuration
ENTITY = "YOUR_WANDB_ENTITY"
PROJECT = "lg-cotrain-humaid"
OUTPUT_FILE = "sweep_ids.txt"

# Data Constants (from verifymatch/make_container.sh)
EVENTS = [
    "california_wildfires_2018",
    "canada_wildfires_2016",
    "cyclone_idai_2019",
    "hurricane_dorian_2019",
    "hurricane_florence_2018",
    "hurricane_harvey_2017",
    "hurricane_irma_2017",
    "hurricane_maria_2017",
    "kaikoura_earthquake_2016",
    "kerala_floods_2018",
]

LBCLS = ["5", "10", "25", "50"]

# Fixed parameters for this batch of sweeps
FIXED_PARAMS = {
    "program": "run_sweep_wrapper.py",
    "dataset": "humaid",
    "hf_model_id_short": "N/A",
    "plm_id": ["clip", "bert-tweet", "roberta-base", "bert-base", "deberta-base", "roberta-large"],
    "metric_combination": "cv",
    "setup_local_logging": False,
    "seed": 1234,
    "pseudo_label_dir": "anh_4o",
    "data_dir": "../../data", # Assuming run from cotrain/
    "cuda_devices": "0,1", # Agents will override visibility, but this is passed to script
    "method": "bayes",
    "metric_name": "val_f1",
    "metric_goal": "maximize",

}

def main():
    sweep_ids = []
    
    print(f"Initializing sweeps for Project: {PROJECT}, Entity: {ENTITY}")
    
    for event in EVENTS:
        for lbcl in LBCLS:
            print(f"Generating sweep for {event} - {lbcl} lbcl...")
            
            # Generate the sweep config dictionary
            sweep_config = generate_sweep_yaml(
                event=event,
                lbcl=lbcl,
                **FIXED_PARAMS
            )
            
            # Create the sweep on WandB
            try:
                sweep_id = wandb.sweep(
                    sweep_config,
                    project=PROJECT,
                    entity=ENTITY
                )
                sweep_ids.append(sweep_id)
                print(f"  -> Created sweep ID: {sweep_id}")
            except Exception as e:
                print(f"  -> FAILED to create sweep: {e}")
                # Optional: break or continue? Continuing for now.
    
    # Save IDs to file
    if sweep_ids:
        with open(OUTPUT_FILE, "w") as f:
            for sid in sweep_ids:
                f.write(f"{sid}\n")
        print(f"\nSuccessfully saved {len(sweep_ids)} sweep IDs to {OUTPUT_FILE}")
    else:
        print("\nNo sweeps were created.")

if __name__ == "__main__":
    main()
