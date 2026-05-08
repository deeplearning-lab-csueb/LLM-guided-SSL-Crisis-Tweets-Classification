import sys
import os
import pandas as pd
import torch # Ensure torch is importable as main script uses it

# Add current dir to path
sys.path.append(os.getcwd())

import data_utils

# Setup paths
# We are likely running from d:\Downloads\Git-Stuff\ssl\llm-co-training-crisismmd-main\cotrain
# Data is at d:\Downloads\Git-Stuff\ssl\data
data_dir = "../../data"
pseudo_dir = "../../data" 

print(f"Testing HumAID data load from {os.path.abspath(data_dir)}...")

try:
    # Attempt to load
    # Note: main_bertweet.py calls it with shots=0 (or whatever)
    t1, t2, test, val, auto = data_utils.load_dataset_helper(
        task_name='humaid', 
        data_dir=data_dir, 
        pseudo_label_dir=pseudo_dir, 
        shots=0, 
        use_correct_labels_only=False
    )
    
    print("SUCCESS: Data loaded.")
    print(f"Training Set 1 size: {len(t1)}")
    print(f"Training Set 2 size: {len(t2)}")
    print(f"Test size: {len(test)}")
    print(f"Dev size: {len(val)}")
    print(f"Auto-labeled size: {len(auto)}")
    
    if len(auto) > 0:
        print("\nSample Auto-labeled Record:")
        print(auto.iloc[0])
        print("\nColumns:", auto.columns.tolist())
        
        # Check label types
        lbl = auto.iloc[0]['label']
        print(f"Label value: {lbl}, Type: {type(lbl)}")
        
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
