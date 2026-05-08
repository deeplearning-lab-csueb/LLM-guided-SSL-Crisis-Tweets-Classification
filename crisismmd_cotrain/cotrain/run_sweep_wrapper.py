#!/usr/bin/env python3
"""
Wrapper script to run co-training for all 3 sets and average the results.
"""

import subprocess
import argparse
import re
import numpy as np
import wandb
import os

def run_single_set(args, set_num):
    """Run the main script for a single set and return the metrics."""
    cmd = [
        "python", "-u", "main_bertweet.py",
        "--dataset", args.dataset,
        "--hf_model_id_short", args.hf_model_id_short,
        "--plm_id", args.plm_id,
        "--metric_combination", args.metric_combination,
        "--seed", str(args.seed),
        "--pseudo_label_dir", args.pseudo_label_dir,
        "--data_dir", args.data_dir,
        "--cuda_devices", args.cuda_devices,
        "--event", args.event,
        "--lbcl", args.lbcl,
        "--set_num", str(set_num),
        "--lr", str(args.lr),
        "--num_epochs", str(args.num_epochs),
        "--epoch_patience", str(args.epoch_patience),
        "--weight_decay", str(args.weight_decay),
        "--max_grad_norm", str(args.max_grad_norm),
        "--batch_size", str(args.batch_size),
        "--use_wandb"
    ]


    
    if args.setup_local_logging:
        cmd.insert(3, "--setup_local_logging")
    
    # Prepare a clean environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Remove WANDB environment variables that might conflict
    # We want the child to start a FRESH run, not inherit the parent's state
    wandb_keys = [key for key in env.keys() if key.startswith("WANDB_")]
    for key in wandb_keys:
        if key not in ["WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_USERNAME"]:
             del env[key]

    # Use Popen to stream output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        universal_newlines=True,
        env=env
    )
    
    full_output = []
    
    # Read output line by line as it is generated
    for line in process.stdout:
        print(line, end='', flush=True)  # Stream to console
        full_output.append(line)
    
    process.wait()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
        
    output = "".join(full_output)
    
    # Parse the final results from output
    # Look for the line with Validation F1, Accuracy, ECE
    # Specific to the helper: val_result_msg in main_bertweet.py
    val_match = re.search(r'Validation .*F1: ([\d.]+), .*Accuracy: ([\d.]+), ECE: ([\d.]+)', output)
    test_match = re.search(r'Test .*F1: ([\d.]+), .*Accuracy: ([\d.]+), ECE: ([\d.]+)', output)
    
    metrics = {}
    
    if val_match:
        metrics['val_f1'] = float(val_match.group(1))
        metrics['val_acc'] = float(val_match.group(2))
        metrics['val_ece'] = float(val_match.group(3))
    else:
        print(f"Failed to parse Validation results for set {set_num}")

    if test_match:
        metrics['test_f1'] = float(test_match.group(1))
        metrics['test_acc'] = float(test_match.group(2))
        metrics['test_ece'] = float(test_match.group(3))
    else:
        print(f"Failed to parse Test results for set {set_num}")
        
    if not metrics:
        return None
        
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run co-training sweep over 3 sets")
    parser.add_argument("--dataset", type=str, default="humaid", help="Dataset name")
    parser.add_argument("--hf_model_id_short", type=str, default="N/A", help="Short HF model id")
    parser.add_argument("--plm_id", type=str, default="roberta-base", help="Backbone PLM id")
    parser.add_argument("--metric_combination", type=str, default="cv", help="Metric combination")
    parser.add_argument("--setup_local_logging", action="store_true", default=False, help="Setup local logging")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--pseudo_label_dir", type=str, default="anh_4o", help="Directory containing LLM pseudo labels")
    parser.add_argument("--data_dir", type=str, default="../../data", help="Base data directory")
    parser.add_argument("--cuda_devices", type=str, default="0,1", help="CUDA devices")
    parser.add_argument("--event", type=str, required=True, help="Event name")
    parser.add_argument("--lbcl", type=str, required=True, help="Labeled count per class")
    parser.add_argument("--set_num", type=str, help="Set number (not used in wrapper)")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--epoch_patience", type=int, required=True, help="Epoch patience")
    parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, required=True, help="Max gradient norm")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")

    
    args = parser.parse_args()
    
    # Initialize wandb for this sweep run
    wandb.init(
        project="lg-cotrain-humaid",
        name=f"{args.event}_{args.lbcl}_lr{args.lr}_ep{args.num_epochs}_pat{args.epoch_patience}",
        config={
            "dataset": args.dataset,
            "hf_model_id_short": args.hf_model_id_short,
            "plm_id": args.plm_id,
            "metric_combination": args.metric_combination,
            "seed": args.seed,
            "pseudo_label_dir": args.pseudo_label_dir,
            "event": args.event,
            "lbcl": args.lbcl,
            "data_dir": args.data_dir,
            "cuda_devices": args.cuda_devices,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "epoch_patience": args.epoch_patience,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "batch_size": args.batch_size,

        }
    )
    
    results = []
    for set_num in [1, 2, 3]:
        print(f"Running set {set_num}")
        metrics = run_single_set(args, set_num)
        if metrics:
            results.append(metrics)
            
            # Log individual set results
            log_dict = {}
            if 'val_f1' in metrics:
                log_dict.update({
                    f"set_{set_num}_f1": metrics['val_f1'],
                    f"set_{set_num}_accuracy": metrics['val_acc'],
                    f"set_{set_num}_ece": metrics['val_ece']
                })
            if 'test_f1' in metrics:
                log_dict.update({
                    f"set_{set_num}_test_f1": metrics['test_f1'],
                    f"set_{set_num}_test_accuracy": metrics['test_acc'],
                    f"set_{set_num}_test_ece": metrics['test_ece']
                })
            
            wandb.log(log_dict)
            
        else:
            print(f"No results for set {set_num}")
    
    if results:
        # Calculate averages for Validation
        val_f1s = [r['val_f1'] for r in results if 'val_f1' in r]
        val_accs = [r['val_acc'] for r in results if 'val_acc' in r]
        val_eces = [r['val_ece'] for r in results if 'val_ece' in r]
        
        avg_val_f1 = np.mean(val_f1s) if val_f1s else 0
        avg_val_acc = np.mean(val_accs) if val_accs else 0
        avg_val_ece = np.mean(val_eces) if val_eces else 0
        
        # Calculate averages for Test
        test_f1s = [r['test_f1'] for r in results if 'test_f1' in r]
        test_accs = [r['test_acc'] for r in results if 'test_acc' in r]
        test_eces = [r['test_ece'] for r in results if 'test_ece' in r]
        
        avg_test_f1 = np.mean(test_f1s) if test_f1s else 0
        avg_test_acc = np.mean(test_accs) if test_accs else 0
        avg_test_ece = np.mean(test_eces) if test_eces else 0
        
        # Log averages
        wandb.log({
            "avg_f1": avg_val_f1,
            "avg_accuracy": avg_val_acc,
            "avg_ece": avg_val_ece,
            "val_f1": avg_val_f1,  # Alias for sweep metric compatibility
            
            "avg_test_f1": avg_test_f1,
            "avg_test_accuracy": avg_test_acc,
            "avg_test_ece": avg_test_ece
        })
        
        print(f"Average Val F1: {avg_val_f1:.4f}, Average Val Accuracy: {avg_val_acc:.4f}, Average Val ECE: {avg_val_ece:.4f}")
        print(f"Average Test F1: {avg_test_f1:.4f}, Average Test Accuracy: {avg_test_acc:.4f}, Average Test ECE: {avg_test_ece:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()