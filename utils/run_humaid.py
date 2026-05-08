import subprocess
import itertools
import sys
import os
import pandas as pd
from pathlib import Path 

# Configurable params
# LB_PER_CLASS = [5]
# SETS = [1]
LB_PER_CLASS = [5, 10, 25, 50]
SETS = [1, 2, 3]

# Paths to your scripts
TRAIN_SCRIPT = r"../verifymatch/train.py"
BERT_FT_SCRIPT = r"../supervised/bert_ft.py"

# You can define your custom args inline here later
TRAIN_ARGS_TEMPLATE = [
    "python", TRAIN_SCRIPT,
    "--model", "vinai/bertweet-base",
    "--task", "HumAID",
    "--epochs", "18",
    "--batch_size", "16",
    "--learning_rate", "5e-5",
    "--do_train", "--do_evaluate",
    "--ssl", "--mixup", "--pseudo_label_by_normalized",
    "--seed", "42",
    "--T", "0.5",
    "--max_seq_length", "128"
]

BERT_FT_ARGS_TEMPLATE = [
    "python", BERT_FT_SCRIPT,
    "--label_col", "class_label",
    "--raw_format", "tsvdir",
    "--lrs", "5e-5",
    "--epochs", "18",
    "--batch_sizes", "16",
    "--max_length", "128",
    "--seed", "42",
]


def run_and_stream(cmd, prefix):
    """Run a command and stream its stdout to this process's stdout."""
    print(f"Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
    finally:
        if proc.stdout:
            proc.stdout.close()
    return proc.wait()

def get_events(tsv_folder):
    files = [os.path.join(tsv_folder, f"{split}.tsv") for split in ["train", "dev", "test"]]
    df = pd.concat(
        (pd.read_csv(f, sep="\t", usecols=[0]) for f in files),
        ignore_index=True
    )
    return set(df["event"].unique())

def separate_event(event, tsv_file, outfile_name):
    """
    Gather all rows with the specified event label from the TSV file.
    Save them to a new TSV file in: ./temp/{event}_{outfile_name}.tsv
    Returns the output file path.
    """
    
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)

    # Build output path safely
    output_path = os.path.join("temp", f"{event}_{outfile_name}.tsv")

    # Check if the input file exists
    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")

    # Read and filter
    df = pd.read_csv(tsv_file, sep="\t")
    if df.empty:
        print(f"Warning: {tsv_file} is empty.")
        return None

    # Assuming the event column is the first column
    event_df = df[df["event"] == event]

    # Only save if there are matching rows
    if event_df.empty:
        print(f"No rows found for event '{event}' in {tsv_file}.")
        return None

    # Save filtered rows
    event_df.to_csv(output_path, sep="\t", index=False)

    return output_path

def separate_event_folder(event, tsv_folder, outfile_name):
    """
    Like separate_event but for all files in a folder.
    Returns files with the desired event in:
        ./temp/{event}_{outfile_name}/{original_filename}.tsv
    """
    # Create output folder safely
    out_folder = os.path.join("temp", f"{event}_{outfile_name}")
    os.makedirs(out_folder, exist_ok=True)

    # Process each split file
    for split in ["train", "dev", "test"]:
        input_path = os.path.join(tsv_folder, f"{split}.tsv")

        # Skip missing files gracefully
        if not os.path.exists(input_path):
            continue

        # Read and filter
        df = pd.read_csv(input_path, sep="\t")
        if df.empty:
            continue

        # Assuming the event column is the first column
        event_df = df[df["event"] == event]

        # Write only if filtered data exists
        if not event_df.empty:
            output_path = os.path.join(out_folder, f"{split}.tsv")
            event_df.to_csv(output_path, sep="\t", index=False)

    return out_folder

def get_paths(event: str, lbcl: int, set_num: int, run_num:int=0) -> dict:
    """
    Returns a dictionary of dataset and artifact paths for a given event, label count, and set number.
    """

    dev_path = separate_event(event, r"../data/humaid/joined/dev.tsv", "dev")
    test_path = separate_event(event, r"../data/humaid/joined/test.tsv", "test")
    joined_path = separate_event_folder(event, r"../data/humaid/joined", "joined")

    train_labeled_path = separate_event(
        event, fr"../data/humaid/anh_4o/sep/{lbcl}lb/{set_num}/labeled.tsv", "labeled"
    )
    train_unlabeled_path = separate_event(
        event, fr"../data/humaid/anh_4o/sep/{lbcl}lb/{set_num}/unlabeled.tsv", "unlabeled"
    )

    vmatch_out = fr"../artifacts/humaid/vmatch{run_num}/humaid_vmatch_run_{event}_{lbcl}_{set_num}"
    os.makedirs(vmatch_out, exist_ok=True)

    return {
        "dev_path": dev_path,
        "test_path": test_path,
        "joined_path": joined_path,
        "train_labeled_path": train_labeled_path,
        "train_unlabeled_path": train_unlabeled_path,
        "vmatch_out": vmatch_out,
    }

def main():
    EVENTS = get_events(r"../data/humaid/joined")

    for lbcl, event, set_num in itertools.product(LB_PER_CLASS, EVENTS, SETS):
        tag = f"{event}_lb{lbcl}_set{set_num}"
        print(f"\n=== Running combo: {tag} ===", flush=True)

        dev_path = separate_event(event, r"../data/humaid/joined/dev.tsv", "dev")
        test_path = separate_event(event, r"../data/humaid/joined/test.tsv", "test")
        joined_path = separate_event_folder(event, r"../data/humaid/joined", "joined")
        train_labeled_path = separate_event(event, fr"../data/humaid/anh_4o/sep/{lbcl}lb/{set_num}/labeled.tsv", "labeled")
        train_unlabeled_path = separate_event(event, fr"../data/humaid/anh_4o/sep/{lbcl}lb/{set_num}/unlabeled.tsv", "unlabeled")

        vmatch_out = fr"../artifacts/humaid/vmatch7/humaid_vmatch_run_{event}_{lbcl}_{set_num}"
        os.makedirs(vmatch_out, exist_ok=True)

        # --------- Run train.py ---------
        train_cmd = TRAIN_ARGS_TEMPLATE + [
            "--ckpt_path", fr"{vmatch_out}/model.pt", 
            "--output_path", fr"{vmatch_out}/preds.json",
            "--dev_path", dev_path, 
            "--test_path", test_path,
            "--labeled_train_path", train_labeled_path,
            "--unlabeled_train_path", train_unlabeled_path,
        ]
        code = run_and_stream(train_cmd, f"train[{tag}]")
        if code != 0:
            print(f"[ERROR] train.py failed for {tag} with exit code {code}", file=sys.stderr)
            exit(code)

        # --------- Run bert_ft.py w/ bertweet ---------
        # bert_cmd = BERT_FT_ARGS_TEMPLATE + [
        #     fr"--output_dir", fr"..\artifacts\humaid\bertweet5\humaid_bertweet_ft_{event}_{lbcl}_{set_num}",
        #     fr"--train_path", train_labeled_path,
        #     fr"--dataset_path", joined_path,
        #     fr"--model_name", fr"vinai/bertweet-base",
        #     fr"--label_order", 
        #     "requests_or_urgent_needs",
        #     "rescue_volunteering_or_donation_effort",
        #     "infrastructure_and_utility_damage",
        #     "missing_or_found_people",
        #     "displaced_people_and_evacuations",
        #     "sympathy_and_support",
        #     "injured_or_dead_people",
        #     "caution_and_advice",
        #     "other_relevant_information",
        #     "not_humanitarian",
        # ]
        # code = run_and_stream(bert_cmd, f"bertweet[{tag}]")
        # if code != 0:
        #     print(f"[ERROR] bert_ft.py (bertweet) failed for {tag} with exit code {code}", file=sys.stderr)
        #     exit(code)
        
        # --------- Run bert_ft.py w/ bert-base-uncased ---------
        # bert_cmd = BERT_FT_ARGS_TEMPLATE + [
        #     fr"--output_dir", fr"..\artifacts\humaid\bert3\humaid_bert_ft_{event}_{lbcl}_{set_num}",
        #     fr"--train_path", train_labeled_path,
        #     fr"--dataset_path", joined_path,
        # ]
        # code = run_and_stream(bert_cmd, f"bert[{tag}]")
        # if code != 0:
        #     print(f"[ERROR] bert_ft.py failed for {tag} with exit code {code}", file=sys.stderr)
        #     exit(code)

    print("\nAll combinations complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
