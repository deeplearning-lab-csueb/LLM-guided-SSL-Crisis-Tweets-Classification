import argparse
import os
from datasets import load_dataset

def hf_dl():
    p = argparse.ArgumentParser(description="Download and save HF dataset as-is")
    p.add_argument("--dataset_name", required=True, help="HF dataset path (e.g., 'tweet_eval' or 'username/ds')")
    p.add_argument("--out_dir", help="Output directory. Defaults to ./[dataset_name]")

    args = p.parse_args()
    if not args.out_dir:
        args.out_dir = "./" + args.dataset_name.replace("/", "_")
    os.makedirs(args.out_dir, exist_ok=True)

    # Download full dataset (not streaming, since we want to save all)
    ds = load_dataset(
        path=args.dataset_name,
    )

    # Save in Hugging Face native Arrow format
    ds.save_to_disk(args.out_dir)
    print(f"Dataset saved to {args.out_dir}")

if __name__ == "__main__":
    hf_dl()
