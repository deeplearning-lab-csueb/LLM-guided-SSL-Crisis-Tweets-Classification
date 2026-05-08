import pandas as pd
import argparse

def keep_label_column(input_tsv, output_tsv):
    # Read TSV
    df = pd.read_csv(input_tsv, sep='\t')
    
    # Keep only 'tweet_text' column
    if 'tweet_text' not in df.columns:
        raise ValueError("The input TSV does not contain a 'tweet_text' column.")
    df = df[['tweet_text']]
    
    # Save to new TSV
    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"Saved file with only 'tweet_text' column to: {output_tsv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop all columns except 'tweet_text' from a TSV file.")
    parser.add_argument("--input_tsv", required=True, help="Path to input TSV file.")
    parser.add_argument("--output_tsv", required=True, help="Path to save output TSV file.")
    args = parser.parse_args()

    keep_label_column(args.input_tsv, args.output_tsv)
