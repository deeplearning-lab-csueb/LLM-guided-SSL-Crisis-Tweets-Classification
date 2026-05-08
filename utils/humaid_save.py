import pandas as pd
import argparse
import os

def join_tsv(tsv1_path, tsv2_path, join_col, output_path):
    # Load TSV files
    df1 = pd.read_csv(tsv1_path, sep="\t")
    df2 = pd.read_csv(tsv2_path, sep="\t")

    # Merge on the join column (inner join by default)
    merged = pd.merge(df1, df2, on=join_col, how="inner")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    # Write to output TSV
    merged.to_csv(output_path, sep="\t", index=False)

    print(f"✅ Joined TSV saved to: {output_path}")
    print(f"Rows: {len(merged)} | Columns: {len(merged.columns)}")

    # Also write to a csv in the form of id,gold,pred
    merged.rename(columns={"tweet_id":"id", "class_label":"gold", "label":"pred"}, inplace=True)

    csv_output_path = output_path.replace(".tsv", ".csv")
    merged.to_csv(csv_output_path, columns=["id", "gold", "pred"], index=False)
    print(f"✅ CSV saved to: {csv_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join two TSV files based on a common column.")
    parser.add_argument("--tsv1", required=True, help="Path to the first TSV file")
    parser.add_argument("--tsv2", required=True, help="Path to the second TSV file")
    parser.add_argument("--join_col", default="tweet_text", help="Column to join on (default: tweet_text)")
    parser.add_argument("--output_path", default="./joined.tsv", help="Output file path (default: ./joined.tsv)")
    
    args = parser.parse_args()
    join_tsv(args.tsv1, args.tsv2, args.join_col, args.output_path)
