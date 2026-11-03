
import pandas as pd
import sys
from pathlib import Path


def split_by_learner(input_csv: str, output_dir: str = "."):
    df = pd.read_csv(input_csv)

    required = {"dataset", "learner", "acc"}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: missing column in input CSV: {missing}")
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for learner, group in df.groupby("learner"):
        result = group[["dataset", "acc"]].reset_index(drop=True)
        path = out / f"{learner}.csv"
        result.to_csv(path, index=False, mode='a', header=False)
        print(f"  {path}  ({len(result)} rows)")


if __name__ == "__main__":
    input_csv = 'baselines/saga/transfer_results.csv'
    output_dir = 'baselines/saga/results'
    split_by_learner(input_csv, output_dir)

'''
python baselines/saga/partition_results.py
'''
