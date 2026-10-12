import argparse
import glob
import os

import pandas as pd


def aggregate_dir(directory, output_name="aggregated.csv"):
    output_path = os.path.join(directory, output_name)
    paths = sorted(
        p for p in glob.glob(os.path.join(directory, "*.csv"))
        if os.path.basename(p) != output_name
    )
    if not paths:
        raise FileNotFoundError(f"No result CSVs found in {directory}")

    frames = [pd.read_csv(p).set_index("test data") for p in paths]
    row_order = frames[0].index

    combined = pd.concat(frames)
    averaged = combined.groupby(level=0).mean().reindex(row_order)
    averaged.insert(0, "n_files_averaged", len(paths))
    averaged.reset_index().to_csv(output_path, index=False)

    print(f"{directory}: averaged {len(paths)} files -> {output_path}")
    for p in paths:
        print(f"  - {os.path.basename(p)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Average the per-fold result CSVs in a directory (same 'test data' rows) into one aggregated.csv in that same directory."
    )
    parser.add_argument("directories", nargs="+", help="one or more directories, each containing the CSVs to average")
    args = parser.parse_args()

    for d in args.directories:
        aggregate_dir(d)
