import argparse
import glob
import os
import re

import pandas as pd

# Maps each training-file naming pattern to its experiment label.
# Order matters: "irony_txc_neg" must be checked before "irony_neg" / "txc_neg"
# since it is a superstring of both.
GROUP_PATTERNS = [
    ("both", re.compile(r"^Ptacek65_irony_txc_neg\d+\.csv$")),
    ("irony", re.compile(r"^Ptacek65_irony_neg\d+\.csv$")),
    ("toxic", re.compile(r"^Ptacek65_txc_neg\d+\.csv$")),
    ("random", re.compile(r"^Ptacek65_random\d+\.csv$")),
]

EXCLUDED_TEST_SETS = {"MSTI"}


def classify(filename):
    for label, pattern in GROUP_PATTERNS:
        if pattern.match(filename):
            return label
    return None


def is_excluded(test_data_value):
    if "(CV heldout)" in test_data_value:
        return True
    return test_data_value in EXCLUDED_TEST_SETS


def aggregate(directory, output_path):
    rows = []
    grouped_files = {label: [] for label, _ in GROUP_PATTERNS}
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        label = classify(os.path.basename(path))
        if label is not None:
            grouped_files[label].append(path)

    for label, _ in GROUP_PATTERNS:
        paths = grouped_files[label]
        if not paths:
            print(f"[skip] no files found for experiment '{label}'")
            continue

        frames = []
        for p in paths:
            df = pd.read_csv(p)
            df = df[~df["test data"].apply(is_excluded)]
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        test_set_order = frames[0]["test data"].tolist()

        # average across the 5 seeds, per test set (kept separate, not collapsed)
        per_test_set = (
            combined.groupby("test data")[["mean f1 binary", "std f1 binary"]]
            .mean()
            .reindex(test_set_order)
            .reset_index()
        )
        per_test_set.insert(0, "experiment", label)
        per_test_set.insert(1, "n_seed_files", len(paths))
        rows.append(per_test_set)
        print(f"[{label}] averaged {len(paths)} seed files across {len(per_test_set)} test sets")

    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nwrote summary -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For each Ptacek_aug65 experiment (irony/toxic/both/random), "
                     "average f1-binary mean/std across the 5 seed files and across "
                     "all test sets except the CV heldout fold and MSTI."
    )
    parser.add_argument(
        "--directory",
        default="scores_results_finetuned/roberta-base/Ptacek_aug65",
        help="directory containing the per-seed result CSVs",
    )
    parser.add_argument(
        "--output",
        default="scores_results_finetuned/roberta-base/Ptacek_aug65_summary.csv",
        help="path to write the aggregated summary CSV",
    )
    args = parser.parse_args()
    aggregate(args.directory, args.output)
