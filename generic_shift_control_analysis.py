"""
Does affective KL divergence explain the in-domain minus cross-domain F1 gap
beyond generic (lexical + embedding) domain shift?

Affective KL and F1 gaps are the ones from sublabels_analysis_demo.ipynb
(irony binned live from new_irony_probs, humor/toxicity distributions and
F1 scores as curated there). Only those 46 train/test pairs are used anywhere
in the analysis. Generic-shift controls come from generic_shift.py
(generic_kl_divergence_results.csv).

Reports, for the affective-aware dimensions (humor, irony, toxicity):
  - parametric zero-order and partial correlations (assume independent pairs)
  - exact QAP permutation p-values (account for pairs sharing a dataset)
  - leave-one-dataset-out partial correlations
FDR family: Benjamini-Hochberg over the 3 dimensions, separately for the
parametric and the QAP p-values.
"""
import numpy as np
import pandas as pd
from sublabels_analysis import kl_div_f1, partial_corr_analysis, qap_partial_corr_test, leave_one_dataset_out

AFFECTIVE_DIMS = ["humor", "irony", "toxicity"]
CONTROLS = ["lexical_js_divergence", "embedding_distance"]
AUX_LABELS_DIR = "scores_results_finetuned/aux_labels"
IRONY_BINS = [0, 0.25, 0.5, 0.75, 1]

# Curated in sublabels_analysis_demo.ipynb (humor/toxicity 4-bin distributions)
HUMOR_TOXICITY = {
    "MUStARD": ([0.11, 0, 0, 0.89], [0.73, 0, 0, 0.27]),
    "SARC": ([0.24, 0.01, 0, 0.75], [0.73, 0, 0, 0.27]),
    "SPIRS": ([0.07, 0, 0, 0.93], [0.72, 0, 0, 0.28]),
    "iSarcasmEval": ([0.26, 0, 0, 0.74], [0.83, 0, 0, 0.17]),
    "multimodal_sarcasm_detection": ([0.1, 0, 0, 0.9], [0.68, 0, 0, 0.32]),
    "SemEval2022": ([0.24, 0, 0, 0.76], [0.69, 0, 0, 0.31]),
    "Ptacek": ([0.02, 0, 0, 0.98], [0.32, 0, 0, 0.68]),
    "Riloff": ([0.02, 0, 0, 0.98], [0.31, 0, 0, 0.69]),
}

MSD = "multimodal_sarcasm_detection"
# (train, test, in_domain_f1, cross_domain_f1), curated in sublabels_analysis_demo.ipynb
F1_PAIRS = [
    ("MUStARD", "SARC", 0.611, 0.59), ("MUStARD", "SPIRS", 0.611, 0.61),
    (MSD, "iSarcasmEval", 0.867, 0.46), (MSD, "SemEval2022", 0.867, 0.639), (MSD, "MUStARD", 0.867, 0.43),
    (MSD, "SPIRS", 0.867, 0.5), (MSD, "SARC", 0.867, 0.396), (MSD, "Riloff", 0.867, 0.811),
    ("iSarcasmEval", "MUStARD", 0.87, 0.547), ("iSarcasmEval", "SPIRS", 0.87, 0.651),
    ("iSarcasmEval", MSD, 0.87, 0.657), ("iSarcasmEval", "Ptacek", 0.87, 0.667),
    ("iSarcasmEval", "Riloff", 0.87, 0.672), ("iSarcasmEval", "SemEval2022", 0.87, 0.672),
    ("iSarcasmEval", "SARC", 0.87, 0.597),
    ("SPIRS", "iSarcasmEval", 0.703, 0.704), ("SPIRS", "MUStARD", 0.703, 0.539), ("SPIRS", "SARC", 0.703, 0.66),
    ("SPIRS", MSD, 0.703, 0.585), ("SPIRS", "SemEval2022", 0.703, 0.663),
    ("SARC", "SemEval2022", 0.679, 0.618), ("SARC", "iSarcasmEval", 0.679, 0.477), ("SARC", "Ptacek", 0.679, 0.513),
    ("SARC", "MUStARD", 0.679, 0.47), ("SARC", MSD, 0.679, 0.428), ("SARC", "Riloff", 0.679, 0.438),
    ("SARC", "SPIRS", 0.679, 0.494),
    ("SemEval2022", "SPIRS", 0.641, 0.528), ("SemEval2022", "iSarcasmEval", 0.641, 0.599),
    ("SemEval2022", "MUStARD", 0.641, 0.506), ("SemEval2022", MSD, 0.641, 0.521), ("SemEval2022", "SARC", 0.641, 0.557),
    ("Ptacek", "Riloff", 0.995, 0.836), ("Ptacek", "SPIRS", 0.995, 0.086), ("Ptacek", "iSarcasmEval", 0.995, 0.03),
    ("Ptacek", "MUStARD", 0.995, 0), ("Ptacek", MSD, 0.995, 0.39), ("Ptacek", "SARC", 0.995, 0.019),
    ("Ptacek", "SemEval2022", 0.995, 0.011),
    ("Riloff", "SPIRS", 0.822, 0.026), ("Riloff", "Ptacek", 0.822, 0.846), ("Riloff", "iSarcasmEval", 0.822, 0.076),
    ("Riloff", "MUStARD", 0.822, 0.06), ("Riloff", MSD, 0.822, 0.073), ("Riloff", "SemEval2022", 0.822, 0.098),
    ("Riloff", "SARC", 0.822, 0.101),
]


def irony_bin_distribution(dataset_name):
    df = pd.read_csv(f"{AUX_LABELS_DIR}/{dataset_name}/test_data_filtered.csv")
    hist, _ = np.histogram(df["new_irony_probs"], bins=IRONY_BINS)
    return (hist / hist.sum()).tolist()


def sublabel_distributions():
    """{dataset: [humor, irony, toxicity] distributions}, as in the demo notebook."""
    return {
        name: [humor, irony_bin_distribution(name), toxicity]
        for name, (humor, toxicity) in HUMOR_TOXICITY.items()
    }


def build_table():
    """The 46 demo pairs with per-dimension KL, f1_gap and generic-shift controls."""
    probs = sublabel_distributions()
    rows = []
    for train, test, in_f1, cross_f1 in F1_PAIRS:
        *kl, f1_gap = kl_div_f1(probs[train], probs[test], in_f1, cross_f1)
        rows.append({"train_dataset": train, "test_dataset": test, "in_domain_f1": in_f1,
                     "cross_domain_f1": cross_f1, "f1_gap": f1_gap, **dict(zip(AFFECTIVE_DIMS, kl))})
    controls = pd.read_csv("generic_kl_divergence_results.csv")
    table = pd.DataFrame(rows).merge(controls, on=["train_dataset", "test_dataset"], how="left", validate="one_to_one")
    assert len(table) == len(F1_PAIRS) and not table.isna().any().any()
    return table


if __name__ == "__main__":
    observed = build_table()
    observed.to_csv("generic_shift_control_analysis_table_demo46.csv", index=False)

    parametric = partial_corr_analysis(observed, AFFECTIVE_DIMS, CONTROLS)
    qap = qap_partial_corr_test(observed, AFFECTIVE_DIMS, CONTROLS)
    results = parametric.merge(qap[["sublabel", "qap_p", "qap_p_fdr", "n_permutations",
                                    "min_pairs_per_permutation", "median_pairs_per_permutation"]], on="sublabel")
    results["n_pairs"] = len(observed)
    results.to_csv("partial_corr_results_demo46.csv", index=False)

    lodo = leave_one_dataset_out(observed, AFFECTIVE_DIMS, CONTROLS)
    lodo.to_csv("partial_corr_leave_one_dataset_out_demo46.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(results.drop(columns="fdr_family").round(4).to_string())
    print("FDR family:", results["fdr_family"].iloc[0])
    print(observed[CONTROLS + ["f1_gap"]].corr().round(3))
    print(lodo.pivot(index="held_out", columns="sublabel", values="partial_r").round(3))
    print(lodo.groupby("held_out")["n_pairs"].first())
