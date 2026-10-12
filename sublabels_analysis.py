"""
Reusable functions extracted from evaluation_datasets_all_sublabels.ipynb.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from itertools import combinations, permutations
from scipy.special import rel_entr, kl_div
from scipy.stats import pearsonr, entropy, ttest_1samp, wilcoxon, mannwhitneyu, ks_2samp, false_discovery_control


def plot_sublabels_prob(dataset, data_name):
    """
    Bucket each of the irony/toxicity/humor probability columns into
    [0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1] bins and plot them as a grouped
    percentage bar chart.
    """
    bins = [0, 0.25, 0.5, 0.75, 1]
    hist_data = {}

    for column in ["new_irony_probs", "toxicity_probs", "humor_probs"]:
        hist, _ = np.histogram(dataset[column], bins=bins)
        hist_percentage = hist / hist.sum() * 100
        hist_data[column] = hist_percentage

    hist_df = pd.DataFrame(hist_data, index=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1'])

    bar_width = 0.15
    bin_labels = hist_df.index
    x = np.arange(len(bin_labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, column in enumerate(hist_df.columns):
        label = column[:-len("_probs")].removeprefix("new_")
        bars = ax.bar(x + i * bar_width, hist_df[column], width=bar_width,
                       label=label, color=colors[i % len(colors)])

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{round(height)}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)

    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(bin_labels)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_ylim(0, 100)

    ax.set_xlabel('Score intervals', fontsize=18)
    ax.set_ylabel('Percentage', fontsize=18)
    ax.legend(loc='upper center', fontsize=16)

    plt.show()


def f1_score_eval(in_A_A, cross_A_B):
    """Gap between in-domain and cross-domain F1 score."""
    return in_A_A - cross_A_B


def kl_div_f1(train_data_prob, test_data_prob, in_A_A, cross_A_B):
    """
    Per-sublabel KL divergence between a test and train probability
    distribution, plus the in-domain/cross-domain F1 gap appended at the end.
    Works with however many sublabels train_data_prob/test_data_prob contain.
    """
    epsilon = 1e-10
    train_data_prob = list(train_data_prob)
    test_data_prob = list(test_data_prob)
    sublabel_kl = []
    for p_dist, q_dist in zip(test_data_prob, train_data_prob):
        p = np.array(p_dist)
        q = np.array(q_dist)
        q = np.where(q == 0, epsilon, q)  # avoid division with zero
        sublabel_kl.append(np.sum(rel_entr(p, q)))
    f1_score = f1_score_eval(in_A_A, cross_A_B)
    sublabel_kl.append(f1_score)
    return sublabel_kl


def corr_p_val_plot(data_prop, sublabels=None):
    """
    Correlate F1 score gap (last row) against each sublabel's KL divergence
    (Pearson r + p-value) and scatter-plot them.
    `sublabels` names the rows in order; defaults to the original 6-sublabel
    ordering (humor, irony, politeness, toxicity, empathy, offensive),
    truncated/generated to match however many sublabel rows data_prop has.
    """
    num_sublabels = len(data_prop) - 1
    if sublabels is None:
        default_labels = ['humor', 'irony', 'politeness', 'toxicity', 'empathy', 'offensive']
        if num_sublabels <= len(default_labels):
            sublabels = default_labels[:num_sublabels]
        else:
            sublabels = [f"sublabel_{i}" for i in range(num_sublabels)]

    f1_scores = data_prop.iloc[-1].tolist()
    for idx, label in enumerate(sublabels):
        kl_values = data_prop.iloc[idx].tolist()
        print(f"{label} : ", pearsonr(f1_scores, kl_values))

    colors = ['blue', 'green', 'red', 'purple', 'yellow', 'orange']

    plt.figure(figsize=(10, 6))

    last_row = data_prop.iloc[-1]
    for idx, label in enumerate(sublabels):
        x_values = data_prop.iloc[idx]
        y_values = last_row
        plt.scatter(x_values, y_values, color=colors[idx % len(colors)], label=label)

    plt.title("Sublabels' KL Divergence and F1 Score of In and Cross Domain")
    plt.xlabel('KL divergence')
    plt.ylabel('F1 score (the gap between in domain and cross domain)')
    plt.legend()

    plt.show()


def _residualize(target, controls):
    """Regress `target` on `controls` (+ intercept) and return the residuals."""
    design = sm.add_constant(np.column_stack(controls))
    fit = sm.OLS(target, design).fit()
    return target - fit.predict(design)


def partial_corr_analysis(analysis_table, sublabel_columns, control_columns):
    """
    Partial Pearson correlation between each affective sublabel's KL
    divergence and the F1 gap, controlling for generic distribution-shift
    covariates (e.g. lexical_js_divergence, embedding_distance from
    generic_shift.py).

    `analysis_table` is a DataFrame with one row per (train, test) pair and
    columns: 'f1_gap', each name in `sublabel_columns`, each name in
    `control_columns`. Controls are the generic-shift metrics to net out.

    Returns a DataFrame with one row per sublabel: zero-order r/p (no
    controls) alongside the partial r/p (controls removed via residualization
    on both the sublabel KL and the F1 gap, then correlating the residuals).
    If a sublabel's partial r stays sizeable and significant while the
    zero-order r shrinks little, that sublabel's KL is not simply standing in
    for the generic-shift controls.

    FDR family: Benjamini-Hochberg is applied over exactly the tests in
    `sublabel_columns` (one partial-correlation test per sublabel), nothing
    else. The family is recorded in the 'fdr_family' column of the output.

    The parametric p-values here assume independent pairs, which (train, test)
    pairs are not - see qap_partial_corr_test for the dependence-aware test.
    """
    f1_gap = analysis_table["f1_gap"].to_numpy(dtype=float)
    controls = [analysis_table[col].to_numpy(dtype=float) for col in control_columns]

    rows = []
    for sublabel in sublabel_columns:
        kl_values = analysis_table[sublabel].to_numpy(dtype=float)

        zero_order_r, zero_order_p = pearsonr(kl_values, f1_gap)

        resid_kl = _residualize(kl_values, controls)
        resid_f1 = _residualize(f1_gap, controls)
        partial_r, partial_p = pearsonr(resid_kl, resid_f1)

        rows.append({
            "sublabel": sublabel,
            "zero_order_r": zero_order_r,
            "zero_order_p": zero_order_p,
            "partial_r": partial_r,
            "partial_p": partial_p,
            "n_controls": len(control_columns),
        })

    results = pd.DataFrame(rows)
    results["partial_p_fdr"] = false_discovery_control(results["partial_p"], method="bh")
    results["fdr_family"] = ", ".join(sublabel_columns)
    return results


def _residual_maker(controls):
    """Annihilator matrix M = I - Z(Z'Z)^-1 Z' for Z = [1, controls]."""
    design = sm.add_constant(np.column_stack(controls))
    return np.eye(design.shape[0]) - design @ np.linalg.pinv(design)


def qap_partial_corr_test(analysis_table, sublabel_columns, control_columns,
                          train_column="train_dataset", test_column="test_dataset"):
    """
    Dependence-aware version of partial_corr_analysis for dyadic data.

    The (train, test) pairs are not independent: each dataset appears in
    2*(k-1) of the k*(k-1) pairs (14 of 56 for k=8), and all pairs sharing a
    train set share its in-domain F1. Treating the 56 pairs as i.i.d. inflates
    significance. Instead, this uses a QAP (node-label) permutation test with
    Dekker et al.'s (2007) double semi-partialling: the sublabel KL is
    residualized on the controls, the residual pair-matrix is permuted by
    relabelling the datasets (same permutation on rows and columns), and the
    partial correlation of the permuted residual with the F1 gap given the
    controls is recomputed. Relabelling datasets preserves all of the
    row/column dependence among pairs, so the null distribution reflects the
    real effective sample size (k datasets, not 56 pairs).

    With k datasets there are k! relabellings; all are enumerated (exact test,
    40,320 for k=8). The statistic |partial r| is monotone in the OLS t for the
    sublabel coefficient, so this is equivalent to DSP-MRQAP on t.

    Only the pairs in `analysis_table` are used. If some ordered pairs are
    absent (e.g. excluded pairs), a relabelling can map an observed pair onto
    an absent one; such cells are treated as missing, and each permutation's
    partial correlation is computed on the pairs observed both before and
    after relabelling (available-case, as in UCINET's QAP with missing data).
    The identity permutation uses all pairs, so the observed statistic is the
    ordinary partial r. Min/median pairs used per permutation are reported.

    FDR family: BH over the QAP p-values of exactly `sublabel_columns`.
    """
    pairs = list(zip(analysis_table[train_column], analysis_table[test_column]))
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    datasets = sorted(set(analysis_table[train_column]) | set(analysis_table[test_column]))
    position = {name: i for i, name in enumerate(datasets)}

    controls = np.column_stack([analysis_table[col].to_numpy(dtype=float) for col in control_columns])
    f1_gap = analysis_table["f1_gap"].to_numpy(dtype=float)
    M = _residual_maker([controls])
    resid_kl = np.column_stack([M @ analysis_table[s].to_numpy(dtype=float) for s in sublabel_columns])

    r_perm, n_used = [], []
    for perm in permutations(range(len(datasets))):  # identity first
        # row that observed pair i's KL residual comes from under this relabelling (-1 = absent pair)
        source = np.array([
            pair_index.get((datasets[perm[position[a]]], datasets[perm[position[b]]]), -1) for a, b in pairs
        ])
        keep = source >= 0
        M_keep = _residual_maker([controls[keep]])
        x = M_keep @ resid_kl[source[keep]]
        y = M_keep @ f1_gap[keep]
        r_perm.append((y @ x) / (np.linalg.norm(y) * np.linalg.norm(x, axis=0)))
        n_used.append(int(keep.sum()))
    r_perm = np.array(r_perm)
    observed_r = r_perm[0]

    results = pd.DataFrame({
        "sublabel": sublabel_columns,
        "partial_r": observed_r,
        "qap_p": np.mean(np.abs(r_perm) >= np.abs(observed_r) - 1e-12, axis=0),
        "n_permutations": len(r_perm),
        "min_pairs_per_permutation": min(n_used),
        "median_pairs_per_permutation": float(np.median(n_used)),
    })
    results["qap_p_fdr"] = false_discovery_control(results["qap_p"], method="bh")
    results["fdr_family"] = ", ".join(sublabel_columns)
    return results


def leave_one_dataset_out(analysis_table, sublabel_columns, control_columns,
                          train_column="train_dataset", test_column="test_dataset"):
    """
    Robustness check for dataset-driven results: drop every pair involving
    one dataset (as train or test), recompute the partial correlations on the
    remaining pairs, and repeat for each dataset. A result carried by a single
    dataset shows up as a large swing in partial r when that dataset is held out.
    """
    datasets = sorted(set(analysis_table[train_column]) | set(analysis_table[test_column]))
    rows = []
    for held_out in datasets:
        keep = (analysis_table[train_column] != held_out) & (analysis_table[test_column] != held_out)
        subset = partial_corr_analysis(analysis_table[keep], sublabel_columns, control_columns)
        subset["held_out"] = held_out
        subset["n_pairs"] = int(keep.sum())
        rows.append(subset[["held_out", "n_pairs", "sublabel", "partial_r", "partial_p"]])
    return pd.concat(rows, ignore_index=True)


def regression_analysis(analysis_table, sublabel_columns, control_columns, cluster_column=None):
    """
    Nested OLS model comparison: does adding the affective-sublabel KL
    divergences to a model already containing the generic-shift controls
    significantly improve fit on the F1 gap?

    Fits:
      - generic-only model:   f1_gap ~ control_columns
      - full model:           f1_gap ~ control_columns + sublabel_columns
    and compares them with an F-test on the incremental R^2. Also reports
    each affective sublabel's coefficient/p-value in the full model, which is
    the number that answers "does affective KL predict the F1 gap after
    accounting for generic shift".

    If `cluster_column` is given (e.g. 'train_dataset'), the full model is
    refit with cluster-robust standard errors on that column, since several
    (train, test) pairs share the same in-domain F1 and are not independent
    observations (see f1_gap_significance's docstring for the same caveat).
    """
    f1_gap = analysis_table["f1_gap"].astype(float)

    X_generic = sm.add_constant(analysis_table[control_columns].astype(float))
    X_full = sm.add_constant(analysis_table[control_columns + sublabel_columns].astype(float))

    model_generic = sm.OLS(f1_gap, X_generic).fit()
    model_full = sm.OLS(f1_gap, X_full).fit()

    f_stat, f_p_value, _ = model_full.compare_f_test(model_generic)

    result = {
        "generic_only_r2": model_generic.rsquared,
        "full_model_r2": model_full.rsquared,
        "incremental_r2": model_full.rsquared - model_generic.rsquared,
        "f_stat": f_stat,
        "f_p_value": f_p_value,
        "full_model_params": model_full.params.to_dict(),
        "full_model_pvalues": model_full.pvalues.to_dict(),
    }

    if cluster_column is not None:
        clustered_full = sm.OLS(f1_gap, X_full).fit(
            cov_type="cluster",
            cov_kwds={"groups": analysis_table[cluster_column]},
        )
        result["full_model_pvalues_cluster_robust"] = clustered_full.pvalues.to_dict()

    return result


def calc_entropy(prob_distribution):
    """Shannon entropy (base 2) of a sublabel probability distribution."""
    return entropy(np.array(prob_distribution), base=2)


def pairwise_sublabel_significance(datasets, columns=None, test="mannwhitney"):
    """
    Test whether each sublabel's raw probability distribution differs
    significantly between every pair of datasets.

    `datasets` is a dict of {dataset_name: dataframe}, each dataframe holding
    the raw per-example sublabel probability columns (as loaded from
    `test_data_filtered.csv`, i.e. before any binning). Since the two samples
    being compared are independent (different examples, usually different
    sizes), this uses two-sample tests rather than a paired test:
    - "mannwhitney": Mann-Whitney U test (rank-based, no normality assumption)
    - "ks": two-sample Kolmogorov-Smirnov test (sensitive to any distribution
      shape difference, not just location)

    Benjamini-Hochberg FDR correction is applied across all pairwise/sublabel
    comparisons at once, since this runs many tests.
    """
    if columns is None:
        columns = ["new_irony_probs", "toxicity_probs", "humor_probs"]
    test_fn = {"mannwhitney": mannwhitneyu, "ks": ks_2samp}[test]

    rows = []
    for name_a, name_b in combinations(datasets, 2):
        df_a, df_b = datasets[name_a], datasets[name_b]
        for column in columns:
            stat, p_value = test_fn(df_a[column].dropna(), df_b[column].dropna())
            rows.append({
                "dataset_a": name_a,
                "dataset_b": name_b,
                "sublabel": column[:-len("_probs")].removeprefix("new_"),
                "statistic": stat,
                "p_value": p_value,
            })

    results = pd.DataFrame(rows)
    results["p_value_fdr"] = false_discovery_control(results["p_value"], method="bh")
    results["significant_fdr_0.05"] = results["p_value_fdr"] < 0.05
    return results.sort_values("p_value").reset_index(drop=True)


def f1_gap_significance(datsets_kl_div_f1, popmean=0):
    """
    Test whether the in-domain vs. cross-domain F1 gap (the last row of
    `datsets_kl_div_f1`, one value per train/test pair) is significantly
    different from `popmean` (0 by default, i.e. "no gap").

    Each pair contributes one paired observation (in-domain F1 minus
    cross-domain F1 for that pair), so testing the gaps against 0 with a
    one-sample test is equivalent to a paired test between the in-domain and
    cross-domain F1 scores. Reports both the parametric paired t-test and
    the non-parametric Wilcoxon signed-rank test as a robustness check.

    Note: several pairs share the same in-domain F1 (e.g. every "MUSTARD_vrs_*"
    pair reuses MUSTARD's in-domain score), so observations are not fully
    independent - keep that in mind when interpreting the p-value.
    """
    gaps = datsets_kl_div_f1.iloc[-1].astype(float)

    t_stat, t_p = ttest_1samp(gaps, popmean=popmean)
    w_stat, w_p = wilcoxon(gaps - popmean)

    n = len(gaps)
    cohens_d = (gaps.mean() - popmean) / gaps.std(ddof=1)

    return {
        "n_pairs": n,
        "mean_gap": gaps.mean(),
        "std_gap": gaps.std(ddof=1),
        "cohens_d": cohens_d,
        "paired_ttest_stat": t_stat,
        "paired_ttest_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }


def plot_kl_divergence_heatmap(sublabels_prob):
    """
    Pairwise dataset dissimilarity heatmap: for every pair of datasets in
    `sublabels_prob` (a DataFrame keyed by dataset name, each column a list
    of per-sublabel probability distributions), average the KL divergence
    across sublabels and plot it as a Reds heatmap. Diagonal is 0.
    """
    epsilon = 1e-10
    dataset_names = list(sublabels_prob.keys())
    kl_matrix = pd.DataFrame(index=dataset_names, columns=dataset_names, dtype=float)

    for dataset1, probs1 in sublabels_prob.items():
        for dataset2, probs2 in sublabels_prob.items():
            if dataset1 == dataset2:
                kl_matrix.loc[dataset1, dataset2] = 0
            else:
                kl_values = [
                    sum(kl_div(np.array(p1) + epsilon, np.array(p2) + epsilon))
                    for p1, p2 in zip(probs1, probs2)
                ]
                kl_matrix.loc[dataset1, dataset2] = np.mean(kl_values)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(kl_matrix, annot=True, cmap="Reds", linewidths=0.5, fmt=".2f", annot_kws={"size": 14})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=14)
    plt.show()
