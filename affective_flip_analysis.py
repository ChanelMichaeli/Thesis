"""
Affective "flip" analysis.

Model A: BERT trained on Ptacek/train_data_filtered3000.csv
Model B: BERT trained on Ptacek/upsampling_train_imbalanced/Ptacek65_irony_txc_neg0.csv
         (base set + sarcastic samples with irony<0.25 and toxicity<0.25)

For every cross-domain sarcastic test sample we place it in one of four groups:
    Fixed      - A wrong, B right   (what the augmentation recovered)
    Both-wrong - A wrong, B wrong
    Both-right - A right, B right
    Broken     - A right, B wrong
and compare the affective profile (humor / irony / toxicity probabilities from
the aux classifiers) of the groups, and relate it to the augmentation profile.

Inputs (all relative to this file):
    datasets/<DS>/test_data_filtered.csv                            labels
    scores_results_finetuned/affective_flip/predictions/<DS>/...   per-sample P(sarcastic) of A and B
    scores_results_finetuned/aux_labels/<DS>/test_data_filtered.csv affective probs (sarcastic rows)
Outputs:
    scores_results_finetuned/affective_flip/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "datasets")
AUX_DIR = os.path.join(BASE, "scores_results_finetuned", "aux_labels")
OUT_DIR = os.path.join(BASE, "scores_results_finetuned", "affective_flip")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# Ptacek test is excluded (433 of its rows were added to Model B's training set),
# MSTI is excluded because it has no aux (affective) labels.
DATASETS = ["SPIRS", "multimodal_sarcasm_detection", "SARC", "SemEval2022",
            "iSarcasmEval", "Riloff", "MUStARD"]
SHORT = {"multimodal_sarcasm_detection": "MMSD", "iSarcasmEval": "iSarcasm",
         "SemEval2022": "SemEval22"}

# Model A retrained with the same code as Model B (the older savedModels_finetuned/Ptacek
# checkpoint does not reproduce the recorded Model A scores)
COL_A = "Ptacek3000_retrained_probs"
COL_B = "Ptacek65_irony_txc_neg0_probs"
# control R: BERT trained on Ptacek65_random0.csv (same number of added sarcastic rows,
# chosen at random instead of by affect)
COL_R = "Ptacek65_random0_probs"
MODELS = ["A", "B", "R"]
FEATURES = ["humor", "irony", "toxicity"]
AUG_FILE = os.path.join(DATA_DIR, "Ptacek", "upsampling_train_imbalanced",
                        "Ptacek65_irony_txc_neg0.csv")
BASE_TRAIN_FILE = os.path.join(DATA_DIR, "Ptacek", "train_data_filtered3000.csv")

GROUPS = ["Fixed", "Both-wrong", "Both-right", "Broken"]
# reference categorical palette, slots 1-4 (fixed order)
GROUP_COLORS = {"Fixed": "#2a78d6", "Both-wrong": "#eb6834",
                "Both-right": "#1baf7a", "Broken": "#eda100"}
BINS = [0, 0.25, 0.5, 0.75, 1.0000001]   # same 4-bin convention as sublabels_analysis.py
LOW = 0.25                                 # augmentation selection threshold


def short(ds):
    return SHORT.get(ds, ds)


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------
def load_aux(path):
    """Aux probs keyed by text; returns columns text, humor, irony, toxicity."""
    aux = pd.read_csv(path)
    tox = "toxicity_probs" if "toxicity_probs" in aux.columns else "toxcity_probs"
    aux = aux.rename(columns={"humor_probs": "humor", "irony_probs": "irony", tox: "toxicity"})
    return aux[["text"] + FEATURES].drop_duplicates("text")


def assign_group(a_ok, b_ok):
    if not a_ok and b_ok:
        return "Fixed"
    if not a_ok and not b_ok:
        return "Both-wrong"
    if a_ok and b_ok:
        return "Both-right"
    return "Broken"


def load_dataset(ds):
    test = pd.read_csv(os.path.join(DATA_DIR, ds, "test_data_filtered.csv"))
    pred = pd.read_csv(os.path.join(PRED_DIR, ds, "test_data_filtered.csv"))
    assert len(test) == len(pred), f"{ds}: prediction rows != test rows"
    assert (test["text"].astype(str).values == pred["text"].astype(str).values).all(), \
        f"{ds}: prediction rows not aligned with test rows"

    df = test[["label", "text"]].copy()
    df["dataset"] = ds
    for m, col in zip(MODELS, [COL_A, COL_B, COL_R]):
        df[f"{m}_prob"] = pred[col].values
        df[f"{m}_pred"] = (df[f"{m}_prob"] >= 0.5).astype(int)
    return df


def build_sarcastic_table(all_rows):
    """Sarcastic rows with group + affect probs; reports aux join coverage."""
    parts, coverage = [], []
    for ds, df in all_rows.groupby("dataset", sort=False):
        pos = df[df["label"] == 1].copy()
        pos["group"] = [assign_group(a == 1, b == 1) for a, b in zip(pos["A_pred"], pos["B_pred"])]
        aux = load_aux(os.path.join(AUX_DIR, ds, "test_data_filtered.csv"))
        merged = pos.merge(aux, on="text", how="left")
        matched = merged[FEATURES].notna().all(axis=1)
        coverage.append({"dataset": ds, "n_sarcastic": len(pos),
                         "n_with_affect": int(matched.sum()),
                         "match_rate": matched.mean()})
        merged.to_csv(os.path.join(OUT_DIR, f"merged_{ds}.csv"), index=False)
        parts.append(merged[matched])
    return pd.concat(parts, ignore_index=True), pd.DataFrame(coverage)


def reference_profiles():
    """Affect probs of (a) Ptacek3000 sarcastic rows, (b) rows added by the augmentation.
    Both are taken from the current aux_labels files (joined by text) so that they live in the
    same feature space as the test-set probs."""
    aux_ptacek = pd.concat([load_aux(os.path.join(AUX_DIR, "Ptacek", f))
                            for f in ["train_data_filtered.csv", "test_data_filtered.csv"]])
    aux_ptacek = aux_ptacek.drop_duplicates("text")

    base = pd.read_csv(BASE_TRAIN_FILE)
    base_pos = base[base["label"] == 1][["text"]].merge(aux_ptacek, on="text", how="inner")

    aug = pd.read_csv(AUG_FILE)
    added = aug[(aug["label"] == 1) & aug["irony_probs"].notna()][["text"]]
    added = added[~added["text"].isin(base["text"])]
    added_aux = added.merge(aux_ptacek, on="text", how="inner")
    print(f"reference profiles: Ptacek3000 sarcastic {len(base_pos)} rows with affect, "
          f"augmented-added {len(added_aux)}/{len(added)} rows with affect")
    return base_pos, added_aux


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def cliffs_delta(x, y):
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


def compare(x, y):
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    return mannwhitneyu(x, y, alternative="two-sided").pvalue, cliffs_delta(x, y)


def binned(values):
    counts, _ = np.histogram(np.clip(values, 0, 1), bins=BINS)
    return counts


def kl(p_counts, q_counts, eps=1e-6):
    p = p_counts / max(p_counts.sum(), 1) + eps
    q = q_counts / max(q_counts.sum(), 1) + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def in_aug_region(df):
    return (df["irony"] < LOW) & (df["toxicity"] < LOW)


def scopes(sarc):
    """(scope name, frame) for every dataset and for the pooled data."""
    for ds in DATASETS:
        sub = sarc[sarc["dataset"] == ds]
        if len(sub):
            yield short(ds), sub
    yield "Pooled", sarc


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------
def performance_and_prior_shift(all_rows):
    from sklearn.metrics import f1_score, recall_score
    rows = []
    for ds, df in all_rows.groupby("dataset", sort=False):
        neg = df[df["label"] == 0]
        r = {"dataset": ds, "n": len(df), "n_sarcastic": int(df["label"].sum())}
        for m in MODELS:
            pred = df[f"{m}_pred"]
            r[f"{m}_f1_binary"] = f1_score(df["label"], pred, zero_division=0)
            r[f"{m}_f1_macro"] = f1_score(df["label"], pred, average="macro", zero_division=0)
            r[f"{m}_recall"] = recall_score(df["label"], pred, zero_division=0)
            r[f"{m}_pos_rate"] = pred.mean()
            r[f"{m}_FPR"] = neg[f"{m}_pred"].mean() if len(neg) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def group_counts(sarc):
    tab = pd.crosstab(sarc["dataset"].map(short), sarc["group"]).reindex(columns=GROUPS, fill_value=0)
    tab.loc["Pooled"] = tab.sum()
    tab["n"] = tab[GROUPS].sum(axis=1)
    tab["fixed_share_of_A_missed"] = tab["Fixed"] / (tab["Fixed"] + tab["Both-wrong"]).replace(0, np.nan)
    return tab


def feature_tests(sarc):
    rows = []
    for scope, df in scopes(sarc):
        fixed = df[df["group"] == "Fixed"]
        contrasts = {"Fixed vs Both-wrong": df[df["group"] == "Both-wrong"],
                     "Fixed vs rest": df[df["group"] != "Fixed"]}
        for cname, other in contrasts.items():
            for f in FEATURES:
                p, d = compare(fixed[f].values, other[f].values)
                rows.append({"scope": scope, "contrast": cname, "feature": f,
                             "n_fixed": len(fixed), "n_other": len(other),
                             "mean_fixed": fixed[f].mean(), "mean_other": other[f].mean(),
                             "median_fixed": fixed[f].median(), "median_other": other[f].median(),
                             "cliffs_delta": d, "p_value": p})
    res = pd.DataFrame(rows)
    res["p_fdr"] = np.nan
    for cname in res["contrast"].unique():
        m = (res["contrast"] == cname) & res["p_value"].notna()
        if m.any():
            res.loc[m, "p_fdr"] = multipletests(res.loc[m, "p_value"], method="fdr_bh")[1]
    return res


def aug_region_tests(sarc):
    rows = []
    regions = {"low irony & low toxicity": in_aug_region,
               "low irony & low toxicity & humor>0.75": lambda d: in_aug_region(d) & (d["humor"] > 0.75)}
    for scope, df in scopes(sarc):
        for rname, fn in regions.items():
            r = {"scope": scope, "region": rname}
            for g in GROUPS:
                sub = df[df["group"] == g]
                r[f"share_{g}"] = fn(sub).mean() if len(sub) else np.nan
            fx, bw = df[df["group"] == "Fixed"], df[df["group"] == "Both-wrong"]
            if len(fx) and len(bw):
                table = [[fn(fx).sum(), (~fn(fx)).sum()], [fn(bw).sum(), (~fn(bw)).sum()]]
                odds, p = fisher_exact(table)
                r.update({"odds_ratio_fixed_vs_bothwrong": odds, "p_value": p})
            rows.append(r)
    return pd.DataFrame(rows)


def kl_profiles(sarc, base_pos, added):
    rows = []
    refs = {"Ptacek3000 sarcastic": base_pos, "Augmented rows": added}
    for scope, df in scopes(sarc):
        for g in GROUPS:
            sub = df[df["group"] == g]
            if len(sub) < 5:
                continue
            for f in FEATURES:
                r = {"scope": scope, "group": g, "feature": f, "n": len(sub)}
                for rname, ref in refs.items():
                    r[f"KL_to_{rname}"] = kl(binned(sub[f]), binned(ref[f]))
                rows.append(r)
        # chi-square: are the binned distributions of the groups different?
        for f in FEATURES:
            table = np.array([binned(df[df["group"] == g][f]) for g in GROUPS if (df["group"] == g).sum() >= 5])
            table = table[:, table.sum(axis=0) > 0]
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                chi2, p, _, _ = chi2_contingency(table)
                rows.append({"scope": scope, "group": "chi2 across groups", "feature": f,
                             "n": int(table.sum()), "chi2": chi2, "p_value": p})
    return pd.DataFrame(rows)


def cell_ids(df):
    """index of the 4x4x4 (humor, irony, toxicity) bin cell of every row."""
    idx = [np.clip(np.digitize(df[f].values, BINS[1:-1]), 0, 3) for f in FEATURES]
    return idx[0] * 16 + idx[1] * 4 + idx[2]


def aug_affinity(sarc, base_pos, added, alpha=0.5):
    """log( P_aug(cell) / P_ptacek3000(cell) ) of the sample's affect cell.
    > 0: the sample sits in a region of the affect space that the augmentation made more frequent
    than in the original Ptacek sarcastic data. (Binned instead of kNN because the aux probs
    saturate at 0/1, which makes nearest-neighbour distances degenerate.)"""
    n_cells = 64
    p_aug = (np.bincount(cell_ids(added), minlength=n_cells) + alpha) / (len(added) + n_cells * alpha)
    p_base = (np.bincount(cell_ids(base_pos), minlength=n_cells) + alpha) / (len(base_pos) + n_cells * alpha)
    sarc = sarc.copy()
    sarc["aug_affinity"] = np.log(p_aug / p_base)[cell_ids(sarc)]

    rows = []
    for scope, df in scopes(sarc):
        fx = df[df["group"] == "Fixed"]["aug_affinity"]
        bw = df[df["group"] == "Both-wrong"]["aug_affinity"]
        p, d = compare(fx.values, bw.values)
        rows.append({"scope": scope, "n_fixed": len(fx), "n_bothwrong": len(bw),
                     "mean_affinity_fixed": fx.mean(), "mean_affinity_bothwrong": bw.mean(),
                     "share_aug_enriched_fixed": (fx > 0).mean() if len(fx) else np.nan,
                     "share_aug_enriched_bothwrong": (bw > 0).mean() if len(bw) else np.nan,
                     "cliffs_delta": d, "p_value": p})
    return pd.DataFrame(rows), sarc


def logit(sarc):
    """Among sarcastic samples A missed: P(B fixes it) ~ affect + A's own prob + dataset."""
    missed = sarc[sarc["group"].isin(["Fixed", "Both-wrong"])].copy()
    missed["fixed"] = (missed["group"] == "Fixed").astype(int)
    # drop datasets where the outcome does not vary (perfect separation of the FE)
    var = missed.groupby("dataset")["fixed"].transform(lambda s: s.nunique() > 1)
    missed = missed[var]
    cols = FEATURES + ["A_prob"]
    for c in cols:   # standardised -> odds ratio per 1 SD
        missed[c + "_z"] = (missed[c] - missed[c].mean()) / missed[c].std()
    formula = "fixed ~ " + " + ".join(c + "_z" for c in cols)
    if missed["dataset"].nunique() > 1:
        formula += " + C(dataset)"
    fit = smf.logit(formula, data=missed).fit(disp=0)
    ci = fit.conf_int()
    res = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values,
                        "odds_ratio": np.exp(fit.params.values),
                        "OR_ci_low": np.exp(ci[0].values), "OR_ci_high": np.exp(ci[1].values),
                        "p_value": fit.pvalues.values})
    res.attrs["n"] = len(missed)
    res.attrs["pseudo_r2"] = fit.prsquared
    return res


CONTROL_GROUPS = ["B only", "R only", "B and R", "Neither"]


def control_contrast(sarc):
    """Random-augmentation control. Among sarcastic samples A missed, which ones does the
    affect-selected model B recover vs the random-augmentation model R?
    If the affective selection matters, 'B only' should be more in the augmentation region
    (low irony, low toxicity) than 'R only'."""
    missed = sarc[sarc["A_pred"] == 0].copy()
    b, r = missed["B_pred"] == 1, missed["R_pred"] == 1
    missed["control_group"] = np.select([b & ~r, ~b & r, b & r], CONTROL_GROUPS[:3], "Neither")

    summary, tests = [], []
    for scope, df in scopes(missed):
        row = {"scope": scope, "n_A_missed": len(df),
               "n_fixed_by_B": int((df["B_pred"] == 1).sum()),
               "n_fixed_by_R": int((df["R_pred"] == 1).sum())}
        for g in CONTROL_GROUPS:
            sub = df[df["control_group"] == g]
            row[f"n_{g}"] = len(sub)
            row[f"aug_region_{g}"] = in_aug_region(sub).mean() if len(sub) else np.nan
            row[f"irony_{g}"] = sub["irony"].mean() if len(sub) else np.nan
        for m in ["B", "R"]:
            fixed = df[df[f"{m}_pred"] == 1]
            row[f"aug_region_all_fixed_by_{m}"] = in_aug_region(fixed).mean() if len(fixed) else np.nan
        summary.append(row)

        bo, ro = df[df["control_group"] == "B only"], df[df["control_group"] == "R only"]
        for f in FEATURES:
            p, d = compare(bo[f].values, ro[f].values)
            tests.append({"scope": scope, "test": f, "n_B_only": len(bo), "n_R_only": len(ro),
                          "value_B_only": bo[f].mean(), "value_R_only": ro[f].mean(),
                          "cliffs_delta": d, "p_value": p})
        if len(bo) and len(ro):
            table = [[in_aug_region(bo).sum(), (~in_aug_region(bo)).sum()],
                     [in_aug_region(ro).sum(), (~in_aug_region(ro)).sum()]]
            odds, p = fisher_exact(table)
            tests.append({"scope": scope, "test": "aug region share", "n_B_only": len(bo),
                          "n_R_only": len(ro), "value_B_only": in_aug_region(bo).mean(),
                          "value_R_only": in_aug_region(ro).mean(), "odds_ratio": odds, "p_value": p})
    return pd.DataFrame(summary), pd.DataFrame(tests), missed


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.spines["bottom"].set_color("#888")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_feature_shares(sarc, base_pos, added, min_n=20):
    """Share of samples with prob > 0.5 per affect dimension (the aux probs are near-binary,
    so shares are more informative than box plots). Reference profiles drawn as lines."""
    scope_list = [(s, d) for s, d in scopes(sarc) if s in ("SPIRS", "MMSD", "Pooled")]
    groups = [g for g in GROUPS if (sarc["group"] == g).sum() >= min_n]
    fig, axes = plt.subplots(1, len(scope_list), figsize=(13, 4.2), sharey=True)
    w = 0.8 / len(groups)
    x = np.arange(len(FEATURES))
    for ax, (scope, df) in zip(axes, scope_list):
        for j, g in enumerate(groups):
            sub = df[df["group"] == g]
            if len(sub) < min_n:
                continue
            vals = [(sub[f] > 0.5).mean() for f in FEATURES]
            ax.bar(x + (j - (len(groups) - 1) / 2) * w, vals, width=w - 0.03,
                   color=GROUP_COLORS[g], label=f"{g}")
        for ref, name, ls in [(base_pos, "Ptacek3000 sarcastic", "-"), (added, "Augmented rows", "--")]:
            for i, f in enumerate(FEATURES):
                ax.hlines((ref[f] > 0.5).mean(), i - 0.45, i + 0.45, colors="#222", linestyles=ls,
                          linewidth=1.5, label=name if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(FEATURES)
        ax.set_title(scope, fontsize=11)
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_ylabel("share with prob > 0.5")
    legend = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(l, h)
    labels, handles = list(legend), list(legend.values())
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "feature_shares.png"), dpi=150)
    plt.close(fig)


def fig_region_share(region_df):
    df = region_df[region_df["region"] == "low irony & low toxicity"].set_index("scope")
    x = np.arange(len(df))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for j, g in enumerate(GROUPS):
        ax.bar(x + (j - 1.5) * w, df[f"share_{g}"].values, width=w - 0.02,
               color=GROUP_COLORS[g], label=g)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("share in augmentation region\n(irony<0.25 & toxicity<0.25)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "aug_region_share.png"), dpi=150)
    plt.close(fig)


def fig_kl(kl_df):
    df = kl_df[(kl_df["scope"] == "Pooled") & kl_df["group"].isin(GROUPS)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, ref in zip(axes, ["Ptacek3000 sarcastic", "Augmented rows"]):
        mat = df.pivot(index="group", columns="feature", values=f"KL_to_{ref}").reindex(GROUPS)[FEATURES]
        im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=np.nanmax(df[["KL_to_Ptacek3000 sarcastic", "KL_to_Augmented rows"]].values))
        for (i, j), v in np.ndenumerate(mat.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 * im.get_clim()[1] else "#222", fontsize=9)
        ax.set_xticks(range(len(FEATURES)))
        ax.set_xticklabels(FEATURES)
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels(GROUPS)
        ax.set_title(f"KL(group ‖ {ref})", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(os.path.join(FIG_DIR, "kl_profiles_pooled.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fmt(x, nd=3):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def write_summary(perf, coverage, counts, ftests, region, kl_df, prox, logit_res, control, sarc):
    L = ["# Affective flip analysis: Model A (Ptacek3000) → Model B (Ptacek65_irony_txc_neg0)", ""]
    L += ["Groups (sarcastic test samples only): **Fixed** = A wrong & B right; **Both-wrong**; "
          "**Both-right**; **Broken** = A right & B wrong. Ptacek test excluded (leakage), MSTI excluded "
          "(no affect labels).", ""]

    L += ["## 1. Performance and prediction-rate shift", "",
          "R = control model trained on Ptacek65_random0 (same amount of added sarcastic rows, random).", "",
          "| dataset | F1 A | F1 B | F1 R | recall A | recall B | recall R | FPR A | FPR B | FPR R |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in perf.iterrows():
        L.append(f"| {short(r['dataset'])} | " + " | ".join(
            fmt(r[f"{m}_{k}"]) for k in ["f1_binary", "recall", "FPR"] for m in MODELS) + " |")
    L += ["", "Affect-label coverage of sarcastic rows: " +
          ", ".join(f"{short(r.dataset)} {r.match_rate:.0%}" for r in coverage.itertuples()), ""]

    L += ["## 2. Group counts", "", counts.reset_index().rename(columns={"dataset": "scope"})
          .to_markdown(index=False, floatfmt=".3f"), ""]

    L += ["## 3. Fixed vs Both-wrong (samples A missed)", "",
          "| scope | feature | mean Fixed | mean Both-wrong | Cliff's δ | p (FDR) |", "|---|---|---|---|---|---|"]
    sub = ftests[ftests["contrast"] == "Fixed vs Both-wrong"]
    for _, r in sub.iterrows():
        L.append(f"| {r['scope']} | {r['feature']} | {fmt(r['mean_fixed'])} | {fmt(r['mean_other'])} | "
                 f"{fmt(r['cliffs_delta'])} | {fmt(r['p_fdr'], 4)} |")
    L += ["", "Cliff's δ > 0 → Fixed samples have higher values.", ""]

    L += ["## 4. Augmentation region (irony<0.25 & toxicity<0.25)", "",
          "| scope | Fixed | Both-wrong | Both-right | Broken | OR Fixed vs Both-wrong | p |", "|---|---|---|---|---|---|---|"]
    for _, r in region[region["region"] == "low irony & low toxicity"].iterrows():
        L.append(f"| {r['scope']} | {fmt(r['share_Fixed'])} | {fmt(r['share_Both-wrong'])} | "
                 f"{fmt(r['share_Both-right'])} | {fmt(r['share_Broken'])} | "
                 f"{fmt(r.get('odds_ratio_fixed_vs_bothwrong'))} | {fmt(r.get('p_value'), 4)} |")
    L.append("")

    L += ["## 5. Binned-profile KL divergence (pooled)", "",
          kl_df[(kl_df["scope"] == "Pooled") & kl_df["group"].isin(GROUPS)]
          [["group", "feature", "n", "KL_to_Ptacek3000 sarcastic", "KL_to_Augmented rows"]]
          .to_markdown(index=False, floatfmt=".3f"), ""]

    L += ["## 6. Affinity to the augmented rows", "",
          "aug_affinity = log(P_augmented(cell) / P_Ptacek3000_sarcastic(cell)) of the sample's "
          "4x4x4 (humor, irony, toxicity) bin cell; > 0 means the sample lies in an affective region "
          "the augmentation made more frequent.", "",
          prox.to_markdown(index=False, floatfmt=".3f"), ""]

    L += ["## 7. Logistic regression (A-missed samples: P(Fixed))", "",
          f"n = {logit_res.attrs['n']}, pseudo-R² = {logit_res.attrs['pseudo_r2']:.3f}; "
          "continuous predictors z-scored (OR per 1 SD).", "",
          logit_res.to_markdown(index=False, floatfmt=".4f"), ""]

    ctrl_sum, ctrl_tests = control
    L += ["## 8. Random-augmentation control (R = Ptacek65_random0)", "",
          "Among sarcastic samples A missed: **B only** = recovered by the affect-selected model only, "
          "**R only** = recovered by the random-augmentation model only.", "",
          "| scope | A missed | fixed by B | fixed by R | B only | R only | B and R | "
          "region share: all B-fixed | all R-fixed | B only | R only |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ctrl_sum.iterrows():
        L.append(f"| {r['scope']} | {r['n_A_missed']} | {r['n_fixed_by_B']} | {r['n_fixed_by_R']} | "
                 f"{r['n_B only']} | {r['n_R only']} | {r['n_B and R']} | "
                 f"{fmt(r['aug_region_all_fixed_by_B'])} | {fmt(r['aug_region_all_fixed_by_R'])} | "
                 f"{fmt(r['aug_region_B only'])} | {fmt(r['aug_region_R only'])} |")
    L += ["", "B only vs R only (means / region share; Cliff's δ > 0 → higher in B only):", "",
          ctrl_tests.to_markdown(index=False, floatfmt=".3f"), ""]

    L += ["## 9. Example Fixed samples", ""]
    for ds in DATASETS:
        ex = sarc[(sarc["dataset"] == ds) & (sarc["group"] == "Fixed")]
        if len(ex) == 0:
            continue
        L.append(f"**{short(ds)}**")
        for _, r in ex.sample(min(4, len(ex)), random_state=0).iterrows():
            txt = str(r["text"]).replace("\n", " ").replace("|", "/")[:200]
            L.append(f"- (h={r['humor']:.2f}, i={r['irony']:.2f}, t={r['toxicity']:.2f}; "
                     f"A={r['A_prob']:.2f}, B={r['B_prob']:.2f}) {txt}")
        L.append("")

    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    all_rows = pd.concat([load_dataset(ds) for ds in DATASETS], ignore_index=True)

    perf = performance_and_prior_shift(all_rows)
    perf.to_csv(os.path.join(OUT_DIR, "prior_shift.csv"), index=False)

    sarc, coverage = build_sarcastic_table(all_rows)
    coverage.to_csv(os.path.join(OUT_DIR, "affect_coverage.csv"), index=False)
    print(coverage.to_string(index=False))

    base_pos, added = reference_profiles()

    counts = group_counts(sarc)
    counts.to_csv(os.path.join(OUT_DIR, "group_counts.csv"))
    print(counts.to_string())

    ftests = feature_tests(sarc)
    ftests.to_csv(os.path.join(OUT_DIR, "feature_tests.csv"), index=False)

    region = aug_region_tests(sarc)
    region.to_csv(os.path.join(OUT_DIR, "aug_region_tests.csv"), index=False)

    kl_df = kl_profiles(sarc, base_pos, added)
    kl_df.to_csv(os.path.join(OUT_DIR, "kl_profiles.csv"), index=False)

    prox, sarc = aug_affinity(sarc, base_pos, added)
    prox.to_csv(os.path.join(OUT_DIR, "aug_affinity_tests.csv"), index=False)

    logit_res = logit(sarc)
    logit_res.to_csv(os.path.join(OUT_DIR, "logit_results.csv"), index=False)

    ctrl_sum, ctrl_tests, missed = control_contrast(sarc)
    ctrl_sum.to_csv(os.path.join(OUT_DIR, "control_random_summary.csv"), index=False)
    ctrl_tests.to_csv(os.path.join(OUT_DIR, "control_random_tests.csv"), index=False)
    sarc = sarc.merge(missed[["dataset", "text", "control_group"]].drop_duplicates(["dataset", "text"]),
                      on=["dataset", "text"], how="left")

    sarc.to_csv(os.path.join(OUT_DIR, "sarcastic_all_groups.csv"), index=False)

    fig_feature_shares(sarc, base_pos, added)
    fig_region_share(region)
    fig_kl(kl_df)
    write_summary(perf, coverage, counts, ftests, region, kl_df, prox, logit_res,
                  (ctrl_sum, ctrl_tests), sarc)
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
