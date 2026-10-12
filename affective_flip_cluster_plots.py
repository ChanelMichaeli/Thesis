"""
Cluster 2 plots for the affective flip analysis (see affective_flip_analysis.py).

Cluster 2 = SPIRS, SemEval2022, iSarcasmEval, multimodal_sarcasm_detection, SARC, MUStARD
(the cross-domain test sets; cluster 1 = Ptacek, Riloff is not plotted).

Models (names used in the plots):
    Ptacek model                              BERT on Ptacek/train_data_filtered3000.csv
    Ptacek model + 50% augmentation           BERT on Ptacek65_irony_txc_neg0.csv (affective selection)
    Ptacek model + 50% random augmentation    BERT on Ptacek65_random0.csv (control)

Plots (scores_results_finetuned/affective_flip/figures/):
    1. cluster2_group_pie.png         outcome shares of the sarcastic samples
    2. cluster2_fixed_affect.png      affect profile of the Fixed samples vs the augmentation samples
                                      and the original Ptacek sarcastic samples
    3. cluster2_random_vs_affect.png  outcome counts: random vs affective augmentation
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import affective_flip_analysis as afa

CLUSTER_NAME = "Cluster 2"
CLUSTER_DATASETS = ["SPIRS", "SemEval2022", "iSarcasmEval", "multimodal_sarcasm_detection",
                    "SARC", "MUStARD"]

BASE_NAME = "Ptacek model"
AUG_NAME = "Ptacek model + 50% augmentation"
RANDOM_NAME = "Ptacek model + 50% random augmentation"
GROUP_LABELS = {
    "Fixed": "Fixed: missed by the Ptacek model, detected after augmentation",
    "Both-wrong": "Both wrong: missed by both",
    "Both-right": "Both right: detected by both",
    "Broken": "Broken: detected by the Ptacek model, missed after augmentation",
}

INK, INK_2, GRID = "#222222", "#555555", "#e5e5e5"
REF_ORIG = "#b9b9b9"     # neutral fill for the original Ptacek reference
RANDOM_COLOR = "#9a9a9a"  # neutral for the random-augmentation control
AFFECT_COLOR = afa.GROUP_COLORS["Fixed"]


def load_sarcastic():
    """Sarcastic rows of the cluster test sets with outcome groups and affect probs.
    group   : Ptacek model -> + 50% augmentation
    group_R : Ptacek model -> + 50% random augmentation"""
    parts = []
    for ds in CLUSTER_DATASETS:
        df = afa.load_dataset(ds)
        pos = df[df["label"] == 1].copy()
        pos["group"] = [afa.assign_group(a == 1, b == 1) for a, b in zip(pos["A_pred"], pos["B_pred"])]
        pos["group_R"] = [afa.assign_group(a == 1, r == 1) for a, r in zip(pos["A_pred"], pos["R_pred"])]
        aux = afa.load_aux(os.path.join(afa.AUX_DIR, ds, "test_data_filtered.csv"))
        parts.append(pos.merge(aux, on="text", how="left"))
    return pd.concat(parts, ignore_index=True)


def style(ax, grid_axis="y"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#888")
    ax.tick_params(colors=INK_2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def cluster_subtitle():
    return f"{CLUSTER_NAME}: " + ", ".join(afa.short(d) for d in CLUSTER_DATASETS)


# --------------------------------------------------------------------------
# 1. pie
# --------------------------------------------------------------------------
def draw_pie(ax, counts, title):
    n = counts.sum()
    nz = counts[counts > 0]
    wedges, _ = ax.pie(nz.values, colors=[afa.GROUP_COLORS[g] for g in nz.index],
                       startangle=90, counterclock=False,
                       wedgeprops={"edgecolor": "white", "linewidth": 2})
    for w, g, v in zip(wedges, nz.index, nz.values):
        share = v / n
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        if share < 0.05:
            # thin wedges sit side by side at the top: push each label out to its own side
            side = 1 if np.cos(ang) >= 0 else -1
            xytext = (0.12 * side, 1.17)
        else:
            xytext = (1.1 * np.cos(ang), 1.1 * np.sin(ang))
        # anchor the text on the side facing away from the pie so it never covers a wedge
        ha = "center" if abs(xytext[0]) < 0.05 else ("left" if xytext[0] > 0 else "right")
        va = "bottom" if xytext[1] > 0.9 else ("top" if xytext[1] < -0.9 else "center")
        ax.annotate(f"{g.replace('-', ' ')}\n{share:.1%} (n={v})" if share < 0.05 else
                    f"{g.replace('-', ' ')}\n{share:.0%} (n={v})",
                    xy=(0.97 * np.cos(ang), 0.97 * np.sin(ang)),
                    xytext=xytext,
                    ha=ha, va=va, fontsize=11, color=INK,
                    arrowprops={"arrowstyle": "-", "color": "#888", "linewidth": 0.8})
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, color=INK, pad=60)
    parts = [f"{g.replace('-', ' ')}: {counts[g]} ({counts[g] / n:.1%})" for g in afa.GROUPS]
    ax.text(0.5, -0.1, "   ".join(parts[:2]) + "\n" + "   ".join(parts[2:]), transform=ax.transAxes,
            ha="center", va="top", fontsize=10, color=INK_2)


def plot_pie(sarc):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    counts_r = sarc["group_R"].value_counts().reindex(afa.GROUPS, fill_value=0)
    counts_b = sarc["group"].value_counts().reindex(afa.GROUPS, fill_value=0)
    draw_pie(axes[0], counts_r, RANDOM_NAME)
    draw_pie(axes[1], counts_b, AUG_NAME)
    handles = [plt.Rectangle((0, 0), 1, 1, color=afa.GROUP_COLORS[g]) for g in afa.GROUPS]
    fig.legend(handles, [GROUP_LABELS[g] for g in afa.GROUPS], loc="lower center", ncol=2,
               frameon=False, fontsize=10)
    fig.suptitle(f"Outcome compared with the {BASE_NAME}: random vs affective augmentation\n"
                 f"{cluster_subtitle()} ({len(sarc)} sarcastic test samples)", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    fig.savefig(os.path.join(afa.FIG_DIR, "cluster2_group_pie.png"), dpi=150)
    plt.close(fig)
    return pd.DataFrame([{"cluster": CLUSTER_NAME, "model": RANDOM_NAME, **counts_r.to_dict(), "n": len(sarc)},
                         {"cluster": CLUSTER_NAME, "model": AUG_NAME, **counts_b.to_dict(), "n": len(sarc)}])


# --------------------------------------------------------------------------
# table: irony / toxicity of the Fixed samples, random vs affective augmentation
# --------------------------------------------------------------------------
def fixed_affect_table(sarc, thr=0.5):
    """Counts of Fixed samples by high/low irony x high/low toxicity (high = prob > thr)."""
    cells = [("low irony, low toxicity", False, False), ("low irony, high toxicity", False, True),
             ("high irony, low toxicity", True, False), ("high irony, high toxicity", True, True)]
    rows = []
    for name, col in [(RANDOM_NAME, "group_R"), (AUG_NAME, "group")]:
        fixed = sarc[sarc[col] == "Fixed"]
        scored = fixed[fixed[["irony", "toxicity"]].notna().all(axis=1)]
        hi_i, hi_t = scored["irony"] > thr, scored["toxicity"] > thr
        row = {"model": name, "Fixed samples": len(fixed), "with affect scores": len(scored)}
        for label, want_i, want_t in cells:
            k = int(((hi_i == want_i) & (hi_t == want_t)).sum())
            row[label] = k
            row[label + " (%)"] = 100 * k / len(scored) if len(scored) else np.nan
        row["high irony (total)"] = int(hi_i.sum())
        row["high toxicity (total)"] = int(hi_t.sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(afa.OUT_DIR, "cluster2_fixed_irony_toxicity_table.csv"), index=False)

    # readable version: "count (percent)"
    show = table[["model", "Fixed samples"]].copy()
    for label, _, _ in cells:
        show[label] = [f"{k} ({p:.1f}%)" for k, p in zip(table[label], table[label + " (%)"])]
    show["high irony (total)"] = table["high irony (total)"]
    show["high toxicity (total)"] = table["high toxicity (total)"]
    with open(os.path.join(afa.OUT_DIR, "cluster2_fixed_irony_toxicity_table.md"), "w", encoding="utf-8") as f:
        f.write(f"Fixed samples ({cluster_subtitle()}): missed by the {BASE_NAME}, detected after "
                f"augmentation. high = probability > {thr}; percentages are of the Fixed samples "
                f"with affect scores.\n\n")
        f.write(show.to_markdown(index=False))
    write_latex_table(table, cells, thr)
    return show


def write_latex_table(table, cells, thr):
    def cell(r, label):
        return f"{r[label]:,} ({r[label + ' (%)']:.1f}\\%)"

    def total(r, col):
        return f"{r[col]:,} ({100 * r[col] / r['with affect scores']:.1f}\\%)"

    names = {RANDOM_NAME: "+ 50\\% random aug.", AUG_NAME: "+ 50\\% affective aug."}
    L = [r"\begin{table}[ht]", r"\centering", r"\small",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{Low irony} & \multicolumn{2}{c}{High irony} \\",
         r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
         r"Ptacek model & Fixed & Low tox. & High tox. & Low tox. & High tox. \\", r"\midrule"]
    for _, r in table.iterrows():
        L.append(f"{names[r['model']]} & {r['Fixed samples']:,} & " +
                 " & ".join(cell(r, label) for label, _, _ in cells) + r" \\")
    L += [r"\midrule",
          r" & & \multicolumn{2}{c}{High irony (total)} & \multicolumn{2}{c}{High toxicity (total)} \\",
          r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}"]
    for _, r in table.iterrows():
        L.append(f"{names[r['model']]} & & \\multicolumn{{2}}{{c}}{{{total(r, 'high irony (total)')}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{total(r, 'high toxicity (total)')}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Irony and toxicity of the sarcastic test samples of Cluster 2 (SPIRS, SemEval-2022, "
          r"iSarcasmEval, MMSD, SARC, MUStARD) that are missed by the Ptacek model and detected after "
          r"augmentation (Fixed), for random and affective augmentation. High = classifier probability "
          f"$> {thr}$; percentages are of the Fixed samples with affect scores." + "}",
          r"\label{tab:fixed_irony_toxicity}", r"\end{table}"]
    path = os.path.join(afa.OUT_DIR, "cluster2_fixed_irony_toxicity_table.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------
# 2. affect profile of Fixed samples vs augmentation profile
# --------------------------------------------------------------------------
def high_share(df):
    """share of samples with prob > 0.5 per affect dimension (aux probs are near-binary)."""
    return np.array([(df[f] > 0.5).mean() if len(df) else np.nan for f in afa.FEATURES])


def plot_fixed_affect(sarc, base_pos, added):
    feats = afa.FEATURES
    df = sarc[sarc[feats].notna().all(axis=1)]
    fixed, bothwrong = df[df["group"] == "Fixed"], df[df["group"] == "Both-wrong"]
    p_orig, p_aug, p_fix, p_bw = high_share(base_pos), high_share(added), high_share(fixed), high_share(bothwrong)
    d_aug, d_orig = np.mean(np.abs(p_fix - p_aug)), np.mean(np.abs(p_fix - p_orig))
    region = [afa.in_aug_region(base_pos).mean(), afa.in_aug_region(added).mean(),
              afa.in_aug_region(fixed).mean(), afa.in_aug_region(bothwrong).mean()]

    series = [(f"Ptacek training set, sarcastic samples (n={len(base_pos)})", REF_ORIG, None),
              (f"Augmentation samples, added to the training set (n={len(added)})", "white", "///"),
              (f"Fixed test samples (n={len(fixed)})", AFFECT_COLOR, None),
              (f"Both-wrong test samples (n={len(bothwrong)})", afa.GROUP_COLORS["Both-wrong"], None)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(feats))
    w = 0.2
    style(ax1)
    for j, ((lab, col, hatch), vals) in enumerate(zip(series, [p_orig, p_aug, p_fix, p_bw])):
        xs = x + (j - 1.5) * w
        ax1.bar(xs, vals, width=w - 0.03, color=col, hatch=hatch,
                edgecolor=INK if hatch else "white", linewidth=1, label=lab)
        for xi, v in zip(xs, vals):
            ax1.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(feats, fontsize=12)
    ax1.set_ylim(0, 1.18)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_ylabel("share of samples with probability > 0.5", color=INK)
    ax1.set_title("High-score share per affective dimension", fontsize=11, color=INK)
    closer = "augmentation samples" if d_aug < d_orig else "Ptacek training set"
    ax1.text(0.99, 0.99, f"Profile distance of the Fixed samples\n"
                         f"to augmentation samples = {d_aug:.2f}\nto Ptacek training set = {d_orig:.2f}\n"
                         f"→ closer to the {closer}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9, color=INK,
             bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID})

    style(ax2)
    for j, ((lab, col, hatch), v) in enumerate(zip(series, region)):
        ax2.bar(j, v, width=0.7, color=col, hatch=hatch, edgecolor=INK if hatch else "white", linewidth=1)
        ax2.text(j, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=10, color=INK)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["Ptacek\ntraining set", "Augmentation\nsamples", "Fixed", "Both wrong"], fontsize=9.5)
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_title("Share in the augmentation region\n(irony < .25 and toxicity < .25)", fontsize=11, color=INK)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"Affective profile of the sarcastic samples fixed by the {AUG_NAME}\n{cluster_subtitle()}",
                 fontsize=12, color=INK)
    fig.text(0.5, 0.115, "profile distance = mean |difference| of the high-score shares over humor, irony and toxicity",
             ha="center", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    fig.savefig(os.path.join(afa.FIG_DIR, "cluster2_fixed_affect.png"), dpi=150)
    plt.close(fig)

    return pd.DataFrame([{"cluster": CLUSTER_NAME, "n_fixed": len(fixed), "n_bothwrong": len(bothwrong),
                          **{f"high_{f}_fixed": v for f, v in zip(feats, p_fix)},
                          **{f"high_{f}_bothwrong": v for f, v in zip(feats, p_bw)},
                          **{f"high_{f}_augmentation_samples": v for f, v in zip(feats, p_aug)},
                          **{f"high_{f}_ptacek_train": v for f, v in zip(feats, p_orig)},
                          "dist_fixed_to_augmentation": d_aug, "dist_fixed_to_ptacek_train": d_orig,
                          "aug_region_ptacek_train": region[0], "aug_region_augmentation_samples": region[1],
                          "aug_region_fixed": region[2], "aug_region_bothwrong": region[3]}])


# --------------------------------------------------------------------------
# 3. random vs affective augmentation
# --------------------------------------------------------------------------
def plot_random_vs_affect(sarc):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(afa.GROUPS))
    w = 0.38
    r_counts = sarc["group_R"].value_counts().reindex(afa.GROUPS, fill_value=0)
    b_counts = sarc["group"].value_counts().reindex(afa.GROUPS, fill_value=0)
    top = max(r_counts.max(), b_counts.max())
    for off, counts, col, lab in [(-w / 2, r_counts, RANDOM_COLOR, RANDOM_NAME),
                                  (w / 2, b_counts, AFFECT_COLOR, AUG_NAME)]:
        ax.bar(x + off, counts.values, width=w - 0.04, color=col, label=lab)
        for xi, v in zip(x + off, counts.values):
            ax.text(xi, v + top * 0.015, str(v), ha="center", va="bottom", fontsize=10, color=INK)
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(["Fixed\n(missed by Ptacek model,\ndetected after augmentation)",
                        "Both wrong", "Both right",
                        "Broken\n(detected by Ptacek model,\nmissed after augmentation)"], fontsize=9.5)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("number of sarcastic test samples", color=INK)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
    fig.suptitle(f"Outcome compared with the {BASE_NAME}: random vs affective augmentation\n"
                 f"{cluster_subtitle()} ({len(sarc)} sarcastic samples)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(afa.FIG_DIR, "cluster2_random_vs_affect.png"), dpi=150)
    plt.close(fig)
    return pd.DataFrame({"group": afa.GROUPS, RANDOM_NAME: r_counts.values, AUG_NAME: b_counts.values})


def main():
    os.makedirs(afa.FIG_DIR, exist_ok=True)
    sarc = load_sarcastic()
    base_pos, added = afa.reference_profiles()

    pie = plot_pie(sarc)
    profile = plot_fixed_affect(sarc, base_pos, added)
    rvb = plot_random_vs_affect(sarc)
    table = fixed_affect_table(sarc)

    pie.to_csv(os.path.join(afa.OUT_DIR, "cluster2_group_counts.csv"), index=False)
    profile.to_csv(os.path.join(afa.OUT_DIR, "cluster2_fixed_affect_profile.csv"), index=False)
    rvb.to_csv(os.path.join(afa.OUT_DIR, "cluster2_random_vs_affect_counts.csv"), index=False)
    print(pie.to_string(index=False))
    print(profile.round(3).T.to_string())
    print(rvb.to_string(index=False))
    print(table.T.to_string())


if __name__ == "__main__":
    main()
