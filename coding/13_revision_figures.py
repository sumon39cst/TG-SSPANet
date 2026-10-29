"""Revision figures (reviewer comments 3-7 and minor 13).

fig_p1_reproduction.png   dot plot: reported in [1] vs reproduced (mean +- s.d., 3 seeds), P1 (non-held-out)
fig_p2_accuracy.png       P2 slice-level accuracy per model: fold values + pooled patient-clustered 95% CI
fig_p2_localisation.png   P2 Grad-CAM++ pointing game / energy-in-mask (mean +- s.d. over folds) + reference maps
fig_p2_faithfulness.png   P2 occlusion accuracies and deletion / insertion curves
fig_p2_ablation.png       P2 ablation and mask-using baselines: accuracy and pointing game
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

REV = os.path.join(C.RESULT, "revision"); FIG = os.path.join(C.RESULT, "figures")
PAPER = r"E:\Life_Nero\paper\figures"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, BLUE, GREY, GREEN, PURPLE = "#C0392B", "#2E86C1", "#95A5A6", "#27AE60", "#8E44AD"
NAMES = json.load(open(os.path.join(REV, "names.json"), encoding="utf8")) if os.path.exists(os.path.join(REV, "names.json")) else {}


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".png"), dpi=400); fig.savefig(os.path.join(FIG, stem + ".pdf"))
    fig.savefig(os.path.join(PAPER, stem + ".png"), dpi=400); plt.close(fig)


FAITH_COL = {"resnet50_none": "#BDC3C7", "resnet50_none_ourrecipe": BLUE, "resnet50_sspanet_ourrecipe": "#85C1E9",
             "resnet50_TG-noTFP": GREEN, "resnet50_TG-SSPANet": RED}


def cat_color(run):
    if "TG-SSPANet" in run and "vgg" not in run or run == "vgg16_TG-SSPANet":
        return RED
    if "oracle" in run or "roicrop" in run or "noTFP" in run or "floor" in run:
        return GREEN
    if "ourrecipe" in run:
        return BLUE
    return GREY


def fig_p1():
    t = pd.read_csv(os.path.join(C.RESULT, "tables", "table_main_comparison.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for ax, bb in zip(axes, ["VGG16", "ResNet50"]):
        d = t[t.Backbone == bb].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        for yi, (_, r) in zip(y, d.iterrows()):
            m, s = [float(v) for v in str(r["Ours-run Acc"]).split(" ± ")]
            ours = "ours" in r.Attention
            ax.errorbar(m, yi, xerr=s, fmt="o", color=RED if ours else "#34495E", ms=4, capsize=2, lw=0.9, zorder=3)
            if not np.isnan(r["Paper Acc"]):
                ax.plot(r["Paper Acc"], yi, "D", mfc="white", mec=GREY, ms=4, zorder=2)
                ax.plot([r["Paper Acc"], m], [yi, yi], color="#D5D8DC", lw=0.8, zorder=1)
        lab = [a.replace(" (paper)", "").replace(" (ours)", " (ours)") for a in d.Attention]
        ax.set_yticks(y); ax.set_yticklabels(lab); ax.set_title(bb, fontsize=8.5)
        ax.set_xlim(0.5, 1.01); ax.grid(axis="x", alpha=0.3); ax.set_xlabel("test accuracy on P1 (image-level, patient-leaky)")
    axes[0].plot([], [], "D", mfc="white", mec=GREY, label="reported in [21]")
    axes[0].errorbar([], [], xerr=[], fmt="o", color="#34495E", label="reproduced (mean ± s.d., 3 seeds)")
    axes[0].errorbar([], [], xerr=[], fmt="o", color=RED, label="TG-SSPANet (mean ± s.d., 3 seeds)")
    axes[0].legend(fontsize=6.5, loc="lower left", frameon=False)
    fig.tight_layout(); save(fig, "fig_p1_reproduction")


def fig_p2_acc():
    T = pd.read_csv(os.path.join(REV, "table_p2_full.csv"))
    order = ["resnet50_none", "resnet50_sspanet", "resnet50_none_ourrecipe", "resnet50_coordinate_ourrecipe", "resnet50_gcnet_ourrecipe",
             "resnet50_spnet_ourrecipe", "resnet50_srmnet_ourrecipe", "resnet50_se_ourrecipe", "resnet50_cbam_ourrecipe",
             "resnet50_sspanet_ourrecipe", "resnet50_TG-noTFP", "resnet50_roicrop", "resnet50_TG-SSPANet",
             "resnet50_roicrop_gtcrop", "resnet50_TG-oracleGT",
             "vgg16_none", "vgg16_sspanet", "vgg16_none_ourrecipe", "vgg16_sspanet_ourrecipe", "vgg16_TG-SSPANet"]
    T = T.set_index("run"); order = [o for o in order if o in T.index]
    folds = {}
    for o in order:
        base = o.replace("_gtcrop", ""); suf = "_gtcrop" if o.endswith("_gtcrop") else ""
        folds[o] = [ (lambda d: (d.true_label == d.pred_label).mean())(pd.read_csv(os.path.join(
            C.PRED, "patient", base, f"test_predictions_patient_fold{k}_seed0{suf}.csv"))) for k in range(1, 6)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(order))[::-1]
    for yi, o in zip(y, order):
        r = T.loc[o]; c = cat_color(o)
        ax.plot([r.acc_ci_lo, r.acc_ci_hi], [yi, yi], color=c, lw=2.2, alpha=0.5, solid_capstyle="round")
        ax.scatter(folds[o], [yi] * 5, s=9, color=c, alpha=0.6, zorder=3)
        ax.plot(r.acc_pooled, yi, "|", color="black", ms=9, mew=1.4, zorder=4)
        ax.text(1.003, yi, f"{r.acc_mean:.4f} ± {r.acc_sd:.4f}   pat. {r.patient_acc:.3f}", va="center", fontsize=6.3,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(y); ax.set_yticklabels([T.loc[o, "model"] for o in order], fontsize=6.8)
    ax.set_xlabel("slice-level test accuracy on P2 (patient-wise 5-fold)"); ax.set_xlim(0.86, 1.0); ax.grid(axis="x", alpha=0.3)
    ax.axhline(y[order.index("vgg16_none")] + 0.5 if "vgg16_none" in order else -1, color="#BBBBBB", lw=0.6, ls="--")
    for c, l in [(GREY, "original recipe [21]"), (BLUE, "same (proposed) recipe"), (GREEN, "mask-using / ablation"), (RED, "TG-SSPANet")]:
        ax.plot([], [], color=c, lw=2.2, label=l)
    ax.scatter([], [], s=9, color="#555555", label="fold"); ax.plot([], [], "|", color="black", ms=9, label="pooled")
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=6)
    fig.subplots_adjust(left=0.30, right=0.76, top=0.97, bottom=0.16)
    fig.text(0.985, 0.985, "mean ± s.d. over folds; pat. = patient-level acc.", ha="right", va="top", fontsize=6)
    save(fig, "fig_p2_accuracy")


def fig_p2_loc():
    L = pd.read_csv(os.path.join(REV, "p2_localisation.csv"))
    runs = ["resnet50_none", "resnet50_sspanet", "resnet50_none_ourrecipe", "resnet50_se_ourrecipe", "resnet50_cbam_ourrecipe",
            "resnet50_sspanet_ourrecipe", "resnet50_TG-noTFP", "resnet50_TG-floor0.5", "resnet50_TG-floor0.3",
            "resnet50_TG-SSPANet", "resnet50_TG-oracleGT", "vgg16_none_ourrecipe", "vgg16_sspanet_ourrecipe", "vgg16_TG-SSPANet"]
    runs = [r for r in runs if r in set(L.model)]
    g = L[(L.method == "GradCAM++") & (L.tol == 7)].groupby("model")
    ref = L[L.method == "reference"].groupby("model")
    dec = L[(L.method == "decoder mask") & (L.tol == 7) & (L.model == "resnet50_TG-SSPANet")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    y = np.arange(len(runs))[::-1]
    for ax, met, lab in zip(axes, ["pointing_game", "energy_in_mask"], ["pointing game (Grad-CAM++)", "energy in mask (Grad-CAM++)"]):
        for yi, r in zip(y, runs):
            m, s = g[met].mean()[r], g[met].std()[r]
            ax.barh(yi, m, xerr=s, color=cat_color(r), alpha=0.85, height=0.7, error_kw=dict(lw=0.7, capsize=1.5))
            ax.text(m + s + 0.01, yi, f"{m:.2f}", va="center", fontsize=6)
        for nm, ls in [("centre prior", "--"), ("random map", ":")]:
            if nm in ref.groups:
                ax.axvline(ref[met].mean()[nm], color="#555555", ls=ls, lw=0.8, label=nm)
        if len(dec):
            ax.axvline(dec[met].mean(), color=GREEN, ls="-.", lw=0.9, label="decoder mask as explanation (TG)")
        ax.set_xlabel(lab); ax.set_xlim(0, 1); ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y); axes[0].set_yticklabels([NAMES.get(r, r) for r in runs], fontsize=6.6)
    axes[1].legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout(); save(fig, "fig_p2_localisation")


def fig_p2_faith():
    """Deletion / insertion curves only (occlusion results with controls: 18_round2_figures.fig_occlusion)."""
    Fd = pd.read_csv(os.path.join(REV, "p2_faithfulness.csv"))
    z = np.load(os.path.join(REV, "p2_curves.npz"), allow_pickle=True); curves = z["curves"].item(); steps = z["steps"]
    runs = ["resnet50_none", "resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe", "resnet50_TG-noTFP", "resnet50_TG-SSPANet"]
    runs = [r for r in runs if r in set(Fd.model)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for axc, (title, idx) in zip(axs, [("(a) Deletion (Grad-CAM++ order)", 0), ("(b) Insertion (Grad-CAM++ order)", 1)]):
        for r in runs:
            cs = [curves[(r, f)]["gradcampp"][idx] for f in range(1, 6) if (r, f) in curves]
            if cs:
                axc.plot(steps, np.mean(cs, 0), color=FAITH_COL.get(r, GREY), lw=1.6 if "TG-SSPANet" in r else 1,
                         label=NAMES.get(r, r))
        rnd = [curves[("resnet50_TG-SSPANet", f)]["random"][idx] for f in range(1, 6) if ("resnet50_TG-SSPANet", f) in curves]
        if rnd:
            axc.plot(steps, np.mean(rnd, 0), color=RED, ls=":", lw=1, label="TG-SSPANet, random order")
        axc.set_xlabel("fraction of pixels " + ("deleted" if idx == 0 else "inserted")); axc.set_ylabel("true-class probability")
        axc.set_ylim(0, 1); axc.grid(alpha=0.3); axc.set_title(title, fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.2, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); save(fig, "fig_p2_faithfulness")


def fig_p2_ablation():
    T = pd.read_csv(os.path.join(REV, "table_p2_full.csv")).set_index("run")
    L = pd.read_csv(os.path.join(REV, "p2_localisation.csv"))
    pg = L[(L.method == "GradCAM++") & (L.tol == 7)].groupby("model").pointing_game
    runs = ["resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe", "resnet50_TG-noTFP", "resnet50_TG-floor0.5",
            "resnet50_TG-floor0.3", "resnet50_TG-SSPANet", "resnet50_TG-oracleGT", "resnet50_roicrop", "resnet50_roicrop_gtcrop"]
    runs = [r for r in runs if r in T.index]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(runs))
    axes[0].bar(x, [T.loc[r, "acc_mean"] for r in runs], yerr=[T.loc[r, "acc_sd"] for r in runs],
                color=[cat_color(r) for r in runs], error_kw=dict(lw=0.7, capsize=1.5))
    for xi, r in zip(x, runs):
        axes[0].text(xi, T.loc[r, "acc_mean"] + T.loc[r, "acc_sd"] + 0.003, f"{T.loc[r, 'acc_mean']:.3f}", ha="center", fontsize=5.8, rotation=90)
    axes[0].set_ylim(0.85, 1.02); axes[0].set_ylabel("slice-level accuracy (P2)"); axes[0].set_title("(a) Accuracy", fontsize=8)
    pr = [r for r in runs if r in pg.mean().index]
    xp = np.arange(len(pr))
    axes[1].bar(xp, [pg.mean()[r] for r in pr], yerr=[pg.std()[r] for r in pr], color=[cat_color(r) for r in pr],
                error_kw=dict(lw=0.7, capsize=1.5))
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("pointing game (Grad-CAM++, P2)"); axes[1].set_title("(b) Localisation", fontsize=8)
    for ax, rr, xx in [(axes[0], runs, x), (axes[1], pr, xp)]:
        ax.set_xticks(xx); ax.set_xticklabels([NAMES.get(r, r) for r in rr], rotation=40, ha="right", fontsize=6.2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); save(fig, "fig_p2_ablation")


if __name__ == "__main__":
    for f in (fig_p1, fig_p2_acc, fig_p2_loc, fig_p2_faith, fig_p2_ablation):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("skip", f.__name__, repr(e))
