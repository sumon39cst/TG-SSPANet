"""Round-2 figures: occlusion-condition examples, occlusion results with controls (replaces the old Fig. 12a),
qualitative explanations on held-out P2 slices (six ResNet50 models, same recipe)."""
import importlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common as C
from xai import METHODS, CAMExtractor

r2 = importlib.import_module("16_round2_eval"); xe = importlib.import_module("03_xai_eval")
OUT = os.path.join(C.RESULT, "figures"); PAPER = r"E:\Life_Nero\paper\figures"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False})
NAMES = {"resnet50_none_ourrecipe": "ResNet50", "resnet50_sspanet_ourrecipe": "ResNet50 + SSPANet",
         "resnet50_TG-noTFP": "Y-Net-style (no TFP)", "resnet50_TG-SSPANet": "TG-SSPANet",
         "resnet50_TG-oracleGT": "Oracle GT gate", "resnet50_roicrop": "ROI crop (two-stage)",
         "resnet50_maskonly": "Mask only (no image)"}
COLS = {"resnet50_none_ourrecipe": "#2E86C1", "resnet50_sspanet_ourrecipe": "#85C1E9", "resnet50_TG-noTFP": "#27AE60",
        "resnet50_TG-SSPANet": "#C0392B", "resnet50_TG-oracleGT": "#F39C12", "resnet50_roicrop": "#8E44AD"}


def save(fig, stem):
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=400); fig.savefig(os.path.join(OUT, stem + ".pdf"))
    fig.savefig(os.path.join(PAPER, stem + ".png"), dpi=400); plt.close(fig)


def fig_conditions(data):
    _, _, te = C.split_patient(data, 1)
    ex = xe.pick_examples(data, te, {})
    r3 = importlib.import_module("19_round3_eval")
    conds = ["original", "tumour removed", "sham removed", "decoy removed", "ring 0", "ring 15", "ring 30", "ring 60"]
    titles = ["original", "tumour\nremoved", "sham\nremoved", "tumour +\ndecoys removed", "tumour only\n(0 px)", "tumour +\n15 px", "tumour +\n30 px", "tumour +\n60 px"]
    fig, axes = plt.subplots(len(ex), len(conds), figsize=(7.2, 3.1))
    for r, i in enumerate(ex):
        c = r3.conditions(data["images"][i], data["masks"][i].astype(bool))
        for k, cn in enumerate(conds):
            ax = axes[r, k]; im = c[cn] if c[cn] is not None else data["images"][i]
            ax.imshow(im, cmap="gray"); ax.contour(data["masks"][i], levels=[0.5], colors="#00E676", linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[k], fontsize=6.2)
        axes[r, 0].set_ylabel(C.CLASSES[data["labels"][i]].capitalize(), fontsize=7)
    fig.tight_layout(pad=0.3); save(fig, "fig_occlusion_conditions")


def fig_occlusion():
    O = pd.read_csv(os.path.join(C.RESULT, "round2", "occlusion.csv"))
    g = O.groupby(["model", "condition"]).acc.agg(["mean", "std"]).reset_index()
    runs = [r for r in COLS if r in set(O.model)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    conds = ["original", "sham removed", "tumour removed", "tumour only"]
    x = np.arange(len(conds)); w = 0.8 / len(runs)
    for k, r in enumerate(runs):
        m = [g[(g.model == r) & (g.condition == c)]["mean"].values[0] for c in conds]
        s = [g[(g.model == r) & (g.condition == c)]["std"].values[0] for c in conds]
        axes[0].bar(x + (k - (len(runs) - 1) / 2) * w, m, w, yerr=s, color=COLS[r], label=NAMES[r],
                    error_kw=dict(lw=0.5, capsize=1))
    axes[0].axhline(1 / 3, color="#888888", ls=":", lw=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["original", "sham\nremoved", "tumour\nremoved", "tumour\nonly"])
    axes[0].set_ylabel("accuracy (P2, mean ± s.d. over folds)"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Occlusion with controls", fontsize=8)
    ring = ["tumour only", "tumour + 15 px ring", "tumour + 30 px ring", "tumour + 60 px ring", "original"]
    xr = [6, 15, 30, 60, 90]
    for r in runs:
        m = [g[(g.model == r) & (g.condition == c)]["mean"].values[0] for c in ring]
        axes[1].plot(xr, m, "o-", color=COLS[r], ms=3, lw=1.4 if r == "resnet50_TG-SSPANet" else 1)
    axes[1].set_xticks(xr); axes[1].set_xticklabels(["6", "15", "30", "60", "full"])
    axes[1].set_xlabel("context kept around the tumour (px ≈ mm)"); axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.05); axes[1].axhline(1 / 3, color="#888888", ls=":", lw=0.7)
    axes[1].set_title("(b) Graded context", fontsize=8); axes[1].grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(runs), fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "fig_occlusion_controls")


def fig_qualitative_p2(data):
    dev = torch.device("cuda")
    _, _, te = C.split_patient(data, 1); tag = "patient_fold1_seed0"
    ex = xe.pick_examples(data, te, {})
    gd = C.GPUData(data, dev); x, m, _ = gd.batch(np.array(ex)); gt = m[:, 0].cpu().numpy() > 0.5
    img = F.interpolate(torch.from_numpy(data["images"][ex]).float()[:, None], size=(224, 224), mode="bilinear",
                        antialias=True, align_corners=False)[:, 0].numpy()
    rows = [("resnet50_none_ourrecipe", "ResNet50"), ("resnet50_coordinate_ourrecipe", "Coordinate"),
            ("resnet50_gcnet_ourrecipe", "GCNet"), ("resnet50_spnet_ourrecipe", "SPNet"),
            ("resnet50_srmnet_ourrecipe", "SRMNet"), ("resnet50_TG-SSPANet", "TG-SSPANet\n(ours)")]
    cams = []
    for name, _ in rows:
        model = xe.load_run("patient", name, tag, dev)[0]
        ext = CAMExtractor(model, model.cam_layer); mp, _ = ext(x); ext.remove(); cams.append(mp)
        del model; torch.cuda.empty_cache()
    mlab = {"GradCAM": "Grad-CAM", "GradCAM++": "Grad-CAM++", "EigenGradCAM": "EigenGrad-CAM"}
    fig = plt.figure(figsize=(7.2, 5.25))
    gs = fig.add_gridspec(len(rows), 9, left=0.095, right=0.995, top=0.905, bottom=0.005, wspace=0.04, hspace=0.05)
    axs = []
    for r, (_, label) in enumerate(rows):
        for t in range(3):
            for j, meth in enumerate(METHODS):
                ax = fig.add_subplot(gs[r, t * 3 + j]); axs.append(ax)
                ax.imshow(img[t], cmap="gray"); ax.imshow(cams[r][meth][t], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                ax.contour(gt[t], levels=[0.5], colors="#00E676", linewidths=0.7); ax.set_xticks([]); ax.set_yticks([])
                ours = "(ours)" in label
                for s in ax.spines.values():
                    s.set_visible(True); s.set_linewidth(1.4 if ours else 0.4); s.set_color("#C0392B" if ours else "#333333")
                if r == 0:
                    ax.set_title(mlab[meth], fontsize=6.5, pad=2)
                if t == 0 and j == 0:
                    ax.set_ylabel(label, fontsize=7, fontweight="bold" if ours else "normal", color="#C0392B" if ours else "black")
    for t in range(3):
        x0 = axs[t * 3].get_position().x0; x1 = axs[t * 3 + 2].get_position().x1
        fig.text((x0 + x1) / 2, 0.955, C.CLASSES[data["labels"][ex[t]]].capitalize(), ha="center", fontsize=8, fontweight="bold")
        fig.add_artist(plt.Line2D([x0 + 0.004, x1 - 0.004], [0.938, 0.938], color="#555555", lw=0.7))
    fig.text(0.5, 0.99, "ResNet50, identical training recipe · held-out patients (P2 fold 1) · green: ground-truth tumour",
             ha="center", va="top", fontsize=7.5, color="#333333")
    save(fig, "fig_xai_qualitative_p2")


if __name__ == "__main__":
    import sys
    data = C.load_data()
    what = sys.argv[1:] or ["conditions", "occlusion", "qualitative"]
    if "conditions" in what:
        fig_conditions(data)
    if "occlusion" in what:
        fig_occlusion()
    if "qualitative" in what:
        fig_qualitative_p2(data)
    print("ok", what)
