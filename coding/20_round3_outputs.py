"""Round-3 figure (occlusion on the matched subset) and markdown tables (main Table 4; Tables S2-S4)."""
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

R3 = os.path.join(C.RESULT, "round3"); FIG = os.path.join(C.RESULT, "figures"); PAPER = r"E:\Life_Nero\paper\figures"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
O = pd.read_csv(os.path.join(R3, "occlusion_matched.csv"))
M = O.pivot_table(index="model", columns="condition", values="acc")
SD = O.pivot_table(index="model", columns="condition", values="acc", aggfunc="std")
Fd = pd.read_csv(os.path.join(C.RESULT, "revision", "p2_faithfulness.csv")).groupby("model").mean(numeric_only=True)
NAMES = {"resnet50_none": "ResNet50 (orig.)", "resnet50_sspanet": "ResNet50 + SSPANet (orig.)", "resnet50_none_ourrecipe": "ResNet50",
         "resnet50_sspanet_ourrecipe": "ResNet50 + SSPANet", "resnet50_TG-noTFP": "Y-Net-style (no TFP)",
         "resnet50_TG-SSPANet": "TG-SSPANet", "resnet50_TG-oracleGT": "Oracle GT gate (positive control)",
         "resnet50_roicrop [pred crop]": "ROI crop, predicted-mask box (positive control)",
         "resnet50_roicrop [gt crop]": "ROI crop, GT-mask box (positive control)", "vgg16_none": "VGG16 (orig.)",
         "vgg16_none_ourrecipe": "VGG16", "vgg16_TG-SSPANet": "VGG16 TG-SSPANet"}
ORDER = list(NAMES)
COLS = {"resnet50_none_ourrecipe": "#2E86C1", "resnet50_sspanet_ourrecipe": "#85C1E9", "resnet50_TG-noTFP": "#27AE60",
        "resnet50_TG-SSPANet": "#C0392B", "resnet50_TG-oracleGT": "#F39C12", "resnet50_roicrop [pred crop]": "#8E44AD",
        "resnet50_none": "#95A5A6"}
SHORT = {"resnet50_none_ourrecipe": "ResNet50", "resnet50_sspanet_ourrecipe": "ResNet50 + SSPANet", "resnet50_TG-noTFP": "Y-Net-style (no TFP)",
         "resnet50_TG-SSPANet": "TG-SSPANet", "resnet50_TG-oracleGT": "Oracle GT gate", "resnet50_roicrop [pred crop]": "ROI crop (two-stage)",
         "resnet50_none": "ResNet50 (original recipe)"}


def figure():
    runs = [r for r in COLS if r in M.index]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.3, 1]))
    conds = ["original", "sham removed", "tumour removed", "decoy removed", "ring 0"]
    x = np.arange(len(conds)); w = 0.84 / len(runs)
    for k, r in enumerate(runs):
        axes[0].bar(x + (k - (len(runs) - 1) / 2) * w, [M.loc[r, c] for c in conds], w, yerr=[SD.loc[r, c] for c in conds],
                    color=COLS[r], label=SHORT[r], error_kw=dict(lw=0.5, capsize=1))
    axes[0].axhline(1 / 3, color="#888888", ls=":", lw=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["original", "sham\nremoved", "tumour\nremoved", "tumour +\ndecoys\nremoved", "tumour\nonly"])
    axes[0].set_ylabel("accuracy (P2, matched subset, mean ± s.d.)"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Occlusion with controls", fontsize=8)
    xr = [0, 6, 15, 30, 60, 90]; xp = np.arange(len(xr))   # categorical, equally spaced positions
    for r in runs:
        axes[1].plot(xp, [M.loc[r, f"ring {q}"] for q in xr[:-1]] + [M.loc[r, "original"]], "o-", color=COLS[r], ms=3,
                     lw=1.4 if r == "resnet50_TG-SSPANet" else 1)
    axes[1].set_xticks(xp); axes[1].set_xticklabels(["0", "6", "15", "30", "60", "full"])
    axes[1].set_xlabel("tissue kept around the tumour (px ≈ mm; categorical axis)"); axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.05); axes[1].axhline(1 / 3, color="#888888", ls=":", lw=0.7); axes[1].grid(alpha=0.3)
    axes[1].set_title("(b) Graded context", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for d in (FIG, PAPER):
        fig.savefig(os.path.join(d, "fig_occlusion_controls.png"), dpi=600)
    fig.savefig(os.path.join(FIG, "fig_occlusion_controls.pdf")); plt.close(fig)


def table4():
    rows = ["resnet50_none", "resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe", "resnet50_TG-noTFP", "resnet50_TG-SSPANet",
            "resnet50_TG-oracleGT", "resnet50_roicrop [pred crop]", "vgg16_none_ourrecipe", "vgg16_TG-SSPANet"]
    L = ["| Model | Orig. | Sham | Rem. | Decoy | Only | +15 | +60 | Deletion AUC | Insertion AUC |",
         "|------------------|------|------|------|------|------|------|------|-----------|-----------|"]
    for r in rows:
        g = M.loc[r]; base = r.split(" ")[0]; de = in_ = "–"
        if base in Fd.index and r != "resnet50_roicrop [pred crop]":
            de = f"{Fd.loc[base, 'deletion_auc']:.3f}"; in_ = f"{Fd.loc[base, 'insertion_auc']:.3f}"
            if not np.isnan(Fd.loc[base, "deletion_auc_random"]):
                de += f" [{Fd.loc[base, 'deletion_auc_random']:.3f}]"; in_ += f" [{Fd.loc[base, 'insertion_auc_random']:.3f}]"
        lab = NAMES[r].replace(", predicted-mask box", "")
        if "TG-SSPANet" in r and "oracle" not in r:
            lab = f"**{lab}**"
        L.append(f"| {lab} | {g['original']:.3f} | {g['sham removed']:.3f} | {g['tumour removed']:.3f} | {g['decoy removed']:.3f} | "
                 f"{g['ring 0']:.3f} | {g['ring 15']:.3f} | {g['ring 60']:.3f} | {de} | {in_} |")
    return "\n".join(L)


def tableS2():
    L = ["| Model | Orig. | Sham | Rem. | Decoy | 0 px | 6 px | 15 px | 30 px | 60 px |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ORDER:
        g, s = M.loc[r], SD.loc[r]
        L.append(f"| {NAMES[r]} | " + " | ".join(f"{g[c]:.3f} ± {s[c]:.3f}" for c in
                 ["original", "sham removed", "tumour removed", "decoy removed", "ring 0", "ring 6", "ring 15", "ring 30", "ring 60"]) + " |")
    return "\n".join(L)


def tableS3():
    cm = pd.read_csv(os.path.join(R3, "confusion_tumour_only.csv"))
    L = ["| Model | True class | → meningioma | → glioma | → pituitary |", "|---|---|---|---|---|"]
    for r in ORDER:
        for _, x in cm[cm.model == r].iterrows():
            L.append(f"| {NAMES[r]} | {x.true} | {x.pred_meningioma} | {x.pred_glioma} | {x.pred_pituitary} |")
    return "\n".join(L)


def tableS4():
    t = pd.read_csv(os.path.join(R3, "temperature_followup.csv"))
    nm = {"resnet50_none": "ResNet50 (orig.)", "resnet50_none_ourrecipe": "ResNet50", "resnet50_sspanet_ourrecipe": "ResNet50 + SSPANet",
          "resnet50_TG-SSPANet": "TG-SSPANet", "vgg16_TG-SSPANet": "VGG16 TG-SSPANet"}
    L = ["| Model | Stage | Slice acc. | ECE | AURC | Acc@90% | Patient acc. |", "|---|---|---|---|---|---|---|"]
    for r in nm:
        for st in ["before", "after"]:
            x = t[(t.model == r) & (t.stage == st)].iloc[0]
            L.append(f"| {nm[r]} | {st} scaling | {x.slice_acc:.3f} | {x.ece:.3f} | {x.aurc:.4f} | {x.acc_at_90cov:.3f} | {x.patient_acc:.3f} |")
    return "\n".join(L)


if __name__ == "__main__":
    figure()
    out = {"T4": table4(), "S2": tableS2(), "S3": tableS3(), "S4": tableS4()}
    io.open(os.path.join(R3, "tables_round3.md"), "w", encoding="utf8").write("\n\n".join(f"{k}\n{v}" for k, v in out.items()))
    print(out["T4"]); print(out["S4"])
