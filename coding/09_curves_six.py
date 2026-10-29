"""Training / validation curves for six models (paper Figs. 5-6): Baseline, Coordinate, GCNet, SPNet, SRMNet, TG-SSPANet."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common as C

ROWS = [("none", "Baseline", "#1f77b4"), ("coordinate", "Coordinate", "#ff7f0e"), ("gcnet", "GCNet", "#2ca02c"),
        ("spnet", "SPNet", "#9467bd"), ("srmnet", "SRMNet", "#8c564b"), ("TG-SSPANet", "TG-SSPANet (ours)", "#C0392B")]
PAPER = r"E:\Life_Nero\paper\figures"
FIG = os.path.join(C.RESULT, "figures")
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})


def main():
    for bb, name, paper_name in [("resnet50", "ResNet50", "fig05_training_curves_resnet50_six.png"),
                                 ("vgg16", "VGG16", "fig06_training_curves_vgg16_six.png")]:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
        for key, label, col in ROWS:
            h = pd.DataFrame(json.load(open(os.path.join(C.RESULT, "main", f"{bb}_{key}", "random_seed0", "history.json"))))
            ours = key == "TG-SSPANet"
            kw = dict(color=col, lw=2.0 if ours else 1.0, label=label, zorder=5 if ours else 2)
            axes[0, 0].plot(h.epoch, h.train_acc, **kw); axes[0, 1].plot(h.epoch, h.val_acc, **kw)
            axes[1, 0].plot(h.epoch, h.train_loss, **kw); axes[1, 1].plot(h.epoch, h.val_loss, **kw)
        titles = ["(a) Training accuracy", "(b) Validation accuracy", "(c) Training loss*", "(d) Validation loss (cross-entropy)"]
        for ax, t in zip(axes.flat, titles):
            ax.set_title(t, fontsize=8.5); ax.set_xlabel("epoch"); ax.grid(alpha=0.3, lw=0.5); ax.set_xlim(0, 41)
        axes[0, 0].set_ylabel("accuracy"); axes[0, 1].set_ylabel("accuracy")
        axes[1, 0].set_ylabel("loss"); axes[1, 1].set_ylabel("loss")
        axes[0, 0].set_ylim(0.6, 1.005); axes[0, 1].set_ylim(0.8, 1.003)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.035))
        fig.text(0.5, 0.008, "*TG-SSPANet training loss includes label smoothing, segmentation and alignment terms, so its "
                 "absolute value is not comparable. Baselines stop early on validation loss, as in [1].",
                 ha="center", fontsize=6.5, style="italic")
        fig.suptitle(f"{name} backbone (protocol P1, seed 0)", fontsize=9)
        fig.tight_layout(rect=(0, 0.075, 1, 0.97))
        stem = f"fig_training_curves_six_{bb}"
        fig.savefig(os.path.join(FIG, stem + ".png"), dpi=400); fig.savefig(os.path.join(FIG, stem + ".pdf"))
        fig.savefig(os.path.join(PAPER, paper_name), dpi=400)
        plt.close(fig)
        print("saved", stem)


if __name__ == "__main__":
    main()
