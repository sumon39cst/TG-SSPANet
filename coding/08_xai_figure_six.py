"""Explainability comparison grids with six models (paper Figs. 11-12).

Rows: Baseline, Coordinate, GCNet, SPNet, SRMNet, TG-SSPANet (ours); columns: tumour type x CAM method.
Uses the seed-0 P1 models and the same three test slices as 03_xai_eval.py.
"""
import importlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import common as C
from xai import METHODS, CAMExtractor

xe = importlib.import_module("03_xai_eval")
ROWS = [("none", "Baseline"), ("coordinate", "Coordinate"), ("gcnet", "GCNet"), ("spnet", "SPNet"),
        ("srmnet", "SRMNet"), ("TG-SSPANet", "TG-SSPANet\n(ours)")]
MLAB = {"GradCAM": "Grad-CAM", "GradCAM++": "Grad-CAM++", "EigenGradCAM": "EigenGrad-CAM"}
OUT_XAI = os.path.join(C.RESULT, "figures", "xai")
PAPER = r"E:\Life_Nero\paper\figures"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7})


def main():
    dev = torch.device("cuda")
    data = C.load_data(); gd = C.GPUData(data, dev)
    _, _, te = C.split_random(data["labels"])
    ex = xe.pick_examples(data, te, {})
    x, m, _ = gd.batch(np.array(ex))
    gt = m[:, 0].cpu().numpy() > 0.5
    img = F.interpolate(torch.from_numpy(data["images"][ex]).float()[:, None], size=(C.IMG, C.IMG), mode="bilinear",
                        antialias=True, align_corners=False)[:, 0].numpy()
    names = [C.CLASSES[data["labels"][k]].capitalize() for k in ex]
    for bb, paper_name in [("resnet50", "fig11_xai_resnet50_six.png"), ("vgg16", "fig12_xai_vgg16_six.png")]:
        cams = []
        for key, _ in ROWS:
            model, _, _ = xe.load_run("main", f"{bb}_{key}", "random_seed0", dev)
            ext = CAMExtractor(model, model.cam_layer)
            maps, _ = ext(x); ext.remove()
            cams.append(maps); del model; torch.cuda.empty_cache()
        nr, nc = len(ROWS), 9
        fig = plt.figure(figsize=(7.2, 5.25))
        gs = fig.add_gridspec(nr, nc, left=0.095, right=0.995, top=0.905, bottom=0.005, wspace=0.04, hspace=0.05)
        for r, (_, label) in enumerate(ROWS):
            for t in range(3):
                for j, meth in enumerate(METHODS):
                    ax = fig.add_subplot(gs[r, t * 3 + j])
                    ax.imshow(img[t], cmap="gray")
                    ax.imshow(cams[r][meth][t], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                    ax.contour(gt[t], levels=[0.5], colors="#00E676", linewidths=0.7)
                    ax.set_xticks([]); ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_linewidth(1.4 if "(ours)" in label else 0.4)
                        s.set_color("#C0392B" if "(ours)" in label else "#333333")
                    if r == 0:
                        ax.set_title(MLAB[meth], fontsize=6.5, pad=2)
                    if t == 0 and j == 0:
                        ax.set_ylabel(label, fontsize=7, fontweight="bold" if "(ours)" in label else "normal",
                                      color="#C0392B" if "(ours)" in label else "black")
        # tumour-type group headers
        for t, n in enumerate(names):
            x0 = fig.axes[t * 3].get_position().x0; x1 = fig.axes[t * 3 + 2].get_position().x1
            fig.text((x0 + x1) / 2, 0.955, n, ha="center", va="center", fontsize=8, fontweight="bold")
            fig.add_artist(plt.Line2D([x0 + 0.004, x1 - 0.004], [0.938, 0.938], color="#555555", lw=0.7))
        fig.text(0.5, 0.99, f"{'ResNet50' if bb == 'resnet50' else 'VGG16'} backbone  ·  green contour: ground-truth tumour",
                 ha="center", va="top", fontsize=7.5, color="#333333")
        stem = f"xai_comparison_six_{bb}"
        fig.savefig(os.path.join(OUT_XAI, stem + ".png"), dpi=400)
        fig.savefig(os.path.join(OUT_XAI, stem + ".pdf"))
        fig.savefig(os.path.join(PAPER, paper_name), dpi=400)
        plt.close(fig)
        print("saved", stem)


if __name__ == "__main__":
    main()
