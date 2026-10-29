"""Prediction figures from the trained TG-SSPANet (saved in E:/Life_Nero/prediction/figures).

* grid of test predictions per class (MRI + GT contour + predicted mask + label/confidence)
* all misclassified test slices
* one overlay PNG for EVERY test slice (prediction/<run>/overlays/)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common as C

OUT = os.path.join(C.PRED, "figures")


def panel(ax, img, gt, pm, row, small=False):
    ax.imshow(img, cmap="gray")
    if pm is not None:
        ax.imshow(np.ma.masked_where(pm < 0.5, pm), cmap="autumn", alpha=0.45, vmin=0, vmax=1)
        ax.contour(pm, levels=[0.5], colors="red", linewidths=0.9)
    ax.contour(gt, levels=[0.5], colors="lime", linewidths=0.9)
    ok = row.true_label == row.pred_label
    conf = row[f"p_{row.pred_label}"]
    ax.set_title(f"True: {row.true_label}\nPred: {row.pred_label} ({conf:.3f})", fontsize=6.5 if small else 8,
                 color="darkgreen" if ok else "red")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main")
    ap.add_argument("--run", default="resnet50_TG-SSPANet")
    ap.add_argument("--tag", default="random_seed0")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = C.load_data()
    p = pd.read_csv(os.path.join(C.PRED, a.group, a.run, f"test_predictions_{a.tag}.csv"))
    mk = np.load(os.path.join(C.PRED, a.group, a.run, f"test_masks_{a.tag}.npz"))
    pmask = dict(zip(mk["index"], mk["mask"].astype(np.float32) / 255))
    idx = p["index"].values
    img = F.interpolate(torch.from_numpy(data["images"][idx]).float()[:, None], size=(C.IMG, C.IMG), mode="bilinear",
                        align_corners=False, antialias=True)[:, 0].numpy()
    gt = F.interpolate(torch.from_numpy(data["masks"][idx]).float()[:, None], size=(C.IMG, C.IMG), mode="bilinear",
                       align_corners=False)[:, 0].numpy()
    pos = {k: i for i, k in enumerate(idx)}
    name = f"{a.run}_{a.tag}"

    # 1) grid: 6 random correct examples per class
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(3, 6, figsize=(12, 7.2))
    for r, cls in enumerate(["glioma", "meningioma", "pituitary"]):
        sel = p[(p.true_label == cls)]
        sel = sel.iloc[rng.choice(len(sel), 6, replace=False)]
        for c, (_, row) in enumerate(sel.iterrows()):
            i = pos[row["index"]]
            panel(axes[r, c], img[i], gt[i], pmask[row["index"]], row)
        axes[r, 0].set_ylabel(cls.capitalize(), fontsize=10)
    fig.suptitle(f"TG-SSPANet test predictions ({a.run.split('_')[0]}) - green: ground-truth tumour, red: predicted tumour mask; "
                 f"title: true / predicted class (confidence)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pred_grid_{name}.png"), dpi=300); fig.savefig(os.path.join(OUT, f"pred_grid_{name}.pdf")); plt.close(fig)

    # 2) all misclassified test slices
    wrong = p[~p.correct]
    if len(wrong):
        n = len(wrong); cols = min(n, 6); rows_ = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_, cols, figsize=(2.2 * cols, 2.5 * rows_), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, row) in zip(axes.flat, wrong.iterrows()):
            ax.axis("on"); i = pos[row["index"]]
            panel(ax, img[i], gt[i], pmask[row["index"]], row, small=True)
            pmk = pmask[row["index"]] > 0.5; g = gt[i] > 0.5
            dice = 2 * (pmk & g).sum() / max(pmk.sum() + g.sum(), 1)
            ax.text(0.03, 0.04, f"Dice {dice:.2f} | patient {row.patient_id}", transform=ax.transAxes, fontsize=5,
                    color="white", va="bottom")
        where = "patient-wise fold " + a.tag.split("fold")[1][0] if "fold" in a.tag else "image-level test set"
        fig.suptitle(f"TG-SSPANet (ResNet50), {where}: all misclassified slices ({len(wrong)} of {len(p)})", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"pred_errors_{name}.png"), dpi=300); plt.close(fig)
    else:
        stale = os.path.join(OUT, f"pred_errors_{name}.png")
        if os.path.exists(stale):
            os.remove(stale)

    # 3) one overlay per test slice
    od = os.path.join(C.PRED, a.group, a.run, f"overlays_{a.tag}")
    os.makedirs(od, exist_ok=True)
    for _, row in p.iterrows():
        i = pos[row["index"]]
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
        axes[0].imshow(img[i], cmap="gray"); axes[0].set_title("MRI (T1-CE)", fontsize=8)
        axes[1].imshow(img[i], cmap="gray"); axes[1].contour(gt[i], levels=[0.5], colors="lime", linewidths=1)
        axes[1].set_title(f"Ground truth: {row.true_label}", fontsize=8)
        panel(axes[2], img[i], gt[i], pmask[row["index"]], row)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(od, f"{int(row.file_id):04d}_{row.true_label}_pred-{row.pred_label}.png"), dpi=110)
        plt.close(fig)
    print("saved figures to", OUT, "and", len(p), "overlays to", od, "; errors:", len(wrong))


if __name__ == "__main__":
    main()
