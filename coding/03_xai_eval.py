"""Explainability evaluation of trained models (qualitative figures + quantitative metrics vs tumorMask).

Usage: python 03_xai_eval.py --group main --tag random_seed0
Writes result/<group>/xai_metrics_<tag>.csv and figures to result/figures/xai/.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common as C
from models import build_model
from xai import METHODS, CAMExtractor, localisation_metrics

DISPLAY = {"none": "Baseline", "coordinate": "Coordinate", "gcnet": "GCNet", "spnet": "SPNet", "srmnet": "SRMNet",
           "sspanet": "SSPANet (paper)", "se": "SE", "cbam": "CBAM"}


def load_run(group, name, tag, device):
    mj = json.load(open(os.path.join(C.RESULT, group, name, tag, "metrics.json")))
    a = mj["args"]
    kw = {}
    if a["kind"] == "proposed":
        kw = dict(stages=tuple(int(s) for s in a["stages"].split(",") if s != ""), seg=not a["no_seg"],
                  tfp=not a["no_tfp"], use_channel=not a["no_channel"], use_strip=not a["no_strip"],
                  use_style=not a["no_style"], floor=a.get("floor", 0.1), oracle=a.get("oracle", False))
    m = build_model(a["kind"], a["backbone"], a["attn"], pretrained=False, **kw).to(device)
    m.load_state_dict(torch.load(os.path.join(C.CKPT, f"{group}__{name}__{tag}.pt"), map_location=device, weights_only=True))
    m.eval()
    label = "TG-SSPANet (ours)" if a["kind"] == "proposed" else DISPLAY.get(a["attn"], a["attn"])
    return m, mj, label


def pick_examples(data, te, prob_ok, per_class=1, seed=3):
    """One correctly classified test slice per class with a reasonably sized tumour (deterministic)."""
    rng = np.random.RandomState(seed)
    out = []
    for c in [1, 0, 2]:  # glioma, meningioma, pituitary (paper order)
        cand = [i for i in te if data["labels"][i] == c and prob_ok.get(i, True) and data["masks"][i].sum() > 600]
        out += list(rng.choice(cand, per_class, replace=False))
    return out


def overlay(ax, img, cam=None, gt=None, title=None):
    ax.imshow(img, cmap="gray")
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    if gt is not None:
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main")
    ap.add_argument("--tag", default="random_seed0")
    ap.add_argument("--backbones", default="resnet50,vgg16")
    ap.add_argument("--all_runs", action="store_true", help="evaluate every run in the group (ablation); label by run name")
    a = ap.parse_args()
    if a.all_runs:
        a.backbones = "resnet50"
    device = torch.device("cuda")
    data = C.load_data()
    gd = C.GPUData(data, device)
    tr, va, te = C.split_random(data["labels"]) if a.tag.startswith("random") else C.split_patient(data, int(a.tag.split("fold")[1][0]))
    fig_dir = os.path.join(C.RESULT, "figures", "xai"); os.makedirs(fig_dir, exist_ok=True)
    gt_all = F.interpolate(torch.from_numpy(data["masks"]).float()[:, None], size=(C.IMG, C.IMG), mode="bilinear",
                           align_corners=False)[:, 0].numpy() > 0.5
    img_all = F.interpolate(torch.from_numpy(data["images"]).float()[:, None], size=(C.IMG, C.IMG), mode="bilinear",
                            align_corners=False, antialias=True)[:, 0].numpy()
    rows = []
    for bb in a.backbones.split(","):
        pattern = "*" if a.all_runs else f"{bb}_*"
        runs = sorted(d for d in glob.glob(os.path.join(C.RESULT, a.group, pattern)) if os.path.exists(os.path.join(d, a.tag, "metrics.json")))
        order = ["none", "coordinate", "gcnet", "spnet", "srmnet", "se", "cbam", "sspanet"]
        runs.sort(key=lambda d: order.index(os.path.basename(d)[len(bb) + 1:]) if os.path.basename(d)[len(bb) + 1:] in order else 99)
        cams_for_grid, labels = {}, []
        examples = None
        for rd in runs:
            name = os.path.basename(rd)
            model, mj, label = load_run(a.group, name, a.tag, device)
            if a.all_runs:
                label = name
            ext = CAMExtractor(model, model.cam_layer)
            stats = {m: [[], [], []] for m in METHODS}
            per_img = {}
            for i in range(0, len(te), 16):
                idx = te[i:i + 16]
                x, _, _ = gd.batch(idx)
                maps, _ = ext(x)
                for j, k in enumerate(idx):
                    per_img[k] = {m: maps[m][j] for m in METHODS}
                    for m in METHODS:
                        r = localisation_metrics(maps[m][j], gt_all[k])
                        if r:
                            for s, v in zip(stats[m], r):
                                s.append(v)
            ext.remove()
            for m in METHODS:
                rows.append(dict(backbone=bb, model=label, run=name, method=m, pointing_game=np.mean(stats[m][0]),
                                 energy_in_mask=np.mean(stats[m][1]), cam_iou=np.mean(stats[m][2])))
            print(bb, label, {m: round(float(np.mean(stats[m][0])), 3) for m in METHODS}, flush=True)
            if examples is None:
                examples = pick_examples(data, te, {})
            cams_for_grid[label] = {k: per_img[k] for k in examples}
            labels.append(label)
            # --- per-model figure (paper Figs. 6/7 layout): rows = tumour types, cols = original + 3 CAMs
            fig, axes = plt.subplots(3, 4, figsize=(8, 6.4))
            for r, k in enumerate(examples):
                overlay(axes[r, 0], img_all[k], gt=gt_all[k], title="Original + GT" if r == 0 else None)
                axes[r, 0].set_ylabel(C.CLASSES[data["labels"][k]].capitalize(), fontsize=9)
                for cidx, m in enumerate(METHODS):
                    overlay(axes[r, cidx + 1], img_all[k], per_img[k][m], gt_all[k], m if r == 0 else None)
            fig.suptitle(f"{bb.upper()} + {label}", fontsize=10)
            fig.tight_layout()
            fn = f"xai_{a.group}_{bb}_{name}_{a.tag}"
            fig.savefig(os.path.join(fig_dir, fn + ".png"), dpi=300); fig.savefig(os.path.join(fig_dir, fn + ".pdf")); plt.close(fig)
            del model; torch.cuda.empty_cache()
        # --- comparison grid (paper Figs. 8/9): rows = models, cols = (tumour type x CAM method)
        if labels:
            fig, axes = plt.subplots(len(labels), 9, figsize=(13.5, 1.55 * len(labels) + 0.6))
            axes = np.atleast_2d(axes)
            for r, lab in enumerate(labels):
                for t, k in enumerate(examples):
                    for mi, m in enumerate(METHODS):
                        ax = axes[r, t * 3 + mi]
                        overlay(ax, img_all[k], cams_for_grid[lab][k][m], gt_all[k],
                                f"{C.CLASSES[data['labels'][k]].capitalize()}\n{m}" if r == 0 else None)
                axes[r, 0].set_ylabel(lab, fontsize=8)
            fig.suptitle(f"Explainability comparison - {bb.upper()} backbone (green = ground-truth tumour contour)", fontsize=10)
            fig.tight_layout()
            fn = f"xai_comparison_{a.group}_{bb}_{a.tag}"
            fig.savefig(os.path.join(fig_dir, fn + ".png"), dpi=300); fig.savefig(os.path.join(fig_dir, fn + ".pdf")); plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULT, a.group, f"xai_metrics_{a.tag}.csv"), index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
