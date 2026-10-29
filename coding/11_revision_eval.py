"""Revision analyses on the patient-wise folds (P2) that need the trained models.

For every model x fold (test fold only, unflipped images, no TTA):
  * localisation of Grad-CAM / Grad-CAM++ / EigenGrad-CAM vs. the tumour mask: pointing game (7-px tolerance and none),
    energy-in-mask, CAM-IoU -- overall, per class and per tumour-size tercile;
  * reference explanations: centre prior, random map, and (for decoder models) the predicted tumour mask itself;
  * occlusion faithfulness: tumour removed (inpainted) and context removed (only the dilated tumour kept);
  * deletion / insertion curves of the true-class probability for Grad-CAM++ ordering and for random ordering.
Outputs: result/revision/p2_localisation.csv, p2_faithfulness.csv, p2_curves.npz
"""
import gc
import importlib
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation

import common as C
from xai import METHODS, CAMExtractor, localisation_metrics

xe = importlib.import_module("03_xai_eval")
OUT = os.path.join(C.RESULT, "revision"); os.makedirs(OUT, exist_ok=True)
MODELS = ["resnet50_none", "resnet50_sspanet", "resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe",
          "resnet50_se_ourrecipe", "resnet50_cbam_ourrecipe", "resnet50_TG-noTFP", "resnet50_TG-floor0.5",
          "resnet50_TG-floor0.3", "resnet50_TG-SSPANet", "resnet50_TG-oracleGT",
          "vgg16_none", "vgg16_sspanet", "vgg16_none_ourrecipe", "vgg16_sspanet_ourrecipe", "vgg16_TG-SSPANet",
          # major revision: P2 component ablation (21_major_revision_runs.py)
          "resnet50_TG-tgsaNoAlign", "resnet50_TG-tgsaAlign", "resnet50_TG-decOnly", "resnet50_TG-decTFP",
          "resnet50_TG-noAlign"]
STEPS = np.linspace(0, 1, 11)
RANDOM_REF = {"resnet50_none_ourrecipe", "resnet50_TG-SSPANet", "vgg16_TG-SSPANet"}  # random-order reference curves
MEAN, STD = C.MEAN.cuda(), C.STD.cuda()


class WithMask(torch.nn.Module):
    """Feeds the ground-truth mask to oracle models so the CAM extractor can call model(x)."""

    def __init__(self, m):
        super().__init__(); self.m = m; self.mask = None; self.cam_layer = m.cam_layer

    def forward(self, x):
        return self.m(x, self.mask)


def norm(x01):
    return (x01.repeat(1, 3, 1, 1) - MEAN) / STD


@torch.no_grad()
def probs(model, x01, y, bs=32):
    out = []
    for i in range(0, len(x01), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(norm(x01[i:i + bs]))
        out.append(torch.softmax(o["logits"].float(), 1).gather(1, y[i:i + bs, None])[:, 0])
    return torch.cat(out)


def centre_prior(n=224, s=0.25):
    g = np.exp(-(((np.arange(n) - n / 2) / (s * n)) ** 2) / 2)
    return np.outer(g, g)


def main():
    dev = torch.device("cuda")
    data = C.load_data()
    img224 = F.interpolate(torch.from_numpy(data["images"]).float()[:, None], size=(224, 224), mode="bilinear",
                           antialias=True, align_corners=False)[:, 0].numpy() / 255.0
    gt224 = F.interpolate(torch.from_numpy(data["masks"]).float()[:, None], size=(224, 224), mode="bilinear",
                          align_corners=False)[:, 0].numpy() > 0.5
    area = gt224.reshape(len(gt224), -1).mean(1)
    terc = np.quantile(area, [1 / 3, 2 / 3])
    size_bin = np.digitize(area, terc)                     # 0 small, 1 medium, 2 large
    cp = centre_prior(); rng = np.random.RandomState(0)
    loc_rows, faith_rows, curves = [], [], {}
    lf, ff = os.path.join(OUT, "p2_localisation.csv"), os.path.join(OUT, "p2_faithfulness.csv")
    done = set(pd.read_csv(lf)[["model", "fold"]].itertuples(index=False, name=None)) if os.path.exists(lf) else set()
    if os.path.exists(lf):
        loc_rows = pd.read_csv(lf).to_dict("records"); faith_rows = pd.read_csv(ff).to_dict("records")
        curves = dict(np.load(os.path.join(OUT, "p2_curves.npz"), allow_pickle=True)["curves"].item())
    for fold in range(1, 6):
        _, _, te = C.split_patient(data, fold)
        tag = f"patient_fold{fold}_seed0"
        y = torch.from_numpy(data["labels"][te]).to(dev)
        x01 = torch.from_numpy(img224[te]).float()[:, None].to(dev)
        gts = gt224[te]
        # occluded inputs (model independent)
        rem, keep = [], []
        for k in range(len(te)):
            im = (img224[te[k]] * 255).astype(np.uint8)
            d = binary_dilation(gts[k], iterations=5).astype(np.uint8)
            rem.append(cv2.inpaint(im, d, 7, cv2.INPAINT_TELEA) / 255.0)
            keep.append(img224[te[k]] * d)
        rem = torch.from_numpy(np.stack(rem)).float()[:, None].to(dev)
        keep = torch.from_numpy(np.stack(keep)).float()[:, None].to(dev)
        blur = torch.from_numpy(np.stack([cv2.GaussianBlur(img224[k], (0, 0), 10) for k in te])).float()[:, None].to(dev)
        # reference explanations (model independent)
        for refname, maps in [("centre prior", [cp] * len(te)), ("random map", [rng.rand(224, 224) for _ in te])]:
            key = (refname, fold)
            if key in done:
                continue
            r = np.array([localisation_metrics(m, g) for m, g in zip(maps, gts)], dtype=float)
            loc_rows.append(dict(model=refname, fold=fold, method="reference", tol=7, pointing_game=r[:, 0].mean(),
                                 energy_in_mask=r[:, 1].mean(), cam_iou=r[:, 2].mean()))
        for name in MODELS:
            if (name, fold) in done or not os.path.exists(os.path.join(C.RESULT, "patient", name, tag, "metrics.json")):
                continue
            model, mj, _ = xe.load_run("patient", name, tag, dev)
            oracle = mj["args"].get("oracle", False)
            net = WithMask(model) if oracle else model
            maps = {m: [] for m in METHODS}; dec = []
            for i in range(0, len(te), 16):
                xb = norm(x01[i:i + 16])
                if oracle:
                    net.mask = torch.from_numpy(gts[i:i + 16]).float()[:, None].to(dev)
                ext = CAMExtractor(net, net.cam_layer)
                mp, _ = ext(xb); ext.remove()
                for m in METHODS:
                    maps[m].append(mp[m])
                if "TG" in name and not oracle and getattr(model, "seg", False):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        dec.append(torch.sigmoid(model(xb)["mask_logits"].float())[:, 0].cpu().numpy())
            maps = {m: np.concatenate(v) for m, v in maps.items()}
            if dec:
                maps["decoder mask"] = np.concatenate(dec)
            for m, mm in maps.items():
                for tol in (7, 0):
                    if tol == 0 and m not in ("GradCAM++", "decoder mask"):
                        continue
                    r = np.array([localisation_metrics(a, g, tol=tol) for a, g in zip(mm, gts)], dtype=float)
                    row = dict(model=name, fold=fold, method=m, tol=tol, pointing_game=r[:, 0].mean(),
                               energy_in_mask=r[:, 1].mean(), cam_iou=r[:, 2].mean())
                    lab = data["labels"][te]; sb = size_bin[te]
                    for c, cn in enumerate(C.CLASSES):
                        row[f"pg_{cn}"] = r[lab == c, 0].mean()
                    for b, bn in enumerate(["small", "medium", "large"]):
                        row[f"pg_{bn}"] = r[sb == b, 0].mean()
                    loc_rows.append(row)
            if oracle:
                done.add((name, fold)); del model, net; torch.cuda.empty_cache(); continue
            torch.cuda.empty_cache()
            # faithfulness: occlusion
            p0 = probs(model, x01, y); pr = probs(model, rem, y); pk = probs(model, keep, y)
            acc = lambda xx: float((torch.cat([torch.softmax(model(norm(xx[i:i + 32]))["logits"].float(), 1).argmax(1)
                                               for i in range(0, len(xx), 32)]) == y).float().mean())
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                a0, ar, ak = acc(x01), acc(rem), acc(keep)
            # deletion / insertion curves ordered by Grad-CAM++ and by random order
            order_cam = torch.from_numpy(maps["GradCAM++"]).to(dev).flatten(1).argsort(1, descending=True)
            order_rnd = torch.rand(len(te), 224 * 224, device=dev).argsort(1)
            cur = {}
            orders = [("gradcampp", order_cam)] + ([("random", order_rnd)] if name in RANDOM_REF else [])
            for oname, order in orders:
                rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(224 * 224, device=dev).expand_as(order))
                rank = rank.view(-1, 1, 224, 224).float() / (224 * 224)
                dl, ins = [], []
                for s in STEPS:
                    msk = (rank < s).float()
                    dl.append(probs(model, x01 * (1 - msk) + blur * msk, y).mean().item())
                    ins.append(probs(model, blur * (1 - msk) + x01 * msk, y).mean().item())
                cur[oname] = (np.array(dl), np.array(ins))
            curves[(name, fold)] = cur
            faith_rows.append(dict(model=name, fold=fold, acc_original=a0, acc_tumour_removed=ar, acc_tumour_only=ak,
                                   p_original=p0.mean().item(), p_tumour_removed=pr.mean().item(), p_tumour_only=pk.mean().item(),
                                   deletion_auc=np.trapezoid(cur["gradcampp"][0], STEPS), insertion_auc=np.trapezoid(cur["gradcampp"][1], STEPS),
                                   deletion_auc_random=np.trapezoid(cur["random"][0], STEPS) if "random" in cur else np.nan,
                                   insertion_auc_random=np.trapezoid(cur["random"][1], STEPS) if "random" in cur else np.nan))
            print(fold, name, f"PG++ {np.mean([localisation_metrics(a, g)[0] for a, g in zip(maps['GradCAM++'], gts)]):.3f}",
                  f"acc {a0:.3f} removed {ar:.3f} only {ak:.3f}", flush=True)
            done.add((name, fold))
            del model, net, ext, order_cam, order_rnd, rank, msk, maps, p0, pr, pk
            gc.collect(); torch.cuda.empty_cache()
            pd.DataFrame(loc_rows).to_csv(lf, index=False); pd.DataFrame(faith_rows).to_csv(ff, index=False)
            np.savez(os.path.join(OUT, "p2_curves.npz"), curves=np.array(curves, dtype=object), steps=STEPS)
    pd.DataFrame(loc_rows).to_csv(lf, index=False); pd.DataFrame(faith_rows).to_csv(ff, index=False)
    np.savez(os.path.join(OUT, "p2_curves.npz"), curves=np.array(curves, dtype=object), steps=STEPS)
    print("done")


if __name__ == "__main__":
    main()
