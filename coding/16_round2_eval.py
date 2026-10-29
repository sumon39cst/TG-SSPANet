"""Second-round analyses on P2 (all at the 256-px working scale, 1 px ~ 0.98 mm).

1. Occlusion with controls: original / tumour removed (inpainted, 6-px margin) / sham removed (same-shaped region
   inpainted elsewhere in the head) / tumour + context ring of r in {6, 15, 30, 60} px (everything else zeroed).
   Models include the positive controls (ROI-crop two-stage classifier, oracle ground-truth gate).
2. Location-only baselines: gradient boosting on mask geometry (no image content); the mask-only CNN is trained by
   17_mask_only.py and summarised here.
3. Temperature scaling fitted on each fold's validation patients; ECE before / after on the pooled test folds.
4. Centre-prior reference per class and per tumour-size tercile.
Outputs: result/round2/*.csv
"""
import importlib
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, distance_transform_edt
from sklearn.ensemble import HistGradientBoostingClassifier

import common as C
from models import build_model
from xai import localisation_metrics

xe = importlib.import_module("03_xai_eval")
roi = importlib.import_module("10_roi_crop")
rs = importlib.import_module("12_revision_stats")
OUT = os.path.join(C.RESULT, "round2"); os.makedirs(OUT, exist_ok=True)
MODELS = ["resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe", "resnet50_TG-noTFP", "resnet50_TG-SSPANet",
          "resnet50_TG-oracleGT", "resnet50_roicrop", "vgg16_TG-SSPANet", "vgg16_none_ourrecipe"]
RINGS = [6, 15, 30, 60]
S = 256


def load_model(name, tag, dev):
    if name == "resnet50_roicrop":
        m = build_model("baseline", "resnet50", "none", pretrained=False).to(dev)
        m.load_state_dict(torch.load(os.path.join(C.CKPT, f"patient__{name}__{tag}.pt"), map_location=dev, weights_only=True))
        return m.eval()
    return xe.load_run("patient", name, tag, dev)[0]


def head_mask(img):
    b = cv2.GaussianBlur(img, (0, 0), 3) > 0.12 * img.max()
    n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8))
    return lab == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else b


def sham_region(img, D):
    """Same-shaped region inside the head, not overlapping the (dilated) tumour: mirror first, then translations."""
    H = head_mask(img); forbid = binary_dilation(D, iterations=8)
    ys, xs = np.nonzero(D); h, w = np.ptp(ys) + 1, np.ptp(xs) + 1

    def shift(dy, dx):
        if ys.min() + dy < 0 or ys.max() + dy >= S or xs.min() + dx < 0 or xs.max() + dx >= S:
            return None                                   # would leave the image: reject (no wrap-around)
        c = np.zeros_like(D); c[ys + dy, xs + dx] = True
        return c

    cands = [np.fliplr(D)]
    for dy, dx in [(0, w + 12), (0, -(w + 12)), (h + 12, 0), (-(h + 12), 0), (h + 12, w + 12), (-(h + 12), -(w + 12)),
                   (h + 12, -(w + 12)), (-(h + 12), w + 12)]:
        cands.append(shift(dy, dx))
    best, bestfrac = None, -1
    for c in cands:
        if c is None or (c & forbid).any():
            continue
        frac = (c & H).sum() / c.sum()
        if frac > bestfrac:
            best, bestfrac = c, frac
        if frac >= 0.95:
            break
    return (best, bestfrac) if bestfrac >= 0.8 else (None, bestfrac)   # sham must lie inside the head


def conditions(img, gt):
    """Dict of occluded 256-px uint8 images for one slice."""
    dist = distance_transform_edt(~gt)                 # Euclidean distance to the tumour (px)
    D = dist <= 6
    out = {"original": img, "tumour removed": cv2.inpaint(img, D.astype(np.uint8), 7, cv2.INPAINT_TELEA)}
    sh, frac = sham_region(img, D)
    out["sham removed"] = cv2.inpaint(img, sh.astype(np.uint8), 7, cv2.INPAINT_TELEA) if sh is not None else None
    for r in RINGS:
        out[f"tumour + {r} px ring" if r > 6 else "tumour only"] = (img * (dist <= r)).astype(np.uint8)
    return out, frac if sh is not None else np.nan


@torch.no_grad()
def acc_p(model, imgs, masks, labels, dev, boxes=None):
    if boxes is not None:
        imgs = np.stack([roi.crop(im, mk, b)[0] for im, mk, b in zip(imgs, masks, boxes)])
    d = {"images": imgs, "masks": masks, "labels": labels}
    prob, _ = C.predict(model, C.GPUData(d, dev), np.arange(len(labels)), tta=False)
    return float((prob.argmax(1) == labels).mean()), float(prob[np.arange(len(labels)), labels].mean())


def occlusion(data, dev):
    rows, sham_ok = [], []
    for fold in range(1, 6):
        _, _, te = C.split_patient(data, fold); tag = f"patient_fold{fold}_seed0"
        imgs, gts, labs = data["images"][te], data["masks"][te].astype(bool), data["labels"][te]
        conds = {}; fracs = []
        for im, g in zip(imgs, gts):
            c, fr = conditions(im, g); fracs.append(fr)
            for k, v in c.items():
                conds.setdefault(k, []).append(v)
        valid = np.array([v is not None for v in conds["sham removed"]])
        sham_ok.append(dict(fold=fold, n=len(te), sham_found=int(valid.sum()), median_in_head=float(np.nanmedian(fracs))))
        conds["sham removed"] = [v if v is not None else im for v, im in zip(conds["sham removed"], imgs)]
        # ROI-crop boxes from the TG-SSPANet decoder of this fold (as in the two-stage pipeline)
        z = np.load(os.path.join(C.PRED, "patient", "resnet50_TG-SSPANet", f"test_masks_{tag}.npz"))
        pm = dict(zip(z["index"], z["mask"]))
        boxes = []
        for i in te:
            m = cv2.resize(pm[i].astype(np.float32) / 255, (S, S)) > 0.5
            boxes.append(roi.box_from_mask(roi.largest_component(m).astype(np.uint8)))
        for name in MODELS:
            if not os.path.exists(os.path.join(C.RESULT, "patient", name, tag, "metrics.json")):
                continue
            model = load_model(name, tag, dev)
            for cname, ims in conds.items():
                a, p = acc_p(model, np.stack(ims), gts.astype(np.uint8), labs, dev,
                             boxes if name == "resnet50_roicrop" else None)
                if cname == "sham removed":   # accuracy restricted to slices with a valid sham region
                    a2, p2 = acc_p(model, np.stack(ims)[valid], gts[valid].astype(np.uint8), labs[valid], dev,
                                   [b for b, v in zip(boxes, valid) if v] if name == "resnet50_roicrop" else None)
                    a, p = a2, p2
                rows.append(dict(model=name, fold=fold, condition=cname, acc=a, p_true=p))
            print(fold, name, {r["condition"]: round(r["acc"], 3) for r in rows if r["model"] == name and r["fold"] == fold}, flush=True)
            del model; torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "occlusion.csv"), index=False)
    pd.DataFrame(sham_ok).to_csv(os.path.join(OUT, "sham_validity.csv"), index=False)


def geometry(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return [np.nan] * 9
    cy, cx = ys.mean() / S, xs.mean() / S; area = len(ys) / S ** 2
    h, w = (np.ptp(ys) + 1) / S, (np.ptp(xs) + 1) / S
    cov = np.cov(np.stack([ys, xs])) if len(ys) > 1 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ecc = np.sqrt(1 - ev[1] / ev[0]) if ev[0] > 0 else 0
    return [cx, cy, abs(cx - 0.5), area, h, w, area / (h * w), ecc, np.arctan2(*np.linalg.eigh(cov)[1][:, -1])]


def location_features(data):
    X = np.array([geometry(m.astype(bool)) for m in data["masks"]]); y = data["labels"]
    rows = []; allp = []
    for fold in range(1, 6):
        tr, va, te = C.split_patient(data, fold); trv = np.concatenate([tr, va])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[trv], y[trv])
        p = clf.predict_proba(X[te]); allp.append(pd.DataFrame({"fold": fold, "y": y[te], "yhat": p.argmax(1),
                                                                "patient_id": data["pids"][te]}))
        rows.append(dict(fold=fold, acc=float((p.argmax(1) == y[te]).mean())))
    P = pd.concat(allp); pl = P.groupby("patient_id").apply(lambda g: np.bincount(g.yhat, minlength=3).argmax() == g.y.iloc[0])
    r = pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "location_features.csv"), index=False)
    return dict(acc_mean=r.acc.mean(), acc_sd=r.acc.std(ddof=1), patient_acc=pl.mean())


def temperature(data, dev):
    runs = ["resnet50_none", "resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe", "resnet50_TG-SSPANet", "vgg16_TG-SSPANet"]
    rows = []
    for name in runs:
        pooled_p, pooled_pT, pooled_y = [], [], []
        for fold in range(1, 6):
            _, va, te = C.split_patient(data, fold); tag = f"patient_fold{fold}_seed0"
            model = load_model(name, tag, dev)
            mj = json.load(open(os.path.join(C.RESULT, "patient", name, tag, "metrics.json")))
            pv, _ = C.predict(model, C.GPUData(data, dev), va, tta=mj["config"]["tta"])
            lv = np.log(np.clip(pv, 1e-8, 1)); yv = data["labels"][va]
            Ts = np.linspace(0.2, 3, 281)
            nll = [-np.mean(np.log(np.clip(torch.softmax(torch.tensor(lv / T), 1).numpy()[np.arange(len(yv)), yv], 1e-12, 1)))
                   for T in Ts]
            T = Ts[int(np.argmin(nll))]
            d = pd.read_csv(os.path.join(C.PRED, "patient", name, f"test_predictions_{tag}.csv"))
            p = d[["p_meningioma", "p_glioma", "p_pituitary"]].values
            pT = torch.softmax(torch.tensor(np.log(np.clip(p, 1e-8, 1)) / T), 1).numpy()
            pooled_p.append(p); pooled_pT.append(pT); pooled_y.append(d.true_label.map({c: i for i, c in enumerate(C.CLASSES)}).values)
            rows.append(dict(model=name, fold=fold, T=T))
            del model; torch.cuda.empty_cache()
        p, pT, y = np.concatenate(pooled_p), np.concatenate(pooled_pT), np.concatenate(pooled_y)
        for tagx, q in [("before", p), ("after", pT)]:
            conf = q.max(1); corr = q.argmax(1) == y
            rows.append(dict(model=name, fold="pooled", T=np.nan, stage=tagx, ece=rs.ece(conf, corr), mean_conf=conf.mean(),
                             acc=corr.mean(), brier=np.mean(np.sum((q - np.eye(3)[y]) ** 2, 1))))
        print(name, [r for r in rows if r["model"] == name and r["fold"] == "pooled"], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "temperature.csv"), index=False)


def centre_prior(data):
    img224 = F.interpolate(torch.from_numpy(data["masks"]).float()[:, None], size=(224, 224), mode="bilinear",
                           align_corners=False)[:, 0].numpy() > 0.5
    area = img224.reshape(len(img224), -1).mean(1); sb = np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))
    cp = importlib.import_module("11_revision_eval").centre_prior()
    hit = np.array([localisation_metrics(cp, g)[0] if g.any() else np.nan for g in img224], dtype=float)
    out = {}
    for fold in range(1, 6):
        _, _, te = C.split_patient(data, fold)
        out[fold] = dict(**{c: np.nanmean(hit[te][data["labels"][te] == k]) for k, c in enumerate(C.CLASSES)},
                         **{b: np.nanmean(hit[te][sb[te] == k]) for k, b in enumerate(["small", "medium", "large"])},
                         all=np.nanmean(hit[te]))
    r = pd.DataFrame(out).T; r.to_csv(os.path.join(OUT, "centre_prior_strata.csv")); return r.mean()


if __name__ == "__main__":
    dev = torch.device("cuda"); data = C.load_data()
    print("centre prior", centre_prior(data).round(3).to_dict(), flush=True)
    print("location features", location_features(data), flush=True)
    temperature(data, dev)
    occlusion(data, dev)
    print("done")
