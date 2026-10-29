"""Third-round analyses (P2, 256-px scale, 1 px ~ 0.98 mm).

1. Occlusion on a matched subset (slices where a valid sham region exists), for all conditions:
   original / sham / tumour removed / decoy (tumour + two same-shaped decoy regions inpainted, so the inpainted
   footprint no longer uniquely marks the tumour location) / tumour + ring r in {0, 6, 15, 30, 60} px.
   Models: same-recipe controls, TG-SSPANet variants, positive controls, and the original-recipe models.
   The ROI-crop classifier is evaluated with two crop sources: the TG-SSPANet decoder mask predicted on the
   *original* slice (as in the two-stage pipeline) and the ground-truth mask.
2. Confusion matrices for the tumour-only (0 px) condition.
3. Temperature scaling: AURC, Acc@90% and patient-level accuracy recomputed after scaling.
Outputs: result/round3/*.csv
"""
import importlib
import os

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt
from sklearn.metrics import confusion_matrix

import common as C

r2 = importlib.import_module("16_round2_eval"); roi = importlib.import_module("10_roi_crop")
rs = importlib.import_module("12_revision_stats")
OUT = os.path.join(C.RESULT, "round3"); os.makedirs(OUT, exist_ok=True)
S = 256
MODELS = ["resnet50_none", "resnet50_sspanet", "resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe", "resnet50_TG-noTFP",
          "resnet50_TG-SSPANet", "resnet50_TG-oracleGT", "resnet50_roicrop", "vgg16_none", "vgg16_none_ourrecipe",
          "vgg16_TG-SSPANet"]
RINGS = [0, 6, 15, 30, 60]


def candidate_regions(img, D, n=2):
    """Up to n same-shaped regions inside the head that overlap neither the tumour nor each other."""
    H = r2.head_mask(img); forbid = binary_dilation(D, iterations=8)
    ys, xs = np.nonzero(D); h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    cands = [np.fliplr(D)]
    for dy, dx in [(0, w + 12), (0, -(w + 12)), (h + 12, 0), (-(h + 12), 0), (h + 12, w + 12), (-(h + 12), -(w + 12)),
                   (h + 12, -(w + 12)), (-(h + 12), w + 12), (0, 2 * (w + 12)), (0, -2 * (w + 12))]:
        if ys.min() + dy < 0 or ys.max() + dy >= S or xs.min() + dx < 0 or xs.max() + dx >= S:
            continue
        c = np.zeros_like(D); c[ys + dy, xs + dx] = True; cands.append(c)
    out = []
    for c in cands:
        if (c & forbid).any() or (c & H).sum() / c.sum() < 0.8:
            continue
        out.append(c); forbid = forbid | binary_dilation(c, iterations=8)
        if len(out) == n:
            break
    return out


def conditions(img, gt):
    dist = distance_transform_edt(~gt); D = dist <= 6
    inp = lambda m: cv2.inpaint(img, m.astype(np.uint8), 7, cv2.INPAINT_TELEA)
    regs = candidate_regions(img, D, 2)
    out = {"original": img, "tumour removed": inp(D)}
    out["sham removed"] = inp(regs[0]) if regs else None
    out["decoy removed"] = inp(D | regs[0] | regs[1]) if len(regs) == 2 else None
    for r in RINGS:
        out[f"ring {r}"] = (img * (gt if r == 0 else dist <= r)).astype(np.uint8)
    return out


def main():
    dev = torch.device("cuda"); data = C.load_data()
    rows, cms, valid_rows = [], {}, []
    for fold in range(1, 6):
        _, _, te = C.split_patient(data, fold); tag = f"patient_fold{fold}_seed0"
        imgs, gts, labs = data["images"][te], data["masks"][te].astype(bool), data["labels"][te]
        conds = [conditions(im, g) for im, g in zip(imgs, gts)]
        valid = np.array([c["sham removed"] is not None and c["decoy removed"] is not None for c in conds])
        valid_rows.append(dict(fold=fold, n=len(te), matched=int(valid.sum())))
        z = np.load(os.path.join(C.PRED, "patient", "resnet50_TG-SSPANet", f"test_masks_{tag}.npz"))
        pm = dict(zip(z["index"], z["mask"]))
        box_pred = [roi.box_from_mask(roi.largest_component(cv2.resize(pm[i].astype(np.float32) / 255, (S, S)) > 0.5)
                                      .astype(np.uint8)) for i in te]
        box_gt = [roi.box_from_mask(g.astype(np.uint8)) for g in gts]
        keys = list(conds[0].keys())
        for name in MODELS:
            if not os.path.exists(os.path.join(C.RESULT, "patient", name, tag, "metrics.json")):
                continue
            model = r2.load_model(name, tag, dev)
            variants = [("pred", box_pred), ("gt", box_gt)] if name == "resnet50_roicrop" else [("", None)]
            for vname, boxes in variants:
                label = name + (f" [{vname} crop]" if vname else "")
                for k in keys:
                    ims = np.stack([c[k] for c, v in zip(conds, valid) if v])
                    bx = [b for b, v in zip(boxes, valid) if v] if boxes is not None else None
                    msk = gts[valid].astype(np.uint8); y = labs[valid]
                    if bx is not None:
                        ims = np.stack([roi.crop(im, mk, b)[0] for im, mk, b in zip(ims, msk, bx)])
                    prob, _ = C.predict(model, C.GPUData({"images": ims, "masks": msk, "labels": y}, dev),
                                        np.arange(len(y)), tta=False)
                    rows.append(dict(model=label, fold=fold, condition=k, acc=float((prob.argmax(1) == y).mean()),
                                     p_true=float(prob[np.arange(len(y)), y].mean())))
                    if k == "ring 0":
                        cms[label] = cms.get(label, 0) + confusion_matrix(y, prob.argmax(1), labels=[0, 1, 2])
            print(fold, name, flush=True)
            del model; torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "occlusion_matched.csv"), index=False)
    pd.DataFrame(valid_rows).to_csv(os.path.join(OUT, "matched_subset.csv"), index=False)
    cm_rows = []
    for k, m in cms.items():
        for i, t in enumerate(C.CLASSES):
            cm_rows.append(dict(model=k, true=t, **{f"pred_{c}": int(m[i, j]) for j, c in enumerate(C.CLASSES)}))
    pd.DataFrame(cm_rows).to_csv(os.path.join(OUT, "confusion_tumour_only.csv"), index=False)
    temperature_followup()
    print("done")


def temperature_followup():
    tmp = pd.read_csv(os.path.join(C.RESULT, "round2", "temperature.csv")); tmp = tmp[tmp.fold != "pooled"]
    out = []
    for name, g in tmp.groupby("model"):
        P, PT, Y, PID = [], [], [], []
        for _, r in g.iterrows():
            d = pd.read_csv(os.path.join(C.PRED, "patient", name, f"test_predictions_patient_fold{int(float(r.fold))}_seed0.csv"))
            p = d[["p_meningioma", "p_glioma", "p_pituitary"]].values
            pt = torch.softmax(torch.tensor(np.log(np.clip(p, 1e-8, 1)) / float(r["T"])), 1).numpy()
            P.append(p); PT.append(pt); Y.append(d.true_label.map({c: i for i, c in enumerate(C.CLASSES)}).values)
            PID.append(d.patient_id.values)
        P, PT, Y, PID = map(np.concatenate, (P, PT, Y, PID))
        for stage, q in [("before", P), ("after", PT)]:
            conf = q.max(1); corr = q.argmax(1) == Y
            cov, risk = rs.risk_coverage(conf, corr)
            df = pd.DataFrame(q, columns=[0, 1, 2]).assign(pid=PID, y=Y)
            pm = df.groupby("pid")[[0, 1, 2]].mean(); py = df.groupby("pid").y.first()
            out.append(dict(model=name, stage=stage, slice_acc=corr.mean(), ece=rs.ece(conf, corr), aurc=np.trapezoid(risk, cov),
                            acc_at_90cov=1 - risk[int(0.9 * len(risk)) - 1],
                            patient_acc=float((pm.values.argmax(1) == py.values).mean())))
    pd.DataFrame(out).to_csv(os.path.join(OUT, "temperature_followup.csv"), index=False)
    print(pd.DataFrame(out).round(4).to_string())


if __name__ == "__main__":
    main()
