"""Major revision (Life, two reviewers): analyses that need no new training.

1. aggregation  - patient-level accuracy under several aggregation rules (Reviewer 1, comment 1)
2. valsplits    - exact composition of the P2 validation sets (Reviewer 1, comment 7)
3. shuffle      - tile-shuffle control on the matched occlusion subset (Reviewer 1, comment 2; Reviewer 2, comment 3):
                  head tissue is cut into 16-px tiles that are permuted, which destroys location and anatomical layout
                  but keeps local tissue statistics (intensity distribution and texture)
                    context shuffled      tumour + 6-px margin kept in place, all other head tiles permuted
                    removed + shuffled    tumour + margin inpainted, then all head tiles permuted (footprint moves too)
                    all shuffled          all head tiles of the original slice permuted (tumour pieces move too)
4. shift        - simulated acquisition shifts on the full P2 test folds (Reviewer 2, comment 2): noise, bias field,
                  gamma contrast and reduced in-plane resolution. A stress test, not external validation.
Outputs: result/major_revision/*.csv
"""
import importlib
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt, gaussian_filter

import common as C

r2 = importlib.import_module("16_round2_eval"); roi = importlib.import_module("10_roi_crop")
r3 = importlib.import_module("19_round3_eval")
OUT = os.path.join(C.RESULT, "major_revision"); os.makedirs(OUT, exist_ok=True)
S, T = 256, 16
CLS = {c: i for i, c in enumerate(C.CLASSES)}
AGG_MODELS = ["resnet50_none", "resnet50_sspanet", "resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe",
              "resnet50_srmnet_ourrecipe", "resnet50_TG-noTFP", "resnet50_roicrop", "resnet50_TG-oracleGT",
              "resnet50_TG-SSPANet", "vgg16_none_ourrecipe", "vgg16_sspanet_ourrecipe", "vgg16_TG-SSPANet"]
EVAL_MODELS = ["resnet50_none", "resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe", "resnet50_TG-noTFP",
               "resnet50_TG-SSPANet", "resnet50_TG-oracleGT", "resnet50_roicrop", "vgg16_none_ourrecipe", "vgg16_TG-SSPANet"]


# ---------------------------------------------------------------------------------------------- 1. aggregation
def load_preds(name):
    ds = []
    for k in range(1, 6):
        f = os.path.join(C.PRED, "patient", name, f"test_predictions_patient_fold{k}_seed0.csv")
        if not os.path.exists(f):
            return None
        ds.append(pd.read_csv(f).assign(fold=k))
    d = pd.concat(ds, ignore_index=True)
    d["y"] = d.true_label.map(CLS); d["yhat"] = d.pred_label.map(CLS)
    return d


def patient_rules(d):
    """Per-patient correctness under each aggregation rule."""
    P = d[["p_meningioma", "p_glioma", "p_pituitary"]].values
    rows = []
    for pid, g in d.groupby("patient_id"):
        idx = g.index.values; p = P[idx]; y = int(g.y.iloc[0]); yh = g.yhat.values
        votes = np.bincount(yh, minlength=3)
        top = np.flatnonzero(votes == votes.max())
        maj = top[0] if len(top) == 1 else top[np.argmax(p.mean(0)[top])]       # ties broken by mean probability
        conf = p.max(1)
        rows.append(dict(pid=pid, y=y, n=len(idx), mean_prob=p.mean(0).argmax() == y, majority=maj == y,
                         most_confident=yh[conf.argmax()] == y, all_slices=bool((yh == y).all()),
                         discordant=len(np.unique(yh)) > 1, any_correct=bool((yh == y).any())))
    return pd.DataFrame(rows)


def boot_ci(v, B=2000, seed=0):
    rng = np.random.RandomState(seed); v = np.asarray(v, float)
    bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(B)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def aggregation():
    out = []
    for name in AGG_MODELS:
        d = load_preds(name)
        if d is None:
            continue
        r = patient_rules(d)
        row = dict(model=name, slice_acc=float((d.y == d.yhat).mean()), patients=len(r))
        for rule in ["mean_prob", "majority", "most_confident", "all_slices", "any_correct", "discordant"]:
            lo, hi = boot_ci(r[rule].values)
            row[rule] = float(r[rule].mean()); row[rule + "_lo"] = lo; row[rule + "_hi"] = hi
        row["n_wrong_mean_prob"] = int((~r.mean_prob).sum()); row["n_wrong_majority"] = int((~r.majority).sum())
        row["n_discordant"] = int(r.discordant.sum())
        out.append(row)
    df = pd.DataFrame(out); df.to_csv(os.path.join(OUT, "patient_aggregation.csv"), index=False)
    print(df[["model", "slice_acc", "mean_prob", "majority", "most_confident", "all_slices", "any_correct", "discordant",
              "n_wrong_mean_prob", "n_wrong_majority", "n_discordant"]].round(4).to_string())


# ---------------------------------------------------------------------------------------------- 2. validation splits
def valsplits():
    data = C.load_data(); rows = []
    for k in range(1, 6):
        tr, va, te = C.split_patient(data, k)
        for part, idx in [("train", tr), ("validation", va), ("test", te)]:
            row = dict(fold=k, part=part, slices=len(idx), patients=len(np.unique(data["pids"][idx])))
            for c, cn in enumerate(C.CLASSES):
                sel = idx[data["labels"][idx] == c]
                row[f"{cn}_patients"] = len(np.unique(data["pids"][sel])); row[f"{cn}_slices"] = len(sel)
            rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "p2_split_composition.csv"), index=False)
    print(df.to_string())


# ---------------------------------------------------------------------------------------------- 3. tile shuffle
def head_tiles(img, exclude=None):
    H = binary_fill_holes(r2.head_mask(img)); tiles = []    # fill holes: dark ventricles must be shuffled too
    for i in range(0, S, T):
        for j in range(0, S, T):
            if H[i:i + T, j:j + T].all() and (exclude is None or not exclude[i:i + T, j:j + T].any()):
                tiles.append((i, j))
    return tiles


def shuffle_tiles(img, tiles, rng):
    out = img.copy(); perm = rng.permutation(len(tiles))
    for (i, j), k in zip(tiles, perm):
        a, b = tiles[k]; out[i:i + T, j:j + T] = img[a:a + T, b:b + T]
    return out


def shuffle_conditions(img, gt, seed):
    dist = distance_transform_edt(~gt); D = dist <= 6
    rng = np.random.RandomState(seed)
    removed = cv2.inpaint(img, D.astype(np.uint8), 7, cv2.INPAINT_TELEA)
    return {"context shuffled": shuffle_tiles(img, head_tiles(img, binary_dilation(D, iterations=2)), rng),
            "removed + shuffled": shuffle_tiles(removed, head_tiles(img), rng),
            "all shuffled": shuffle_tiles(img, head_tiles(img), rng)}


# ---------------------------------------------------------------------------------------------- 4. acquisition shift
def bias_field(shape, amp, seed):
    f = gaussian_filter(np.random.RandomState(seed).randn(*shape), 48)
    f = (f - f.min()) / (np.ptp(f) + 1e-8) * 2 - 1
    return 1 + amp * f


def shift_conditions(img, seed):
    x = img.astype(np.float32); rng = np.random.RandomState(seed)
    clip = lambda a: np.clip(a, 0, 255).astype(np.uint8)
    low = cv2.resize(cv2.resize(x, (S // 2, S // 2), interpolation=cv2.INTER_AREA), (S, S), interpolation=cv2.INTER_LINEAR)
    return {"original": img,
            "noise 2%": clip(x + rng.randn(S, S) * 0.02 * 255), "noise 5%": clip(x + rng.randn(S, S) * 0.05 * 255),
            "bias field 20%": clip(x * bias_field((S, S), 0.2, seed)), "bias field 40%": clip(x * bias_field((S, S), 0.4, seed)),
            "gamma 0.7": clip(255 * (x / 255) ** 0.7), "gamma 1.5": clip(255 * (x / 255) ** 1.5),
            "half resolution": clip(low),
            "combined": clip(cv2.resize(cv2.resize(255 * (np.clip(x * bias_field((S, S), 0.2, seed), 0, 255) / 255) ** 1.2,
                                                   (S // 2, S // 2), interpolation=cv2.INTER_AREA), (S, S))
                             + rng.randn(S, S) * 0.02 * 255)}


# ---------------------------------------------------------------------------------------------- shared evaluation loop
def evaluate(kind):
    dev = torch.device("cuda"); data = C.load_data(); rows = []
    for fold in range(1, 6):
        _, _, te = C.split_patient(data, fold); tag = f"patient_fold{fold}_seed0"
        imgs, gts, labs = data["images"][te], data["masks"][te].astype(bool), data["labels"][te]
        if kind == "shuffle":
            occ = [r3.conditions(im, g) for im, g in zip(imgs, gts)]
            keep = np.array([c["sham removed"] is not None and c["decoy removed"] is not None for c in occ])
            conds = [dict(original=im, **shuffle_conditions(im, g, int(i))) for im, g, i in zip(imgs, gts, te)]
        else:
            keep = np.ones(len(te), bool)
            conds = [shift_conditions(im, int(i)) for im, i in zip(imgs, te)]
        conds = [c for c, v in zip(conds, keep) if v]; gts_k, y = gts[keep], labs[keep]
        keys = list(conds[0].keys())
        z = np.load(os.path.join(C.PRED, "patient", "resnet50_TG-SSPANet", f"test_masks_{tag}.npz"))
        pm = dict(zip(z["index"], z["mask"]))
        box_orig = [roi.box_from_mask(roi.largest_component(cv2.resize(pm[i].astype(np.float32) / 255, (S, S)) > 0.5)
                                      .astype(np.uint8)) for i, v in zip(te, keep) if v]
        box_cond = {}
        for name in EVAL_MODELS:
            if not os.path.exists(os.path.join(C.RESULT, "patient", name, tag, "metrics.json")):
                continue
            model = r2.load_model(name, tag, dev)
            for k in keys:
                ims = np.stack([c[k] for c in conds]); msk = gts_k.astype(np.uint8)
                if name == "resnet50_roicrop":
                    # shuffle: crop box from the decoder on the unperturbed slice (as in the occlusion tests);
                    # shift: the two-stage pipeline segments the perturbed slice itself
                    bx = box_orig if kind == "shuffle" else box_cond[k]
                    ims = np.stack([roi.crop(im, mk, b)[0] for im, mk, b in zip(ims, msk, bx)])
                prob, pmask = C.predict(model, C.GPUData({"images": ims, "masks": msk, "labels": y}, dev),
                                        np.arange(len(y)), tta=False)
                if name == "resnet50_TG-SSPANet" and kind == "shift":
                    box_cond[k] = [roi.box_from_mask(roi.largest_component(cv2.resize(m.astype(np.float32), (S, S)) > 0.5)
                                                     .astype(np.uint8)) for m in pmask]
                rows.append(dict(model=name, fold=fold, condition=k, n=len(y), acc=float((prob.argmax(1) == y).mean()),
                                 p_true=float(prob[np.arange(len(y)), y].mean())))
            print(kind, fold, name, flush=True)
            del model; torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{kind}_control.csv"), index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["model", "condition"]).acc.agg(["mean", "std"]).unstack().round(3).to_string())


def save_examples():
    """Example images of the new conditions for the supplementary figure."""
    data = C.load_data(); _, _, te = C.split_patient(data, 1)
    picks = [te[np.flatnonzero(data["labels"][te] == c)[3]] for c in range(3)]
    ex = {}
    for i in picks:
        im, g = data["images"][i], data["masks"][i].astype(bool)
        ex[int(i)] = dict(original=im, **shuffle_conditions(im, g, int(i)),
                          **{k: v for k, v in shift_conditions(im, int(i)).items() if k != "original"})
    np.save(os.path.join(OUT, "examples.npy"), ex, allow_pickle=True)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["aggregation", "valsplits", "shuffle", "shift", "examples"]:
        {"aggregation": aggregation, "valsplits": valsplits, "shuffle": lambda: evaluate("shuffle"),
         "shift": lambda: evaluate("shift"), "examples": save_examples}[w]()
