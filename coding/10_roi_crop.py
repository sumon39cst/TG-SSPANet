"""Two-stage mask-using baseline (revision, reviewer comment 5): classify the tumour ROI crop.

Stage 1 = tumour mask (training/validation: ground truth; test: the mask predicted for that fold by the
TG-SSPANet decoder, i.e. a realistic segment-then-classify pipeline).  Stage 2 = ResNet50 classifier trained with
the proposed recipe on square ROI crops (tumour box + 25 % margin, at least 64 px at 256-px scale, resized to 256).
Also reports the oracle variant that crops with the ground-truth mask at test time.

python 10_roi_crop.py --fold 1
"""
import argparse
import copy
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

import common as C
from models import build_model

S = 256


def box_from_mask(m, margin=0.25, min_side=64):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return 0, 0, S, S                                     # no tumour predicted: use the full slice
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0); side = int(max(min_side, side * (1 + 2 * margin)))
    side = min(side, S)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    ty = int(np.clip(round(cy - side / 2), 0, S - side)); tx = int(np.clip(round(cx - side / 2), 0, S - side))
    return ty, tx, ty + side, tx + side


def crop(img, msk, box):
    y0, x0, y1, x1 = box
    ci = cv2.resize(img[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_LINEAR)
    cm = cv2.resize(msk[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_NEAREST)
    return ci, cm


def largest_component(m):
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    name, group, tag = "resnet50_roicrop", "patient", f"patient_fold{a.fold}_seed{a.seed}"
    run_dir = os.path.join(C.RESULT, group, name, tag); pred_dir = os.path.join(C.PRED, group, name)
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        print("skip", tag); return
    os.makedirs(run_dir, exist_ok=True); os.makedirs(pred_dir, exist_ok=True)
    ckpt = os.path.join(C.CKPT, f"{group}__{name}__{tag}.pt")
    logf = open(os.path.join(run_dir, "train_log.txt"), "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    C.seed_all(a.seed); dev = torch.device("cuda")
    data = C.load_data()
    tr, va, te = C.split_patient(data, a.fold)
    # ground-truth crops for every slice
    gt = copy.deepcopy(data)
    for i in range(len(data["labels"])):
        gt["images"][i], gt["masks"][i] = crop(data["images"][i], data["masks"][i], box_from_mask(data["masks"][i]))
    # predicted-mask crops for the test slices (stage 1 = TG-SSPANet decoder of the same fold)
    pm = np.load(os.path.join(C.PRED, "patient", "resnet50_TG-SSPANet", f"test_masks_{tag}.npz"))
    pmap = dict(zip(pm["index"], pm["mask"]))
    pr = copy.deepcopy(data); empty = 0
    for i in te:
        m = cv2.resize(pmap[i].astype(np.float32) / 255, (S, S), interpolation=cv2.INTER_LINEAR) > 0.5
        m = largest_component(m); empty += int(m.sum() == 0)
        pr["images"][i], pr["masks"][i] = crop(data["images"][i], data["masks"][i], box_from_mask(m.astype(np.uint8)))
    log(f"{name} | {tag} | train/val/test = {[len(tr), len(va), len(te)]} | empty predicted masks: {empty}")
    cfg = dict(C.RECIPES["ours"]); cfg["w_seg"] = 0.0; cfg["w_att"] = 0.0
    model = build_model("baseline", "resnet50", "none").to(dev)
    model, hist = C.train(model, C.GPUData(gt, dev), (tr, va, te), cfg, run_dir, ckpt, log)
    prob, _ = C.predict(model, C.GPUData(pr, dev), te, tta=True)
    prob_gt, _ = C.predict(model, C.GPUData(gt, dev), te, tta=True)
    y = data["labels"][te]
    res = C.metrics(y, prob)
    res["accuracy_gtcrop"] = float((prob_gt.argmax(1) == y).mean())
    res.update(dict(name=name, protocol="patient", fold=a.fold, seed=a.seed, recipe="ours", group=group, epochs_run=len(hist),
                    empty_pred_masks=empty, config=cfg, args=dict(kind="roicrop", backbone="resnet50", attn="none")))
    json.dump(res, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)
    for p, suffix in [(prob, ""), (prob_gt, "_gtcrop")]:
        pd.DataFrame({"index": te, "file_id": data["ids"][te], "patient_id": data["pids"][te],
                      "true_label": [C.CLASSES[i] for i in y], "pred_label": [C.CLASSES[i] for i in p.argmax(1)],
                      "correct": p.argmax(1) == y, "p_meningioma": p[:, 0], "p_glioma": p[:, 1], "p_pituitary": p[:, 2]}
                     ).to_csv(os.path.join(pred_dir, f"test_predictions_{tag}{suffix}.csv"), index=False)
    log(f"TEST acc (predicted-mask crop) {res['accuracy']:.4f} | acc (GT crop, oracle) {res['accuracy_gtcrop']:.4f}")


if __name__ == "__main__":
    main()
