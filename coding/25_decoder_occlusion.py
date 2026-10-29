"""Reviewer 3, comment 5: what does the TG-SSPANet decoder predict on tumour-removed and decoy images?

If the decoder finds no tumour, p ~ 0 everywhere, the gate is uniform (g = phi) and TG-SSPANet pools the whole slice.
For every matched-subset slice (P2 test folds, 256-px occlusions as in Table 4) we record, per condition:
  * p_max: maximum of the 7 x 7 gate input p (decoder probability, max-pooled), captured from the gate itself;
  * detected: p_max >= 0.5 (at least one gated cell is opened);
  * cells: number of 7 x 7 cells with p >= 0.5;
  * where the predicted mask (decoder probability >= 0.5, 224 px) lies: in the inpainted tumour region, in the decoy
    regions, or elsewhere;
  * whether the class prediction is correct.
Outputs: result/major_revision/decoder_occlusion_slices.csv and decoder_occlusion_summary.csv
"""
import importlib
import os

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt

import common as C

r2 = importlib.import_module("16_round2_eval"); r3 = importlib.import_module("19_round3_eval")
OUT = os.path.join(C.RESULT, "major_revision")
MODELS = ["resnet50_TG-SSPANet", "vgg16_TG-SSPANet"]
CONDS = ["original", "sham removed", "tumour removed", "decoy removed"]


def to224(m):
    return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)


def main():
    dev = torch.device("cuda"); data = C.load_data(); rows = []
    for fold in range(1, 6):
        _, _, te = C.split_patient(data, fold); tag = f"patient_fold{fold}_seed0"
        items = []
        for i in te:
            img, gt = data["images"][i], data["masks"][i].astype(bool)
            D = distance_transform_edt(~gt) <= 6
            regs = r3.candidate_regions(img, D, 2)
            if len(regs) < 2:
                continue                                           # matched subset, as in Table 4
            inp = lambda m: cv2.inpaint(img, m.astype(np.uint8), 7, cv2.INPAINT_TELEA)
            items.append(dict(i=i, y=data["labels"][i], D=to224(D), decoy=to224(regs[0] | regs[1]),
                              ims={"original": img, "sham removed": inp(regs[0]), "tumour removed": inp(D),
                                   "decoy removed": inp(D | regs[0] | regs[1])}))
        y = np.array([it["y"] for it in items])
        for name in MODELS:
            model = r2.load_model(name, tag, dev)
            grabbed = []
            h = model.gate.register_forward_hook(lambda mod, inp_, out: grabbed.append(inp_[1].detach().float().cpu()))
            for c in CONDS:
                grabbed.clear()
                ims = np.stack([it["ims"][c] for it in items])
                gd = C.GPUData({"images": ims, "masks": np.zeros_like(ims), "labels": y}, dev)
                prob, pmask = C.predict(model, gd, np.arange(len(y)), tta=False)
                p7 = torch.cat(grabbed).numpy()[:, 0]
                for k, it in enumerate(items):
                    pm = pmask[k] >= 0.5; area = pm.sum()
                    rows.append(dict(model=name, fold=fold, condition=c, index=int(it["i"]), y=int(it["y"]),
                                     correct=bool(prob[k].argmax() == it["y"]), p_max=float(p7[k].max()),
                                     cells=int((p7[k] >= 0.5).sum()), mask_px=int(area),
                                     frac_in_tumour=float((pm & it["D"]).sum() / area) if area else np.nan,
                                     frac_in_decoy=float((pm & it["decoy"]).sum() / area) if area else np.nan))
            h.remove(); del model; torch.cuda.empty_cache()
            print(fold, name, flush=True)
    df = pd.DataFrame(rows); df["detected"] = df.p_max >= 0.5
    df.to_csv(os.path.join(OUT, "decoder_occlusion_slices.csv"), index=False)
    g = df.groupby(["model", "condition"])
    s = pd.DataFrame({
        "n": g.size(), "detected": g.detected.mean(), "median_p_max": g.p_max.median(), "mean_cells": g.cells.mean(),
        "frac_in_tumour": g.frac_in_tumour.mean(), "frac_in_decoy": g.frac_in_decoy.mean(), "acc": g.correct.mean(),
        "acc_detected": df[df.detected].groupby(["model", "condition"]).correct.mean(),
        "acc_not_detected": df[~df.detected].groupby(["model", "condition"]).correct.mean(),
        "n_not_detected": (~df.detected).groupby([df.model, df.condition]).sum()})
    s.to_csv(os.path.join(OUT, "decoder_occlusion_summary.csv"))
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
