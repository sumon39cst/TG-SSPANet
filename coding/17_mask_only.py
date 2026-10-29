"""Round-2 control: location-only classifier (reviewer 2, comment 1).

Trains a ResNet50 (proposed recipe) on the binary tumour mask alone, with no image content, on each P2 fold.
Its accuracy bounds how much class information is carried by tumour location, size and shape.
python 17_mask_only.py --fold 1
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch

import common as C
from models import build_model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); a = ap.parse_args()
    name, group, tag = "resnet50_maskonly", "patient", f"patient_fold{a.fold}_seed0"
    run_dir = os.path.join(C.RESULT, group, name, tag); pred_dir = os.path.join(C.PRED, group, name)
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        print("skip", tag); return
    os.makedirs(run_dir, exist_ok=True); os.makedirs(pred_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "train_log.txt"), "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    C.seed_all(0); dev = torch.device("cuda")
    data = C.load_data(); tr, va, te = C.split_patient(data, a.fold)
    md = copy.deepcopy(data); md["images"] = (data["masks"] * 255).astype(np.uint8)   # input = binary mask only
    cfg = dict(C.RECIPES["ours"]); cfg["w_seg"] = 0.0; cfg["w_att"] = 0.0
    model = build_model("baseline", "resnet50", "none").to(dev)
    log(f"{name} | {tag} | train/val/test = {[len(tr), len(va), len(te)]}")
    model, hist = C.train(model, C.GPUData(md, dev), (tr, va, te), cfg, run_dir,
                          os.path.join(C.CKPT, f"{group}__{name}__{tag}.pt"), log)
    prob, _ = C.predict(model, C.GPUData(md, dev), te, tta=True)
    y = data["labels"][te]; res = C.metrics(y, prob)
    res.update(dict(name=name, protocol="patient", fold=a.fold, seed=0, recipe="ours", group=group, epochs_run=len(hist),
                    config=cfg, args=dict(kind="maskonly", backbone="resnet50", attn="none")))
    json.dump(res, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)
    pd.DataFrame({"index": te, "file_id": data["ids"][te], "patient_id": data["pids"][te],
                  "true_label": [C.CLASSES[i] for i in y], "pred_label": [C.CLASSES[i] for i in prob.argmax(1)],
                  "correct": prob.argmax(1) == y, "p_meningioma": prob[:, 0], "p_glioma": prob[:, 1], "p_pituitary": prob[:, 2]}
                 ).to_csv(os.path.join(pred_dir, f"test_predictions_{tag}.csv"), index=False)
    log(f"TEST acc (mask only) {res['accuracy']:.4f}")


if __name__ == "__main__":
    main()
