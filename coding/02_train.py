"""Train + test one configuration.

Examples
  python 02_train.py --kind baseline --backbone resnet50 --attn sspanet --recipe paper
  python 02_train.py --kind proposed --backbone resnet50 --recipe ours --name TG-SSPANet
  python 02_train.py --kind proposed --backbone resnet50 --recipe ours --protocol patient --fold 3
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import common as C
from models import build_model
from torch.utils.flop_counter import FlopCounterMode


def complexity(model, device):
    model.eval()
    x = torch.randn(1, 3, C.IMG, C.IMG, device=device)
    params = sum(p.numel() for p in model.parameters())
    if getattr(model, "needs_mask", False):
        return dict(params_M=params / 1e6, gflops=float("nan"), ms_per_image=float("nan"))
    with FlopCounterMode(display=False) as fc:
        model(x)
    flops = fc.get_total_flops() / 2  # MACs -> report GFLOPs as multiply-adds, like most papers
    xb = torch.randn(32, 3, C.IMG, C.IMG, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            model(xb)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(10):
            model(xb)
        torch.cuda.synchronize()
    return dict(params_M=params / 1e6, gflops=flops / 1e9, ms_per_image=(time.time() - t) / 320 * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="baseline", choices=["baseline", "proposed"])
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--attn", default="none")
    ap.add_argument("--recipe", default="paper")
    ap.add_argument("--protocol", default="random", choices=["random", "patient"])
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--group", default="main")
    # proposed-model switches (ablation)
    ap.add_argument("--stages", default="1,2,3")
    ap.add_argument("--no_seg", action="store_true")
    ap.add_argument("--no_tfp", action="store_true")
    ap.add_argument("--no_channel", action="store_true")
    ap.add_argument("--no_strip", action="store_true")
    ap.add_argument("--no_style", action="store_true")
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--oracle", action="store_true", help="gate with the ground-truth mask (upper bound)")
    ap.add_argument("--w_att", type=float, default=None)
    ap.add_argument("--w_seg", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()

    cfg = dict(C.RECIPES[a.recipe])
    for k in ("w_att", "w_seg", "epochs"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    name = a.name or (f"{a.backbone}_{a.attn}" if a.kind == "baseline" else f"{a.backbone}_TG-SSPANet")
    tag = f"{a.protocol}" + (f"_fold{a.fold}" if a.protocol == "patient" else "") + f"_seed{a.seed}"
    run_dir = os.path.join(C.RESULT, a.group, name, tag)
    pred_dir = os.path.join(C.PRED, a.group, name)
    os.makedirs(run_dir, exist_ok=True); os.makedirs(pred_dir, exist_ok=True); os.makedirs(C.CKPT, exist_ok=True)
    ckpt = os.path.join(C.CKPT, f"{a.group}__{name}__{tag}.pt")
    logf = open(os.path.join(run_dir, "train_log.txt"), "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    C.seed_all(a.seed)
    device = torch.device("cuda")
    data = C.load_data()
    split = C.split_random(data["labels"]) if a.protocol == "random" else C.split_patient(data, a.fold)
    log(f"{name} | {tag} | recipe={a.recipe} | train/val/test = {[len(s) for s in split]}")
    gd = C.GPUData(data, device)

    kw = {}
    if a.kind == "proposed":
        kw = dict(stages=tuple(int(s) for s in a.stages.split(",") if s != ""), seg=not a.no_seg, tfp=not a.no_tfp,
                  use_channel=not a.no_channel, use_strip=not a.no_strip, use_style=not a.no_style, floor=a.floor, oracle=a.oracle)
        if a.no_seg:
            cfg["w_seg"] = 0.0
    model = build_model(a.kind, a.backbone, a.attn, **kw).to(device)
    comp = complexity(model, device)
    log(f"complexity {comp}")
    model, hist = C.train(model, gd, split, cfg, run_dir, ckpt, log)

    te = split[2]
    prob, pmask = C.predict(model, gd, te, tta=cfg["tta"])
    y = data["labels"][te]
    res = C.metrics(y, prob)
    if pmask is not None:
        gt = torch.nn.functional.interpolate(torch.from_numpy(data["masks"][te]).float()[:, None], size=(C.IMG, C.IMG),
                                             mode="bilinear", align_corners=False)[:, 0].numpy() > 0.5
        res["seg_dice"], res["seg_iou"] = C.dice_iou(pmask, gt)
    res.update(comp)
    res.update(dict(name=name, protocol=a.protocol, fold=a.fold, seed=a.seed, recipe=a.recipe, group=a.group,
                    epochs_run=len(hist), config=cfg, args=vars(a)))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    pd.DataFrame({"index": te, "file_id": data["ids"][te], "patient_id": data["pids"][te],
                  "true_label": [C.CLASSES[i] for i in y], "pred_label": [C.CLASSES[i] for i in prob.argmax(1)],
                  "correct": prob.argmax(1) == y, "p_meningioma": prob[:, 0], "p_glioma": prob[:, 1], "p_pituitary": prob[:, 2]}
                 ).to_csv(os.path.join(pred_dir, f"test_predictions_{tag}.csv"), index=False)
    if pmask is not None:
        np.savez_compressed(os.path.join(pred_dir, f"test_masks_{tag}.npz"), index=te, mask=(pmask * 255).astype(np.uint8))
    log(f"TEST acc {res['accuracy']:.4f} f1 {res['f1_macro']:.4f} kappa {res['cohen_kappa']:.4f} mcc {res['mcc']:.4f}"
        + (f" dice {res['seg_dice']:.4f}" if pmask is not None else ""))


if __name__ == "__main__":
    main()
