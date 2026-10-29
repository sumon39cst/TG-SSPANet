"""Data, splits, GPU augmentation, training loop and metrics shared by all experiments."""
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

ROOT = r"E:\Life_Nero"
DATA = os.path.join(ROOT, "dataset", "processed", "figshare_256.npz")
RESULT = os.path.join(ROOT, "result")
PRED = os.path.join(ROOT, "prediction")
CKPT = os.path.join(RESULT, "checkpoints")
CLASSES = ["meningioma", "glioma", "pituitary"]
IMG = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_data():
    d = np.load(DATA)
    return {k: d[k] for k in d.files}


def split_random(labels, seed=42):
    """Paper protocol (Table 1): stratified image-level 80/10/10 -> 2450 / 307 / 307."""
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in range(3):
        idx = rng.permutation(np.where(labels == c)[0])
        n = len(idx); n_te = int(round(n * 0.1)); n_va = n_te
        te += list(idx[:n_te]); va += list(idx[n_te:n_te + n_va]); tr += list(idx[n_te + n_va:])
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


def split_patient(data, fold, seed=42):
    """Leakage-free protocol: official patient-wise fold `fold` (1..5) is the test set,
    ~10% of the remaining *patients* form the validation set."""
    cv, pids, labels = data["cvfold"], data["pids"], data["labels"]
    te = np.where(cv == fold)[0]
    rest = np.where(cv != fold)[0]
    rng = np.random.RandomState(seed + fold)
    pat = np.unique(pids[rest])
    # stratify patients by their (majority) label
    plab = np.array([np.bincount(labels[rest][pids[rest] == p]).argmax() for p in pat])
    val_p = []
    for c in range(3):
        pc = rng.permutation(pat[plab == c]); val_p += list(pc[:max(1, int(round(len(pc) * 0.1)))])
    isval = np.isin(pids[rest], val_p)
    va, tr = rest[isval], rest[~isval]
    assert not set(pids[te]) & set(pids[tr]) and not set(pids[va]) & set(pids[tr])
    return np.sort(tr), np.sort(va), np.sort(te)


class GPUData:
    """Keeps the whole (small) dataset on the GPU; batches are built by indexing."""

    def __init__(self, data, device):
        self.img = torch.from_numpy(data["images"]).to(device)
        self.mask = torch.from_numpy(data["masks"]).to(device)
        self.lab = torch.from_numpy(data["labels"]).to(device)
        self.device = device

    def batch(self, idx, augment=False, strong=False):
        idx = torch.as_tensor(idx, device=self.device)
        x = self.img[idx].float().unsqueeze(1) / 255.0
        m = self.mask[idx].float().unsqueeze(1)
        y = self.lab[idx]
        b = x.shape[0]
        if augment:
            ang = (torch.rand(b, device=x.device) * 2 - 1) * (15 if strong else 0) * math.pi / 180
            sc = 1 + (torch.rand(b, device=x.device) * 2 - 1) * (0.1 if strong else 0)
            tx = (torch.rand(b, 2, device=x.device) * 2 - 1) * (0.06 if strong else 0)
            flip = torch.where(torch.rand(b, device=x.device) < 0.5, -1.0, 1.0)
            cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
            theta = torch.stack([torch.stack([cos * flip, -sin, tx[:, 0]], 1),
                                 torch.stack([sin * flip, cos, tx[:, 1]], 1)], 1)
            grid = F.affine_grid(theta, (b, 1, IMG, IMG), align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            m = (F.grid_sample(m, grid, mode="bilinear", align_corners=False) > 0.5).float()
            if strong:
                c = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * 0.15
                br = (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * 0.08
                x = ((x - 0.5) * c + 0.5 + br).clamp(0, 1)
        else:
            x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False, antialias=True)
            m = (F.interpolate(m, size=(IMG, IMG), mode="bilinear", align_corners=False) > 0.5).float()
        x = (x.repeat(1, 3, 1, 1) - MEAN.to(x.device)) / STD.to(x.device)
        return x, m, y


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits.float())
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def compute_loss(out, y, m, cfg, class_w):
    ce = F.cross_entropy(out["logits"].float(), y, weight=class_w, label_smoothing=cfg.get("label_smoothing", 0.0))
    loss, parts = ce, {"ce": ce.item()}
    if "mask_logits" in out and cfg.get("w_seg", 0) > 0:
        ml = out["mask_logits"].float()
        seg = F.binary_cross_entropy_with_logits(ml, m) + dice_loss(ml, m)
        loss = loss + cfg["w_seg"] * seg; parts["seg"] = seg.item()
    if out.get("attn_logits") and cfg.get("w_att", 0) > 0:
        att = 0
        for a in out["attn_logits"]:
            t = F.adaptive_avg_pool2d(m, a.shape[-2:])
            att = att + F.binary_cross_entropy_with_logits(a.float(), t)
        att = att / len(out["attn_logits"])
        loss = loss + cfg["w_att"] * att; parts["att"] = att.item()
    return loss, parts


@torch.no_grad()
def predict(model, gd, idx, bs=64, tta=False):
    model.eval()
    probs, masks = [], []
    for i in range(0, len(idx), bs):
        x, m, _ = gd.batch(idx[i:i + bs])
        nm = getattr(model, "needs_mask", False)
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(x, m) if nm else model(x)
            p = torch.softmax(o["logits"].float(), 1)
            mk = torch.sigmoid(o["mask_logits"].float()) if "mask_logits" in o else None
            if tta:
                of = model(torch.flip(x, [3]), torch.flip(m, [3])) if nm else model(torch.flip(x, [3]))
                p = (p + torch.softmax(of["logits"].float(), 1)) / 2
                if mk is not None:
                    mk = (mk + torch.flip(torch.sigmoid(of["mask_logits"].float()), [3])) / 2
        probs.append(p.cpu())
        if mk is not None:
            masks.append(mk.cpu())
    return torch.cat(probs).numpy(), (torch.cat(masks).numpy()[:, 0] if masks else None)


def metrics(y, prob):
    pred = prob.argmax(1)
    r = {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
    }
    try:
        r["auc_macro_ovr"] = roc_auc_score(y, prob, multi_class="ovr", average="macro")
    except ValueError:
        r["auc_macro_ovr"] = float("nan")
    for c, n in enumerate(CLASSES):
        tp = ((pred == c) & (y == c)).sum()
        r[f"recall_{n}"] = tp / max((y == c).sum(), 1)
        r[f"precision_{n}"] = tp / max((pred == c).sum(), 1)
    r["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    return {k: (float(v) if not isinstance(v, list) else v) for k, v in r.items()}


def dice_iou(pm, gt, thr=0.5):
    p = pm > thr
    inter = (p & gt).sum((1, 2)); s = p.sum((1, 2)) + gt.sum((1, 2)); u = (p | gt).sum((1, 2))
    return float(np.mean((2 * inter + 1e-6) / (s + 1e-6))), float(np.mean((inter + 1e-6) / (u + 1e-6)))


RECIPES = {
    # faithful to Sec. 3.6 of the paper: Adam, lr 1e-4, batch 32, 30 epochs, CE, early stopping on val loss
    "paper": dict(epochs=30, lr=1e-4, opt="adam", wd=0.0, bs=32, patience=7, select="val_loss",
                  augment=False, strong=False, label_smoothing=0.0, class_weight=False, ema=False, tta=False,
                  w_seg=0.0, w_att=0.0),
    # proposed training recipe (also used for every ablation so gains are attributable to the model)
    "ours": dict(epochs=40, lr=2e-4, head_lr_mult=5.0, opt="adamw", wd=1e-4, bs=32, patience=100, select="val_acc",
                 augment=True, strong=True, label_smoothing=0.1, class_weight=True, ema=True, tta=True,
                 w_seg=1.0, w_att=0.2, warmup=2),
}


def param_groups(model, lr, mult):
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return [{"params": bb, "lr": lr}, {"params": hd, "lr": lr * mult}]


def train(model, gd, split, cfg, run_dir, ckpt_path, log=print):
    device = gd.device
    tr, va, te = split
    model.to(device)
    if cfg["opt"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    else:
        opt = torch.optim.AdamW(param_groups(model, cfg["lr"], cfg.get("head_lr_mult", 1.0)), weight_decay=cfg["wd"])
    steps_per_epoch = math.ceil(len(tr) / cfg["bs"])
    total = steps_per_epoch * cfg["epochs"]
    warm = steps_per_epoch * cfg.get("warmup", 0)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda")
    ema = None
    if cfg["ema"]:
        ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995),
                                                  use_buffers=True)
    labels_tr = gd.lab[torch.as_tensor(tr, device=device)]
    class_w = None
    if cfg["class_weight"]:
        cnt = torch.bincount(labels_tr, minlength=3).float()
        class_w = (cnt.sum() / (3 * cnt)).to(device)
    hist, best, bad, step = [], None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train(); t0 = time.time()
        perm = np.random.permutation(tr)
        tl, tc, n = 0.0, 0, 0
        for i in range(0, len(perm), cfg["bs"]):
            bi = perm[i:i + cfg["bs"]]
            if len(bi) < 2:
                continue
            x, m, y = gd.batch(bi, augment=cfg["augment"], strong=cfg["strong"])
            if cfg["opt"] != "adam":
                f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                for g, bl in zip(opt.param_groups, base_lrs):
                    g["lr"] = bl * f
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x, m) if getattr(model, "needs_mask", False) else model(x)
                loss, _ = compute_loss(out, y, m, cfg, class_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); step += 1
            if ema is not None:
                ema.update_parameters(model)
            tl += loss.item() * len(bi); tc += (out["logits"].argmax(1) == y).sum().item(); n += len(bi)
        evm = ema.module if ema is not None else model
        vp, _ = predict(evm, gd, va)
        yv = gd.lab[torch.as_tensor(va, device=device)].cpu().numpy()
        vloss = float(F.nll_loss(torch.log(torch.from_numpy(vp).clamp_min(1e-8)), torch.from_numpy(yv)))
        vacc = float((vp.argmax(1) == yv).mean())
        h = dict(epoch=ep + 1, train_loss=tl / n, train_acc=tc / n, val_loss=vloss, val_acc=vacc, time=time.time() - t0)
        hist.append(h)
        log(f"ep {ep + 1:02d} loss {h['train_loss']:.4f} acc {h['train_acc']:.4f} | val loss {vloss:.4f} acc {vacc:.4f} | {h['time']:.1f}s")
        key = (-vloss,) if cfg["select"] == "val_loss" else (vacc, -vloss)
        if best is None or key > best:
            best, bad = key, 0
            torch.save(evm.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log("early stopping"); break
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model, hist
