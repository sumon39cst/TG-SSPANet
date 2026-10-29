"""Extract real data / model internals used by the schematic figures (07_paper_figures.py).

Saves E:/Life_Nero/result/figures/diagrams_v2/assets.npz with, for one P1 test slice per class:
input image, GT mask, TG-SSPANet class probabilities, decoder mask probability (224 and 56),
TFP tumour probability p (7x7), TGSA spatial attention sigma(A_s) at stages 2-4, learned gammas,
Grad-CAM++ of TG-SSPANet (on F'') and of the reproduced SSPANet, plus augmented views of one slice.
"""
import importlib
import os

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from xai import CAMExtractor

OUT = os.path.join(C.RESULT, "figures", "diagrams_v2")
os.makedirs(OUT, exist_ok=True)
xe = importlib.import_module("03_xai_eval")


def main():
    dev = torch.device("cuda")
    data = C.load_data(); gd = C.GPUData(data, dev)
    _, _, te = C.split_patient(data, 1)                      # held-out patients of P2 fold 1
    ex = xe.pick_examples(data, te, {})                      # glioma, meningioma, pituitary (same as XAI figures)
    ours, _, _ = xe.load_run("patient", "resnet50_TG-SSPANet", "patient_fold1_seed0", dev)
    base, _, _ = xe.load_run("patient", "resnet50_sspanet_ourrecipe", "patient_fold1_seed0", dev)   # same recipe
    x, m, y = gd.batch(np.array(ex))
    img = F.interpolate(torch.from_numpy(data["images"][ex]).float()[:, None], size=(224, 224), mode="bilinear",
                        antialias=True, align_corners=False)[:, 0].numpy() / 255.0
    gt = m[:, 0].cpu().numpy() > 0.5

    grabbed = {}
    ours.gate.register_forward_hook(lambda mod, i, o: grabbed.__setitem__("p", i[1].detach()))
    with torch.no_grad():
        o = ours(x)
        prob = torch.softmax(o["logits"].float(), 1).cpu().numpy()
        mask224 = torch.sigmoid(o["mask_logits"].float())[:, 0].cpu().numpy()
        mask56 = F.interpolate(torch.sigmoid(o["mask_logits"].float()), size=(56, 56), mode="area")[:, 0].cpu().numpy()
        att = [torch.sigmoid(t.float())[:, 0].cpu().numpy() for t in o["attn_logits"]]   # stages 2,3,4
        p = grabbed["p"].float()[:, 0].cpu().numpy()
        bprob = torch.softmax(base(x)["logits"].float(), 1).cpu().numpy()
    gammas = np.array([float(ours.tgsa[i].gamma) for i in (1, 2, 3)])
    cam_o, _ = CAMExtractor(ours, ours.cam_layer)(x)
    cam_b, _ = CAMExtractor(base, base.cam_layer)(x)

    torch.manual_seed(7)
    xa, ma, _ = gd.batch(np.array([ex[0]] * 4), augment=True, strong=True)
    aug = (xa[:, 0] * C.STD[0, 0].item() + C.MEAN[0, 0].item()).clamp(0, 1).cpu().numpy()
    aug_m = ma[:, 0].cpu().numpy() > 0.5

    np.savez_compressed(os.path.join(OUT, "assets.npz"), ids=data["ids"][ex], labels=y.cpu().numpy(), img=img, gt=gt,
                        prob=prob, bprob=bprob, mask224=mask224, mask56=mask56, p=p,
                        att2=att[0], att3=att[1], att4=att[2], gammas=gammas,
                        cam_ours=cam_o["GradCAM++"], cam_base=cam_b["GradCAM++"], aug=aug, aug_m=aug_m)
    print("examples", data["ids"][ex], "labels", y.tolist(), "prob", prob.round(3).tolist(), "gammas", gammas.round(3))


if __name__ == "__main__":
    main()
