"""Grad-CAM, Grad-CAM++ and EigenGrad-CAM + quantitative localisation metrics against tumorMask."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation

METHODS = ["GradCAM", "GradCAM++", "EigenGradCAM"]


class CAMExtractor:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self.h = layer.register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        self.acts = o
        o.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        self.h.remove()

    def __call__(self, x, target=None):
        """Returns dict method -> (B,H,W) maps in [0,1], plus logits."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            out = self.model(x)
            logits = out["logits"].float()
            tgt = logits.argmax(1) if target is None else target
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, tgt[:, None]).sum().backward()
        A, G = self.acts.detach().float(), self.grads.detach().float()
        maps = {}
        maps["GradCAM"] = F.relu((G.mean((2, 3), keepdim=True) * A).sum(1))
        g2, g3 = G ** 2, G ** 3
        alpha = g2 / (2 * g2 + (A.sum((2, 3), keepdim=True) * g3) + 1e-7)
        alpha = torch.where(G != 0, alpha, torch.zeros_like(alpha))
        w = (alpha * F.relu(G)).sum((2, 3), keepdim=True)
        maps["GradCAM++"] = F.relu((w * A).sum(1))
        b, k, h, wd = A.shape
        M = (A * G).view(b, k, h * wd).transpose(1, 2)                    # B,HW,K
        M = M - M.mean(1, keepdim=True)
        _, _, V = torch.linalg.svd(M, full_matrices=False)
        proj = (M @ V[:, 0, :, None]).squeeze(-1)                          # B,HW
        ref = (A * G).sum(1).view(b, -1)                                   # orient with gradient-weighted sum
        sign = torch.sign((proj * ref).sum(1, keepdim=True)); sign[sign == 0] = 1
        maps["EigenGradCAM"] = F.relu(proj * sign).view(b, h, wd)
        res = {}
        for k_, m in maps.items():
            m = F.interpolate(m[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            mn = m.flatten(1).min(1)[0][:, None, None]; mx = m.flatten(1).max(1)[0][:, None, None]
            res[k_] = ((m - mn) / (mx - mn + 1e-7)).cpu().numpy()
        return res, logits.detach().cpu()


def localisation_metrics(cam, gt, tol=7):
    """cam (H,W) in [0,1]; gt bool (H,W). Pointing game with tolerance, energy-based pointing game, IoU@0.5."""
    if gt.sum() == 0:
        return None
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    # note: scipy dilates until convergence when iterations=0, so tolerance 0 must use the mask directly
    hit = bool((binary_dilation(gt, iterations=tol) if tol > 0 else gt)[y, x])
    energy = float((cam * gt).sum() / (cam.sum() + 1e-7))
    b = cam >= 0.5
    iou = float((b & gt).sum() / ((b | gt).sum() + 1e-7))
    return hit, energy, iou
