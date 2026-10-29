"""Attention modules.

Baselines compared in Hasan et al. (BSPC 2026, 108636): Coordinate, GCNet, SPNet (strip pooling),
SRMNet (style recalibration) and their proposed SSPANet. SE and CBAM are added because the paper
discusses them theoretically but never evaluates them.

Proposed: TGSA (Tumour-Guided Strip-Style Attention), used by TG-SSPANet (see models.py).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _std(x, dims, eps=1e-5):
    x = x.float()
    return torch.sqrt(x.var(dim=dims, keepdim=True, unbiased=False) + eps)


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean((2, 3)))
        return x * w[:, :, None, None]


class CBAM(nn.Module):
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c))
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, k, padding=k // 2, bias=False), nn.BatchNorm2d(1))

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3))))
        x = x * ca[:, :, None, None]
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class CoordAtt(nn.Module):
    """Hou et al., CVPR 2021."""

    def __init__(self, c, r=32):
        super().__init__()
        m = max(8, c // r)
        self.conv1 = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.BatchNorm2d(m), nn.Hardswish())
        self.conv_h = nn.Conv2d(m, c, 1)
        self.conv_w = nn.Conv2d(m, c, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        xh = x.mean(3, keepdim=True)                      # B,C,H,1
        xw = x.mean(2, keepdim=True).permute(0, 1, 3, 2)  # B,C,W,1
        y = self.conv1(torch.cat([xh, xw], 2))
        yh, yw = torch.split(y, [h, w], 2)
        ah = torch.sigmoid(self.conv_h(yh))
        aw = torch.sigmoid(self.conv_w(yw.permute(0, 1, 3, 2)))
        return x * ah * aw


class GCBlock(nn.Module):
    """Cao et al., GCNet 2019 (attention pooling + bottleneck transform + add)."""

    def __init__(self, c, r=16):
        super().__init__()
        self.key = nn.Conv2d(c, 1, 1)
        self.transform = nn.Sequential(nn.Conv2d(c, c // r, 1), nn.LayerNorm([c // r, 1, 1]),
                                       nn.ReLU(inplace=True), nn.Conv2d(c // r, c, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        attn = torch.softmax(self.key(x).view(b, 1, h * w), -1)          # B,1,HW
        ctx = torch.bmm(x.view(b, c, h * w), attn.transpose(1, 2))        # B,C,1
        return x + self.transform(ctx.view(b, c, 1, 1))


class StripPooling(nn.Module):
    """Strip Pooling Module of SPNet (Hou et al., CVPR 2020)."""

    def __init__(self, c, r=4):
        super().__init__()
        m = c // r
        self.reduce = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(inplace=True))
        self.conv_h = nn.Sequential(nn.Conv2d(m, m, (1, 3), padding=(0, 1), bias=False), nn.BatchNorm2d(m))
        self.conv_v = nn.Sequential(nn.Conv2d(m, m, (3, 1), padding=(1, 0), bias=False), nn.BatchNorm2d(m))
        self.fuse = nn.Conv2d(m, c, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        y = self.reduce(x)
        yh = self.conv_h(F.adaptive_avg_pool2d(y, (1, w))).expand(-1, -1, h, w)
        yv = self.conv_v(F.adaptive_avg_pool2d(y, (h, 1))).expand(-1, -1, h, w)
        return x * torch.sigmoid(self.fuse(F.relu(yh + yv)))


class SRM(nn.Module):
    """Style-based Recalibration Module (Lee et al., ICCV 2019)."""

    def __init__(self, c):
        super().__init__()
        self.cfc = nn.Conv1d(c, c, 2, groups=c, bias=False)
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        b, c = x.shape[:2]
        mu = x.float().mean((2, 3))
        sd = _std(x, (2, 3)).view(b, c)
        g = torch.sigmoid(self.bn(self.cfc(torch.stack([mu, sd], -1).to(x.dtype)).squeeze(-1)))
        return x * g[:, :, None, None]


class SSPANet(nn.Module):
    """Re-implementation of the paper's SSPANet (Sec. 3.5, Eqs. 17-23).

    Channel: Z-pool (avg ; max over H,W) -> conv -> BN -> sigmoid  (M_c)
    Spatial: strip pooling (H and W) -> 1x1 conv  +  style pooling (mean, std over C) -> conv,
             summed -> 1x1 conv -> sigmoid (M_s);  Y = M_s * (M_c * X) with residual fusion.
    """

    def __init__(self, c, r=16, k=3):
        super().__init__()
        m = max(8, c // r)
        self.ch_conv = nn.Conv1d(2, 1, k, padding=k // 2, bias=False)
        self.ch_bn = nn.BatchNorm1d(1)
        self.strip_h = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.BatchNorm2d(m))
        self.strip_v = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.BatchNorm2d(m))
        self.style = nn.Sequential(nn.Conv2d(2, m, 3, padding=1, bias=False), nn.BatchNorm2d(m))
        self.out = nn.Conv2d(m, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        z = torch.stack([x.mean((2, 3)), x.amax((2, 3))], 1)              # B,2,C
        mc = torch.sigmoid(self.ch_bn(self.ch_conv(z))).view(b, c, 1, 1)
        x1 = x * mc
        sh = self.strip_h(x1.mean(2, keepdim=True)).expand(-1, -1, h, w)    # pooled over H
        sv = self.strip_v(x1.mean(3, keepdim=True)).expand(-1, -1, h, w)    # pooled over W
        style = torch.cat([x1.float().mean(1, keepdim=True), _std(x1, 1)], 1).to(x.dtype)
        ms = torch.sigmoid(self.out(F.relu(sh + sv + self.style(style))))
        return x + ms * x1


# --------------------------------------------------------------------------------------------
# Proposed module
# --------------------------------------------------------------------------------------------
class TGSA(nn.Module):
    """Tumour-Guided Strip-Style Attention (proposed).

    Differences from SSPANet (each addresses a limitation identified in the paper):
      * channel branch uses tri-statistic pooling (avg, max, std) with an ECA-style adaptive 1-D
        kernel -> second-order channel statistics that SSPANet only uses spatially;
      * strip branch pools each strip with avg AND max and models dependencies *along* the strip
        with depthwise 1-D convolutions (SSPANet only applies 1x1 convs to the strips);
      * style branch uses mean/std/max over channels with a dilated 7x7 receptive field;
      * the spatial logit map is returned so it can be supervised by the tumour mask
        (attention-alignment loss) -- the paper's attention is never checked against masks;
      * zero-initialised residual gate (gamma) so the module can be inserted at several stages
        of an ImageNet-pretrained backbone without destroying pretrained features.
    Toggles (use_channel / use_strip / use_style) are used for the ablation study.
    """

    def __init__(self, c, r=16, use_channel=True, use_strip=True, use_style=True):
        super().__init__()
        self.use_channel, self.use_strip, self.use_style = use_channel, use_strip, use_style
        k = int(abs((math.log2(c) + 1) / 2)); k = k if k % 2 else k + 1
        self.ch_conv = nn.Conv1d(3, 1, k, padding=k // 2, bias=False)
        self.ch_bn = nn.BatchNorm1d(1)
        m = max(16, c // r)
        self.reduce = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(inplace=True))
        self.strip_h = nn.Sequential(nn.Conv1d(2 * m, m, 7, padding=3, groups=m, bias=False), nn.BatchNorm1d(m))
        self.strip_v = nn.Sequential(nn.Conv1d(2 * m, m, 7, padding=3, groups=m, bias=False), nn.BatchNorm1d(m))
        self.style = nn.Sequential(nn.Conv2d(3, m, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(m),
                                   nn.ReLU(inplace=True), nn.Conv2d(m, m, 3, padding=1, bias=False), nn.BatchNorm2d(m))
        self.out = nn.Conv2d(m, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_logits = None

    def forward(self, x):
        b, c, h, w = x.shape
        if self.use_channel:
            z = torch.stack([x.float().mean((2, 3)), x.float().amax((2, 3)), _std(x, (2, 3)).view(b, c)], 1)
            mc = torch.sigmoid(self.ch_bn(self.ch_conv(z.to(x.dtype)))).view(b, c, 1, 1)
            x1 = x * mc
        else:
            x1 = x
        s = 0
        if self.use_strip:
            y = self.reduce(x1)
            row = torch.cat([y.mean(3), y.amax(3)], 1)          # B,2m,H  (pool along W)
            col = torch.cat([y.mean(2), y.amax(2)], 1)          # B,2m,W  (pool along H)
            # interleave avg/max per channel for grouped conv
            row = row.view(b, 2, -1, h).transpose(1, 2).reshape(b, -1, h)
            col = col.view(b, 2, -1, w).transpose(1, 2).reshape(b, -1, w)
            s = s + self.strip_h(row)[..., None].expand(-1, -1, h, w) + self.strip_v(col)[:, :, None, :].expand(-1, -1, h, w)
        if self.use_style:
            st = torch.cat([x1.float().mean(1, keepdim=True), _std(x1, 1), x1.float().amax(1, keepdim=True)], 1)
            s = s + self.style(st.to(x.dtype))
        if not (self.use_strip or self.use_style):
            self.last_logits = None
            return x + self.gamma * (x1 - x) if self.use_channel else x
        logits = self.out(F.relu(s))
        self.last_logits = logits
        return x + self.gamma * (torch.sigmoid(logits) * x1)


BASELINE_ATTN = {
    "none": None, "se": SE, "cbam": CBAM, "coordinate": CoordAtt, "gcnet": GCBlock,
    "spnet": StripPooling, "srmnet": SRM, "sspanet": SSPANet, "tgsa": TGSA,
}
