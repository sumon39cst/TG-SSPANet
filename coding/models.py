"""Backbones, the paper's baseline configuration and the proposed TG-SSPANet."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from attention import BASELINE_ATTN, TGSA

NUM_CLASSES = 3


class Backbone(nn.Module):
    """Returns 4 feature stages (strides 4, 8, 16, 32) of an ImageNet-pretrained CNN."""

    def __init__(self, name, pretrained=True):
        super().__init__()
        self.name = name
        if name == "resnet50":
            m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            self.stages = nn.ModuleList([
                nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1),
                m.layer2, m.layer3, m.layer4])
            self.channels = [256, 512, 1024, 2048]
        elif name == "vgg16":
            f = torchvision.models.vgg16(weights="IMAGENET1K_V1" if pretrained else None).features
            self.stages = nn.ModuleList([f[:10], f[10:17], f[17:24], f[24:31]])
            self.channels = [128, 256, 512, 512]
        else:
            raise ValueError(name)

    def forward(self, x, hooks=None):
        feats = []
        for i, s in enumerate(self.stages):
            x = s(x)
            if hooks is not None and hooks[i] is not None:
                x = hooks[i](x)
            feats.append(x)
        return feats


class BaselineNet(nn.Module):
    """Configuration evaluated in the paper: backbone -> attention module -> GAP -> FC."""

    def __init__(self, backbone="resnet50", attn="none", pretrained=True):
        super().__init__()
        self.backbone = Backbone(backbone, pretrained)
        c = self.backbone.channels[-1]
        cls = BASELINE_ATTN[attn]
        self.attn = cls(c) if cls is not None else nn.Identity()
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(c, NUM_CLASSES)
        self.cam_layer = self.attn  # Grad-CAM target: attention-refined last feature map

    def forward(self, x):
        f = self.attn(self.backbone(x)[-1])
        return {"logits": self.fc(self.drop(f.mean((2, 3))))}


class FPNDecoder(nn.Module):
    """Light top-down decoder predicting the tumour mask at stride 4."""

    def __init__(self, channels, d=128):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, d, 1) for c in channels])
        self.smooth = nn.Sequential(nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True),
                                    nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(d, 1, 1)

    def forward(self, feats):
        y = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            y = F.interpolate(y, size=feats[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](feats[i])
        return self.head(self.smooth(y))


class TumourGate(nn.Module):
    """Tumour-focused pooling gate: F'' = F * (floor + (1 - floor) * p), p = predicted tumour probability at
    feature resolution. Evidence is pooled mainly inside the predicted tumour (context still reaches it through
    the receptive field of stage-4 features). Followed by GAP + FC, so Grad-CAM on F'' is exact."""

    def __init__(self, floor=0.1):
        super().__init__()
        self.floor = floor

    def forward(self, f, p):
        return f * (self.floor + (1 - self.floor) * p)


class TGSSPANet(nn.Module):
    """Proposed Tumour-Guided Strip-Style Pooling Attention Network.

    * TGSA modules inserted after stages 2, 3 and 4 (multi-scale attention; the paper uses one module)
    * FPN decoder -> tumour-mask prediction (multi-task; uses the masks the paper said did not exist)
    * attention-alignment: TGSA spatial logits are supervised with the down-sampled mask
    * Tumour-focused pooling (TFP): the predicted mask gates the final features before GAP (TumourGate)
    Flags allow every component to be switched off for the ablation study.
    """

    def __init__(self, backbone="resnet50", pretrained=True, stages=(1, 2, 3), seg=True, tfp=True,
                 use_channel=True, use_strip=True, use_style=True, floor=0.1, oracle=False):
        super().__init__()
        self.needs_mask = oracle  # oracle upper bound: gate with the ground-truth mask instead of the prediction
        self.backbone = Backbone(backbone, pretrained)
        ch = self.backbone.channels
        self.tgsa = nn.ModuleList([TGSA(ch[i], use_channel=use_channel, use_strip=use_strip, use_style=use_style)
                                   if i in stages else nn.Identity() for i in range(4)])
        self.seg, self.tfp = seg, tfp and seg
        if seg:
            self.decoder = FPNDecoder(ch)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(ch[-1], NUM_CLASSES)
        if self.tfp:
            self.gate = TumourGate(floor)
            self.cam_layer = self.gate
        else:
            self.cam_layer = self.tgsa[3] if 3 in stages else self.backbone.stages[3]

    def forward(self, x, mask=None):
        feats = self.backbone(x, hooks=list(self.tgsa))
        f = feats[-1]
        out = {"attn_logits": [m.last_logits for m in self.tgsa if isinstance(m, TGSA) and m.last_logits is not None]}
        if self.seg:
            mask_logits = self.decoder(feats)
            out["mask_logits"] = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if self.tfp:
                # probability that each stage-4 cell contains tumour (pool probabilities, not logits);
                # 56x56 -> 7x7 is an 8x8 window with stride 8
                src = mask.float() if self.needs_mask else torch.sigmoid(mask_logits.float())
                p = F.adaptive_max_pool2d(src, f.shape[-2:]).to(f.dtype)
                f = self.gate(f, p)
        out["logits"] = self.fc(self.drop(f.mean((2, 3))))
        return out


def build_model(kind, backbone, attn="none", pretrained=True, **kw):
    if kind == "baseline":
        return BaselineNet(backbone, attn, pretrained)
    if kind == "proposed":
        return TGSSPANet(backbone, pretrained, **kw)
    raise ValueError(kind)
