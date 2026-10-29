"""Pipeline figure + architecture figures (overall TG-SSPANet, TGSA module, mask-guided decoder / TFP head).
Saved to E:/Life_Nero/result/figures/diagrams as PNG (300 dpi), PDF and SVG."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = r"E:\Life_Nero\result\figures\diagrams"
# DIAGRAM_MODE=paper: no in-image titles, cross-references renumbered to the manuscript figures, PNGs to paper/figures
PAPER = os.environ.get("DIAGRAM_MODE") == "paper"
PAPER_OUT = r"E:\Life_Nero\paper\figures"
PAPER_NAMES = {"fig_pipeline": "fig01_pipeline", "fig_architecture_A_overall": "fig02_architecture_overall",
               "fig_architecture_B_TGSA_module": "fig03_tgsa_module", "fig_architecture_C_decoder_TFP_head": "fig04_decoder_tfp"}


def T(s):
    """Figure title text (suppressed in paper mode, where the caption carries the title)."""
    return "" if PAPER else s
ASPECT = 1.0
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

COL = {
    "data": "#DCEAF7", "prep": "#E3F1E0", "model": "#FCE8D5", "prop": "#FAD4D4", "train": "#EDE3F5",
    "eval": "#FFF4CC", "xai": "#D9F2F2", "att": "#F7C9A9", "seg": "#CDE8C4", "head": "#D6D6F5", "op": "#FFFFFF",
    "loss": "#FFE0E0", "edge": "#3A3A3A",
}


def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False, ec=None, lw=1.0, r=0.02, ls="-"):
    if PAPER:
        text = text.replace("(Fig. B)", "(Fig. 3)").replace("(Fig. C)", "(Fig. 4)")
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}", fc=fc, ec=ec or COL["edge"], lw=lw,
                       ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            wrap=True, linespacing=1.25)
    return p


def arrow(ax, p0, p1, color="#333333", lw=1.2, style="-|>", ls="-", rad=0.0, text=None, fs=7.5, toff=(0, 0.012)):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=10, color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1)
    ax.add_patch(a)
    if text:
        ax.text((p0[0] + p1[0]) / 2 + toff[0], (p0[1] + p1[1]) / 2 + toff[1], text, ha="center", va="bottom", fontsize=fs,
                color=color, style="italic")


def circle(ax, x, y, s, r=0.011, fc="white"):
    c = matplotlib.patches.Ellipse((x, y), 2 * r, 2 * r * ASPECT, fc=fc, ec=COL["edge"], lw=1.0, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, s, ha="center", va="center", fontsize=10, zorder=6)


def save(fig, name):
    if PAPER:
        fig.savefig(os.path.join(PAPER_OUT, PAPER_NAMES[name] + ".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------------------------------------------------------
def pipeline():
    fig, ax = canvas(15, 7.2)
    ax.text(0.5, 0.965, T("Experimental pipeline: paper reproduction + research-gap-driven TG-SSPANet"), ha="center",
            fontsize=14, fontweight="bold")
    # row 1
    y1, h1 = 0.62, 0.26
    blocks = [
        (0.015, "① Dataset\n\nFigshare brain tumour\n(Cheng et al.)\n3,064 T1-CE MRI slices\n233 patients · 3 classes\nimage + label + PID\n+ tumorMask", COL["data"]),
        (0.215, "② Preprocessing\n\n.mat (HDF5) → array\nmin–max intensity norm.\nresize 512 → 256 → 224\ngray → 3-ch, ImageNet norm.\nmask co-registered", COL["prep"]),
        (0.415, "③ Data protocols\n\nP1 (paper): stratified\nimage-level 80/10/10\n2450 / 307 / 307\nP2 (gap): official\npatient-wise 5-fold CV\n(no patient leakage)", COL["prep"]),
        (0.615, "④ GPU augmentation\n\nh-flip · rotate ±15°\nscale ±10% · shift ±6%\ncontrast/brightness\n(masks transformed\njointly with images)", COL["prep"]),
        (0.815, "⑤ Models\n\nBaselines (paper):\nVGG16 / ResNet50 +\nCoord · GCNet · SPNet ·\nSRMNet · SSPANet\n(+ SE, CBAM)\nProposed: TG-SSPANet", COL["model"]),
    ]
    for x, t, c in blocks:
        box(ax, x, y1, 0.17, h1, t, c, fs=8.6)
    for x, _, _ in blocks[:-1]:
        arrow(ax, (x + 0.172, y1 + h1 / 2), (x + 0.213, y1 + h1 / 2), lw=1.6)
    # row 2
    y2, h2 = 0.2, 0.3
    b2 = [
        (0.815, "⑥ Training\n\nL = L_CE(label-smooth,\nclass-weighted)\n+ λ_seg (BCE + Dice)\n+ λ_att · L_align\nAdamW · cosine · warm-up\nEMA · AMP · 40 epochs", COL["train"]),
        (0.615, "⑦ Classification eval.\n\nAcc · P · R · F1\nCohen κ · MCC · AUC\nper-class recall\nconfusion matrix · ROC\nmean ± sd (3 seeds)\npaired t-test · McNemar", COL["eval"]),
        (0.415, "⑧ Explainability\n\nGrad-CAM · Grad-CAM++\nEigenGrad-CAM\nquantitative vs. GT mask:\npointing game · energy-\nin-mask · CAM-IoU\n+ predicted tumour mask", COL["xai"]),
        (0.215, "⑨ Robustness & cost\n\npatient-wise 5-fold CV\ncomponent ablation\n(channel/strip/style,\nstages, L_align, TFP)\nparams · GMACs · latency", COL["eval"]),
        (0.015, "⑩ Outputs\n\nresult/: metrics, tables,\ncurves, figures\nprediction/: per-image\nclass probabilities +\npredicted masks\ncoding/: all scripts", COL["data"]),
    ]
    for x, t, c in b2:
        box(ax, x, y2, 0.17, h2, t, c, fs=8.6)
    arrow(ax, (0.9, y1 - 0.005), (0.9, y2 + h2 + 0.005), lw=1.6)
    for x, _, _ in b2[:-1]:
        arrow(ax, (x - 0.002, y2 + h2 / 2), (x - 0.043, y2 + h2 / 2), lw=1.6)
    # research gaps banner
    box(ax, 0.015, 0.035, 0.97, 0.11,
        "Research gaps addressed:  G1 masks exist but were unused → mask-supervised attention + quantitative XAI   ·   G2 image-level split leaks patients → patient-wise CV\n"
        "G3 single-stage attention, no ablation → multi-stage TGSA + full ablation   ·   "
        "G4 imbalance not handled → class-weighted, label-smoothed loss\nG5 SE/CBAM never tested → added   ·   "
        "G6 no measured cost / variance → params, GMACs, latency, mean ± sd, significance tests",
        "#F4F4F4", fs=8.8, ls="--")
    save(fig, "fig_pipeline")


# ----------------------------------------------------------------------------------------------------------------
def overall():
    fig, ax = canvas(16, 7.6)
    ax.text(0.5, 0.965, T("TG-SSPANet: Tumour-Guided Strip-Style Pooling Attention Network (overall architecture)"),
            ha="center", fontsize=14, fontweight="bold")
    yb, hb = 0.58, 0.2
    box(ax, 0.01, yb, 0.075, hb, "Input MRI\n224×224×3\n(T1-CE)", COL["data"], fs=8.5)
    xs = [0.11, 0.27, 0.46, 0.66]
    names = ["Stage 1\nconv1+pool\n+layer1\n256 @ 56×56", "Stage 2\nlayer2\n512 @ 28×28", "Stage 3\nlayer3\n1024 @ 14×14",
             "Stage 4\nlayer4\n2048 @ 7×7"]
    for i, (x, n) in enumerate(zip(xs, names)):
        box(ax, x, yb, 0.085, hb, n, "#E6E6E6", fs=8.2)
    tg = [0.2, 0.555, 0.755]
    tg = [xs[1] + 0.095, xs[2] + 0.095, xs[3] + 0.095]
    for x in tg:
        box(ax, x, yb + 0.03, 0.07, hb - 0.06, "TGSA\n(Fig. B)", COL["att"], fs=8.3, bold=True)
    arrow(ax, (0.086, yb + hb / 2), (xs[0], yb + hb / 2))
    arrow(ax, (xs[0] + 0.086, yb + hb / 2), (xs[1], yb + hb / 2))
    for i in range(3):
        arrow(ax, (xs[i + 1] + 0.086, yb + hb / 2), (tg[i], yb + hb / 2))
        if i < 2:
            arrow(ax, (tg[i] + 0.071, yb + hb / 2), (xs[i + 2], yb + hb / 2))
    ax.text(0.47, yb + hb + 0.035, "ImageNet-pre-trained ResNet50 (or VGG16) backbone with TGSA inserted after stages 2, 3, 4 "
            "(zero-initialised residual gate γ)", ha="center", fontsize=9, style="italic")
    # decoder
    yd, hd = 0.28, 0.14
    box(ax, 0.2, yd, 0.42, hd, "Mask-guided FPN decoder (Fig. C)\nlateral 1×1 (d=128) · top-down upsample + add · 2×[3×3 conv-BN-ReLU] · 1×1 → mask logits (56×56)",
        COL["seg"], fs=8.6)
    for x in [xs[0] + 0.04, tg[0] + 0.035, tg[1] + 0.035, tg[2] + 0.035]:
        arrow(ax, (x, yb - 0.005), (min(max(x, 0.21), 0.61), yd + hd + 0.003), color="#2E7D32", lw=1.0)
    box(ax, 0.2, 0.07, 0.19, 0.12, "Predicted tumour mask\n(upsampled 224×224)", COL["seg"], fs=8.5)
    arrow(ax, (0.3, yd - 0.003), (0.3, 0.19), color="#2E7D32")
    box(ax, 0.43, 0.07, 0.19, 0.12, "L_seg = BCE + Dice\n(vs. tumorMask)", COL["loss"], fs=8.5)
    arrow(ax, (0.39, 0.13), (0.43, 0.13), color="#C62828")
    # head
    xh = 0.86
    box(ax, xh, 0.6, 0.12, 0.16, "Tumour-focused\npooling (TFP) gate\nF″ = F4′ ⊙ (0.1 + 0.9·p)\n2048 @ 7×7", COL["head"], fs=8.4)
    arrow(ax, (tg[2] + 0.071, yb + hb / 2), (xh, yb + hb / 2))
    arrow(ax, (0.62, yd + hd / 2), (xh, 0.63), color="#2E7D32", rad=0.0)
    ax.text(0.745, 0.5, "p = maxpool 7×7 (σ(mask logits)) → TFP", ha="center", fontsize=7.8, color="#2E7D32", style="italic")
    box(ax, xh, 0.25, 0.12, 0.14, "GAP → 2048-d\nDropout 0.3\nFC → Softmax\n(Grad-CAM target: F″)", COL["head"], fs=8.2)
    arrow(ax, (xh + 0.06, 0.6), (xh + 0.06, 0.392))
    box(ax, xh, 0.07, 0.12, 0.12, "Meningioma /\nGlioma / Pituitary\nL_cls = CE (LS 0.1,\nclass-weighted)", COL["loss"], fs=8.0)
    arrow(ax, (xh + 0.06, 0.25), (xh + 0.06, 0.19))
    # attention alignment loss
    box(ax, 0.66, 0.16, 0.15, 0.09, "L_align = BCE(TGSA spatial\nlogits, pooled mask)", COL["loss"], fs=8.2)
    for x in tg:
        arrow(ax, (x + 0.035, yb + 0.03), (0.735, 0.25), color="#C62828", ls="--", lw=0.9)
    ax.text(0.735, 0.1, "L = L_cls + 1.0·L_seg + 0.2·L_align", ha="center", fontsize=10, fontweight="bold", color="#8B0000")
    ax.set_ylim(0.03, 0.9)
    ax.texts[0].set_position((0.5, 0.87))
    save(fig, "fig_architecture_A_overall")


# ----------------------------------------------------------------------------------------------------------------
def tgsa_module():
    fig, ax = canvas(18, 8)
    asp = 18 / 8

    def node(x, y, s):
        e = matplotlib.patches.Ellipse((x, y), 0.022, 0.022 * asp, fc="white", ec=COL["edge"], lw=1.0, zorder=5)
        ax.add_patch(e)
        ax.text(x, y, s, ha="center", va="center", fontsize=11, zorder=6)

    def poly(pts, color="#333333", ls="-", lw=1.2):
        xs, ys = zip(*pts[:-1])
        ax.plot(list(xs) + [pts[-1][0]], list(ys) + [pts[-1][1]], color=color, ls=ls, lw=lw, zorder=1)
        arrow(ax, pts[-2], pts[-1], color=color, ls=ls, lw=lw)

    ax.text(0.5, 0.975, T("Fig. B - TGSA: Tumour-Guided Strip-Style Attention module (proposed sub-module 1)"), ha="center",
            fontsize=14, fontweight="bold")
    ym = 0.55
    box(ax, 0.01, ym - 0.06, 0.06, 0.12, "Input X\nC×H×W", COL["data"], fs=9.5)
    # ---------------- channel branch (top)
    ax.text(0.245, 0.905, "Channel branch: tri-statistic Z-pool", ha="center", fontsize=10.5, fontweight="bold", color="#8A4B08")
    for i, t in enumerate(["AvgPool (H,W)", "MaxPool (H,W)", "StdPool (H,W)"]):
        yy = 0.83 - i * 0.065
        box(ax, 0.09, yy - 0.025, 0.08, 0.05, t, COL["att"], fs=8.2)
        poly([(0.07, ym + 0.03), (0.08, ym + 0.03), (0.08, yy), (0.09, yy)], lw=0.9)
        arrow(ax, (0.17, yy), (0.19, 0.765), lw=0.9)
    box(ax, 0.19, 0.725, 0.045, 0.08, "Stack\n3×C", COL["op"], fs=8.5)
    box(ax, 0.255, 0.725, 0.075, 0.08, "Conv1D (adaptive\nk, ECA rule) + BN", COL["att"], fs=8.2)
    arrow(ax, (0.235, 0.765), (0.255, 0.765))
    box(ax, 0.35, 0.725, 0.05, 0.08, "σ → M_c\nC×1×1", COL["att"], fs=8.5)
    arrow(ax, (0.33, 0.765), (0.35, 0.765))
    # ---------------- main path
    node(0.4, ym, "⊗")
    arrow(ax, (0.07, ym), (0.389, ym))
    poly([(0.375, 0.725), (0.375, 0.64), (0.4, 0.64), (0.4, ym + 0.026)])
    ax.text(0.455, ym + 0.02, "X′ = M_c ⊗ X", fontsize=9)
    node(0.8, ym, "⊗")
    arrow(ax, (0.411, ym), (0.789, ym))
    box(ax, 0.825, ym - 0.035, 0.045, 0.07, "× γ\n(init 0)", COL["op"], fs=8.5)
    arrow(ax, (0.811, ym), (0.825, ym))
    node(0.895, ym, "⊕")
    arrow(ax, (0.87, ym), (0.884, ym))
    box(ax, 0.92, ym - 0.06, 0.07, 0.12, "Output Y\nC×H×W", COL["data"], fs=9.5)
    arrow(ax, (0.906, ym), (0.92, ym))
    poly([(0.04, ym + 0.06), (0.04, 0.935), (0.895, 0.935), (0.895, ym + 0.026)], lw=1.0)
    ax.text(0.65, 0.943, "identity (residual) path", ha="center", fontsize=8.5, style="italic")
    # ---------------- spatial branch (bottom)
    ax.text(0.6, 0.49, "Spatial branch: directional strips + style statistics", ha="center", fontsize=10.5,
            fontweight="bold", color="#1F5F8B")
    xb = 0.43
    box(ax, 0.42, 0.34, 0.07, 0.08, "1×1 conv\nC→m, BN, ReLU", COL["xai"], fs=8)
    poly([(xb, ym), (xb, 0.46), (0.455, 0.46), (0.455, 0.42)], lw=1.0)
    box(ax, 0.51, 0.39, 0.085, 0.055, "H-strip: avg‖max\nover W → 2m×H", COL["xai"], fs=7.9)
    box(ax, 0.51, 0.31, 0.085, 0.055, "W-strip: avg‖max\nover H → 2m×W", COL["xai"], fs=7.9)
    arrow(ax, (0.49, 0.39), (0.51, 0.4175), lw=0.9); arrow(ax, (0.49, 0.37), (0.51, 0.3375), lw=0.9)
    box(ax, 0.61, 0.39, 0.09, 0.055, "DW-Conv1D k=7, BN\nexpand → m×H×W", COL["xai"], fs=7.9)
    box(ax, 0.61, 0.31, 0.09, 0.055, "DW-Conv1D k=7, BN\nexpand → m×H×W", COL["xai"], fs=7.9)
    arrow(ax, (0.595, 0.4175), (0.61, 0.4175)); arrow(ax, (0.595, 0.3375), (0.61, 0.3375))
    box(ax, 0.42, 0.15, 0.11, 0.07, "Style pool over C:\nmean ‖ std ‖ max → 3×H×W", COL["xai"], fs=7.9)
    poly([(xb - 0.02, ym), (xb - 0.02, 0.185), (0.42, 0.185)], lw=1.0)
    box(ax, 0.55, 0.15, 0.15, 0.07, "3×3 dilated(2) conv-BN-ReLU\n+ 3×3 conv-BN → m×H×W", COL["xai"], fs=7.9)
    arrow(ax, (0.53, 0.185), (0.55, 0.185))
    node(0.73, 0.3, "⊕")
    arrow(ax, (0.7, 0.4175), (0.722, 0.31), lw=1.0); arrow(ax, (0.7, 0.3375), (0.719, 0.303), lw=1.0)
    arrow(ax, (0.7, 0.185), (0.724, 0.289), lw=1.0)
    box(ax, 0.755, 0.27, 0.09, 0.06, "ReLU → 1×1 conv\n→ logits A_s (1×H×W)", COL["xai"], fs=7.9)
    arrow(ax, (0.741, 0.3), (0.755, 0.3))
    box(ax, 0.775, 0.38, 0.05, 0.05, "σ → M_s", COL["xai"], fs=8.5)
    arrow(ax, (0.8, 0.33), (0.8, 0.38)); arrow(ax, (0.8, 0.43), (0.8, ym - 0.026))
    box(ax, 0.87, 0.25, 0.12, 0.1, "L_align = BCE(A_s,\navg-pooled tumour\nmask)  [training only]", COL["loss"], fs=8)
    arrow(ax, (0.845, 0.3), (0.87, 0.3), color="#C62828", ls="--")
    # ---------------- notes
    ax.text(0.015, 0.25, "Y = X + γ · σ(A_s) ⊗ (M_c ⊗ X)\n\nDifferences from the paper's SSPANet:\n"
            "• std statistic added to the channel Z-pool\n• strips pooled with avg AND max\n"
            "• depthwise 1-D conv along each strip (long-range context)\n• max added to style descriptor; dilated conv\n"
            "• spatial logits supervised by tumorMask (L_align)\n• zero-initialised gate γ → safe insertion at\n  several stages of a pre-trained backbone",
            ha="left", va="center", fontsize=8.8, bbox=dict(boxstyle="round", fc="#FAFAFA", ec="#999999"))
    ax.set_ylim(0.03, 1.0)
    save(fig, "fig_architecture_B_TGSA_module")


# ----------------------------------------------------------------------------------------------------------------
def decoder_head():
    fig, ax = canvas(15, 6.8)
    global ASPECT; ASPECT = 15 / 6.8
    ax.text(0.5, 0.96, T("Fig. C - Mask-guided FPN decoder and Tumour-Focused Pooling head (proposed sub-module 2)"),
            ha="center", fontsize=14, fontweight="bold")
    ys = [0.72, 0.54, 0.36, 0.18]
    names = ["F1  256@56²", "F2′ 512@28²", "F3′ 1024@14²", "F4′ 2048@7²"]
    for y, n in zip(ys, names):
        box(ax, 0.02, y, 0.12, 0.09, n, "#E6E6E6", fs=8.5)
        box(ax, 0.19, y, 0.1, 0.09, "1×1 conv\n→ 128", COL["seg"], fs=8.3)
        arrow(ax, (0.14, y + 0.045), (0.19, y + 0.045))
    for i, y in enumerate(ys[:-1]):
        circle(ax, 0.37, y + 0.045, "⊕")
        arrow(ax, (0.29, y + 0.045), (0.353, y + 0.045))
    arrow(ax, (0.29, ys[3] + 0.045), (0.37, ys[3] + 0.045), style="-")
    for i in range(3, 0, -1):
        arrow(ax, (0.37, ys[i] + 0.045 + (0 if i == 3 else 0.017)), (0.37, ys[i - 1] + 0.028), text="×2 up", toff=(0.03, -0.01))
    box(ax, 0.43, 0.7, 0.14, 0.13, "2 × [3×3 conv\n-BN-ReLU] (128)", COL["seg"], fs=8.5)
    arrow(ax, (0.387, ys[0] + 0.045), (0.43, 0.765))
    box(ax, 0.61, 0.7, 0.1, 0.13, "1×1 conv\n→ mask logits\n1@56²", COL["seg"], fs=8.5)
    arrow(ax, (0.57, 0.765), (0.61, 0.765))
    box(ax, 0.76, 0.76, 0.2, 0.09, "bilinear ↑ 224² → predicted mask\nL_seg = BCE + Dice", COL["loss"], fs=8.3)
    arrow(ax, (0.71, 0.79), (0.76, 0.805))
    # TFP
    ax.text(0.68, 0.62, "Tumour-Focused Pooling (TFP) head", ha="center", fontsize=10, fontweight="bold", color="#3F3F9F")
    box(ax, 0.61, 0.46, 0.14, 0.11, "σ(·) → AdaptiveMaxPool\n7×7 → p (1@7²)", COL["head"], fs=8.3)
    arrow(ax, (0.66, 0.7), (0.66, 0.57))
    box(ax, 0.61, 0.25, 0.14, 0.12, "Tumour gate\nF″ = F4′ ⊙ (0.1 + 0.9·p)\n(floor 0.1 keeps context)", COL["head"], fs=8.3)
    arrow(ax, (0.68, 0.46), (0.68, 0.37))
    arrow(ax, (0.1, ys[3]), (0.61, 0.28), rad=0.18)
    box(ax, 0.61, 0.06, 0.14, 0.1, "GAP(F″)\n2048-d descriptor", COL["head"], fs=8.3)
    arrow(ax, (0.68, 0.25), (0.68, 0.16))
    box(ax, 0.8, 0.04, 0.17, 0.14, "Dropout 0.3 → FC(3)\nSoftmax →\nmeningioma / glioma /\npituitary", COL["head"], fs=8.3)
    arrow(ax, (0.75, 0.11), (0.8, 0.11))
    ax.text(0.79, 0.4, "Why: the paper pools the whole slice\n(GAP), so skull/background texture\ncan drive the decision. "
            "The gate\npools evidence mainly where the\ndecoder (trained with the tumorMask\nlabels the paper left unused) predicts\n"
            "tumour. The head stays linear in F″,\nso Grad-CAM on F″ is exact.", fontsize=8.5, va="center",
            bbox=dict(boxstyle="round", fc="#FAFAFA", ec="#999999"))
    save(fig, "fig_architecture_C_decoder_TFP_head")


if __name__ == "__main__":
    pipeline(); overall(); tgsa_module(); decoder_head()
    print("saved to", OUT)
