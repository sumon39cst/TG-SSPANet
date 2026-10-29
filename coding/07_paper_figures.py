"""Publication-style schematic figures (v2) built with real data from the trained TG-SSPANet.

Fig. 1 pipeline, Fig. 2 overall architecture, Fig. 3 TGSA module (+ learned attention maps),
Fig. 4 mask-guided decoder / tumour-focused pooling (+ real walk-through).
Outputs: E:/Life_Nero/result/figures/diagrams_v2/*.{pdf,svg,png}  and  E:/Life_Nero/paper/figures/fig0[1-4]_*.png
Requires assets.npz from 07_figure_assets.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = r"E:\Life_Nero\result\figures\diagrams_v2"
PAPER = r"E:\Life_Nero\paper\figures"
A = np.load(os.path.join(OUT, "assets.npz"))
CLASSES = ["meningioma", "glioma", "pituitary"]

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 6.5, "mathtext.fontset": "stixsans",
                     "axes.linewidth": 0.5})
MM = 1 / 25.4
# colour-blind-safe, muted palette
BB, BB_E = "#D5DAE1", "#5B6572"
ATT, ATT_E = "#F6B26B", "#B45F06"
DEC, DEC_E = "#9FD5B0", "#2E7D4F"
HEAD, HEAD_E = "#B4C0EA", "#3949AB"
DATA, DATA_E = "#A9CCE3", "#1F5F8B"
RED = "#C0392B"
GREY = "#555555"


# ------------------------------------------------------------------------------------------------ helpers
def canvas(w, h):
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def shade(c, f):
    c = np.array(matplotlib.colors.to_rgb(c)); return tuple(np.clip(c * f if f < 1 else c + (1 - c) * (f - 1), 0, 1))


def rbox(ax, x, y, w, h, text="", fc="white", ec=GREY, fs=6.2, bold=False, lw=0.6, ls="-", r=1.0, color="black", z=2,
         ha="center", italic=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z))
    if text:
        ax.text(x + w / 2 if ha == "center" else x + 1, y + h / 2, text, ha=ha, va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", style="italic" if italic else "normal", color=color,
                zorder=z + 1, linespacing=1.15)


def arrow(ax, p0, p1, color="#333333", lw=0.7, ls="-", rad=0.0, head=4, z=1, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=head, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def path(ax, pts, color="#333333", lw=0.7, ls="-", z=1):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, z=z)


def node(ax, x, y, s, r=2.0, fc="white"):
    ax.add_patch(Circle((x, y), r, fc=fc, ec="#333333", lw=0.6, zorder=6))
    ax.text(x, y - 0.1, s, ha="center", va="center", fontsize=7.5, zorder=7)


def cuboid(ax, x, y, w, h, d, fc, ec, label=None, sub=None, lw=0.5, z=3, img=None, cmap="gray", above=False):
    """Feature-map block: front face w (channels) x h (spatial), oblique depth d (spatial)."""
    dx, dy = d * 0.55, d * 0.42
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=shade(fc, 1.35), ec=ec, lw=lw, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=shade(fc, 0.82), ec=ec, lw=lw, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))
    if img is not None:
        ax.imshow(img, cmap=cmap, extent=(x, x + w, y, y + h), zorder=z + 0.5, interpolation="bilinear")
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=z + 0.6))
    if label and above:
        ax.text(x + (w + dx) / 2, y + h + dy + 1.2, label, ha="center", va="bottom", fontsize=5.8, linespacing=1.1, zorder=z)
    elif label:
        ax.text(x + (w + dx) / 2, y - 1.4, label, ha="center", va="top", fontsize=5.8, linespacing=1.1, zorder=z)
    return x + w + dx, y + h + dy


def image(ax, arr, x, y, w, h, cmap="gray", vmin=0, vmax=1, gt=None, over=None, ocmap="jet", oalpha=0.5,
          interp="bilinear", ec="#333333", title=None, tfs=5.8, pred=None):
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x, x + w, y, y + h), zorder=4, interpolation=interp)
    if over is not None:
        ax.imshow(over, cmap=ocmap, vmin=0, vmax=1, alpha=oalpha, extent=(x, x + w, y, y + h), zorder=4.2,
                  interpolation=interp)
    if gt is not None:
        ax.contour(np.flipud(gt.astype(float)), levels=[0.5], colors="#00E676", linewidths=0.55,
                   extent=(x, x + w, y, y + h), zorder=4.4)
    if pred is not None:
        ax.contour(np.flipud(pred.astype(float)), levels=[0.5], colors="#FF1744", linewidths=0.55,
                   extent=(x, x + w, y, y + h), zorder=4.4)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=0.5, zorder=4.5))
    if title:
        ax.text(x + w / 2, y + h + 1.0, title, ha="center", va="bottom", fontsize=tfs, linespacing=1.1)


def plabel(ax, x, y, s):
    ax.text(x, y, s, ha="left", va="top", fontsize=9, fontweight="bold")


def group(ax, x, y, w, h, title, ec, fc=None, fs=6.2, tx=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc or "none", ec=ec,
                                lw=0.6, ls=(0, (3, 2)), zorder=0.5))
    ax.text(x + tx, y + h - 1.2, title, ha="left", va="top", fontsize=fs, fontweight="bold", color=ec, zorder=1)


def legend(ax, x, y, items, ncol=1, colw=40, rowh=4.2, fs=5.8):
    for i, (kind, a, b, text) in enumerate(items):
        cx, cy = x + (i % ncol) * colw, y - (i // ncol) * rowh
        if kind == "patch":
            ax.add_patch(Rectangle((cx, cy - 1.3), 4, 2.6, fc=a, ec=b, lw=0.5, zorder=5))
        elif kind == "line":
            ax.plot([cx, cx + 4], [cy, cy], color=a, ls=b, lw=0.8, zorder=5)
        elif kind == "sym":
            node(ax, cx + 2, cy, a, r=1.4)
        ax.text(cx + 5.5, cy, text, ha="left", va="center", fontsize=fs, zorder=5)


def crop(fig, ax, ymin, ymax=None):
    """Trim empty canvas below ymin (and above ymax) while keeping mm units."""
    y0, y1 = ax.get_ylim(); ymax = ymax or y1
    ax.set_ylim(ymin, ymax)
    w = ax.get_xlim()[1] - ax.get_xlim()[0]
    fig.set_size_inches(w * MM, (ymax - ymin) * MM)


def save(fig, stem, paper_name):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, f"{stem}.png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(PAPER, paper_name), dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def overlay_rgb(img, mask, color=(1.0, 0.1, 0.27), alpha=0.45):
    rgb = np.repeat(img[..., None], 3, -1)
    m = mask[..., None]
    return rgb * (1 - alpha * m) + np.array(color) * alpha * m


# ------------------------------------------------------------------------------------------------ Fig. 2
def fig_overall():
    W, H = 180, 118
    fig, ax = canvas(W, H)
    i = 0  # glioma example
    img, gt = A["img"][i], A["gt"][i]
    yc = 86
    # input (3 stacked slices)
    for k in (2, 1, 0):
        image(ax, img, 2 + k * 1.2, yc - 11 + k * 1.2, 22, 22)
    ax.text(14, yc - 13.5, "Input T1-CE slice\n3 × 224 × 224", ha="center", va="top", fontsize=5.8)
    # encoder
    s1 = cuboid(ax, 33, yc - 12, 3.5, 24, 12, BB, BB_E, "Stage 1\n256 × 56 × 56", above=True)
    s2 = cuboid(ax, 51, yc - 8.5, 4.5, 17, 8.5, BB, BB_E, "Stage 2\n512 × 28 × 28", above=True)
    t2 = (61.5, yc - 6, 8, 12)
    s3 = cuboid(ax, 74, yc - 6, 5.5, 12, 6, BB, BB_E, "Stage 3\n1024 × 14 × 14", above=True)
    t3 = (84.5, yc - 6, 8, 12)
    s4 = cuboid(ax, 97, yc - 4, 7, 8, 4, BB, BB_E, "Stage 4\n2048 × 7 × 7", above=True)
    t4 = (107.5, yc - 6, 8, 12)
    for (x, y, w, h), g in zip((t2, t3, t4), A["gammas"]):
        rbox(ax, x, y, w, h, "", ATT, ATT_E)
        ax.text(x + w / 2, y + h * 0.62, "TGSA", ha="center", va="center", fontsize=6, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * 0.28, f"γ={g:.3f}", ha="center", va="center", fontsize=4.8, color="#5A2E00", zorder=4)
    arrow(ax, (25.5, yc), (33, yc))
    arrow(ax, (s1[0] + 0.3, yc), (51, yc))
    arrow(ax, (s2[0] + 0.2, yc), (t2[0], yc))
    arrow(ax, (t2[0] + t2[2], yc), (74, yc))
    arrow(ax, (s3[0] + 0.2, yc), (t3[0], yc))
    arrow(ax, (t3[0] + t3[2], yc), (97, yc))
    arrow(ax, (s4[0] + 0.2, yc), (t4[0], yc))
    # L_align tags above TGSA
    for (x, y, w, h) in (t2, t3, t4):
        arrow(ax, (x + w / 2, y + h), (x + w / 2, 104.5), color=RED, ls=(0, (2, 1.5)), lw=0.6)
    rbox(ax, 64, 104.5, 53, 5, r"$\mathcal{L}_{align}$ = BCE(spatial logits $A_s$, pooled mask)", "#FDEDEC", RED,
         fs=5.8, color=RED)
    # brace labels
    ax.plot([31, 117], [114, 114], color=GREY, lw=0.6); ax.plot([31, 31], [112.5, 114], color=GREY, lw=0.6)
    ax.plot([117, 117], [112.5, 114], color=GREY, lw=0.6)
    ax.text(74, 115, "Encoder: ImageNet-pre-trained ResNet50 (or VGG16) + TGSA after stages 2–4 (Fig. 3)",
            ha="center", va="bottom", fontsize=6.2, fontweight="bold")
    ax.plot([120, 178], [114, 114], color=GREY, lw=0.6); ax.plot([120, 120], [112.5, 114], color=GREY, lw=0.6)
    ax.plot([178, 178], [112.5, 114], color=GREY, lw=0.6)
    ax.text(149, 115, "Tumour-focused classification head (Fig. 4)", ha="center", va="bottom", fontsize=6.2,
            fontweight="bold")
    # TFP gate + head
    node(ax, 122, yc, "⊙")
    arrow(ax, (t4[0] + t4[2], yc), (120, yc))
    f2 = cuboid(ax, 127, yc - 4, 7, 8, 4, HEAD, HEAD_E, "F″ (gated)\n2048 × 7 × 7")
    arrow(ax, (124, yc), (127, yc))
    ax.text(128.5, yc + 9.5, "Grad-CAM\ntarget", ha="center", va="bottom", fontsize=5, color=HEAD_E, style="italic")
    rbox(ax, 139.5, yc - 9, 3, 18, "", HEAD, HEAD_E, r=0.4)
    ax.text(141, yc - 10.5, "GAP\n2048", ha="center", va="top", fontsize=5.8)
    arrow(ax, (f2[0] + 0.2, yc), (139.5, yc))
    # class probabilities (real)
    prob = A["prob"][i]; bx, by = 152, yc - 7
    arrow(ax, (142.5, yc), (bx - 1, yc))
    ax.text(bx - 0.5, yc + 7.5, "FC + softmax", ha="left", va="bottom", fontsize=5.8)
    for j, (c, pv) in enumerate(zip(CLASSES, prob)):
        yy = by + (2 - j) * 4.8
        ax.add_patch(Rectangle((bx, yy), 24 * pv, 3.2, fc=HEAD_E if j == 1 else HEAD, ec=HEAD_E, lw=0.4, zorder=4))
        ax.add_patch(Rectangle((bx, yy), 24, 3.2, fc="none", ec="#999999", lw=0.3, zorder=4))
        ax.text(bx + 0.6, yy + 1.6, f"{c}  {pv:.2f}", ha="left", va="center", fontsize=5.2, zorder=5,
                color="white" if (j == 1 and pv > 0.3) else "black")
    rbox(ax, 150, yc - 16.5, 28, 5, r"$\mathcal{L}_{cls}$: CE, LS 0.1, class-weighted", "#FDEDEC", RED, fs=5.4, color=RED)
    # decoder (top-down)
    yd = 36
    dec = [(38, 16, 8), (62, 12, 6), (86, 9, 4.5), (104, 6, 3)]  # x, h, d  for P1..P4
    srcx = [34.75, t2[0] + 4, t3[0] + 4, t4[0] + 4]
    srcy = [yc - 12 - 5.5, t2[1] - 3.5, t3[1] - 3.5, t4[1] - 3.5]
    for k, ((x, h, d), sx, sy) in enumerate(zip(dec, srcx, srcy)):
        cuboid(ax, x, yd - h / 2, 2.5, h, d, DEC, DEC_E, f"P{k + 1}\n128 × {[56, 28, 14, 7][k]}²")
        # lateral 1×1 projection: into the ⊕ of P1–P3 (summed with the upsampled coarser map); P4 is the lateral alone
        tx, ty = (x + 2.5 + d * 0.55 + 3.5, yd + 1.7) if k < 3 else (x + 1.25, yd + h / 2 + d * 0.42 + 0.3)
        path(ax, [(sx, sy + (5.5 if k == 0 else 3.5)), (sx, yd + h / 2 + d * 0.42 + 6), (tx, yd + h / 2 + d * 0.42 + 6),
                  (tx, ty)], color=DEC_E, lw=0.6)
    ax.text(37, 62, "lateral 1×1 conv → 128", fontsize=5.2, color=DEC_E, style="italic")
    for k in (3, 2, 1):
        x0 = dec[k][0]; x1 = dec[k - 1][0] + 2.5 + dec[k - 1][2] * 0.55
        node(ax, x1 + 3.5, yd, "⊕", r=1.6)
        arrow(ax, (x0 - 0.3, yd), (x1 + 5.2, yd), color=DEC_E)
        arrow(ax, (x1 + 1.9, yd), (x1 + 0.2, yd), color=DEC_E)
        ax.text((x0 + x1 + 5) / 2, yd + 1.2, "×2 ↑", ha="center", va="bottom", fontsize=5.2, color=DEC_E)
    rbox(ax, 4, yd - 6, 25, 12, "2 × [Conv3×3–BN–ReLU]\nConv1×1 → L\n1 × 56 × 56", DEC, DEC_E, fs=5.6)
    arrow(ax, (37.8, yd), (29, yd), color=DEC_E)
    ax.text(16.5, yd + 7.5, "Mask-guided FPN decoder (Fig. 4)", ha="center", va="bottom", fontsize=6.2,
            fontweight="bold", color=DEC_E)
    # predicted mask (real)
    pm = A["mask224"][i]
    image(ax, overlay_rgb(img, pm), 5, 3, 21, 21, gt=gt)
    arrow(ax, (16.5, yd - 6), (16.5, 24.5), color=DEC_E)
    ax.text(27.5, 17.4, "predicted tumour mask (red)\nvs. ground truth (green)", fontsize=5.2, va="center")
    rbox(ax, 27, 4, 27, 5, r"$\mathcal{L}_{seg}$ = BCE + Dice", "#FDEDEC", RED, fs=5.6, color=RED)
    # p map route to gate
    pmap = A["p"][i]
    path(ax, [(26.5, 13.5), (58, 13.5), (58, 8), (118, 8), (118, 52)], color=DEC_E, lw=0.6)
    ax.text(88, 9, "σ(L) → max-pool (8×8 windows, stride 8) → p", ha="center", va="bottom", fontsize=5.6, color=DEC_E)
    image(ax, pmap, 113, 52, 10, 10, cmap="viridis", interp="nearest")
    ax.text(124, 57, "p ∈ [0,1]\n1 × 7 × 7", fontsize=5.4, va="center")
    arrow(ax, (118, 62.3), (121.2, yc - 2), color=DEC_E)
    ax.text(124.5, 69, "gate: 0.1 + 0.9·p", fontsize=5.4, color=HEAD_E, style="italic")
    # total loss + legend
    ax.text(88, 2.2, r"$\mathcal{L} = \mathcal{L}_{cls} + 1.0\,\mathcal{L}_{seg} + 0.2\,\mathcal{L}_{align}$",
            ha="center", va="bottom", fontsize=6.8, color=RED)
    legend(ax, 132, 42, [("patch", BB, BB_E, "backbone stage"), ("patch", ATT, ATT_E, "TGSA attention"),
                         ("patch", DEC, DEC_E, "decoder"), ("patch", HEAD, HEAD_E, "classification head"),
                         ("line", RED, (0, (2, 1.5)), "training-only supervision"), ("sym", "⊙", None, "element-wise product"),
                         ("sym", "⊕", None, "element-wise sum")], ncol=1, rowh=4.3)
    rbox(ax, 129.5, 12, 48.5, 34, "", "none", "#BBBBBB", lw=0.4)
    save(fig, "fig02_architecture_overall", "fig02_architecture_overall.png")


# ------------------------------------------------------------------------------------------------ Fig. 3
def fig_tgsa():
    W, H = 180, 142
    fig, ax = canvas(W, H)
    plabel(ax, 1, H - 1, "(a)")
    ym = 104
    cuboid(ax, 4, ym - 8, 3, 16, 7, BB, BB_E, "X\nC × H × W", above=True)
    # channel branch
    group(ax, 16, 114, 78, 25, "Channel attention: tri-statistic pooling + ECA-style 1-D conv", ATT_E, "#FEF5EC")
    for j, t in enumerate(["GAP (mean)", "GMP (max)", "GSP (std)"]):
        yy = 129.5 - j * 6
        rbox(ax, 19, yy - 2.2, 16, 4.4, t, ATT, ATT_E, fs=5.6)
        path(ax, [(10.9, ym + 4), (13.5, ym + 4), (13.5, yy), (19, yy)], lw=0.5)
        arrow(ax, (35, yy), (39, 123.5), lw=0.5)
    rbox(ax, 39, 119, 10, 9, "stack\n3 × C", "white", ATT_E, fs=5.6)
    rbox(ax, 53, 119, 17, 9, "Conv1D (k = ψ(C))\n+ BN", ATT, ATT_E, fs=5.6)
    arrow(ax, (49, 123.5), (53, 123.5))
    node(ax, 74.5, 123.5, "σ", r=2)
    arrow(ax, (70, 123.5), (72.5, 123.5))
    rbox(ax, 79, 119, 12, 9, "$M_c$\nC × 1 × 1", "white", ATT_E, fs=5.6)
    arrow(ax, (76.5, 123.5), (79, 123.5))
    # main line
    node(ax, 85, ym, "⊙")
    arrow(ax, (85, 119), (85, ym + 2))
    arrow(ax, (11, ym), (83, ym))
    ax.text(90, ym + 1.2, "$X' = M_c \\odot X$", fontsize=6, va="bottom")
    node(ax, 150, ym, "⊙")
    arrow(ax, (87, ym), (148, ym))
    rbox(ax, 155, ym - 3.5, 9, 7, "× γ\n(init 0)", "white", GREY, fs=5.6)
    arrow(ax, (152, ym), (155, ym))
    node(ax, 168.5, ym, "⊕")
    arrow(ax, (164, ym), (166.5, ym))
    cuboid(ax, 172.5, ym - 8, 3, 16, 3, ATT, ATT_E, "TGSA(X)\nC × H × W")
    arrow(ax, (170.5, ym), (172.5, ym))
    path(ax, [(5.5, ym - 8), (5.5, 55), (168.5, 55), (168.5, ym - 2)], lw=0.6)
    ax.text(88, 55.8, "identity (residual) path", ha="center", va="bottom", fontsize=5.6, style="italic")
    # spatial branch
    group(ax, 88, 59, 72, 40, "Spatial attention: directional strips + style statistics", "#1F5F8B", "#EEF5FB", tx=8)
    path(ax, [(92, ym), (92, 88.5)], lw=0.6)
    rbox(ax, 90, 80, 17, 8.5, "Conv1×1 C→m\nBN, ReLU", DATA, DATA_E, fs=5.6)
    ax.text(98.5, 78.8, "m = max(16, C/16)", ha="center", va="top", fontsize=5, style="italic")
    rbox(ax, 111, 88, 19, 7, "avg ‖ max over W\n→ 2m × H", DATA, DATA_E, fs=5.4)
    rbox(ax, 111, 78.5, 19, 7, "avg ‖ max over H\n→ 2m × W", DATA, DATA_E, fs=5.4)
    arrow(ax, (107, 86), (111, 91.5), lw=0.5); arrow(ax, (107, 82.5), (111, 82), lw=0.5)
    rbox(ax, 133, 88, 17, 7, "GConv1D k=7, BN\nbroadcast", DATA, DATA_E, fs=5.4)
    rbox(ax, 133, 78.5, 17, 7, "GConv1D k=7, BN\nbroadcast", DATA, DATA_E, fs=5.4)
    arrow(ax, (130, 91.5), (133, 91.5)); arrow(ax, (130, 82), (133, 82))
    path(ax, [(90.5, ym), (90.5, 67.5), (93, 67.5)], lw=0.6)
    rbox(ax, 93, 63.5, 22, 8, "mean ‖ std ‖ max over C\n→ 3 × H × W", DATA, DATA_E, fs=5.4)
    rbox(ax, 118, 63.5, 27, 8, "Conv3×3 (dil. 2)–BN–ReLU\nConv3×3–BN → m×H×W", DATA, DATA_E, fs=5.2)
    arrow(ax, (115, 67.5), (118, 67.5))
    node(ax, 154, 76, "⊕", r=1.8)
    arrow(ax, (150, 91.5), (152.8, 77.4), lw=0.5); arrow(ax, (150, 82), (152.4, 76.9), lw=0.5)
    arrow(ax, (145, 69), (152.6, 74.8), lw=0.5)
    rbox(ax, 147, 60.5, 15.5, 6.5, "ReLU, Conv1×1\n→ $A_s$ (1×H×W)", DATA, DATA_E, fs=5.0)
    arrow(ax, (154, 74.2), (154, 67))
    rbox(ax, 139.5, 97.5 - 3, 0.1, 0.1, "", "none", "none")
    path(ax, [(162.5, 63.5), (165, 63.5), (165, 92), (151.5, 92), (150.6, ym - 2)], lw=0.6)
    ax.text(166, 80, "σ → $M_s$", fontsize=5.8, va="center", rotation=90)
    rbox(ax, 164, 57, 15, 5.5, "$\\mathcal{L}_{align}$", "#FDEDEC", RED, fs=6, color=RED)
    arrow(ax, (162.5, 62), (164, 60.5), color=RED, ls=(0, (2, 1.5)))
    ax.text(W / 2, 48.5, r"$\mathrm{TGSA}(X) = X + \gamma\,M_s\odot(M_c\odot X)$,  $M_s=\sigma(A_s)$,   $k=\psi(C)=\mathrm{odd}\left(\left|\frac{\log_2 C+1}{2}\right|\right)$",
            ha="center", va="center", fontsize=7)
    # (b) learned attention maps
    plabel(ax, 1, 44.3, "(b)")
    ax.text(10, 43.6, "Learned TGSA spatial attention $M_s=\\sigma(A_s)$ of the trained ResNet50 TG-SSPANet on held-out test slices (P2 fold 1; green: ground-truth tumour)",
            ha="left", va="top", fontsize=6.2)
    names = ["Glioma", "Meningioma", "Pituitary"]
    iw, gap, gx = 13.2, 0.8, 3
    for i in range(3):
        x0 = gx + i * (4 * iw + 3 * gap + 4)
        img, gt = A["img"][i], A["gt"][i]
        tiles = [(None, "input"), (A["att2"][i], f"stage 2 (28²)\nγ={A['gammas'][0]:.3f}"),
                 (A["att3"][i], f"stage 3 (14²)\nγ={A['gammas'][1]:.3f}"), (A["att4"][i], f"stage 4 (7²)\nγ={A['gammas'][2]:.3f}")]
        for j, (att, t) in enumerate(tiles):
            xx = x0 + j * (iw + gap)
            over = None
            if att is not None:
                over = np.kron(att, np.ones((224 // att.shape[0], 224 // att.shape[1])))
                over = (over - over.min()) / (np.ptp(over) + 1e-6)
            image(ax, img, xx, 22, iw, iw, gt=gt, over=over, ocmap="jet", oalpha=0.45, title=t, tfs=5,
                  interp="nearest" if att is not None else "bilinear")
        ax.text(x0 + 2 * (iw + gap) - gap / 2, 20, names[i], ha="center", va="top", fontsize=6.2, fontweight="bold")
    crop(fig, ax, 15.5)
    save(fig, "fig03_tgsa_module", "fig03_tgsa_module.png")


# ------------------------------------------------------------------------------------------------ Fig. 4
def fig_decoder():
    W, H = 180, 150
    fig, ax = canvas(W, H)
    plabel(ax, 1, H + 4.5, "(a)")
    # bottom-up (left) and top-down (right) pyramids
    lv = [(66, 16, 3.5, "F$_1$  256 × 56²"), (88, 12, 4.5, "F$_2'$  512 × 28²"), (106, 8, 5.5, "F$_3'$  1024 × 14²"),
          (121, 6, 7, "F$_4'$  2048 × 7²")]
    for (y, h, w, lab) in lv:
        cuboid(ax, 6, y, w, h, h * 0.5, BB if "F$_1$" in lab else ATT, BB_E if "F$_1$" in lab else ATT_E)
        ax.text(4.5, y + h / 2, lab, ha="right", va="center", fontsize=5.6)
    ax.text(10, 145, "encoder features", ha="center", fontsize=5.8, style="italic")
    for k, (y, h, w, lab) in enumerate(lv):
        cuboid(ax, 44, y, 2.5, h, h * 0.5, DEC, DEC_E)
        ax.text(47 + h * 0.3, y + h + h * 0.21 + 1.0, f"P$_{k + 1}$  128 × {[56, 28, 14, 7][k]}²", ha="left", va="bottom", fontsize=5.4)
        arrow(ax, (6 + w + h * 0.28 + 1, y + h / 2), (39.5 if k < 3 else 44, y + h / 2), color=DEC_E)
        if k < 3:
            node(ax, 41.5, y + h / 2, "⊕", r=1.6)
            arrow(ax, (43.1, y + h / 2), (44, y + h / 2), color=DEC_E)
    ax.text(26, 129.5, "lateral\n1×1 conv", ha="center", fontsize=5.4, color=DEC_E, style="italic")
    for k in (3, 2, 1):
        y_from = lv[k][0]; y_to = lv[k - 1][0] + lv[k - 1][1] / 2 + 1.7
        # top-down path: leaves the P block above (its bottom face) and enters the ⊕ of the next finer level
        path(ax, [(45.25, y_from), (45.25, y_from - 1.6), (41.5, y_from - 1.6), (41.5, y_to)], color=DEC_E, lw=0.6)
        ax.text(40.5, (y_from - 1.6 + y_to) / 2, "×2 ↑", ha="right", va="center", fontsize=5.2, color=DEC_E)
    ax.text(49, 145, "top-down pathway", ha="center", fontsize=5.8, style="italic")
    rbox(ax, 70, 67, 26, 13, "2 × [Conv3×3–BN–ReLU]\n(128)", DEC, DEC_E, fs=5.8)
    arrow(ax, (51.2, 74), (70, 74), color=DEC_E)
    rbox(ax, 100, 67, 20, 13, "Conv1×1\n→ mask logits L\n1 × 56 × 56", DEC, DEC_E, fs=5.8)
    arrow(ax, (96, 73.5), (100, 73.5), color=DEC_E)
    rbox(ax, 126, 67, 25, 13, "σ, bilinear ↑\n→ tumour mask M̂\n1 × 224 × 224", DEC, DEC_E, fs=5.8)
    arrow(ax, (120, 73.5), (126, 73.5), color=DEC_E)
    rbox(ax, 155, 70, 23, 7, "$\\mathcal{L}_{seg}$ = BCE + Dice", "#FDEDEC", RED, fs=5.8, color=RED)
    arrow(ax, (151, 73.5), (155, 73.5), color=RED, ls=(0, (2, 1.5)))
    # TFP head
    group(ax, 72, 88, 107, 50, "Tumour-focused pooling (TFP) head", HEAD_E, "#F3F5FC")
    rbox(ax, 104, 92, 30, 11, "σ(L) → max-pool (8×8, stride 8)\n→ p $\\in [0,1]^{1\\times7\\times7}$", HEAD, HEAD_E, fs=5.6)
    arrow(ax, (110, 80), (110, 92), color=DEC_E)
    rbox(ax, 138, 92, 18, 11, "g = φ + (1 − φ)·p\nφ = 0.1", HEAD, HEAD_E, fs=5.6)
    arrow(ax, (134, 97.5), (138, 97.5))
    path(ax, [(10.5, 128.6), (10.5, 141.5), (144, 141.5), (144, 125.5)], lw=0.6)
    ax.text(100, 142.2, "F$_4'$ (after TGSA)", ha="center", va="bottom", fontsize=5.6)
    node(ax, 147, 118, "⊙")
    arrow(ax, (147, 103), (147, 116))
    arrow(ax, (144, 125.5), (145.6, 119.5))
    f2 = cuboid(ax, 152, 114, 6, 8, 4, HEAD, HEAD_E)
    arrow(ax, (149, 118), (152, 118))
    ax.text(157, 112.5, "F″ = F$_4' \\odot$ g", ha="center", va="top", fontsize=5.6)
    rbox(ax, 163, 110, 2.6, 16, "", HEAD, HEAD_E, r=0.3)
    arrow(ax, (f2[0], 118), (163, 118))
    ax.text(164.3, 108.5, "GAP", ha="center", va="top", fontsize=5.6)
    rbox(ax, 168, 113, 10, 10, "FC\nsoftmax", HEAD, HEAD_E, fs=5.6)
    arrow(ax, (165.6, 118), (168, 118))
    ax.text(108, 120, "Linear head in F″ ⇒ Grad-CAM on F″ equals the exact class-activation map\n$\\sum_k W_{ck}F''_k/49$, up to ReLU and min–max normalisation",
            ha="center", va="center", fontsize=5.8, style="italic", color=HEAD_E)
    ax.text(108, 110, "p pools probabilities (not logits), so small tumours are not averaged away",
            ha="center", va="center", fontsize=5.4, color=GREY)
    # (b) walk-through
    plabel(ax, 1, 60, "(b)")
    ax.text(10, 59.3, "Walk-through on held-out test slices (P2 fold 1; green contour: ground truth; red: prediction)",
            ha="left", va="top", fontsize=6.2)
    cols = ["input", "decoder σ(L)\n56 × 56", "p = max-pool (8×8, stride 8)", "gate g = 0.1 + 0.9 p",
            "Grad-CAM++ on F″\nTG-SSPANet (ours)", "Grad-CAM++ (GAP) SSPANet\n(same recipe, P2 fold 1)"]
    iw, x0 = 22, 14
    for r, (i, name) in enumerate([(0, "Glioma"), (2, "Pituitary")]):
        y = 28 - r * 26
        img, gt = A["img"][i], A["gt"][i]
        g = 0.1 + 0.9 * A["p"][i]
        tiles = [dict(arr=img, gt=gt), dict(arr=img, over=np.kron(A["mask56"][i], np.ones((4, 4))), ocmap="Reds", gt=gt),
                 dict(arr=A["p"][i], cmap="viridis", interp="nearest"), dict(arr=g, cmap="viridis", interp="nearest"),
                 dict(arr=img, over=A["cam_ours"][i], gt=gt), dict(arr=img, over=A["cam_base"][i], gt=gt)]
        for j, t in enumerate(tiles):
            xx = x0 + j * (iw + 6.5)
            image(ax, x=xx, y=y, w=iw, h=iw, title=cols[j] if r == 0 else None, tfs=5.6, **t)
            if j < 5:
                arrow(ax, (xx + iw + 0.8, y + iw / 2), (xx + iw + 5.7, y + iw / 2), lw=0.6, head=3.5,
                      color="#999999" if j == 4 else "#333333", ls=(0, (2, 1.5)) if j == 4 else "-")
        pr = A["prob"][i]; bp = A["bprob"][i]; lab = int(A["labels"][i])
        ax.text(x0 - 2, y + iw / 2, f"{name}\nP(true class):\nours {pr[lab]:.2f}\nSSPANet {bp[lab]:.2f}", ha="right", va="center",
                fontsize=5.4, linespacing=1.2)
    ax.text(x0 + 4 * (iw + 6.5) + iw + 3.2, 52.5, "vs.", ha="center", fontsize=5.6, color="#999999")
    crop(fig, ax, 0, H + 5)
    save(fig, "fig04_decoder_tfp", "fig04_decoder_tfp.png")


# ------------------------------------------------------------------------------------------------ Fig. 1
def fig_pipeline():
    W, H = 180, 132
    fig, ax = canvas(W, H)
    cols = [(1.5, "1  Data & pre-processing", DATA, DATA_E), (46.5, "2  Evaluation protocols", "#C8E6C9", DEC_E),
            (91.5, "3  Models & training", ATT, ATT_E), (136.5, "4  Evaluation & explainability", HEAD, HEAD_E)]
    cw = 42
    for x, t, fc, ec in cols:
        rbox(ax, x, 122, cw, 8, t, fc, ec, fs=6.8, bold=True, r=1.2)
        rbox(ax, x, 14, cw, 106, "", "#FAFAFA", "#CCCCCC", lw=0.4, r=1.2, z=0.2)
    for x, _, _, _ in cols[:-1]:
        ax.add_patch(Polygon([(x + cw + 0.4, 71), (x + cw + 2.6, 67), (x + cw + 0.4, 63)], fc="#777777", ec="none", zorder=5))
    # ---- column 1
    x = 1.5
    names = ["glioma", "meningioma", "pituitary"]
    for j, i in enumerate([0, 1, 2]):
        image(ax, A["img"][i], x + 1.5 + j * 13.5, 101, 12.5, 12.5, gt=A["gt"][i], title=names[j], tfs=5.2)
    ax.text(x + 21, 99.5, "Figshare dataset [2,3]: 3,064 T1-CE slices,\n233 patients; fields: image, label,\n"
            "patient ID, tumorMask (green)", ha="center", va="top", fontsize=5.5, linespacing=1.2)
    counts = [708, 1426, 930]  # official class order: meningioma, glioma, pituitary
    for j, (n, c) in enumerate(zip(counts, CLASSES)):
        yy = 84 - j * 4
        ax.add_patch(Rectangle((x + 16, yy - 1.4), 22 * n / 1426, 2.8, fc=DATA, ec=DATA_E, lw=0.4))
        ax.text(x + 15, yy, c, ha="right", va="center", fontsize=5.2)
        ax.text(x + 16.6 + 22 * n / 1426, yy, str(n), ha="left", va="center", fontsize=5.0)
    steps = ["min–max intensity normalisation", "resize 512 → 256 (area); masks nearest",
             "grey → 3 channels, ImageNet statistics", "224 × 224 network input"]
    for j, s in enumerate(steps):
        ax.text(x + 2, 69 - j * 3.6, "•  " + s, fontsize=5.3, va="center")
    ax.text(x + 21, 52.5, "joint image–mask GPU augmentation", ha="center", fontsize=5.6, fontweight="bold")
    for j in range(4):
        image(ax, A["aug"][j], x + 1.5 + j * 10, 40, 9.3, 9.3, gt=A["aug_m"][j])
    ax.text(x + 21, 38.3, "flip · rotate ±15° · scale ±10% ·\nshift ±6% · contrast/brightness", ha="center",
            va="top", fontsize=5.2)
    # ---- column 2: split schematics
    x = 46.5
    rng = np.random.RandomState(3)
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    ax.text(x + 21, 116, "P1  image-level 80/10/10 (as in [21])", ha="center", va="top", fontsize=5.8, fontweight="bold")
    pats = np.repeat(np.arange(12), 5); rng.shuffle(pats)
    rows = [("train", pats[:44]), ("val", pats[44:52]), ("test", pats[52:60])]
    for r, (nm, pp) in enumerate(rows):
        yy = 108 - r * 7 if r == 0 else (108 - 7 - 3.2 * 1) - (r - 1) * 4.5
        per = 22
        for q, pidx in enumerate(pp):
            ax.add_patch(Rectangle((x + 9 + (q % per) * 1.45, yy - (q // per) * 1.8), 1.25, 1.5, fc=pal[pidx], ec="none"))
        ax.text(x + 8, yy + 0.7, nm, ha="right", va="center", fontsize=5.2)
    ax.text(x + 21, 89.5, "slices of the same patient (colour) fall in\ntrain and test → patient leakage;\n"
            "here all 160 test patients are also in train", ha="center", va="top", fontsize=5.2, color=RED)
    ax.text(x + 21, 76, "P2  official patient-wise 5-fold CV", ha="center", va="top", fontsize=5.8, fontweight="bold")
    for f in range(5):
        yy = 69 - f * 4.2
        for q, pidx in enumerate(range(f * 4, f * 4 + 4)):
            ax.add_patch(Rectangle((x + 9 + q * 6.5, yy - 1.2), 6.1, 2.4, fc=pal[pidx % 20], ec="none"))
        ax.text(x + 8, yy, f"fold {f + 1}", ha="right", va="center", fontsize=5.2)
        if f == 1:
            ax.add_patch(Rectangle((x + 8.6, yy - 1.7), 26.8, 3.4, fc="none", ec=DEC_E, lw=0.7))
            ax.text(x + 36.5, yy, "test", ha="left", va="center", fontsize=5.2, color=DEC_E)
    ax.text(x + 21, 46, "each patient in exactly one fold; ~10% of\ntraining patients form the validation set;\n"
            "542–679 test slices per fold", ha="center", va="top", fontsize=5.2)
    ax.text(x + 21, 32, "P1 and P2 splits are fixed across all models and seeds", ha="center", va="top", fontsize=5.2, style="italic")
    # ---- column 3: models
    x = 91.5
    ax.text(x + 21, 116, "Reproduced baselines [21]", ha="center", va="top", fontsize=5.8, fontweight="bold")
    chain = [("VGG16 / ResNet50", BB, BB_E), ("one attention module\nCoord · GCNet · SPNet · SRMNet\nSSPANet · SE · CBAM", ATT, ATT_E),
             ("GAP → FC", HEAD, HEAD_E)]
    yy = 111
    for t, fc, ec in chain:
        hh = 8 if "\n" in t else 4.5
        rbox(ax, x + 5, yy - hh, 32, hh, t, fc, ec, fs=5.2)
        yy -= hh + 2.2
        if t != "GAP → FC":
            arrow(ax, (x + 21, yy + 2.2), (x + 21, yy + 0.2), lw=0.5)
    ax.text(x + 21, yy - 0.3, "original recipe: Adam 1e-4, ≤30 epochs,\nearly stopping, no augmentation",
            ha="center", va="top", fontsize=5.1)
    ax.text(x + 21, 79, "Proposed TG-SSPANet", ha="center", va="top", fontsize=5.8, fontweight="bold")
    chain2 = [("backbone + TGSA × 3 (Fig. 3)", ATT, ATT_E), ("mask-guided FPN decoder (Fig. 4)", DEC, DEC_E),
              ("tumour-focused pooling gate", HEAD, HEAD_E), ("GAP → FC", HEAD, HEAD_E)]
    yy = 74
    for t, fc, ec in chain2:
        rbox(ax, x + 5, yy - 4.5, 32, 4.5, t, fc, ec, fs=5.2)
        yy -= 6.7
        if t != "GAP → FC":
            arrow(ax, (x + 21, yy + 2.2), (x + 21, yy + 0.2), lw=0.5)
    ax.text(x + 21, 45.5, r"$\mathcal{L}=\mathcal{L}_{cls}+\mathcal{L}_{seg}+0.2\,\mathcal{L}_{align}$",
            ha="center", va="top", fontsize=6.2, color=RED)
    ax.text(x + 21, 39.5, "proposed recipe: AdamW + cosine, 40 epochs,\naugmentation, label smoothing, class\n"
            "weights, EMA, flip-TTA (also used for all\nablations and same-recipe controls)", ha="center", va="top",
            fontsize=5.1)
    # ---- column 4: evaluation
    x = 136.5
    ax.text(x + 21, 116, "Classification", ha="center", va="top", fontsize=5.8, fontweight="bold")
    cm = np.array([[658, 20, 30], [35, 1390, 1], [9, 3, 918]])   # TG-SSPANet, P2, pooled folds
    cs = 5.2; cx0, cy0 = x + 13, 94
    for r in range(3):
        for c in range(3):
            v = cm[r, c] / cm[r].sum()
            ax.add_patch(Rectangle((cx0 + c * cs, cy0 + (2 - r) * cs), cs, cs, fc=plt.get_cmap("Blues")(0.15 + 0.8 * v),
                                   ec="white", lw=0.4))
            ax.text(cx0 + c * cs + cs / 2, cy0 + (2 - r) * cs + cs / 2, str(cm[r, c]), ha="center", va="center",
                    fontsize=4.8, color="white" if v > 0.5 else "black")
    for k, n in enumerate(["M", "G", "P"]):
        ax.text(cx0 - 1, cy0 + (2 - k) * cs + cs / 2, n, ha="right", va="center", fontsize=5)
        ax.text(cx0 + k * cs + cs / 2, cy0 - 1, n, ha="center", va="top", fontsize=5)
    ax.text(x + 40.5, 101.8, "TG-SSPANet\nP2, 5 folds", ha="right", va="center", fontsize=4.8, color=GREY)
    ax.text(x + 21, 90.5, "Acc · F1 · κ · MCC · AUC · patient-level acc.\npatient-clustered bootstrap CIs\ncorrected resampled t-tests · calibration",
            ha="center", va="top", fontsize=5.2)
    ax.text(x + 21, 77, "Explanations vs. tumour masks", ha="center", va="top", fontsize=5.8, fontweight="bold")
    image(ax, A["img"][0], x + 5, 55, 15, 15, over=A["cam_base"][0], gt=A["gt"][0], title="SSPANet (same recipe)", tfs=5.2)
    image(ax, A["img"][0], x + 22.5, 55, 15, 15, over=A["cam_ours"][0], gt=A["gt"][0], title="TG-SSPANet", tfs=5.2)
    ax.text(x + 21, 53.5, "Grad-CAM · Grad-CAM++ · EigenGrad-CAM\npointing game · energy-in-mask · CAM-IoU",
            ha="center", va="top", fontsize=5.2)
    ax.text(x + 21, 44, "Faithfulness, ablation & cost", ha="center", va="top", fontsize=5.8, fontweight="bold")
    ax.text(x + 21, 40, "sham / decoy inpainting · context rings ·\nlocation-only control · deletion/insertion ·\nablation · params · GMACs · latency",
            ha="center", va="top", fontsize=5.2)
    # gap chips
    chips = [(1.5, ["G1 masks used", "G4 class weights"]), (46.5, ["G2 leakage-free P2"]),
             (91.5, ["G3 multi-stage + ablation", "G5 SE/CBAM added"]), (136.5, ["G1 quantitative XAI", "G6 cost + variance"])]
    for x, cl in chips:
        for j, c in enumerate(cl):
            rbox(ax, x + 2, 23.5 - j * 6, cw - 4, 4.5, c, "#FDEDEC", RED, fs=5.4, color=RED, r=1)
    ax.text(W / 2, 8.5, "Research gaps of [21] addressed (red): G1 unused tumour masks · G2 patient leakage · "
            "G3 single-stage attention without ablation ·\nG4 unhandled class imbalance · G5 SE/CBAM not evaluated · "
            "G6 unmeasured cost and run-to-run variance", ha="center", va="center", fontsize=5.6, color=RED)
    crop(fig, ax, 4, H)
    save(fig, "fig01_pipeline", "fig01_pipeline.png")


if __name__ == "__main__":
    fig_pipeline(); fig_overall(); fig_tgsa(); fig_decoder()
    print("saved to", OUT, "and", PAPER)
