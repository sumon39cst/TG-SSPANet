"""Editable draw.io versions of Figures 1-4 (bold, clear text; real images embedded).

Outputs to E:/Life_Nero/paper/drawio/:  fig01_pipeline.drawio, fig02_architecture_overall.drawio,
fig03_tgsa_module.drawio, fig04_decoder_tfp.drawio  (+ *_preview.png layout checks).
Requires result/figures/diagrams_v2/assets.npz (07_figure_assets.py).
"""
import matplotlib.pyplot as plt
import numpy as np

from drawio_lib import Diagram, png_from_array

OUT = r"E:\Life_Nero\paper\drawio"
A = np.load(r"E:\Life_Nero\result\figures\diagrams_v2\assets.npz")
CLASSES = ["meningioma", "glioma", "pituitary"]
BB, BB_E = "#D5DAE1", "#5B6572"
ATT, ATT_E = "#F6B26B", "#B45F06"
DEC, DEC_E = "#9FD5B0", "#2E7D4F"
HEAD, HEAD_E = "#B4C0EA", "#3949AB"
DATA, DATA_E = "#A9CCE3", "#1F5F8B"
RED, RED_BG, GREY = "#C0392B", "#FDEDEC", "#555555"


def overlay_rgb(img, mask, color=(1.0, 0.1, 0.27), alpha=0.45):
    rgb = np.repeat(img[..., None], 3, -1); m = mask[..., None]
    return np.clip(rgb * (1 - alpha * m) + np.array(color) * alpha * m, 0, 1)


def up(att):
    o = np.kron(att, np.ones((224 // att.shape[0], 224 // att.shape[1])))
    return (o - o.min()) / (np.ptp(o) + 1e-6)


def hexc(c):
    return "#%02x%02x%02x" % tuple(int(255 * v) for v in c[:3])


# ------------------------------------------------------------------------------------------------ Fig. 1
def fig_pipeline():
    W, H = 180, 132
    d = Diagram(W, H, "fig01_pipeline", ymin=3)
    cols = [(1.5, "1  Data & pre-processing", DATA, DATA_E), (46.5, "2  Evaluation protocols", "#C8E6C9", DEC_E),
            (91.5, "3  Models & training", ATT, ATT_E), (136.5, "4  Evaluation & explainability", HEAD, HEAD_E)]
    cw = 42
    for x, t, fc, ec in cols:
        d.box(x, 14, cw, 106, "", "#FAFAFA", "#CCCCCC", lw=0.8)
        d.box(x, 122, cw, 8, t, fc, ec, fs=6.6)
    for x, _, _, _ in cols[:-1]:
        d.edge([(x + cw + 0.2, 67), (x + cw + 2.8, 67)], color="#777777", lw=3)
    # column 1
    x = 1.5
    names = ["glioma", "meningioma", "pituitary"]
    for j in range(3):
        d.image(x + 1.5 + j * 13.5, 101, 12.5, 12.5, png_from_array(A["img"][j], gt=A["gt"][j]))
        d.text(x + 1.5 + j * 13.5 + 6.25, 114.3, names[j], fs=5.2, va="bottom")
    d.text(x + 21, 99.5, "Figshare dataset: 3,064 T1-CE slices,\n233 patients; fields: image, label,\npatient ID, tumorMask (green)",
           fs=5.3, va="top")
    for j, (n, c) in enumerate(zip([708, 1426, 930], CLASSES)):
        yy = 84 - j * 4
        d.box(x + 16, yy - 1.4, 22 * n / 1426, 2.8, "", DATA, DATA_E, rounded=False, lw=0.8)
        d.text(x + 15, yy, c, fs=5.2, ha="right")
        d.text(x + 16.6 + 22 * n / 1426, yy, str(n), fs=5.0, ha="left")
    for j, s in enumerate(["min–max intensity normalisation", "resize 256 × 256 (masks: nearest)",
                           "grey → 3 channels, ImageNet norm.", "224 × 224 network input"]):
        d.text(x + 2, 69 - j * 3.6, "•  " + s, fs=5.2, ha="left")
    d.text(x + 21, 52.5, "joint image–mask augmentation", fs=5.6)
    for j in range(4):
        d.image(x + 1.5 + j * 10, 40, 9.3, 9.3, png_from_array(A["aug"][j], gt=A["aug_m"][j], size=140))
    d.text(x + 21, 38.3, "flip · rotate ±15° · scale ±10% ·\nshift ±6% · contrast/brightness", fs=5.1, va="top")
    # column 2
    x = 46.5
    rng = np.random.RandomState(3); pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    d.text(x + 21, 116, "P1  image-level 80/10/10\n(as in the original study)", fs=5.6, va="top")
    pats = np.repeat(np.arange(12), 5); rng.shuffle(pats)
    for nm, pp, yy in [("train", pats[:44], 106), ("val", pats[44:52], 97.8), ("test", pats[52:60], 93.3)]:
        for q, pidx in enumerate(pp):
            d.box(x + 9 + (q % 22) * 1.45, yy - (q // 22) * 1.8, 1.25, 1.5, "", hexc(pal[pidx]), "none", rounded=False, lw=0)
        d.text(x + 8, yy + 0.7, nm, fs=5.2, ha="right")
    d.text(x + 21, 89.5, "slices of the same patient (colour) fall in\ntrain and test → patient leakage;\n"
           "all 160 test patients are also in train", fs=5.0, va="top", color=RED)
    d.text(x + 21, 76, "P2  official patient-wise 5-fold CV", fs=5.6, va="top")
    for f in range(5):
        yy = 69 - f * 4.2
        for q, pidx in enumerate(range(f * 4, f * 4 + 4)):
            d.box(x + 9 + q * 6.5, yy - 1.2, 6.1, 2.4, "", hexc(pal[pidx % 20]), "none", rounded=False, lw=0)
        d.text(x + 8, yy, f"fold {f + 1}", fs=5.2, ha="right")
        if f == 1:
            d.box(x + 8.6, yy - 1.7, 26.8, 3.4, "", "none", DEC_E, rounded=False, lw=1.4)
            d.text(x + 36.5, yy, "test", fs=5.2, ha="left", color=DEC_E)
    d.text(x + 21, 46, "each patient in exactly one fold;\n~10% of training patients form the\nvalidation set; "
           "542–679 test slices/fold", fs=5.0, va="top")
    d.text(x + 21, 35, "P1 and P2 splits are fixed\nacross all models and seeds", fs=5.0, va="top", italic=True)
    # column 3
    x = 91.5
    d.text(x + 21, 116, "Reproduced baselines", fs=5.8, va="top")
    yy = 111
    for k, (t, fc, ec, hh) in enumerate([("VGG16 / ResNet50", BB, BB_E, 4.5),
                                         ("one attention module\nCoord · GCNet · SPNet · SRMNet\nSSPANet · SE · CBAM", ATT, ATT_E, 8.5),
                                         ("GAP → FC", HEAD, HEAD_E, 4.5)]):
        d.box(x + 4, yy - hh, 34, hh, t, fc, ec, fs=5.0)
        yy -= hh + 2.2
        if k < 2:
            d.edge([(x + 21, yy + 2.2), (x + 21, yy + 0.2)], lw=1.1)
    d.text(x + 21, yy - 0.3, "original recipe: Adam 1e-4, ≤30 epochs,\nearly stopping, no augmentation", fs=4.9, va="top")
    d.text(x + 21, 79, "Proposed TG-SSPANet", fs=5.8, va="top")
    yy = 74
    for k, (t, fc, ec) in enumerate([("backbone + TGSA × 3 (Fig. 3)", ATT, ATT_E), ("mask-guided decoder (Fig. 4)", DEC, DEC_E),
                                     ("tumour-focused pooling gate", HEAD, HEAD_E), ("GAP → FC", HEAD, HEAD_E)]):
        d.box(x + 4, yy - 4.5, 34, 4.5, t, fc, ec, fs=5.0)
        yy -= 6.7
        if k < 3:
            d.edge([(x + 21, yy + 2.2), (x + 21, yy + 0.2)], lw=1.1)
    d.text(x + 21, 45.5, "ℒ = ℒ_{cls} + ℒ_{seg} + 0.2 ℒ_{align}", fs=6.0, va="top", color=RED)
    d.text(x + 21, 39.5, "proposed recipe: AdamW + cosine,\n40 epochs, augmentation, label\nsmoothing, class weights, EMA,\n"
           "flip-TTA (also for all controls)", fs=4.9, va="top")
    # column 4
    x = 136.5
    d.text(x + 21, 116, "Classification", fs=5.8, va="top")
    cm = np.array([[658, 20, 30], [35, 1390, 1], [9, 3, 918]]); cs = 5.2; cx0, cy0 = x + 11, 94   # TG-SSPANet, P2, pooled folds
    for r in range(3):
        for c in range(3):
            v = cm[r, c] / cm[r].sum()
            d.box(cx0 + c * cs, cy0 + (2 - r) * cs, cs, cs, str(cm[r, c]), hexc(plt.get_cmap("Blues")(0.15 + 0.8 * v)),
                  "#FFFFFF", fs=4.8, rounded=False, lw=1, color="#FFFFFF" if v > 0.5 else "#000000")
    for k, n in enumerate(["M", "G", "P"]):
        d.text(cx0 - 1, cy0 + (2 - k) * cs + cs / 2, n, fs=5, ha="right")
        d.text(cx0 + k * cs + cs / 2, cy0 - 1, n, fs=5, va="top")
    d.text(x + 41, 101.8, "TG-SSPANet\nP2, 5 folds", fs=4.6, ha="right", color=GREY)
    d.text(x + 21, 90.5, "Acc · F1 · κ · MCC · AUC · patient-level acc.\npatient-clustered bootstrap CIs\ncorrected resampled t-tests · calibration",
           fs=4.9, va="top")
    d.text(x + 21, 77, "Explanations vs. tumour masks", fs=5.5, va="top")
    d.image(x + 4.5, 55, 15, 15, png_from_array(A["img"][0], over=A["cam_base"][0], gt=A["gt"][0]))
    d.image(x + 22.5, 55, 15, 15, png_from_array(A["img"][0], over=A["cam_ours"][0], gt=A["gt"][0]))
    d.text(x + 12, 71, "SSPANet (same recipe)", fs=4.6, va="bottom"); d.text(x + 30, 71, "TG-SSPANet", fs=5.2, va="bottom")
    d.text(x + 21, 53.5, "Grad-CAM · Grad-CAM++ · EigenGrad-CAM\npointing game · energy-in-mask\nCAM-IoU · reference maps",
           fs=4.8, va="top")
    d.text(x + 21, 42.5, "Faithfulness, ablation & cost", fs=5.5, va="top")
    d.text(x + 21, 38.5, "sham / decoy inpainting · context rings ·\nlocation-only control · deletion/insertion ·\nablation · params · GMACs · latency", fs=4.9, va="top")
    for x, cl in [(1.5, ["G1 masks used", "G4 class weights"]), (46.5, ["G2 leakage-free P2"]),
                  (91.5, ["G3 multi-stage + ablation", "G5 SE/CBAM added"]), (136.5, ["G1 quantitative XAI", "G6 cost + variance"])]:
        for j, c in enumerate(cl):
            d.box(x + 2, 23.5 - j * 6, cw - 4, 4.5, c, RED_BG, RED, fs=5.4, color=RED)
    d.text(W / 2, 8.5, "Research gaps of [21] addressed (red): G1 unused tumour masks · G2 patient leakage ·\n"
           "G3 single-stage attention without ablation · G4 unhandled class imbalance · G5 SE/CBAM not evaluated ·\n"
           "G6 unmeasured cost and run-to-run variance", fs=5.2, color=RED)
    return d.save(OUT)


# ------------------------------------------------------------------------------------------------ Fig. 2
def fig_overall():
    W, H = 180, 118
    d = Diagram(W, H, "fig02_architecture_overall")
    i = 0; img, gt = A["img"][i], A["gt"][i]; yc = 86
    png = png_from_array(img)
    for k in (2, 1, 0):
        d.image(2 + k * 1.2, yc - 11 + k * 1.2, 22, 22, png)
    d.text(14, yc - 13.5, "Input T1-CE slice\n3 × 224 × 224", fs=5.8, va="top")
    d.cube(33, yc - 12, 3.5, 24, 12, BB, BB_E, "Stage 1\n256 × 56 × 56", above=True)
    d.cube(51, yc - 8.5, 4.5, 17, 8.5, BB, BB_E, "Stage 2\n512 × 28 × 28", above=True)
    d.cube(74, yc - 6, 5.5, 12, 6, BB, BB_E, "Stage 3\n1024 × 14 × 14", above=True)
    d.cube(97, yc - 4, 7, 8, 4, BB, BB_E, "Stage 4\n2048 × 7 × 7", above=True)
    T = [(61.3, yc - 6, 9, 12), (84.2, yc - 6, 9, 12), (107.2, yc - 6, 9, 12)]
    for (x, y, w, h), g in zip(T, A["gammas"]):
        d.box(x, y, w, h, f"TGSA\nγ={g:.3f}", ATT, ATT_E, fs=5.4)
    for a, b in [(25.5, 33), (42.5, 51), (60, 61.3), (70.3, 74), (82.5, 84.2), (93.2, 97), (106, 107.2), (116.2, 120)]:
        d.edge([(a, yc), (b, yc)])
    for (x, y, w, h) in T:
        d.edge([(x + w / 2, y + h), (x + w / 2, 104.5)], color=RED, dashed=True, lw=1.2)
    d.box(64, 104.5, 53, 5.2, "ℒ_{align} = BCE(spatial logits A_{s}, pooled mask)", RED_BG, RED, fs=5.6, color=RED)
    d.edge([(31, 112.5), (31, 114), (117, 114), (117, 112.5)], color=GREY, lw=1, arrow=False)
    d.text(74, 115, "Encoder: pre-trained ResNet50 / VGG16 + TGSA (Fig. 3)", fs=6.2, va="bottom")
    d.edge([(120, 112.5), (120, 114), (178, 114), (178, 112.5)], color=GREY, lw=1, arrow=False)
    d.text(149, 115, "Classification head (Fig. 4)", fs=6.2, va="bottom")
    d.node(122, yc, "⊙")
    d.cube(127, yc - 4, 7, 8, 4, HEAD, HEAD_E, "F″ (gated)\n2048 × 7 × 7")
    d.edge([(124, yc), (127, yc)])
    d.text(128.5, yc + 9.5, "Grad-CAM\ntarget", fs=5.2, va="bottom", color=HEAD_E, italic=True)
    d.box(139.5, yc - 9, 3, 18, "", HEAD, HEAD_E, r=0.4)
    d.text(141, yc - 10.5, "GAP\n2048", fs=5.8, va="top")
    d.edge([(136.2, yc), (139.5, yc)])
    prob = A["prob"][i]; bx, by = 152, yc - 7
    d.edge([(142.5, yc), (bx - 1, yc)])
    d.text(bx - 0.5, yc + 7.5, "FC + softmax", fs=5.8, ha="left", va="bottom")
    for j, (c, pv) in enumerate(zip(CLASSES, prob)):
        yy = by + (2 - j) * 4.8
        d.box(bx, yy, 24, 3.4, "", "#FFFFFF", "#999999", rounded=False, lw=0.8)
        d.box(bx, yy, max(24 * pv, 0.6), 3.4, "", HEAD_E if j == 1 else HEAD, HEAD_E, rounded=False, lw=0.8)
        d.text(bx + 0.8, yy + 1.7, f"{c}  {pv:.2f}", fs=5.0, ha="left", color="#FFFFFF" if j == 1 else "#000000")
    d.box(148, yc - 16.5, 30, 5, "ℒ_{cls}: CE, LS 0.1, weighted", RED_BG, RED, fs=5.3, color=RED)
    # decoder
    yd = 36
    dec = [(38, 16, 8), (62, 12, 6), (86, 9, 4.5), (104, 6, 3)]
    srcx = [34.75] + [t[0] + 4.5 for t in T]; srcy = [yc - 12] + [t[1] for t in T]
    for k, ((x, h, dd), sx, sy) in enumerate(zip(dec, srcx, srcy)):
        d.cube(x, yd - h / 2, 2.5, h, dd, DEC, DEC_E, f"P{k + 1}\n128 × {[56, 28, 14, 7][k]}²")
        top = yd + h / 2 + dd * 0.5
        # lateral 1×1 projection: into the ⊕ of P1–P3 (summed with the upsampled coarser map); P4 is the lateral alone
        tx, ty = (x + 2.5 + dd * 0.5 + 3.5, yd + 1.7) if k < 3 else (x + 1.25, top + 0.3)
        d.edge([(sx, sy), (sx, top + 6), (tx, top + 6), (tx, ty)], color=DEC_E, lw=1.2)
    d.text(37, 62, "lateral 1×1 conv → 128", fs=5.2, ha="left", color=DEC_E, italic=True)
    for k in (3, 2, 1):
        x0 = dec[k][0]; x1 = dec[k - 1][0] + 2.5 + dec[k - 1][2] * 0.5
        d.edge([(x0 - 0.3, yd), (x1 + 5.1, yd)], color=DEC_E)
        d.edge([(x1 + 1.9, yd), (x1 + 0.2, yd)], color=DEC_E)
        d.node(x1 + 3.5, yd, "⊕", r=1.6, fs=7)
        d.text((x0 + x1 + 5) / 2, yd + 1.2, "×2 ↑", fs=5.2, va="bottom", color=DEC_E)
    d.box(4, yd - 6, 25, 12, "2 × [Conv3×3–BN–ReLU]\nConv1×1 → L\n1 × 56 × 56", DEC, DEC_E, fs=5.3)
    d.edge([(37.8, yd), (29, yd)], color=DEC_E)
    d.text(3, yd + 7.5, "Mask-guided\nFPN decoder (Fig. 4)", fs=6.0, ha="left", va="bottom", color=DEC_E)
    d.image(5, 3, 21, 21, png_from_array(overlay_rgb(img, A["mask224"][i]), rgb=True, gt=gt))
    d.edge([(16.5, yd - 6), (16.5, 24.2)], color=DEC_E)
    d.text(27.5, 17.4, "predicted tumour mask (red)\nvs. ground truth (green)", fs=5.2, ha="left")
    d.box(27, 4, 27, 5, "ℒ_{seg} = BCE + Dice", RED_BG, RED, fs=5.6, color=RED)
    d.edge([(26.2, 13.5), (58, 13.5), (58, 8), (118, 8), (118, 52)], color=DEC_E, lw=1.2)
    d.text(88, 9, "σ(L) → max-pool (8×8 windows, stride 8) → p", fs=5.6, va="bottom", color=DEC_E)
    d.image(113, 52, 10, 10, png_from_array(A["p"][i], cmap="viridis", interp="nearest", size=140))
    d.text(124, 57, "p ∈ [0,1]\n1 × 7 × 7", fs=5.4, ha="left")
    d.edge([(118, 62.2), (121.3, yc - 2)], color=DEC_E)
    d.text(124.5, 69, "gate: 0.1 + 0.9·p", fs=5.4, ha="left", color=HEAD_E, italic=True)
    d.text(88, 2.2, "ℒ = ℒ_{cls} + 1.0 ℒ_{seg} + 0.2 ℒ_{align}", fs=6.8, va="bottom", color=RED)
    d.box(129.5, 12, 48.5, 34, "", "#FFFFFF", "#BBBBBB", lw=0.8)
    for k, (fc, ec, t) in enumerate([(BB, BB_E, "backbone stage"), (ATT, ATT_E, "TGSA attention"), (DEC, DEC_E, "decoder"),
                                      (HEAD, HEAD_E, "classification head")]):
        yy = 42 - k * 4.3
        d.box(132, yy - 1.3, 4, 2.6, "", fc, ec, rounded=False, lw=1)
        d.text(137.5, yy, t, fs=5.8, ha="left")
    d.edge([(132, 42 - 4 * 4.3), (136, 42 - 4 * 4.3)], color=RED, dashed=True, arrow=False, lw=1.6)
    d.text(137.5, 42 - 4 * 4.3, "training-only supervision", fs=5.8, ha="left")
    d.node(134, 42 - 5 * 4.3, "⊙", r=1.4, fs=6); d.text(137.5, 42 - 5 * 4.3, "element-wise product", fs=5.8, ha="left")
    d.node(134, 42 - 6 * 4.3, "⊕", r=1.4, fs=6); d.text(137.5, 42 - 6 * 4.3, "element-wise sum", fs=5.8, ha="left")
    return d.save(OUT)


# ------------------------------------------------------------------------------------------------ Fig. 3
def fig_tgsa():
    W, H = 180, 142
    d = Diagram(W, H, "fig03_tgsa_module", ymin=15)
    d.text(1, H - 1, "(a)", fs=9, ha="left", va="top")
    ym = 104
    d.box(16, 114, 78, 25, "", "#FEF5EC", ATT_E, dashed=True)
    d.text(17.5, 137.8, "Channel attention: tri-statistic pooling + ECA-style 1-D conv", fs=6.0, ha="left", va="top", color=ATT_E)
    d.cube(4, ym - 8, 3, 16, 7, BB, BB_E, "X\nC × H × W", above=True)
    for j, t in enumerate(["GAP (mean)", "GMP (max)", "GSP (std)"]):
        yy = 129.5 - j * 6
        d.edge([(10.9, ym + 4), (13.5, ym + 4), (13.5, yy), (19, yy)], lw=1.1)
        d.edge([(35, yy), (38, 123.5)], lw=1.1)
        d.box(19, yy - 2.2, 16, 4.4, t, ATT, ATT_E, fs=5.4)
    d.box(38, 119, 9, 9, "stack\n3 × C", "#FFFFFF", ATT_E, fs=5.4)
    d.box(50, 119, 20, 9, "Conv1D (k = ψ(C))\n+ BN", ATT, ATT_E, fs=5.4)
    d.edge([(47, 123.5), (50, 123.5)])
    d.edge([(70, 123.5), (71.5, 123.5)])
    d.node(73.5, 123.5, "σ", r=2, fs=7)
    d.edge([(75.5, 123.5), (77, 123.5)])
    d.box(77, 119, 12, 9, "M_{c}\nC × 1 × 1", "#FFFFFF", ATT_E, fs=5.4)
    d.edge([(83, 119), (83, ym + 2)])
    d.edge([(11, ym), (81, ym)])
    d.node(83, ym, "⊙")
    d.text(88, ym + 1.2, "X′ = M_{c} ⊙ X", fs=6.0, ha="left", va="bottom")
    d.edge([(5.5, ym - 8), (5.5, 55), (168.5, 55), (168.5, ym - 2)], lw=1.2)
    d.text(88, 55.8, "identity (residual) path", fs=5.8, va="bottom", italic=True)
    d.box(88, 59, 72, 40, "", "#EEF5FB", "#1F5F8B", dashed=True)
    d.text(96, 97.8, "Spatial attention: directional strips + style statistics", fs=6.0, ha="left", va="top", color="#1F5F8B")
    d.edge([(85, ym), (148, ym)])
    d.edge([(92, ym), (92, 88.5)], lw=1.2)
    d.edge([(90.5, ym), (90.5, 67.5), (93, 67.5)], lw=1.2)
    d.box(90, 80, 17, 8.5, "Conv1×1 C→m\nBN, ReLU", DATA, DATA_E, fs=5.4)
    d.text(98.5, 78.8, "m = max(16, C/16)", fs=4.8, va="top", italic=True)
    d.edge([(107, 86), (111, 91.5)], lw=1.1); d.edge([(107, 82.5), (111, 82)], lw=1.1)
    d.box(111, 88, 19, 7, "avg ‖ max over W\n→ 2m × H", DATA, DATA_E, fs=5.2)
    d.box(111, 78.5, 19, 7, "avg ‖ max over H\n→ 2m × W", DATA, DATA_E, fs=5.2)
    d.edge([(130, 91.5), (133, 91.5)]); d.edge([(130, 82), (133, 82)])
    d.box(133, 88, 17, 7, "GConv1D k=7\nBN, broadcast", DATA, DATA_E, fs=5.0)
    d.box(133, 78.5, 17, 7, "GConv1D k=7\nBN, broadcast", DATA, DATA_E, fs=5.0)
    d.box(93, 63.5, 24, 8, "mean ‖ std ‖ max over C\n→ 3 × H × W", DATA, DATA_E, fs=5.0)
    d.edge([(117, 67.5), (119.5, 67.5)])
    d.box(119.5, 63.5, 26.5, 8, "Conv3×3 (dil. 2)–BN–ReLU\nConv3×3–BN → m×H×W", DATA, DATA_E, fs=4.8)
    d.edge([(150, 91.5), (152.8, 77.4)], lw=1.1); d.edge([(150, 82), (152.4, 76.9)], lw=1.1)
    d.edge([(146, 69), (152.6, 74.8)], lw=1.1)
    d.node(154, 76, "⊕", r=1.8, fs=7)
    d.edge([(154, 74.2), (154, 67)])
    d.box(146.5, 60.5, 16.5, 6.5, "ReLU, Conv1×1\n→ A_{s} (1×H×W)", DATA, DATA_E, fs=4.7)
    d.edge([(163, 63.5), (165.5, 63.5), (165.5, 92), (151.5, 92), (150.6, ym - 2)], lw=1.2)
    d.text(166.5, 80, "σ → M_{s}", fs=5.6, ha="left")
    d.edge([(163, 62), (164.5, 60.5)], color=RED, dashed=True)
    d.box(164.5, 56.5, 14.5, 5.5, "ℒ_{align}", RED_BG, RED, fs=6, color=RED)
    d.node(150, ym, "⊙")
    d.edge([(152, ym), (155, ym)])
    d.box(155, ym - 3.5, 9, 7, "× γ\n(init 0)", "#FFFFFF", GREY, fs=5.4)
    d.edge([(164, ym), (166.5, ym)])
    d.node(168.5, ym, "⊕")
    d.edge([(170.5, ym), (172.5, ym)])
    d.cube(172.5, ym - 8, 3, 16, 3, ATT, ATT_E, "TGSA(X)\nC×H×W")
    d.text(W / 2, 48.5, "TGSA(X) = X + γ · M_{s} ⊙ (M_{c} ⊙ X),  M_{s} = σ(A_{s}),     k = ψ(C) = odd( |(log₂C + 1) / 2| )", fs=6.8)
    d.text(1, 44.3, "(b)", fs=9, ha="left", va="top")
    d.text(10, 43.6, "Learned TGSA spatial attention M_{s} = σ(A_{s}) of the trained ResNet50 TG-SSPANet on held-out test slices (P2 fold 1; green: ground truth)",
           fs=6.0, ha="left", va="top")
    iw, gap, gx = 13.2, 0.8, 3
    for i, nm in enumerate(["Glioma", "Meningioma", "Pituitary"]):
        x0 = gx + i * (4 * iw + 3 * gap + 4)
        img, gt = A["img"][i], A["gt"][i]
        tiles = [(None, "input"), (A["att2"][i], f"stage 2 (28²)\nγ={A['gammas'][0]:.3f}"),
                 (A["att3"][i], f"stage 3 (14²)\nγ={A['gammas'][1]:.3f}"), (A["att4"][i], f"stage 4 (7²)\nγ={A['gammas'][2]:.3f}")]
        for j, (att, t) in enumerate(tiles):
            xx = x0 + j * (iw + gap)
            png = png_from_array(img, gt=gt) if att is None else png_from_array(img, over=up(att), gt=gt, interp="nearest")
            d.image(xx, 22, iw, iw, png)
            d.text(xx + iw / 2, 22 + iw + 0.8, t, fs=4.8, va="bottom")
        d.text(x0 + 2 * (iw + gap) - gap / 2, 20, nm, fs=6.2, va="top")
    return d.save(OUT)


# ------------------------------------------------------------------------------------------------ Fig. 4
def fig_decoder():
    W, H = 180, 155
    d = Diagram(W, H, "fig04_decoder_tfp")
    d.text(1, H - 0.5, "(a)", fs=9, ha="left", va="top")
    lv = [(66, 16, 3.5, "F_{1}  256 × 56²"), (88, 12, 4.5, "F′_{2}  512 × 28²"), (106, 8, 5.5, "F′_{3}  1024 × 14²"),
          (121, 6, 7, "F′_{4}  2048 × 7²")]
    d.box(80, 88, 99, 50, "", "#F3F5FC", HEAD_E, dashed=True)
    d.text(81.5, 136.8, "Tumour-focused pooling (TFP) head", fs=6.4, ha="left", va="top", color=HEAD_E)
    d.edge([(24, 128.6), (24, 142), (144, 142), (144, 125.5), (145.6, 119.5)], lw=1.2)
    d.text(100, 142.6, "F′_{4} (after TGSA)", fs=5.6, va="bottom")
    for k, (y, h, w, lab) in enumerate(lv):
        d.cube(20, y, w, h, h * 0.5, BB if k == 0 else ATT, BB_E if k == 0 else ATT_E)
        d.text(18.5, y + h / 2, lab, fs=5.4, ha="right")
    d.text(24, 146.5, "encoder features", fs=5.8, italic=True)
    for k, (y, h, w, lab) in enumerate(lv):
        d.edge([(20 + w + h * 0.25 + 0.8, y + h / 2), (47.9 if k < 3 else 52, y + h / 2)], color=DEC_E)
        if k < 3:
            d.edge([(51.1, y + h / 2), (52, y + h / 2)], color=DEC_E)
            d.node(49.5, y + h / 2, "⊕", r=1.6, fs=7)
        d.cube(52, y, 2.5, h, h * 0.5, DEC, DEC_E)
        d.text(55 + h * 0.3, y + h + h * 0.25 + 1.0, f"P_{{{k + 1}}}  128 × {[56, 28, 14, 7][k]}²", fs=5.2, ha="left", va="bottom")
    d.text(37, 129.5, "lateral\n1×1 conv", fs=5.2, color=DEC_E, italic=True)
    for k in (3, 2, 1):
        y_from = lv[k][0]; y_to = lv[k - 1][0] + lv[k - 1][1] / 2 + 1.7
        # top-down path: leaves the P block above (its bottom face) and enters the ⊕ of the next finer level
        d.edge([(53.25, y_from), (53.25, y_from - 1.6), (49.5, y_from - 1.6), (49.5, y_to)], color=DEC_E)
        d.text(48.5, (y_from - 1.6 + y_to) / 2, "×2 ↑", fs=5.0, ha="right", color=DEC_E)
    d.text(53, 146.5, "top-down pathway", fs=5.8, italic=True)
    d.edge([(59, 74), (68, 74)], color=DEC_E)
    d.box(68, 67, 29, 13, "2 × [Conv3×3–BN–ReLU]\n(128)", DEC, DEC_E, fs=5.4)
    d.edge([(97, 73.5), (100, 73.5)], color=DEC_E)
    d.box(100, 67, 20, 13, "Conv1×1\n→ mask logits L\n1 × 56 × 56", DEC, DEC_E, fs=5.4)
    d.edge([(120, 73.5), (126, 73.5)], color=DEC_E)
    d.box(126, 67, 25, 13, "σ, bilinear ↑\n→ tumour mask M̂\n1 × 224 × 224", DEC, DEC_E, fs=5.4)
    d.edge([(151, 73.5), (155, 73.5)], color=RED, dashed=True)
    d.box(155, 70, 23, 7, "ℒ_{seg} = BCE + Dice", RED_BG, RED, fs=5.6, color=RED)
    d.edge([(110, 80), (110, 92)], color=DEC_E)
    d.box(100, 92, 34, 11, "σ(L) → max-pool\n(8×8 windows, stride 8)\n→ p ∈ [0,1]^{1×7×7}", HEAD, HEAD_E, fs=5.0)
    d.edge([(134, 97.5), (137, 97.5)])
    d.box(137, 92, 20, 11, "g = φ + (1 − φ)·p\nφ = 0.1", HEAD, HEAD_E, fs=5.3)
    d.edge([(147, 103), (147, 116)])
    d.node(147, 118, "⊙")
    d.edge([(149, 118), (152, 118)])
    d.cube(152, 114, 6, 8, 4, HEAD, HEAD_E)
    d.text(156, 112.5, "F″ = F′_{4} ⊙ g", fs=5.4, va="top")
    d.edge([(160, 118), (163, 118)])
    d.box(163, 110, 2.6, 16, "", HEAD, HEAD_E, r=0.3)
    d.text(164.3, 108.5, "GAP", fs=5.4, va="top")
    d.edge([(165.6, 118), (168, 118)])
    d.box(168, 113, 10, 10, "FC\nsoftmax", HEAD, HEAD_E, fs=5.4)
    d.text(110.5, 121, "Linear head in F″  ⇒  Grad-CAM on F″ equals the exact\nclass-activation map  Σ_{k} W_{ck} F″_{k} / 49,\n"
           "up to ReLU and min–max normalisation", fs=5.2, color=HEAD_E, italic=True)
    d.text(112, 109, "p pools probabilities, not logits:\nsmall tumours are not averaged away", fs=5.2, color=GREY)
    d.text(1, 61, "(b)", fs=9, ha="left", va="top")
    d.text(10, 60.3, "Walk-through on held-out test slices (P2 fold 1; green contour: ground truth; red: prediction)",
           fs=6.0, ha="left", va="top")
    cols = ["input", "decoder σ(L)\n56 × 56", "p = max-pool (8×8, stride 8)", "gate g = 0.1 + 0.9 p",
            "Grad-CAM++ on F″\nTG-SSPANet (ours)", "Grad-CAM++ (GAP)\nSSPANet, same recipe"]
    iw, x0, step = 21, 24, 26.5
    for r, (i, name) in enumerate([(0, "Glioma"), (2, "Pituitary")]):
        y = 28 - r * 26
        img, gt = A["img"][i], A["gt"][i]; g = 0.1 + 0.9 * A["p"][i]
        pngs = [png_from_array(img, gt=gt),
                png_from_array(img, over=np.kron(A["mask56"][i], np.ones((4, 4))), ocmap="Reds", oalpha=0.55, gt=gt),
                png_from_array(A["p"][i], cmap="viridis", interp="nearest"),
                png_from_array(g, cmap="viridis", interp="nearest"),
                png_from_array(img, over=A["cam_ours"][i], gt=gt), png_from_array(img, over=A["cam_base"][i], gt=gt)]
        for j, png in enumerate(pngs):
            xx = x0 + j * step
            d.image(xx, y, iw, iw, png)
            if r == 0:
                d.text(xx + iw / 2, y + iw + 1.0, cols[j], fs=5.3, va="bottom")
            if j < 5:
                d.edge([(xx + iw + 0.6, y + iw / 2), (xx + iw + 4.9, y + iw / 2)],
                       color="#999999" if j == 4 else "#333333", dashed=j == 4, lw=1.2)
        pr = A["prob"][i]; bp = A["bprob"][i]; lab = int(A["labels"][i])
        d.text(x0 - 1.5, y + iw / 2, f"{name}\nP(true class):\nours {pr[lab]:.2f}\nSSPANet {bp[lab]:.2f}", fs=5.0, ha="right")
    return d.save(OUT)


if __name__ == "__main__":
    for f in (fig_pipeline, fig_overall, fig_tgsa, fig_decoder):
        print("saved", f())
