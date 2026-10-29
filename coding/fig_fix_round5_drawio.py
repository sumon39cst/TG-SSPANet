"""Round-5 corrections to the draw.io diagrams (15_drawio_figures.py) and the matplotlib Fig. 4 TFP box."""
import io

P = r"E:\Life_Nero\coding\15_drawio_figures.py"
s = io.open(P, encoding="utf8").read()
R = [
    ('cm = np.array([[70, 1, 0], [0, 143, 0], [0, 0, 93]])', 'cm = np.array([[658, 20, 30], [35, 1390, 1], [9, 3, 918]])   # TG-SSPANet, P2, pooled folds'),
    ('"TG-SSPANet\\nP1, seed 0"', '"TG-SSPANet\\nP2, 5 folds"'),
    ('"Acc · P · R · F1 · κ · MCC · AUC\\npatient-level accuracy, bootstrap CIs\\ncorrected t-tests · calibration"',
     '"Acc · F1 · κ · MCC · AUC · patient-level acc.\\npatient-clustered bootstrap CIs\\ncorrected resampled t-tests · calibration"'),
    ('d.text(x + 12, 71, "SSPANet", fs=5.2', 'd.text(x + 12, 71, "SSPANet (same recipe)", fs=4.6'),
    ('"occlusion · deletion / insertion\\nablations · params · GMACs · latency"',
     '"sham / decoy inpainting · context rings ·\\nlocation-only control · deletion/insertion ·\\nablation · params · GMACs · latency"'),
    ('"σ(L) → max-pool 7×7 → p"', '"σ(L) → max-pool (8×8 windows, stride 8) → p"'),
    ('"p\\n1 × 7 × 7"', '"p ∈ [0,1]\\n1 × 7 × 7"'),
    ('"Y\\nC×H×W"', '"TGSA(X)\\nC×H×W"'),
    ('"Y = X + γ · σ(A_{s}) ⊙ (M_{c} ⊙ X),     k', '"TGSA(X) = X + γ · M_{s} ⊙ (M_{c} ⊙ X),  M_{s} = σ(A_{s}),     k'),
    ('"Learned TGSA spatial attention σ(A_{s}) of the trained model on test slices (green: ground-truth tumour)"',
     '"Learned TGSA spatial attention M_{s} = σ(A_{s}) of the trained ResNet50 TG-SSPANet on held-out test slices (P2 fold 1; green: ground truth)"'),
    ('"σ(L) → AdaptiveMaxPool 7×7\\n(8×8 windows, stride 8)\\n→ p ∈ [0,1]^{1×7×7}"',
     '"σ(L) → max-pool\\n(8×8 windows, stride 8)\\n→ p ∈ [0,1]^{1×7×7}"'),
    ('"Linear head in F″  ⇒  Grad-CAM on F″ equals the exact\\nclass-activation map  Σ_{k} W_{ck} F″_{k} / 49"',
     '"Linear head in F″  ⇒  Grad-CAM on F″ equals the exact class-activation\\nmap  Σ_{k} W_{ck} F″_{k} / 49, up to ReLU and min–max normalisation"'),
    ('"Walk-through on test slices with the trained model (green contour: ground truth; red: prediction)"',
     '"Walk-through on held-out test slices (P2 fold 1; green contour: ground truth; red: prediction)"'),
    ('"p = max-pool 7×7"', '"p = max-pool (8×8, stride 8)"'),
    ('"Grad-CAM++ (GAP)\\nSSPANet"]', '"Grad-CAM++ (GAP) SSPANet\\n(same recipe, P2 fold 1)"]'),
    ('f"{name}\\nours p = {pr[lab]:.2f}\\nSSPANet p = {bp[lab]:.2f}"', 'f"{name}\\nP(true class):\\nours {pr[lab]:.2f}\\nSSPANet {bp[lab]:.2f}"'),
    ('"Research gaps of the original study addressed', '"Research gaps of [21] addressed'),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
io.open(P, "w", encoding="utf8").write(s)

P = r"E:\Life_Nero\coding\07_paper_figures.py"
s = io.open(P, encoding="utf8").read()
a = '"σ(L) → AdaptiveMaxPool 7×7\\n→ p $\\\\in [0,1]^{1\\\\times7\\\\times7}$"'
assert a in s
s = s.replace(a, '"σ(L) → max-pool (8×8, stride 8)\\n→ p $\\\\in [0,1]^{1\\\\times7\\\\times7}$"')
io.open(P, "w", encoding="utf8").write(s)
print("ok", len(R) + 1)
