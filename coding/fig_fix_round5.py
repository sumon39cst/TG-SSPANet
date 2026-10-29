"""Round-5 corrections to the schematic figures (07_paper_figures.py and 15_drawio_figures.py)."""
import io

P = r"E:\Life_Nero\coding\07_paper_figures.py"
s = io.open(P, encoding="utf8").read()
R = [
    ('ax.text(88, 9, "σ(L) → max-pool 7×7 → p"', 'ax.text(88, 9, "σ(L) → max-pool (8×8 windows, stride 8) → p"'),
    ('ax.text(124, 57, "p\\n1 × 7 × 7", fontsize=5.4, va="center")', 'ax.text(124, 57, "p ∈ [0,1]\\n1 × 7 × 7", fontsize=5.4, va="center")'),
    ('cuboid(ax, 172.5, ym - 8, 3, 16, 3, ATT, ATT_E, "Y\\nC × H × W")', 'cuboid(ax, 172.5, ym - 8, 3, 16, 3, ATT, ATT_E, "TGSA(X)\\nC × H × W")'),
    (r'r"$Y = X + \gamma\,\sigma(A_s)\odot(M_c\odot X)$,', r'r"$\mathrm{TGSA}(X) = X + \gamma\,M_s\odot(M_c\odot X)$,  $M_s=\sigma(A_s)$,'),
    ('"Learned TGSA spatial attention σ($A_s$) of the trained model on test slices (green: ground-truth tumour)"',
     '"Learned TGSA spatial attention $M_s=\\\\sigma(A_s)$ of the trained ResNet50 TG-SSPANet on held-out test slices (P2 fold 1; green: ground-truth tumour)"'),
    ('"Linear head in F″ ⇒ Grad-CAM on F″ equals the exact\\nclass-activation map  $\\\\sum_k W_{ck}F\'\'_k/49$"',
     '"Linear head in F″ ⇒ Grad-CAM on F″ equals the exact class-activation map\\n$\\\\sum_k W_{ck}F\'\'_k/49$, up to ReLU and min–max normalisation"'),
    ('"Walk-through on test slices with the trained model (green contour: ground truth; red: prediction)"',
     '"Walk-through on held-out test slices (P2 fold 1; green contour: ground truth; red: prediction)"'),
    ('"p = max-pool 7×7"', '"p = max-pool (8×8, stride 8)"'),
    ('"Grad-CAM++ (GAP)\\nSSPANet [1]"', '"Grad-CAM++ (GAP) SSPANet\\n(same recipe, P2 fold 1)"'),
    ('f"{name}\\nours p={pr[lab]:.2f}\\n[1] p={bp[lab]:.2f}"', 'f"{name}\\nP(true class):\\nours {pr[lab]:.2f}\\nSSPANet {bp[lab]:.2f}"'),
    ('"P1  image-level 80/10/10 (as in [1])"', '"P1  image-level 80/10/10 (as in [21])"'),
    ('"Reproduced baselines [1]"', '"Reproduced baselines [21]"'),
    ('cm = np.array([[70, 1, 0], [0, 143, 0], [0, 0, 93]])', 'cm = np.array([[658, 20, 30], [35, 1390, 1], [9, 3, 918]])   # TG-SSPANet, P2, pooled folds'),
    ('"TG-SSPANet\\nP1, seed 0"', '"TG-SSPANet\\nP2, 5 folds"'),
    ('"Acc · P · R · F1 · κ · MCC · AUC\\nmean ± s.d. (3 seeds / 5 folds)\\npaired t-test · McNemar"',
     '"Acc · F1 · κ · MCC · AUC · patient-level acc.\\npatient-clustered bootstrap CIs\\ncorrected resampled t-tests · calibration"'),
    ('title="SSPANet [1]", tfs=5.2)', 'title="SSPANet (same recipe)", tfs=5.2)'),
    ('"12 ablation variants · params ·\\nGMACs · latency · error analysis"',
     '"Faithfulness & controls: sham / decoy\\ninpainting · context rings · location-only\\n· deletion/insertion · ablation · cost"'),
    ('"Research gaps of [1] addressed (red):', '"Research gaps of [21] addressed (red):'),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
io.open(P, "w", encoding="utf8").write(s)
print("07 ok", len(R))
