# TG-SSPANet — reproduction of Hasan et al. (BSPC 112, 2026, 108636) + research-gap extension

Dataset: Figshare brain tumour (Cheng et al.), 3,064 T1-CE slices, 233 patients, meningioma / glioma / pituitary.
All numbers below are on the TEST set and were produced by the scripts in `E:\Life_Nero\coding`.

## Research gaps found in the paper and how they were addressed
| # | Gap in the paper | What this work does |
|---|---|---|
| G1 | States the dataset "lacks spatial annotations" — false: every .mat file has `tumorMask`. XAI judged only by eye. | Mask-supervised attention (L_align), FPN tumour-mask decoder, tumour-focused pooling (TFP); XAI scored quantitatively (pointing game, energy-in-mask, CAM-IoU). |
| G2 | Random image-level split of slices from 233 patients → the same patient appears in both training and test (leakage). | Additional leakage-free evaluation on the official patient-wise 5-fold split (`cvind.mat`). |
| G3 | One attention module at one position; no ablation. | Multi-stage TGSA + 12-variant ablation. |
| G4 | Class imbalance mentioned but not handled. | Class-weighted, label-smoothed loss. |
| G5 | SE / CBAM discussed theoretically, never tested. | Added as baselines. |
| G6 | No measured cost, single numbers, no variance. | Params, GMACs, latency; 3 seeds (mean ± sd); paired t-test, McNemar. |

## 1. Paper protocol (image-level 80/10/10 = 2450/307/307), mean ± sd over 3 seeds
| Model | Paper-reported Acc | This work Acc | Kappa | MCC |
|---|---|---|---|---|
| ResNet50 + SSPANet (paper's best) | 0.97 | 0.9848 ± 0.0068 | 0.9761 | 0.9763 |
| ResNet50 + SE (best reproduced baseline) | – | 0.9924 ± 0.0019 | 0.9881 | 0.9881 |
| **ResNet50 + TG-SSPANet (ours)** | – | **0.9967 ± 0.0033** | **0.9949** | **0.9949** |
| VGG16 + SSPANet (paper) | 0.93 | 0.9924 ± 0.0038 | 0.9881 | 0.9881 |
| **VGG16 + TG-SSPANet (ours)** | – | **0.9924 ± 0.0050** | 0.9881 | 0.9881 |

Full table: `tables/table_main_comparison.csv`; significance: `tables/table_significance.csv`.
Honest notes: (i) our reproduced baselines are far above the paper's reported values (ImageNet-pretrained backbones);
(ii) on this split accuracy saturates, so gaps between the top models are 1–2 test images and not statistically
significant with 3 seeds (paired t-test p ≥ 0.057); on VGG16 TG-SSPANet ties SRMNet/SSPANet.

## 2. Leakage-free patient-wise 5-fold CV (ResNet50)
| Model | Acc | Kappa | MCC | Folds ours better | paired t p |
|---|---|---|---|---|---|
| **TG-SSPANet (ours)** | **0.9682 ± 0.0095** | **0.9499** | **0.9503** | – | – |
| SSPANet, our recipe | 0.9582 ± 0.0150 | 0.9342 | 0.9348 | 4/5 | 0.247 |
| ResNet50, our recipe | 0.9561 ± 0.0185 | 0.9312 | 0.9317 | 3/5 | 0.188 |
| ResNet50, paper recipe | 0.9321 ± 0.0191 | 0.8924 | 0.8935 | 5/5 | 0.018 |
| SSPANet, paper recipe | 0.9197 ± 0.0248 | 0.8747 | 0.8761 | 5/5 | 0.010 |

## 3. Explainability vs ground-truth tumour masks (Grad-CAM++, seed 0)
| Backbone | Best baseline pointing game | SSPANet (paper) | TG-SSPANet (ours) |
|---|---|---|---|
| ResNet50 | 0.274 (SPNet) | 0.300 | **0.827** |
| VGG16 | 0.404 (SE) | 0.362 | **0.808** |

## 4. Ablation (ResNet50, same recipe, seed 0) — key finding
Accuracy is 99.35–100 % for every variant (saturated). Localisation is driven by the mask-guided TFP gate:
no TFP 0.130 → full model 0.844 pointing game; removing channel / strip / style branches of TGSA changes it by ≤ 0.01;
a higher gate floor trades localisation away (0.1 → 0.844, 0.3 → 0.769, 0.5 → 0.648). See `tables/table_ablation.csv`.

## Development history (kept for transparency)
`dev_v1_concatTFP`, `dev_v2_gate1plusp`, `dev_v3_final_design`: earlier head designs. v1/v2 reached 100 % but their
Grad-CAM pointed away from the tumour (context shortcut); v2 also had a pooling bug (logits instead of probabilities).
The final design was fixed before the main experiments; the test set was not used to pick the gate floor.

## Where things are
- `figures/diagrams/` pipeline + architecture figures (A overall, B TGSA module, C decoder/TFP head) — PNG/PDF/SVG
- `figures/` comparison bar, training curves, confusion matrices, ROC, ablation, XAI quantitative
- `figures/xai/` Grad-CAM / Grad-CAM++ / EigenGrad-CAM figures per model + comparison grids (paper Figs. 6–9 layout)
- `tables/` all CSV tables; each run: `<group>/<model>/<tag>/metrics.json, history.json, train_log.txt`
- `E:\Life_Nero\prediction\` per-image test predictions (CSV), predicted masks (NPZ), overlays for every test slice, `figures/`
