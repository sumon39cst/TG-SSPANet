# TG-SSPANet reproducibility package

**Trained checkpoints (209 models, 17 GB):** https://zenodo.org/records/23732120
**Paper:** `paper/TG-SSPANet_manuscript_life-4637166.pdf` (revised manuscript, Life, MDPI)

## Repository layout
* `coding/` - all code (preprocessing, training, evaluation, statistics, figures)
* `splits/` - exact P1 and P2 splits with patient identifiers
* `results/` - metrics, training logs, tables and figures of every run
* `predictions/` - per-image test predictions and predicted tumour masks
* `paper/` - manuscript
The checkpoints are too large for GitHub and are on Zenodo; the MRI data are not redistributed.

Code, exact data splits, metrics, per-image predictions and trained checkpoints for the article

"Tumour-Guided Attention with Mask-Supervised Pooling for Leakage-Aware and Explainable Brain Tumour Classification in MRI: A Reproducibility and Evaluation Study"
Life (MDPI), manuscript life-4637166.

## Files in this record
* `TG-SSPANet_reproducibility.zip` - code, splits, metrics, training logs, per-image predictions and predicted masks
* `checkpoints_P1_image-level_split.zip` - trained models on the P1 image-level split (74 files)
* `checkpoints_P2_patient-wise_part1.zip`, `checkpoints_P2_patient-wise_part2.zip` - trained models on the
  P2 patient-wise folds (135 files)
* `checkpoints_list.txt` - which checkpoint is in which archive

To use the checkpoints, unzip all three archives into `result/checkpoints/`.
File names follow `<experiment>__<model>__<initialisation/fold>.pt`.

## Data
The MRI data are not redistributed. Download the Figshare brain tumour dataset (Cheng et al.),
https://doi.org/10.6084/m9.figshare.1512427, place the four zip files and cvind.mat in `dataset/`,
unzip the .mat files into `dataset/raw/`, then run `python coding/01_preprocess.py`.

## Splits
* `splits/p1_image_level_split.csv`  - P1 (image-level 80/10/10, seed 42); patient-leaky, reproduces the original protocol.
* `splits/p2_patient_wise_folds.csv` - P2 (official patient-wise 5-fold, cvind.mat) with the validation patients per fold.
Each row: slice index (0-based, file id = index + 1), file id, patient id, label, split/fold assignment.

## Reproducing
```
python coding/run_experiments.py main ablation patient patient_fair revision round2
python coding/03_xai_eval.py --group main --tag random_seed0
python coding/11_revision_eval.py      # P2 localisation, reference maps, occlusion, deletion/insertion
python coding/16_round2_eval.py        # location-only features, temperature scaling
python coding/19_round3_eval.py        # matched-subset occlusion: sham, decoys, rings, positive controls
python coding/20_round3_outputs.py
python coding/21_major_revision_runs.py
python coding/22_major_revision_eval.py  # aggregation rules, tile-shuffle, simulated acquisition shifts
python coding/24_determinism.py
python coding/25_decoder_occlusion.py
python coding/26_paired_differences.py
python coding/12_revision_stats.py     # P2 statistics, calibration, corrected tests
python coding/04_report.py             # P1 tables/figures
```
Environment: Python 3.12, PyTorch 2.5.1 + CUDA 12.1, torchvision 0.20.1 (see `environment.txt`).
Training is not bit-wise deterministic on GPU; reruns can differ by 1-2 test slices.

## Licence
Code and files: CC BY 4.0. Please cite the article above when using this package.
