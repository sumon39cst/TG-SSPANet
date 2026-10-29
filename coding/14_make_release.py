"""Build the reproducibility package (code, exact splits, metrics, per-image predictions) for deposit.

Output: E:/Life_Nero/release/TG-SSPANet_reproducibility.zip   (checkpoints are listed but not included: ~6 GB)
"""
import glob
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

import common as C

REL = os.path.join(C.ROOT, "release"); os.makedirs(REL, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))

README = """# TG-SSPANet reproducibility package

Code, data splits, metrics and per-image predictions for
"Tumour-Guided Attention with Mask-Supervised Pooling for Leakage-Aware and Explainable Brain Tumour Classification in MRI".

## Data
Figshare brain tumour dataset (Cheng et al.), https://doi.org/10.6084/m9.figshare.1512427 (not redistributed here).
Place the four zip files and cvind.mat in `dataset/`, unzip the .mat files into `dataset/raw/`, then run
`python coding/01_preprocess.py`.

## Splits
* `splits/p1_image_level_split.csv`   - P1 (image-level 80/10/10, seed 42), as in Hasan et al. (2026); patient-leaky.
* `splits/p2_patient_wise_folds.csv`  - P2 (official patient-wise 5-fold, cvind.mat) with the validation patients used per fold.
Each row: slice index (0-based, file id = index + 1), file id, patient id, label, split/fold assignment.

## Reproducing
```
python coding/run_experiments.py main ablation patient patient_fair revision round2
python coding/03_xai_eval.py --group main --tag random_seed0
python coding/11_revision_eval.py      # P2 localisation, reference maps, occlusion, deletion/insertion
python coding/16_round2_eval.py        # location-only features, temperature scaling
python coding/19_round3_eval.py        # matched-subset occlusion: sham, decoys, rings 0-60 px, positive controls
python coding/20_round3_outputs.py     # round-3 tables and figure
python coding/12_revision_stats.py     # P2 statistics, calibration, corrected tests
python coding/04_report.py             # P1 tables/figures
```
Environment: Python 3.12, PyTorch 2.5.1 + CUDA 12.1, torchvision 0.20.1 (see `environment.txt`).
Training is not bit-wise deterministic on GPU (cuDNN / atomic adds); reruns can differ by 1-2 test slices on P1.

## Contents
* `coding/`        all scripts
* `results/`       metrics.json + history.json + train_log.txt of every run; summary tables
* `predictions/`   per-image test probabilities of every run (CSV) and predicted tumour masks (NPZ)
* `checkpoints.txt` list of trained checkpoints (available from the authors / separate deposit)
"""


def main():
    data = C.load_data()
    tr, va, te = C.split_random(data["labels"])
    split = np.array([""] * len(data["labels"]), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    base = pd.DataFrame({"index": np.arange(len(split)), "file_id": data["ids"], "patient_id": data["pids"],
                         "label": [C.CLASSES[i] for i in data["labels"]]})
    os.makedirs(os.path.join(REL, "splits"), exist_ok=True)
    base.assign(p1_split=split).to_csv(os.path.join(REL, "splits", "p1_image_level_split.csv"), index=False)
    p2 = base.assign(official_fold=data["cvfold"])
    for k in range(1, 6):
        a, b, c = C.split_patient(data, k)
        col = np.array([""] * len(split), dtype=object); col[a], col[b], col[c] = "train", "val", "test"
        p2[f"fold{k}_role"] = col
    p2.to_csv(os.path.join(REL, "splits", "p2_patient_wise_folds.csv"), index=False)
    env = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    keep = [l for l in env.splitlines() if l.split("==")[0].lower() in
            {"torch", "torchvision", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "opencv-python", "h5py"}]
    open(os.path.join(REL, "environment.txt"), "w").write("python " + sys.version.split()[0] + "\n" + "\n".join(keep) + "\n")
    open(os.path.join(REL, "README.md"), "w", encoding="utf8").write(README)
    ck = sorted(os.path.basename(p) for p in glob.glob(os.path.join(C.CKPT, "*.pt")))
    open(os.path.join(REL, "checkpoints.txt"), "w").write("\n".join(ck) + "\n")
    zp = os.path.join(REL, "TG-SSPANet_reproducibility.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["README.md", "environment.txt", "checkpoints.txt"] + [os.path.join("splits", s) for s in os.listdir(os.path.join(REL, "splits"))]:
            z.write(os.path.join(REL, f), f)
        for f in glob.glob(os.path.join(HERE, "*.py")) + glob.glob(os.path.join(HERE, "*.ps1")):
            z.write(f, os.path.join("coding", os.path.basename(f)))
        for f in glob.glob(os.path.join(C.RESULT, "**", "*"), recursive=True):
            if os.path.isfile(f) and "checkpoints" not in f and os.path.splitext(f)[1] in (".json", ".csv", ".txt", ".md", ".npz"):
                z.write(f, os.path.join("results", os.path.relpath(f, C.RESULT)))
        for f in glob.glob(os.path.join(C.PRED, "**", "*"), recursive=True):
            if os.path.isfile(f) and os.path.splitext(f)[1] in (".csv", ".npz"):
                z.write(f, os.path.join("predictions", os.path.relpath(f, C.PRED)))
    print("wrote", zp, round(os.path.getsize(zp) / 1e6, 1), "MB;", len(ck), "checkpoints listed")


if __name__ == "__main__":
    main()
