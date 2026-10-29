"""Determinism (Reviewer 1, comment 7): runs with identical configuration and seed on P1 (307 test slices).

TG-SSPANet: main/random_seed0, ablation/A6 (full), determinism rep1, rep2  -> 4 runs
ResNet50 (proposed recipe): ablation/A0, determinism rep1, rep2            -> 3 runs
"""
import itertools
import os

import numpy as np
import pandas as pd

import common as C

RUNS = {"TG-SSPANet": [("main", "resnet50_TG-SSPANet"), ("ablation", "A6_full-TG-SSPANet"),
                       ("determinism", "resnet50_TG-SSPANet_rep1"), ("determinism", "resnet50_TG-SSPANet_rep2")],
        "ResNet50": [("ablation", "A0_ResNet50"), ("determinism", "resnet50_none_ourrecipe_rep1"),
                     ("determinism", "resnet50_none_ourrecipe_rep2")]}


def main():
    rows = []
    for model, runs in RUNS.items():
        preds = []
        for g, n in runs:
            d = pd.read_csv(os.path.join(C.PRED, g, n, "test_predictions_random_seed0.csv")).sort_values("index")
            preds.append(d)
        accs = [float(d.correct.mean()) for d in preds]
        diffs = [int((a.pred_label.values != b.pred_label.values).sum()) for a, b in itertools.combinations(preds, 2)]
        pdiff = [float(np.abs(a[["p_meningioma", "p_glioma", "p_pituitary"]].values -
                              b[["p_meningioma", "p_glioma", "p_pituitary"]].values).max()) for a, b in itertools.combinations(preds, 2)]
        errs = [int((~d.correct).sum()) for d in preds]
        rows.append(dict(model=model, runs=len(runs), errors=errs, acc_min=min(accs), acc_max=max(accs),
                         pair_pred_diffs=diffs, max_prob_diff=max(pdiff)))
    df = pd.DataFrame(rows); print(df.to_string())
    df.to_csv(os.path.join(C.RESULT, "major_revision", "determinism.csv"), index=False)


if __name__ == "__main__":
    main()
