"""Reviewer 3, comment 7: paired accuracy differences with 95% confidence intervals (P2, fold-paired).

CI of the mean fold difference with the Nadeau-Bengio corrected variance:
    se = sqrt((1/K + n_test/n_train) * var(d)),  CI = mean +- t(0.975, K-1) * se,   K = 5 folds.
Output: result/major_revision/paired_differences.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import common as C

R = os.path.join(C.RESULT, "patient")
PAIRS = [  # (model, reference)
    ("resnet50_coordinate_ourrecipe", "resnet50_none_ourrecipe"), ("resnet50_gcnet_ourrecipe", "resnet50_none_ourrecipe"),
    ("resnet50_spnet_ourrecipe", "resnet50_none_ourrecipe"), ("resnet50_srmnet_ourrecipe", "resnet50_none_ourrecipe"),
    ("resnet50_se_ourrecipe", "resnet50_none_ourrecipe"), ("resnet50_cbam_ourrecipe", "resnet50_none_ourrecipe"),
    ("resnet50_sspanet_ourrecipe", "resnet50_none_ourrecipe"), ("vgg16_sspanet_ourrecipe", "vgg16_none_ourrecipe"),
    ("resnet50_TG-SSPANet", "resnet50_none_ourrecipe"), ("resnet50_TG-SSPANet", "resnet50_sspanet_ourrecipe"),
    ("resnet50_TG-SSPANet", "resnet50_srmnet_ourrecipe"), ("resnet50_TG-SSPANet", "resnet50_TG-noTFP"),
    ("resnet50_TG-SSPANet", "resnet50_roicrop"), ("resnet50_TG-SSPANet", "resnet50_TG-decTFP"),
    ("vgg16_TG-SSPANet", "vgg16_none_ourrecipe"), ("vgg16_TG-SSPANet", "vgg16_sspanet_ourrecipe")]


def fold_stats(name):
    acc, ntr, nte = [], [], []
    for k in range(1, 6):
        mj = json.load(open(os.path.join(R, name, f"patient_fold{k}_seed0", "metrics.json")))
        acc.append(mj["accuracy"])
    return np.array(acc)


def main():
    data = C.load_data()
    ratio = np.mean([len(C.split_patient(data, k)[2]) / len(C.split_patient(data, k)[0]) for k in range(1, 6)])
    rows = []
    for m, ref in PAIRS:
        d = (fold_stats(m) - fold_stats(ref)) * 100
        K = len(d); se = np.sqrt((1 / K + ratio) * d.var(ddof=1)); t = stats.t.ppf(0.975, K - 1)
        p = 2 * stats.t.sf(abs(d.mean() / se), K - 1) if se > 0 else 1.0
        rows.append(dict(model=m, reference=ref, mean_diff=d.mean(), ci_lo=d.mean() - t * se, ci_hi=d.mean() + t * se,
                         p=p, folds_better=int((d > 0).sum()), fold_diffs=" ".join(f"{v:+.2f}" for v in d)))
    df = pd.DataFrame(rows)
    att = df.index[:7]                                          # the seven ResNet50 attention modules vs the plain backbone
    order = df.loc[att, "p"].sort_values().index; adj, run = {}, 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * df.loc[i, "p"])); adj[i] = run
    df["p_holm_modules"] = pd.Series(adj)
    df.to_csv(os.path.join(C.RESULT, "major_revision", "paired_differences.csv"), index=False)
    print("n_test/n_train", round(ratio, 3)); print(df.round(2).to_string())


if __name__ == "__main__":
    main()
