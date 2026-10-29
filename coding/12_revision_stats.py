"""Revision statistics from saved predictions (no models needed).

* P2 table for every model with all 5 folds: fold mean +- s.d., pooled slice-level accuracy with patient-clustered
  bootstrap 95% CI, patient-level accuracy (mean probability per patient) with bootstrap CI, macro-F1, kappa, MCC.
* Nadeau-Bengio corrected resampled t-test vs. TG-SSPANet (same backbone) with Holm correction.
* Calibration: ECE (15 bins), Brier score; selective prediction: AURC and accuracy at 90% / 80% coverage.
* Holm correction of the P1 tests (table_significance.csv).
* Per-error Dice of TG-SSPANet on P2 and clustering of errors within patients.
Outputs in result/revision/ (tables) and result/figures/ (calibration figure).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef

import common as C

OUT = os.path.join(C.RESULT, "revision"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(C.RESULT, "figures")
LABEL = {c: i for i, c in enumerate(C.CLASSES)}
NAMES = {
    "resnet50_none": "ResNet50, original recipe", "resnet50_sspanet": "ResNet50 + SSPANet, original recipe",
    "resnet50_none_ourrecipe": "ResNet50", "resnet50_sspanet_ourrecipe": "ResNet50 + SSPANet",
    "resnet50_se_ourrecipe": "ResNet50 + SE", "resnet50_cbam_ourrecipe": "ResNet50 + CBAM",
    "resnet50_TG-noTFP": "TG-SSPANet w/o TFP (Y-Net-style)", "resnet50_TG-floor0.5": "TG-SSPANet, φ = 0.5",
    "resnet50_TG-floor0.3": "TG-SSPANet, φ = 0.3", "resnet50_TG-SSPANet": "TG-SSPANet (ours), φ = 0.1",
    "resnet50_TG-oracleGT": "TG-SSPANet, GT-mask gate (oracle)", "resnet50_roicrop": "ROI-crop two-stage (predicted mask)",
    "resnet50_roicrop_gtcrop": "ROI-crop two-stage (GT mask, oracle)",
    "resnet50_maskonly": "Mask only (no image content)", "resnet50_coordinate_ourrecipe": "ResNet50 + Coordinate",
    "resnet50_gcnet_ourrecipe": "ResNet50 + GCNet", "resnet50_spnet_ourrecipe": "ResNet50 + SPNet",
    "resnet50_srmnet_ourrecipe": "ResNet50 + SRMNet",
    "vgg16_none": "VGG16, original recipe", "vgg16_sspanet": "VGG16 + SSPANet, original recipe",
    "vgg16_none_ourrecipe": "VGG16", "vgg16_sspanet_ourrecipe": "VGG16 + SSPANet", "vgg16_TG-SSPANet": "VGG16 TG-SSPANet (ours)",
}
B = 2000


def load_preds():
    out = {}
    for d in sorted(glob.glob(os.path.join(C.PRED, "patient", "*"))):
        run = os.path.basename(d)
        for suffix in ("", "_gtcrop"):
            fs = [os.path.join(d, f"test_predictions_patient_fold{k}_seed0{suffix}.csv") for k in range(1, 6)]
            if all(os.path.exists(f) for f in fs):
                df = pd.concat([pd.read_csv(f).assign(fold=k) for k, f in enumerate(fs, 1)], ignore_index=True)
                df["y"] = df.true_label.map(LABEL); df["yhat"] = df.pred_label.map(LABEL)
                out[run + suffix] = df
    return out


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def risk_coverage(conf, correct):
    o = np.argsort(-conf); c = correct[o].astype(float)
    cov = np.arange(1, len(c) + 1) / len(c); risk = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return cov, risk


def cluster_boot(df, fn, rng):
    pats = df.patient_id.unique(); groups = {p: g for p, g in df.groupby("patient_id")}
    vals = [fn(pd.concat([groups[p] for p in rng.choice(pats, len(pats), replace=True)])) for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def patient_level(df):
    P = df.groupby("patient_id")[["p_meningioma", "p_glioma", "p_pituitary"]].mean()
    y = df.groupby("patient_id").y.first()
    return (P.values.argmax(1) == y.values)


def main():
    data = C.load_data()
    preds = load_preds()
    rng = np.random.RandomState(0)
    ntr = np.mean([len(C.split_patient(data, k)[0]) for k in range(1, 6)])
    nte = np.mean([len(C.split_patient(data, k)[2]) for k in range(1, 6)])
    rows = []
    for run, df in preds.items():
        fa = df.groupby("fold").apply(lambda g: (g.y == g.yhat).mean())
        conf = df[["p_meningioma", "p_glioma", "p_pituitary"]].values.max(1); corr = (df.y == df.yhat).values
        cov, risk = risk_coverage(conf, corr)
        pl = patient_level(df)
        pats = df.patient_id.unique(); pl_s = pd.Series(pl, index=df.groupby("patient_id").y.first().index)
        pl_ci = np.percentile([pl_s.loc[rng.choice(pats, len(pats), replace=True)].mean() for _ in range(B)], [2.5, 97.5])
        ci = cluster_boot(df, lambda g: (g.y == g.yhat).mean(), rng)
        onehot = np.eye(3)[df.y.values]
        rows.append(dict(run=run, model=NAMES.get(run, run), backbone=run.split("_")[0], folds=df.fold.nunique(),
                         acc_mean=fa.mean(), acc_sd=fa.std(ddof=1), acc_pooled=corr.mean(), acc_ci_lo=ci[0], acc_ci_hi=ci[1],
                         patient_acc=pl.mean(), patient_ci_lo=pl_ci[0], patient_ci_hi=pl_ci[1], n_patients=len(pl),
                         f1_macro=f1_score(df.y, df.yhat, average="macro"), kappa=cohen_kappa_score(df.y, df.yhat),
                         mcc=matthews_corrcoef(df.y, df.yhat), ece=ece(conf, corr),
                         brier=np.mean(np.sum((df[["p_meningioma", "p_glioma", "p_pituitary"]].values - onehot) ** 2, 1)),
                         mean_conf=conf.mean(), aurc=np.trapezoid(risk, cov),
                         acc_at_90cov=1 - risk[int(0.9 * len(risk)) - 1], acc_at_80cov=1 - risk[int(0.8 * len(risk)) - 1]))
    T = pd.DataFrame(rows)
    # corrected resampled t-test vs TG-SSPANet of the same backbone, Holm within backbone
    tests = []
    for bb, ref in [("resnet50", "resnet50_TG-SSPANet"), ("vgg16", "vgg16_TG-SSPANet")]:
        if ref not in preds:
            continue
        a = preds[ref].groupby("fold").apply(lambda g: (g.y == g.yhat).mean())
        for run, df in preds.items():
            if run == ref or not run.startswith(bb) or "oracle" in run or "gtcrop" in run:
                continue
            b = df.groupby("fold").apply(lambda g: (g.y == g.yhat).mean())
            d = (a - b).values; k = len(d)
            var = np.var(d, ddof=1) * (1 / k + nte / ntr)
            t = d.mean() / np.sqrt(var) if var > 0 else np.nan
            p = 2 * stats.t.sf(abs(t), k - 1) if var > 0 else np.nan
            tests.append(dict(backbone=bb, vs=NAMES.get(run, run), mean_diff=d.mean(), folds_better=int((d > 0).sum()),
                              t_corrected=t, p_corrected=p))
    Tt = pd.DataFrame(tests)
    for bb, g in Tt.groupby("backbone"):
        order = g.p_corrected.sort_values().index; m = len(order); running = 0
        for rank, i in enumerate(order):
            running = max(running, min(1, (m - rank) * Tt.loc[i, "p_corrected"])); Tt.loc[i, "p_holm"] = running
    # pre-specified primary family (2 comparisons per backbone): same-recipe backbone and same-recipe SSPANet
    primary = {"resnet50": ["ResNet50", "ResNet50 + SSPANet"], "vgg16": ["VGG16", "VGG16 + SSPANet"]}
    Tt["primary"] = [v in primary[b] for b, v in zip(Tt.backbone, Tt.vs)]
    for bb, g in Tt[Tt.primary].groupby("backbone"):
        order = g.p_corrected.sort_values().index; m = len(order); running = 0
        for rank, i in enumerate(order):
            running = max(running, min(1, (m - rank) * Tt.loc[i, "p_corrected"])); Tt.loc[i, "p_holm_primary"] = running
    T.to_csv(os.path.join(OUT, "table_p2_full.csv"), index=False); Tt.to_csv(os.path.join(OUT, "table_p2_tests.csv"), index=False)
    # Holm for P1 tests
    s = pd.read_csv(os.path.join(C.RESULT, "tables", "table_significance.csv"))
    for col in ["t-test p", "McNemar p (seed0)"]:
        for bb, g in s.groupby("Backbone"):
            order = g[col].sort_values().index; m = len(order); running = 0
            for rank, i in enumerate(order):
                running = max(running, min(1, (m - rank) * s.loc[i, col])); s.loc[i, col + " (Holm)"] = running
    s.to_csv(os.path.join(OUT, "table_p1_significance_holm.csv"), index=False)
    # per-error Dice and clustering for TG-SSPANet on P2
    gt224 = F.interpolate(torch.from_numpy(data["masks"]).float()[:, None], size=(224, 224), mode="bilinear",
                          align_corners=False)[:, 0].numpy() > 0.5
    err = []
    df = preds.get("resnet50_TG-SSPANet")
    if df is not None:
        for k in range(1, 6):
            z = np.load(os.path.join(C.PRED, "patient", "resnet50_TG-SSPANet", f"test_masks_patient_fold{k}_seed0.npz"))
            mk = dict(zip(z["index"], z["mask"]))
            for _, r in df[(df.fold == k) & (df.y != df.yhat)].iterrows():
                pm = mk[r["index"]] > 127; g = gt224[r["index"]]
                dice = 2 * (pm & g).sum() / max(pm.sum() + g.sum(), 1)
                err.append(dict(fold=k, index=r["index"], file_id=r.file_id, patient=r.patient_id, true=r.true_label,
                                pred=r.pred_label, conf=max(r.p_meningioma, r.p_glioma, r.p_pituitary), dice=dice,
                                overlap_frac=(pm & g).sum() / max(g.sum(), 1)))
    E = pd.DataFrame(err); E.to_csv(os.path.join(OUT, "table_p2_errors_tg.csv"), index=False)
    # dice over all slices for reference
    # calibration / selective-prediction figure
    show = [r for r in ["resnet50_none", "resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe", "resnet50_TG-SSPANet"] if r in preds]
    cols = ["#95A5A6", "#5D6D7E", "#2E86C1", "#C0392B"]
    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5}):
        fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9))
        edges = np.linspace(0, 1, 11); bw = 0.1 / (len(show) + 1)
        for kk, (r, c) in enumerate(zip(show, cols)):
            d = preds[r]; conf = d[["p_meningioma", "p_glioma", "p_pituitary"]].values.max(1); corr = (d.y == d.yhat).values
            mids, accs = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 5:                                   # bins with < 5 slices are not plotted in (a)
                    mids.append(conf[m].mean()); accs.append(corr[m].mean())
            cnt = np.histogram(conf, bins=edges)[0]
            ax[2].bar(edges[:-1] + 0.005 + kk * bw + bw / 2, np.where(cnt > 0, cnt, np.nan), bw, color=c)
            e = T.set_index("run").loc[r, "ece"]
            ax[0].plot(mids, accs, "o-", color=c, ms=3, lw=1.2, label=f"{NAMES[r]} (ECE {e:.3f})")
            cov, risk = risk_coverage(conf, corr)
            ax[1].plot(cov, risk, color=c, lw=1.2)
            if r == "resnet50_TG-SSPANet":
                for lo, n in zip(edges[:-1], cnt):
                    if n > 0 and lo >= 0.3:
                        ax[2].text(lo + 0.05, n * 1.6, str(n), fontsize=5.2, color=c, ha="center", va="bottom")
        ax[0].plot([0, 1], [0, 1], ":", color="grey"); ax[0].set_xlabel("confidence"); ax[0].set_ylabel("accuracy")
        ax[0].set_title("(a) Reliability (P2, pooled folds)", fontsize=7.5)
        ax[0].set_xlim(0.3, 1.0); ax[0].set_ylim(0.3, 1.02)
        ax[1].set_xlabel("coverage"); ax[1].set_ylabel("selective risk (error rate)"); ax[1].set_title("(b) Risk–coverage", fontsize=7.5)
        ax[1].set_ylim(0, 0.1)
        ax[2].set_yscale("log"); ax[2].set_xlim(0.3, 1.0); ax[2].set_ylim(0.7, 2e4); ax[2].set_xlabel("confidence bin (width 0.1)")
        ax[2].set_ylabel("slices per bin (log scale)"); ax[2].set_title("(c) Slices per bin\n(numbers: TG-SSPANet)", fontsize=7.5)
        for a_ in ax:
            a_.grid(alpha=0.3); a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=6.3, frameon=False)
        fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(os.path.join(FIG, "fig_calibration_p2.png"), dpi=600); plt.close(fig)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T[["model", "folds", "acc_mean", "acc_sd", "acc_ci_lo", "acc_ci_hi", "patient_acc", "patient_ci_lo", "patient_ci_hi",
             "kappa", "ece", "aurc", "acc_at_90cov"]].round(4).to_string(index=False))
    print(Tt.round(4).to_string(index=False))
    if len(E):
        print("TG errors:", len(E), "median dice", E.dice.median().round(3), "overlap>0:", (E.overlap_frac > 0).mean().round(3),
              "patients:", E.patient.nunique(), "errors in patients with >1 error:", (E.groupby('patient').size() > 1).sum())


if __name__ == "__main__":
    main()
