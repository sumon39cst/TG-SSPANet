"""Aggregate every run into tables + result figures (E:/Life_Nero/result/tables, E:/Life_Nero/result/figures)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc, roc_curve

import common as C

TAB = os.path.join(C.RESULT, "tables"); FIG = os.path.join(C.RESULT, "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Tables 2 and 3 of Hasan et al. (2026): accuracy, precision, recall, F1, Cohen kappa, MCC
PAPER = {
    "vgg16": {"none": [.89, .89, .89, .89, .82, .82], "coordinate": [.90, .90, .90, .90, .83, .83],
              "gcnet": [.55, .55, .55, .55, .00, .00], "spnet": [.89, .89, .89, .89, .81, .82],
              "srmnet": [.90, .90, .90, .90, .83, .83], "sspanet": [.93, .93, .93, .93, .88, .89]},
    "resnet50": {"none": [.90, .90, .90, .90, .83, .83], "coordinate": [.88, .88, .88, .88, .81, .81],
                 "gcnet": [.84, .84, .84, .84, .74, .74], "spnet": [.93, .93, .93, .93, .89, .89],
                 "srmnet": [.93, .93, .93, .93, .88, .88], "sspanet": [.97, .97, .97, .97, .95, .95]},
}
DISPLAY = {"none": "Baseline", "coordinate": "Coordinate", "gcnet": "GCNet", "spnet": "SPNet", "srmnet": "SRMNet",
           "se": "SE", "cbam": "CBAM", "sspanet": "SSPANet (paper)", "TG-SSPANet": "TG-SSPANet (ours)"}
ORDER = ["none", "coordinate", "gcnet", "spnet", "srmnet", "se", "cbam", "sspanet", "TG-SSPANet"]
BB = {"resnet50": "ResNet50", "vgg16": "VGG16"}
KEYS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "cohen_kappa", "mcc"]
KEYN = ["Acc", "Prec", "Rec", "F1", "Kappa", "MCC"]
OURS_C, BASE_C = "#C0392B", "#7F8C8D"


def load(group):
    rows = []
    for f in glob.glob(os.path.join(C.RESULT, group, "*", "*", "metrics.json")):
        m = json.load(open(f))
        m["run"] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        m["tag"] = os.path.basename(os.path.dirname(f))
        rows.append(m)
    return pd.DataFrame(rows)


def to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def split_name(run):
    bb, att = run.split("_", 1)
    return bb, att


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"


def mcnemar(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    return b, c, (stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0)


def pred_csv(group, run, tag):
    return pd.read_csv(os.path.join(C.PRED, group, run, f"test_predictions_{tag}.csv"))


# --------------------------------------------------------------------------------------------------
def main_tables(df):
    df = df[df.tag.str.startswith("random")].copy()
    df["bb"], df["att"] = zip(*df.run.map(split_name))
    out, sig = [], []
    for bb in ["vgg16", "resnet50"]:
        ours = df[(df.bb == bb) & (df.att == "TG-SSPANet")].sort_values("seed")
        for att in ORDER:
            d = df[(df.bb == bb) & (df.att == att)].sort_values("seed")
            if d.empty:
                continue
            r = {"Backbone": BB[bb], "Attention": DISPLAY[att], "n_seeds": len(d)}
            p = PAPER[bb].get(att)
            for k, n in zip(KEYS, KEYN):
                r[f"Paper {n}"] = p[KEYN.index(n)] if p else np.nan
            for k, n in zip(KEYS, KEYN):
                r[f"Ours-run {n}"] = ms(d[k])
                r[f"_{n}"] = d[k].mean()
            r["AUC"] = ms(d["auc_macro_ovr"])
            r["Params (M)"] = round(d.params_M.iloc[0], 2); r["GMACs"] = round(d.gflops.iloc[0], 2)
            r["ms/img"] = round(d.ms_per_image.iloc[0], 2)
            out.append(r)
            if att != "TG-SSPANet" and not ours.empty:
                common_seeds = sorted(set(ours.seed) & set(d.seed))
                a = ours.set_index("seed").loc[common_seeds, "accuracy"].values
                b = d.set_index("seed").loc[common_seeds, "accuracy"].values
                t, pv = stats.ttest_rel(a, b) if len(common_seeds) > 1 and np.any(a != b) else (np.nan, np.nan)
                po = pred_csv("main", f"{bb}_TG-SSPANet", "random_seed0"); pb = pred_csv("main", f"{bb}_{att}", "random_seed0")
                b_, c_, pm = mcnemar(po.correct.values, pb.correct.values)
                sig.append({"Backbone": BB[bb], "Ours vs": DISPLAY[att], "seeds": len(common_seeds),
                            "mean acc diff": a.mean() - b.mean(), "paired t": t, "t-test p": pv,
                            "McNemar ours-only-correct": b_, "McNemar baseline-only-correct": c_, "McNemar p (seed0)": pm})
    t = pd.DataFrame(out)
    t.drop(columns=[c for c in t.columns if c.startswith("_")]).to_csv(os.path.join(TAB, "table_main_comparison.csv"), index=False)
    with open(os.path.join(TAB, "table_main_comparison.md"), "w", encoding="utf8") as f:
        f.write(to_md(t.drop(columns=[c for c in t.columns if c.startswith("_")])))
    s = pd.DataFrame(sig); s.to_csv(os.path.join(TAB, "table_significance.csv"), index=False)
    return t, s, df


def fig_comparison(t):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, bb in zip(axes, ["VGG16", "ResNet50"]):
        d = t[t.Backbone == bb]
        labels = list(d.Attention)
        x = np.arange(len(labels)); w = 0.26
        paper = d["Paper Acc"].values
        ax.bar(x - w, np.nan_to_num(paper), w, label="Reported in paper", color="#BDC3C7")
        ax.bar(x, d["_Acc"], w, label="Accuracy (this work)", color=[OURS_C if "ours" in l else "#5D6D7E" for l in labels])
        ax.bar(x + w, d["_MCC"], w, label="MCC (this work)", color=[("#E6B0AA" if "ours" in l else "#AEB6BF") for l in labels])
        for i, v in enumerate(d["_Acc"]):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_title(bb)
        ax.axhline(0.97, ls="--", lw=0.8, color="k"); ax.text(len(x) - 0.5, 0.972, "paper best 0.97", ha="right", fontsize=7)
        ax.set_ylim(0.5, 1.04)
    axes[0].set_ylabel("score"); axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_comparison_bar.png"), dpi=300); fig.savefig(os.path.join(FIG, "fig_comparison_bar.pdf")); plt.close(fig)


def fig_curves(df):
    for bb in ["vgg16", "resnet50"]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        for att in ORDER:
            f = os.path.join(C.RESULT, "main", f"{bb}_{att}", "random_seed0", "history.json")
            if not os.path.exists(f):
                continue
            h = pd.DataFrame(json.load(open(f)))
            ours = att == "TG-SSPANet"
            kw = dict(lw=2.2 if ours else 1.1, color=OURS_C if ours else None, label=DISPLAY[att])
            axes[0, 0].plot(h.epoch, h.train_acc, **kw); axes[0, 1].plot(h.epoch, h.val_acc, **kw)
            axes[1, 0].plot(h.epoch, h.train_loss, **kw); axes[1, 1].plot(h.epoch, h.val_loss, **kw)
        for ax, t in zip(axes.flat, ["(a) Training accuracy", "(b) Validation accuracy", "(c) Training loss*", "(d) Validation loss (CE)"]):
            ax.set_title(t); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        axes[0, 1].set_ylim(0.8, 1.005); axes[0, 0].set_ylim(0.6, 1.005)
        axes[0, 1].legend(fontsize=7, ncol=2)
        fig.text(0.5, 0.005, "*TG-SSPANet training loss includes label smoothing, segmentation and alignment terms, so its absolute value is not comparable.",
                 ha="center", fontsize=7.5, style="italic")
        fig.suptitle(f"Accuracy and loss of attention networks with {BB[bb]} (seed 0)")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(os.path.join(FIG, f"fig_training_curves_{bb}.png"), dpi=300); fig.savefig(os.path.join(FIG, f"fig_training_curves_{bb}.pdf")); plt.close(fig)


def fig_confusion_roc():
    runs = [("resnet50_TG-SSPANet", "ResNet50 + TG-SSPANet (ours)"), ("vgg16_TG-SSPANet", "VGG16 + TG-SSPANet (ours)"),
            ("resnet50_sspanet", "ResNet50 + SSPANet (paper, reproduced)"), ("vgg16_sspanet", "VGG16 + SSPANet (paper, reproduced)")]
    runs = [r for r in runs if os.path.exists(os.path.join(C.PRED, "main", r[0], "test_predictions_random_seed0.csv"))]
    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 3.4))
    for ax, (r, lab) in zip(np.atleast_1d(axes), runs):
        p = pred_csv("main", r, "random_seed0")
        cm = pd.crosstab(pd.Categorical(p.true_label, C.CLASSES), pd.Categorical(p.pred_label, C.CLASSES), dropna=False).values
        ax.imshow(cm, cmap="Reds" if "ours" in lab else "Greys")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(C.CLASSES, rotation=30); ax.set_yticklabels(C.CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{lab}\nacc = {p.correct.mean():.4f}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_confusion_matrices.png"), dpi=300); fig.savefig(os.path.join(FIG, "fig_confusion_matrices.pdf")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, bb in zip(axes, ["resnet50", "vgg16"]):
        for r, lab, ls in [(f"{bb}_TG-SSPANet", "TG-SSPANet", "-"), (f"{bb}_sspanet", "SSPANet (paper)", "--")]:
            f = os.path.join(C.PRED, "main", r, "test_predictions_random_seed0.csv")
            if not os.path.exists(f):
                continue
            p = pd.read_csv(f)
            y = p.true_label.map({c: i for i, c in enumerate(C.CLASSES)}).values
            for c, col in zip(range(3), ["#1f77b4", "#d62728", "#2ca02c"]):
                fpr, tpr, _ = roc_curve(y == c, p[f"p_{C.CLASSES[c]}"])
                ax.plot(fpr, tpr, ls=ls, color=col, lw=1.4, label=f"{lab} - {C.CLASSES[c]} (AUC {auc(fpr, tpr):.4f})")
        ax.plot([0, 1], [0, 1], ":", color="grey"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC - {BB[bb]}")
        ax.legend(fontsize=6.5, loc="lower right")
        ins = ax.inset_axes([0.35, 0.35, 0.35, 0.3]); ins.set_xlim(0, 0.1); ins.set_ylim(0.9, 1.0)
        for line in ax.get_lines()[:-1]:
            ins.plot(*line.get_data(), ls=line.get_linestyle(), color=line.get_color(), lw=1)
        ins.set_xticks([0, 0.05, 0.1]); ins.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_roc_curves.png"), dpi=300); fig.savefig(os.path.join(FIG, "fig_roc_curves.pdf")); plt.close(fig)


def ablation():
    df = load("ablation")
    if df.empty:
        return None
    rows = []
    for run, d in df.groupby("run"):
        r = {"Variant": run, "n": len(d)}
        for k, n in zip(KEYS + ["auc_macro_ovr"], KEYN + ["AUC"]):
            r[n] = ms(d[k]); r["_" + n] = d[k].mean()
        r["Seg Dice"] = ms(d["seg_dice"]) if "seg_dice" in d and d["seg_dice"].notna().all() else "-"
        r["Params (M)"] = round(d.params_M.iloc[0], 2); r["GMACs"] = round(d.gflops.iloc[0], 2)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("Variant")
    xf = os.path.join(C.RESULT, "ablation", "xai_metrics_random_seed0.csv")
    if os.path.exists(xf):
        x = pd.read_csv(xf)
        x = x[x.method == "GradCAM++"].set_index("run")
        t["Grad-CAM++ pointing game"] = t.Variant.map(x.pointing_game).round(3)
        t["Grad-CAM++ energy-in-mask"] = t.Variant.map(x.energy_in_mask).round(3)
    t.drop(columns=[c for c in t.columns if c.startswith("_")]).to_csv(os.path.join(TAB, "table_ablation.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(t)); labels = [v.replace("_", " ", 1) for v in t.Variant]
    colors = [OURS_C if "full-TG" in v else "#5D6D7E" for v in t.Variant]
    axes[0].bar(x, t["_Acc"], color=colors)
    for i, v in enumerate(t["_Acc"]):
        axes[0].text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=6.5, rotation=90)
    axes[0].set_ylim(0.985, 1.004); axes[0].set_ylabel("test accuracy"); axes[0].set_title("(a) Classification accuracy (saturated)")
    if "Grad-CAM++ pointing game" in t:
        axes[1].bar(x, t["Grad-CAM++ pointing game"], color=colors)
        for i, v in enumerate(t["Grad-CAM++ pointing game"]):
            axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        axes[1].set_ylim(0, 1); axes[1].set_ylabel("pointing-game hit rate"); axes[1].set_title("(b) Grad-CAM++ localisation vs. tumour mask")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablation of TG-SSPANet (ResNet50, identical training recipe, seed 0)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ablation.png"), dpi=300); fig.savefig(os.path.join(FIG, "fig_ablation.pdf")); plt.close(fig)
    return t


def patient_cv():
    df = load("patient")
    if df.empty:
        return None
    rows = []
    ours = df[df.run == "resnet50_TG-SSPANet"].set_index("fold").accuracy
    names = {"resnet50_TG-SSPANet": "ResNet50 + TG-SSPANet (ours)", "resnet50_none": "ResNet50 (paper recipe)",
             "resnet50_sspanet": "ResNet50 + SSPANet (paper recipe)", "resnet50_none_ourrecipe": "ResNet50 (our recipe)",
             "resnet50_sspanet_ourrecipe": "ResNet50 + SSPANet (our recipe)"}
    for run, d in df.groupby("run"):
        r = {"Model": names.get(run, run), "folds": len(d)}
        for k, n in zip(KEYS + ["auc_macro_ovr"], KEYN + ["AUC"]):
            r[n] = ms(d[k]); r["_" + n] = d[k].mean()
        acc = d.set_index("fold").accuracy
        if run != "resnet50_TG-SSPANet" and len(acc) == len(ours) > 1:
            t_, p_ = stats.ttest_rel(ours.loc[acc.index], acc)
            r["folds ours better"] = int((ours.loc[acc.index] > acc).sum())
            r["paired t-test p (acc)"] = round(float(p_), 4)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("_Acc", ascending=False).drop(columns=[c for c in ["_" + n for n in KEYN + ["AUC"]]])
    t.to_csv(os.path.join(TAB, "table_patientwise_5fold.csv"), index=False)
    piv = df.pivot_table(index="fold", columns="run", values="accuracy")
    order = ["resnet50_sspanet", "resnet50_none", "resnet50_none_ourrecipe", "resnet50_sspanet_ourrecipe", "resnet50_TG-SSPANet"]
    order = [o for o in order if o in piv.columns]
    cols = ["#BDC3C7", "#95A5A6", "#7F8C8D", "#5D6D7E", OURS_C]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    w = 0.16
    for i, (o, c) in enumerate(zip(order, cols)):
        ax.bar(piv.index + (i - 2) * w, piv[o], w, color=c, label=f"{names[o]}: {piv[o].mean():.4f}")
    ax.set_xticks(piv.index); ax.set_xticklabels([f"Fold {f}" for f in piv.index]); ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("test accuracy"); ax.set_title("Patient-wise 5-fold cross-validation (no patient overlap), legend = mean accuracy")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1)); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_patientwise_folds.png"), dpi=300); fig.savefig(os.path.join(FIG, "fig_patientwise_folds.pdf")); plt.close(fig)
    return t


def xai_table():
    f = os.path.join(C.RESULT, "main", "xai_metrics_random_seed0.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d.to_csv(os.path.join(TAB, "table_xai_localisation.csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, met, name in zip(axes, ["pointing_game", "energy_in_mask", "cam_iou"], ["Pointing game (hit rate)", "Energy in tumour mask", "CAM IoU@0.5"]):
        xs, labs, pos = [], [], 0
        for bb in ["resnet50", "vgg16"]:
            s = d[(d.backbone == bb) & (d.method == "GradCAM++")]
            x = np.arange(len(s)) + pos
            ax.bar(x, s[met], color=[OURS_C if "ours" in m else BASE_C for m in s.model])
            for xi, v in zip(x, s[met]):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
            xs += list(x); labs += [f"{BB[bb]}: {m}" for m in s.model]; pos += len(s) + 1
        ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=6.5)
        ax.set_title(name + " - Grad-CAM++"); ax.set_ylim(0, 1.1)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_xai_quantitative.png"), dpi=300); fig.savefig(os.path.join(FIG, "fig_xai_quantitative.pdf")); plt.close(fig)
    return d


if __name__ == "__main__":
    t, s, df = main_tables(load("main"))
    fig_comparison(t); fig_curves(df); fig_confusion_roc()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    cols = ["Backbone", "Attention", "n_seeds"] + [f"Paper {n}" for n in KEYN] + [f"Ours-run {n}" for n in ["Acc", "F1", "Kappa", "MCC"]] + ["Params (M)", "GMACs"]
    print(t[cols].to_string(index=False))
    print(s.to_string(index=False))
    a = ablation()
    if a is not None:
        print(a.drop(columns=[c for c in a.columns if c.startswith("_")]).to_string(index=False))
    p = patient_cv()
    if p is not None:
        print(p.to_string(index=False))
    x = xai_table()
    if x is not None:
        print(x.to_string(index=False))
