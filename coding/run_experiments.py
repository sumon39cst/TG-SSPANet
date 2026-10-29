"""Experiment queue (resumable: runs whose metrics.json exists are skipped).

python run_experiments.py main       # paper reproduction (8 attention settings x 2 backbones) + proposed, seeds 0-2
python run_experiments.py ablation   # component ablation of TG-SSPANet (ResNet50, our recipe)
python run_experiments.py patient    # leakage-free patient-wise 5-fold CV
"""
import os
import subprocess
import sys

import common as C

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ATTNS = ["none", "coordinate", "gcnet", "spnet", "srmnet", "se", "cbam", "sspanet"]


def run(args, name, group, tag):
    if os.path.exists(os.path.join(C.RESULT, group, name, tag, "metrics.json")):
        print("skip", group, name, tag, flush=True); return
    cmd = [PY, os.path.join(HERE, "02_train.py"), "--group", group, "--name", name] + args
    print(" ".join(cmd), flush=True)
    for attempt in range(3):  # retry transient native crashes (e.g. 0xC0000409 from the CUDA driver)
        if subprocess.run(cmd).returncode == 0:
            return
        print(f"attempt {attempt + 1} failed, retrying", flush=True)
    raise RuntimeError(f"run failed 3 times: {group}/{name}/{tag}")


def main_exp(seeds=(0, 1, 2)):
    for seed in seeds:
        for bb in ["resnet50", "vgg16"]:
            run(["--kind", "proposed", "--backbone", bb, "--recipe", "ours", "--seed", str(seed)],
                f"{bb}_TG-SSPANet", "main", f"random_seed{seed}")
            for at in ATTNS:
                run(["--kind", "baseline", "--backbone", bb, "--attn", at, "--recipe", "paper", "--seed", str(seed)],
                    f"{bb}_{at}", "main", f"random_seed{seed}")


ABLATION = [
    # name, extra args  (all: ResNet50, recipe "ours", random split, seed 0)
    ("A0_ResNet50", ["--kind", "baseline", "--attn", "none"]),
    ("A1_ResNet50+SSPANet(paper)", ["--kind", "baseline", "--attn", "sspanet"]),
    ("A2_TGSA-last-stage", ["--kind", "proposed", "--stages", "3", "--no_seg", "--w_att", "0"]),
    ("A3_TGSA-multi-stage", ["--kind", "proposed", "--no_seg", "--w_att", "0"]),
    ("A4_+att-alignment", ["--kind", "proposed", "--no_seg"]),
    ("A5_+seg-decoder(no TFP)", ["--kind", "proposed", "--no_tfp"]),
    ("A6_full-TG-SSPANet", ["--kind", "proposed"]),
    ("B1_full-no-channel", ["--kind", "proposed", "--no_channel"]),
    ("B2_full-no-strip", ["--kind", "proposed", "--no_strip"]),
    ("B3_full-no-style", ["--kind", "proposed", "--no_style"]),
    ("C1_gate-floor-0.3", ["--kind", "proposed", "--floor", "0.3"]),
    ("C2_gate-floor-0.5", ["--kind", "proposed", "--floor", "0.5"]),
]


def ablation_exp(seeds=(0,)):
    for seed in seeds:
        for name, extra in ABLATION:
            run(extra + ["--backbone", "resnet50", "--recipe", "ours", "--seed", str(seed)], name, "ablation", f"random_seed{seed}")


def patient_exp():
    for fold in range(1, 6):
        run(["--kind", "proposed", "--backbone", "resnet50", "--recipe", "ours", "--protocol", "patient", "--fold", str(fold)],
            "resnet50_TG-SSPANet", "patient", f"patient_fold{fold}_seed0")
        for at in ["none", "sspanet"]:
            run(["--kind", "baseline", "--backbone", "resnet50", "--attn", at, "--recipe", "paper", "--protocol", "patient",
                 "--fold", str(fold)], f"resnet50_{at}", "patient", f"patient_fold{fold}_seed0")


def patient_fair_exp():
    """Same-recipe control for the patient-wise CV: baselines trained with the proposed recipe."""
    for fold in range(1, 6):
        for at in ["none", "sspanet"]:
            run(["--kind", "baseline", "--backbone", "resnet50", "--attn", at, "--recipe", "ours", "--protocol", "patient",
                 "--fold", str(fold)], f"resnet50_{at}_ourrecipe", "patient", f"patient_fold{fold}_seed0")


def revision_exp():
    """Major-revision experiments, all on the leakage-free patient-wise folds (P2)."""
    P = ["--protocol", "patient"]
    for fold in range(1, 6):
        f = ["--fold", str(fold)]; tag = f"patient_fold{fold}_seed0"
        rn = ["--kind", "proposed", "--backbone", "resnet50", "--recipe", "ours"] + P + f
        run(rn + ["--no_tfp"], "resnet50_TG-noTFP", "patient", tag)                 # A5 / Y-Net-style joint learning
        run(rn + ["--floor", "0.3"], "resnet50_TG-floor0.3", "patient", tag)
        run(rn + ["--floor", "0.5"], "resnet50_TG-floor0.5", "patient", tag)
        run(rn + ["--oracle"], "resnet50_TG-oracleGT", "patient", tag)              # ground-truth-mask gate (upper bound)
        if not os.path.exists(os.path.join(C.RESULT, "patient", "resnet50_roicrop", tag, "metrics.json")):
            for attempt in range(3):
                if subprocess.run([PY, os.path.join(HERE, "10_roi_crop.py"), "--fold", str(fold)]).returncode == 0:
                    break
    for fold in range(1, 6):
        f = ["--fold", str(fold)]; tag = f"patient_fold{fold}_seed0"
        for at in ["se", "cbam"]:
            run(["--kind", "baseline", "--backbone", "resnet50", "--attn", at, "--recipe", "ours"] + P + f,
                f"resnet50_{at}_ourrecipe", "patient", tag)
    for fold in range(1, 6):
        f = ["--fold", str(fold)]; tag = f"patient_fold{fold}_seed0"
        run(["--kind", "proposed", "--backbone", "vgg16", "--recipe", "ours"] + P + f, "vgg16_TG-SSPANet", "patient", tag)
        for at in ["none", "sspanet"]:
            run(["--kind", "baseline", "--backbone", "vgg16", "--attn", at, "--recipe", "ours"] + P + f,
                f"vgg16_{at}_ourrecipe", "patient", tag)
            run(["--kind", "baseline", "--backbone", "vgg16", "--attn", at, "--recipe", "paper"] + P + f,
                f"vgg16_{at}", "patient", tag)


def round2_exp():
    """Second-round review: location-only control and the remaining attention modules under the same recipe on P2."""
    for fold in range(1, 6):
        if not os.path.exists(os.path.join(C.RESULT, "patient", "resnet50_maskonly", f"patient_fold{fold}_seed0", "metrics.json")):
            for attempt in range(3):
                if subprocess.run([PY, os.path.join(HERE, "17_mask_only.py"), "--fold", str(fold)]).returncode == 0:
                    break
    for fold in range(1, 6):
        for at in ["coordinate", "gcnet", "spnet", "srmnet"]:
            run(["--kind", "baseline", "--backbone", "resnet50", "--attn", at, "--recipe", "ours", "--protocol", "patient",
                 "--fold", str(fold)], f"resnet50_{at}_ourrecipe", "patient", f"patient_fold{fold}_seed0")


if __name__ == "__main__":
    what = sys.argv[1:] or ["main"]
    for w in what:
        {"main": main_exp, "main1": lambda: main_exp((0,)), "ablation": ablation_exp, "patient": patient_exp,
         "patient_fair": patient_fair_exp, "revision": revision_exp, "round2": round2_exp}[w]()
