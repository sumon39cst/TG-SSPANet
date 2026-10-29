"""Major revision (Life, two reviewers): extra training runs.

* P2 component ablation (Reviewer 2, comment 1): every combination needed to separate TGSA, the attention-alignment
  loss, the segmentation decoder and tumour-focused pooling, ResNet50, proposed recipe, five patient-wise folds.
  Existing runs reused: resnet50_none_ourrecipe (nothing), resnet50_TG-noTFP (TGSA + align + decoder),
  resnet50_TG-SSPANet (full).
* Determinism (Reviewer 1, comment 7): repeated runs with an identical configuration and seed on P1.
"""
import sys

from run_experiments import run

P2_ABLATION = [
    ("resnet50_TG-tgsaNoAlign", ["--no_seg", "--w_att", "0"]),      # TGSA only, no supervision of its logits
    ("resnet50_TG-tgsaAlign", ["--no_seg"]),                        # TGSA + alignment loss, no decoder, no TFP
    ("resnet50_TG-decOnly", ["--stages", "", "--no_tfp"]),          # decoder only (joint segmentation), no TGSA
    ("resnet50_TG-decTFP", ["--stages", ""]),                       # decoder + TFP, no TGSA
    ("resnet50_TG-noAlign", ["--w_att", "0"]),                      # full model without the alignment loss
]


def p2_ablation():
    for fold in range(1, 6):
        for name, extra in P2_ABLATION:
            run(["--kind", "proposed", "--backbone", "resnet50", "--recipe", "ours", "--protocol", "patient",
                 "--fold", str(fold)] + extra, name, "patient", f"patient_fold{fold}_seed0")


def determinism():
    for rep in (1, 2):
        run(["--kind", "proposed", "--backbone", "resnet50", "--recipe", "ours", "--seed", "0"],
            f"resnet50_TG-SSPANet_rep{rep}", "determinism", "random_seed0")
        run(["--kind", "baseline", "--backbone", "resnet50", "--attn", "none", "--recipe", "ours", "--seed", "0"],
            f"resnet50_none_ourrecipe_rep{rep}", "determinism", "random_seed0")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["ablation", "determinism"]:
        {"ablation": p2_ablation, "determinism": determinism}[w]()
