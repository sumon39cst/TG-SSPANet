"""Major revision: supplementary figure with examples of the tile-shuffle and acquisition-shift conditions."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C

OUT = os.path.join(C.RESULT, "major_revision")
PAPER = os.path.join(C.ROOT, "paper", "figures")
COLS = ["original", "context shuffled", "removed + shuffled", "all shuffled", "noise 5%", "bias field 40%", "gamma 0.7",
        "gamma 1.5", "half resolution", "combined"]
TITLES = ["original", "context\nshuffled", "tumour removed\n+ shuffled", "all\nshuffled", "noise 5%", "bias field\n±40%",
          "gamma 0.7", "gamma 1.5", "half\nresolution", "combined"]


def main():
    ex = np.load(os.path.join(OUT, "examples.npy"), allow_pickle=True).item()
    data = C.load_data()
    fig, axes = plt.subplots(len(ex), len(COLS), figsize=(7.2, 0.8 * len(ex) + 0.55))
    for r, (i, conds) in enumerate(ex.items()):
        for c, k in enumerate(COLS):
            ax = axes[r, c]; ax.imshow(conds[k], cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.contour(data["masks"][i], levels=[0.5], colors="#2ECC71", linewidths=0.5)
                ax.set_ylabel(C.CLASSES[data["labels"][i]], fontsize=6.5)
            if r == 0:
                ax.set_title(TITLES[c], fontsize=5.6)
    fig.tight_layout(pad=0.2, w_pad=0.15, h_pad=0.15)
    for d in (OUT, PAPER):
        fig.savefig(os.path.join(d, "figS8_shuffle_shift_examples.png"), dpi=600)
    plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
