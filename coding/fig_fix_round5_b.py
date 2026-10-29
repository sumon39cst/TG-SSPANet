"""Round-5: Fig. 11b categorical context axis; Fig. 14 reliability bin counts; 600 dpi."""
import io

P = r"E:\Life_Nero\coding\12_revision_stats.py"; s = io.open(P, encoding="utf8").read()
R = [('    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))\n    for r, c in zip(show, cols):',
      '    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))\n    for kk, (r, c) in enumerate(zip(show, cols)):'),
     ('        edges = np.linspace(0, 1, 11); mids, accs = [], []\n        for lo, hi in zip(edges[:-1], edges[1:]):\n            m = (conf > lo) & (conf <= hi)\n            if m.sum() >= 5:\n                mids.append(conf[m].mean()); accs.append(corr[m].mean())',
      '        edges = np.linspace(0, 1, 11); mids, accs, ns = [], [], []\n        for lo, hi in zip(edges[:-1], edges[1:]):\n            m = (conf > lo) & (conf <= hi)\n            if m.sum() >= 5:\n                mids.append(conf[m].mean()); accs.append(corr[m].mean()); ns.append(int(m.sum()))\n'
      '        cnt = np.histogram(conf, bins=edges)[0]; bw = 0.1 / (len(show) + 1)\n'
      '        ax[2].bar(edges[:-1] + 0.005 + kk * bw + bw / 2, np.maximum(cnt, 0.8), bw, color=c, label=NAMES[r])\n'
      '        if r == "resnet50_TG-SSPANet":\n            for xm, ya, n in zip(mids, accs, ns):\n'
      '                ax[0].annotate(f"n={n}", (xm, ya), textcoords="offset points", xytext=(0, -9), fontsize=4.6, color=c, ha="center")'),
     ('    ax[1].set_ylim(0, 0.1)\n', '    ax[1].set_ylim(0, 0.1)\n    ax[2].set_yscale("log"); ax[2].set_xlim(0.3, 1.0); ax[2].set_xlabel("confidence (10 equal-width bins)")\n'
      '    ax[2].set_ylabel("slices per bin (log)"); ax[2].set_title("(c) Bin counts", fontsize=8.5)\n'),
     ('"fig_calibration_p2.png"), dpi=400)', '"fig_calibration_p2.png"), dpi=600)'),
     ('ax[0].legend(fontsize=5.8, loc="upper left")', 'ax[0].legend(fontsize=5.0, loc="upper left")')]
for a, b in R:
    assert a in s, a[:60]; s = s.replace(a, b)
io.open(P, "w", encoding="utf8").write(s)
P = r"E:\Life_Nero\coding\20_round3_outputs.py"; s = io.open(P, encoding="utf8").read()
R = [('    xr = [0, 6, 15, 30, 60, 90]\n', '    xr = [0, 6, 15, 30, 60, 90]; xp = np.arange(len(xr))   # categorical, equally spaced positions\n'),
     ('axes[1].plot(xr, [', 'axes[1].plot(xp, ['),
     ('axes[1].set_xticks(xr)', 'axes[1].set_xticks(xp)'),
     ('"tissue kept around the tumour (px ≈ mm)"', '"tissue kept around the tumour (px ≈ mm; categorical axis)"'),
     ('"fig_occlusion_controls.png"), dpi=400)', '"fig_occlusion_controls.png"), dpi=600)')]
for a, b in R:
    assert a in s, a[:60]; s = s.replace(a, b)
io.open(P, "w", encoding="utf8").write(s)
print("ok")
