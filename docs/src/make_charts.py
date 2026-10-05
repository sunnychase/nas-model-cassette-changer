"""Regenerate docs/images/fit-chart.png and docs/images/demo-library.png (needs matplotlib)."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "images")
sys.path.insert(0, os.path.join(HERE, "..", "..", "deck")); import mcc_deck as D

BG, FG, DIM, GRID = "#0a0d13", "#e6edf3", "#8b95a5", "#1f2633"
TC = {1: "#22c55e", 2: "#38b6ff", 3: "#a78bfa", 4: "#f59e0b", None: "#ef4444"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
                     "xtick.color": DIM, "ytick.color": DIM, "font.size": 11, "axes.titleweight": "bold"})

# 1 — how many nodes a unit needs, by size
F = D.DEFAULTS["fit"]; xs = [x / 2 for x in range(1, 2 * 460)]
fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
prev, start = None, 0.5
for x in xs + [None]:
    k = D.nodes_needed(x * 1e9, F) if x else "end"
    if k != prev and prev is not None or x is None:
        end = x or xs[-1]
        ax.axvspan(start, end, color=TC[prev], alpha=.16, lw=0)
        ax.text((start + end) / 2, 0.6, f"{prev} node{'s' if prev and prev > 1 else ''}" if prev else "beyond", ha="center", color=TC[prev], fontsize=11, weight="bold")
        start = end
    prev = k
ys = [D.nodes_needed(x * 1e9, F) or 5 for x in xs]
ax.step(xs, ys, where="post", color=FG, lw=2)
for name, gb, dy in [("gpt-oss-20b MXFP4", 12.1, 70), ("Qwen3-32B Q4_K_M", 19.8, 50), ("Llama-3.3-70B Q4_K_M", 42.5, 30), ("GLM-4.5-Air Q4_K_M", 72.9, 12),
                     ("Llama-3.3-70B BF16", 141.1, -18), ("Llama-4-Scout BF16", 217, -18), ("Kimi-K2 UD-Q2_K_XL", 381, -22)]:
    k = D.nodes_needed(gb * 1e9, F) or 5
    ax.plot([gb], [k], "o", color=TC.get(D.nodes_needed(gb * 1e9, F)), ms=8, mec=BG, mew=1.5, zorder=5)
    ax.annotate(f"{name} · {gb:g} GB", (gb, k), xytext=(4, dy), textcoords="offset points", fontsize=8.5, color=FG,
                arrowprops=dict(arrowstyle="-", color=DIM, lw=.7) if abs(dy) > 25 else None)
ax.set_xlim(0, 460); ax.set_ylim(0.4, 5.1); ax.set_yticks([1, 2, 3, 4, 5]); ax.set_yticklabels(["1", "2", "3", "4", "beyond"])
ax.set_xlabel("weights of the unit you insert (GB)"); ax.set_ylabel("GPU nodes needed")
ax.set_title("Fit estimate: max(W + 5.6·k + 8, 1.15·W) ≤ 95.69·k GiB  (a 128 GB unified-memory node)", fontsize=11.5, loc="left", pad=12)
#: max(W + 5.6·k + 8, 1.15·W) ≤ 95.69·k GiB  (128 GB unified-memory node)", fontsize=11.5, loc="left")
ax.grid(axis="x", color=GRID); [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fit-chart.png")); plt.close(fig)

# 2 — the demo shelf: models per tier, split by category
rows = {}
for cat, maker, model, var, pub, fmt, q, gb, quants in D.DEMO_SHELF:
    sizes = [g for _, g in quants] if quants else [gb]; k = min([D.nodes_needed(s * 1e9, F) or 9 for s in sizes]); k = None if k == 9 else k
    group = "Text LLMs" if cat in ("LLMs", "Coding", "Small-On-Device") else "Image / video" if cat in ("Image", "Video") else "Speech / OCR / embeddings"
    rows.setdefault(group, {}); rows[group][k] = rows[group].get(k, 0) + 1
tiers = [1, 2, 3, 4, None]; labels = ["1 node", "2 nodes", "3 nodes", "4 nodes", "beyond"]
cols = {"Text LLMs": "#38b6ff", "Image / video": "#f472b6", "Speech / OCR / embeddings": "#a78bfa"}
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160); bottom = [0] * 5
for g, c in cols.items():
    vals = [rows.get(g, {}).get(t, 0) for t in tiers]
    ax.bar(labels, vals, bottom=bottom, color=c, label=g, width=.62, edgecolor=BG)
    bottom = [b + v for b, v in zip(bottom, vals)]
for i, b in enumerate(bottom): ax.text(i, b + .3, str(b), ha="center", color=FG, weight="bold")
ax.set_ylabel("models"); ax.set_axisbelow(True); ax.set_title("A sample shelf (the demo library): where each model lands", loc="left", fontsize=12)
ax.legend(frameon=False, labelcolor=FG); ax.grid(axis="y", color=GRID); [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
fig.tight_layout(); fig.savefig(os.path.join(OUT, "demo-library.png")); plt.close(fig)
print("charts written")
