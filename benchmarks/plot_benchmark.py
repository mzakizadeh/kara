"""
Redraw the KARA benchmark figure from benchmarks/wiki_benchmark.csv.

Decoupled from the (slow) data-collection run so styling can be iterated
freely. Reads cumulative embedding-operation counts and renders the hero
figure comparing KARA against LangChain's Index API.

Usage:
    python benchmarks/plot_benchmark.py                 # KARA vs LangChain
    python benchmarks/plot_benchmark.py --show-vanilla  # include vanilla too
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

CSV_PATH = "benchmarks/wiki_benchmark.csv"
OUT_PATH = "benchmarks/wiki_benchmark.png"

# palette
KARA_COLOR = "#2E86AB"
LC_COLOR = "#F3A712"
VANILLA_COLOR = "#E4572E"


def load_csv(path):
    cols = {"cycle": [], "vanilla": [], "langchain": [], "kara": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return cols


def smooth(x, y, n=400, log=False):
    """Monotone PCHIP interpolation: smooth, no overshoot on cumulative data.

    In log mode, interpolate in log-space (over positive points only) so the
    curve stays smooth and strictly positive.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if log:
        mask = y > 0
        x, y = x[mask], y[mask]
        if len(x) < 3:
            return x, y
        xs = np.linspace(x.min(), x.max(), n)
        ys = np.power(10.0, PchipInterpolator(x, np.log10(y))(xs))
        return xs, ys
    if len(x) < 3:
        return x, y
    xs = np.linspace(x.min(), x.max(), n)
    return xs, PchipInterpolator(x, y)(xs)


def draw(cols, show_vanilla=False, log=False, max_cycle=None):
    if max_cycle:
        keep = [i for i, c in enumerate(cols["cycle"]) if c <= max_cycle]
        cols = {k: [v[i] for i in keep] for k, v in cols.items()}
    x = cols["cycle"]

    series = [
        ("LangChain Indexing", cols["langchain"], LC_COLOR, False),
        ("KARA", cols["kara"], KARA_COLOR, True),
    ]
    # log scale comfortably fits vanilla too, so include it by default there
    if show_vanilla or log:
        series.insert(0, ("Vanilla", cols["vanilla"], VANILLA_COLOR, False))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ymax = max(max(y) for _, y, _, _ in series)

    for label, y, color, emphasis in series:
        xs, ys = smooth(x, y, log=log)
        ax.plot(xs, ys, color=color, lw=4.5 if emphasis else 2.8,
                zorder=4 if emphasis else 3, solid_capstyle="round",
                alpha=1.0 if emphasis else 0.9)
        # direct end-of-line label (no legend box)
        y_end = ys[-1] if len(ys) else y[-1]
        ax.text(x[-1] + 0.4, y_end, label, color=color, va="center", ha="left",
                fontsize=13.5, fontweight="bold" if emphasis else "normal")

    ax.set_xlabel("Document edits over time  →", fontsize=13, labelpad=8)
    ax.set_xticks([])
    ax.set_xlim(x[0], x[-1] + (x[-1] - x[0]) * 0.24)  # headroom for end labels

    if log:
        # log axis must show ticks to stay honest; keep them subtle
        ax.set_yscale("log")
        ax.set_ylabel("Re-embedding cost  (log scale)", fontsize=13, labelpad=8)
        ax.set_ylim(0.8, ymax * 1.6)
        ax.grid(True, axis="y", which="major", color="#e6e6e6", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=10, colors="#666666")
    else:
        ax.set_ylabel("Re-embedding cost  →", fontsize=13, labelpad=8)
        ax.set_yticks([])
        ax.set_ylim(0, ymax * 1.10)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
        ax.spines[side].set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")

    # headline numbers for the caption
    k, l, v = cols["kara"][-1], cols["langchain"][-1], cols["vanilla"][-1]
    print(f"  KARA={k:.0f}  LangChain={l:.0f}  Vanilla={v:.0f}")
    print(f"  KARA uses {l / max(k, 1):.1f}x fewer embeddings than LangChain, "
          f"{v / max(k, 1):.1f}x fewer than vanilla")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--show-vanilla", action="store_true")
    p.add_argument("--log", action="store_true", help="log y-axis (fits all three)")
    p.add_argument("--max-cycle", type=int, default=None,
                   help="plot only the first N update cycles")
    a = p.parse_args()
    draw(load_csv(CSV_PATH), show_vanilla=a.show_vanilla, log=a.log,
         max_cycle=a.max_cycle)
