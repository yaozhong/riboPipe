"""Minimal amino-acid sequence-logo rendering (matplotlib only).

Letter heights are the signed per-position weights passed in (e.g. motif-filter
amino-acid weights, or ISM attribution): positive letters stack upward from 0, negative
letters stack downward, so the most enriched residue sits furthest from the axis.  Colours
follow the standard physico-chemical grouping.  No external logo library required.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .aa import AA20

# physico-chemical colour groups
AACOL = {}
for _a in "AVLIMFW":
    AACOL[_a] = "#2a8f2a"      # hydrophobic (green)
for _a in "STNQCGY":
    AACOL[_a] = "#7a5cc0"      # polar (purple)
for _a in "KRH":
    AACOL[_a] = "#1f6fd0"      # basic (blue)
for _a in "DE":
    AACOL[_a] = "#d02a2a"      # acidic (red)
AACOL["P"] = "#e07b00"          # proline (orange)


def _letter(ax, ch, x, y0, h, w, color):
    from matplotlib.text import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    from matplotlib.font_manager import FontProperties
    if abs(h) < 1e-6:
        return
    tp = TextPath((0, 0), ch, size=1, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    e = tp.get_extents()
    sx = w / (e.width or 1)
    sy = abs(h) / (e.height or 1)
    t = (Affine2D().translate(-e.x0, -e.y0).scale(sx, sy).translate(x - w / 2, y0) + ax.transData)
    ax.add_patch(PathPatch(tp, transform=t, color=color, lw=0))


def draw_weight_logo(ax, mat: np.ndarray, center: int, w: float = 0.85, labels=AA20):
    """Draw one logo panel. ``mat`` is ``(k, 20)`` signed weights; E/P/A at center-2/-1/0."""
    k = mat.shape[0]
    positions = list(range(k))
    ymax = ymin = 0.0
    for p in positions:
        f = mat[p]
        pos = [(j, f[j]) for j in range(len(f)) if f[j] > 0]
        neg = [(j, f[j]) for j in range(len(f)) if f[j] < 0]
        y = 0.0
        for j, h in sorted(pos, key=lambda z: z[1]):     # smallest first -> largest on top
            _letter(ax, labels[j], p, y, h, w, AACOL.get(labels[j], "#444"))
            y += h
        ymax = max(ymax, y)
        y = 0.0
        for j, h in sorted(neg, key=lambda z: -z[1]):
            y += h
            _letter(ax, labels[j], p, y, h, w, AACOL.get(labels[j], "#444"))
        ymin = min(ymin, y)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_ylim(min(ymin * 1.15, -0.02), max(ymax * 1.15, 0.02))
    lab = {center - 2: "E", center - 1: "P", center: "A"}
    ax.set_xticks(positions)
    ax.set_xticklabels([lab.get(p, str(p - center)) for p in positions])
    for c, co in [(center - 2, "#666"), (center - 1, "#e07b00"), (center, "#1f6fd0")]:
        if 0 <= c < k:
            ax.axvspan(c - 0.5, c + 0.5, color=co, alpha=0.10)
    return ymax


def draw_logo_grid(mats: List[np.ndarray], titles, center: int, out_png: str,
                   ylabel: str = "weight"):
    """Stack several logo panels vertically and save to ``out_png``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(mats)
    fig, axes = plt.subplots(n, 1, figsize=(6, 1.8 * n + 0.4), squeeze=False)
    for i, (mat, title) in enumerate(zip(mats, titles)):
        ax = axes[i, 0]
        draw_weight_logo(ax, np.asarray(mat), center)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel(ylabel, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
