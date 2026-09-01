"""Self-contained statistical baselines (codon-mean, tri-codon, ridge).

These are the non-deep controls from the paper benchmark, computed directly on the
loaded ``(id, codon_idx, count)`` items.  The published neural baselines (iXnos,
RiboMIMO, RiboExp, RiboGL) are external code bases run under the same gene-level
protocol; see the paper's supplement and their upstream repositories.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def codon_mean_lookup(tr_items, te_items, order: int = 1):
    """RUST-style codon (order=1) or E/P/A tri-codon (order=3) mean-``log1p(pause)`` lookup."""
    s = defaultdict(lambda: [0.0, 0])

    def key(idx, i):
        if order == 1:
            return int(idx[i])
        return tuple(int(idx[j]) if 0 <= j < len(idx) else -1 for j in (i - 1, i, i + 1))

    for _k, idx, cnt in tr_items:
        m = cnt.mean()
        if m <= 0:
            continue
        lp = np.log1p(cnt / m)
        for i in range(len(idx)):
            kk = key(idx, i)
            s[kk][0] += lp[i]
            s[kk][1] += 1
    table = {k: (v[0] / v[1]) for k, v in s.items() if v[1] > 0}
    glob = np.mean([v[0] / v[1] for v in s.values()]) if s else 0.0
    pred = {}
    for k, idx, _cnt in te_items:
        pred[k] = np.expm1(np.array([table.get(key(idx, i), glob) for i in range(len(idx))]))
    return pred


def ridge_window(tr_items, te_items, K: int = 5, alpha: float = 10.0):
    """Ridge on a flattened one-hot codon window [-K, +K] (iXnos-linear / scikit-ribo style)."""
    from sklearn.linear_model import Ridge

    nC = 64
    W = 2 * K + 1

    def feat(idx):
        F = np.zeros((len(idx), W * nC), np.float32)
        for i in range(len(idx)):
            for j, off in enumerate(range(-K, K + 1)):
                p = i + off
                if 0 <= p < len(idx) and idx[p] >= 0:
                    F[i, j * nC + int(idx[p])] = 1.0
        return F

    Xtr, ytr = [], []
    for _k, idx, cnt in tr_items:
        m = cnt.mean()
        if m <= 0:
            continue
        Xtr.append(feat(idx))
        ytr.append(np.log1p(cnt / m))
    Xtr = np.concatenate(Xtr)
    ytr = np.concatenate(ytr)
    reg = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred = {}
    for k, idx, _cnt in te_items:
        pred[k] = np.expm1(reg.predict(feat(idx)))
    return pred
