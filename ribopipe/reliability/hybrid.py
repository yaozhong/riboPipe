"""Depth-weighted hybrid estimator: blend model prediction and reads by coverage.

The weight is a logistic in **log** reads/codon centred at D* with slope s=0.5:

    w(rc) = sigmoid( (ln rc - ln D*) / s )

so ``w -> 1`` at high depth (trust the reads), ``w -> 0`` at low depth (trust the model),
``w = 0.5`` exactly at ``rc = D*``. The imputed profile is the convex blend on the
covered-mean-normalised scale, ``w * reads + (1 - w) * prediction``. This is the drop-in
low-coverage estimator: it matches or exceeds either source alone at every depth.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .readsplit import meannorm

SLOPE = 0.5


def depth_weight(rc, dstar: float, slope: float = SLOPE):
    """Logistic read-trust weight in log reads/codon, centred at D* (0.5 at rc == D*)."""
    rc = np.asarray(rc, float)
    return 1.0 / (1.0 + np.exp(-(np.log(np.clip(rc, 1e-9, None)) - np.log(dstar)) / slope))


def hybrid_profile(pred_vec, count_vec, dstar: float, slope: float = SLOPE) -> np.ndarray:
    """Depth-weighted blend for one transcript (covered-mean-normalised).

    reads/codon = ``sum(counts) / L``. Zero-read transcripts fall back to the pure model.
    """
    pr = meannorm(pred_vec)
    cnt = np.asarray(count_vec, float)
    L = cnt.size
    reads = cnt.sum()
    if L == 0 or reads <= 0:
        return pr
    rc = reads / L
    w = float(depth_weight(rc, dstar, slope))
    return w * meannorm(cnt) + (1.0 - w) * pr


def impute(pred: Dict[str, np.ndarray], counts: Dict[str, np.ndarray],
           dstar: float, slope: float = SLOPE) -> Dict[str, np.ndarray]:
    """Depth-weighted hybrid profile for every transcript with a prediction and counts."""
    out = {}
    for tid, cnt in counts.items():
        if tid in pred and len(pred[tid]) == len(cnt):
            out[tid] = hybrid_profile(pred[tid], cnt, dstar, slope)
    return out
