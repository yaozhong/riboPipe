"""Independent-read-split crossover: the depth D* below which the model beats the reads.

For a grid of target depths in **reads/codon**, each transcript's input half is thinned to
``D = round(rc * L)`` and scored (covered-mean-norm Pearson) against the held-out reference
half; the model prediction is scored once (depth-independent). Taking the median across
transcripts gives a raw-accuracy curve that rises with depth and a flat model level; their
crossover is D*. Above D* the reads are the better estimator, below it the model is.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .readsplit import MIN_NONZERO, binomial_split, downsample, meannorm, pear

# geometric-mean centres of the reads/codon bins used in the paper (Fig. 3 axis)
RC_EDGES = np.array([0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.22, 0.32, 0.47, 0.68,
                     1.0, 1.5, 2.2, 3.3, 5, 8, 12])
RC_CENT = np.sqrt(RC_EDGES[:-1] * RC_EDGES[1:])


def read_split_curves(pred: Dict[str, np.ndarray], counts: Dict[str, np.ndarray],
                      rc_grid=RC_CENT, seeds: int = 10, seed0: int = 2000) -> dict:
    """Median model / raw accuracy vs reads/codon from the binomial read split.

    Returns ``{"rc": grid, "model": scalar, "raw": array_like(grid), "n": count}``.
    """
    rc_grid = np.asarray(rc_grid, float)
    model_vals = []
    raw_acc = [[] for _ in rc_grid]
    for tid, cnt in counts.items():
        pr = pred.get(tid)
        if pr is None:
            continue
        n = np.asarray(cnt).astype(np.int64)
        L = len(n)
        if len(pr) != L:
            continue
        for s in range(seeds):
            rng = np.random.RandomState(seed0 + s)
            inp, ref = binomial_split(n, rng)
            if int((ref > 0).sum()) < MIN_NONZERO:
                continue
            refn = meannorm(ref)
            mc = pear(pr, refn)
            if np.isnan(mc):
                continue
            model_vals.append(mc)
            for gi, rc in enumerate(rc_grid):
                ds = downsample(inp, int(round(rc * L)), rng)
                if ds is None:
                    continue
                rcorr = pear(meannorm(ds), refn)
                if not np.isnan(rcorr):
                    raw_acc[gi].append(rcorr)
    model_med = float(np.nanmedian(model_vals)) if model_vals else np.nan
    raw_med = np.array([np.nanmedian(a) if a else np.nan for a in raw_acc])
    return {"rc": rc_grid, "model": model_med, "raw": raw_med, "n": len(model_vals)}


def crossover_dstar(rc, model_level: float, raw_curve) -> Optional[float]:
    """Locate D* = the reads/codon where the raw curve first overtakes the model level."""
    rc = np.asarray(rc, float)
    diff = np.asarray(raw_curve, float) - float(model_level)   # raw - model
    for j in range(1, len(diff)):
        if np.isnan(diff[j - 1]) or np.isnan(diff[j]):
            continue
        if diff[j - 1] < 0 <= diff[j]:
            return float(np.interp(0.0, [diff[j - 1], diff[j]], [rc[j - 1], rc[j]]))
    return None


def estimate_dstar(pred, counts, **kw) -> dict:
    """Convenience: read-split curves + the crossover D* (reads/codon)."""
    c = read_split_curves(pred, counts, **kw)
    c["dstar_codon"] = crossover_dstar(c["rc"], c["model"], c["raw"])
    return c


def invert_dstar(rc, raw_curve, model_level: float) -> Optional[float]:
    """D* = r^{-1}(m): locate the crossover from the raw curve + one model level, no sweep.

    Interpolates the (monotone) raw-vs-depth curve to the depth at which raw accuracy equals
    the model level -- the deployment recipe (a model-free coverage curve + one evaluation).
    """
    rc = np.asarray(rc, float)
    raw = np.asarray(raw_curve, float)
    ok = ~np.isnan(raw)
    if ok.sum() < 2:
        return None
    return float(np.interp(float(model_level), raw[ok], rc[ok]))


def fraction_below(counts, dstar_codon: float) -> dict:
    """Model-favoured fraction: expressed transcripts with reads/codon below D*.

    Returns ``{"frac_expressed": ..., "frac_all": ..., "n_expressed": ...}`` -- the
    ``frac_expressed`` figure is the paper's "52-82% of expressed genes".
    """
    rc_per_tx = []
    for cnt in counts.values():
        n = np.asarray(cnt, float)
        L = n.size
        rc_per_tx.append(n.sum() / L if L else 0.0)
    rc_per_tx = np.array(rc_per_tx)
    expr = rc_per_tx > 0
    frac_expr = float((rc_per_tx[expr] < dstar_codon).mean()) if expr.any() else np.nan
    frac_all = float((rc_per_tx < dstar_codon).mean())
    return {"frac_expressed": frac_expr, "frac_all": frac_all, "n_expressed": int(expr.sum())}
