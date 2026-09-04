"""Tests for the low-coverage reliability toolkit (crossover D* + depth-weighted hybrid)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from ribopipe.reliability.readsplit import meannorm
from ribopipe.reliability.crossover import (read_split_curves, crossover_dstar,
                                            estimate_dstar, invert_dstar, fraction_below)
from ribopipe.reliability.hybrid import depth_weight, hybrid_profile, impute


def test_depth_weight_logistic():
    d = 0.4
    assert abs(depth_weight(d, d) - 0.5) < 1e-9          # 0.5 exactly at rc == D*
    assert depth_weight(d * 10, d) > 0.9                  # high depth -> trust reads
    assert depth_weight(d / 10, d) < 0.1                  # low depth  -> trust model
    grid = np.array([0.05, 0.1, 0.4, 1.0, 5.0])
    w = depth_weight(grid, d)
    assert np.all(np.diff(w) > 0)                         # monotone increasing in depth


def test_hybrid_profile_blend():
    rng = np.random.RandomState(0)
    pred = meannorm(rng.rand(60) + 0.1)
    counts = (rng.poisson(3.0, size=60) + 1).astype(float)  # ~3 reads/codon
    dstar = 0.4
    out = hybrid_profile(pred, counts, dstar)
    rc = counts.sum() / counts.size
    w = depth_weight(rc, dstar)
    expected = w * meannorm(counts) + (1 - w) * pred
    assert np.allclose(out, expected)
    # zero reads -> pure model
    assert np.allclose(hybrid_profile(pred, np.zeros(60), dstar), pred)


def _synthetic(n=150, seed=0):
    rng = np.random.RandomState(seed)
    pred, counts = {}, {}
    for t in range(n):
        L = int(rng.randint(60, 120))
        truth = np.exp(rng.randn(L))                      # positive, peaky
        tn = meannorm(truth)
        pred[f"tx{t}"] = meannorm(tn + 0.6 * rng.randn(L))   # moderate correlation
        lam = 10.0                                        # ~10 reads/codon (deep enough to thin)
        counts[f"tx{t}"] = rng.poisson(tn * lam).astype(np.int64)
    return pred, counts


def test_crossover_and_fraction():
    pred, counts = _synthetic()
    res = estimate_dstar(pred, counts, seeds=3)
    assert np.isfinite(res["model"])
    assert res["raw"].shape == res["rc"].shape
    d = res["dstar_codon"]
    assert d is None or (0 < d < 12)                      # a plausible reads/codon crossover
    inv = invert_dstar(res["rc"], res["raw"], res["model"])
    assert inv is None or inv > 0
    fb = fraction_below(counts, dstar_codon=0.5)
    assert 0.0 <= fb["frac_expressed"] <= 1.0
    assert fb["n_expressed"] > 0


def test_impute_dict():
    pred, counts = _synthetic(n=20)
    imp = impute(pred, counts, dstar=0.4)
    assert len(imp) == 20
    for tid, prof in imp.items():
        assert prof.shape == counts[tid].shape
        assert np.isfinite(prof).all()
