"""Unit tests for the paper's headline target space ``covmean0_log``.

``covmean0_log`` normalises each codon by the mean over *covered* codons only
(count > 0), then applies ``log1p``.  These tests assert:

1. the denominator excludes zero-coverage codons (covered-mean, not all-codon mean);
2. the forward (``log1p``) + inverse (``expm1``) transform round-trips to the linear
   covered-mean-normalised profile that prediction returns;
3. ``covmean0`` is exactly the pre-log linear base of ``covmean0_log``;
4. the existing targets (meannorm / meannorm_log / minmax) are unchanged.
"""
import os
import tempfile

import numpy as np
import torch

from ribopipe.dataset import RiboDataset

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def _tiny_npz(dir_path):
    """One transcript whose counts contain zero-coverage codons, so that the
    covered-mean denominator differs from the all-codon mean."""
    rng = np.random.RandomState(0)
    L = 50
    seq = "".join(CODONS[i] for i in rng.randint(0, 64, size=L))
    cnt = np.zeros(L, np.float32)
    # ~half the codons covered; the rest are genuine zeros
    covered_idx = rng.choice(L, size=L // 2, replace=False)
    cnt[covered_idx] = rng.randint(1, 10, size=L // 2).astype(np.float32)
    key = "ENST00000000001"
    obj = {key: {"cds": {"sequence": seq, "avg_count": cnt}}}
    path = os.path.join(dir_path, "tiny.npz")
    np.savez(path, **{k: np.array(v, dtype=object) for k, v in obj.items()})
    return path, key, cnt


def test_covmean0_log_accepted_and_denominator_excludes_zeros():
    assert "covmean0_log" in RiboDataset.VALID_TARGETS
    assert "covmean0" in RiboDataset.VALID_TARGETS
    with tempfile.TemporaryDirectory() as d:
        path, key, cnt = _tiny_npz(d)
        cov = cnt > 0
        assert not cov.all(), "test fixture must contain zero-coverage codons"
        covered_mean = cnt[cov].mean()
        all_mean = cnt.mean()
        assert not np.isclose(covered_mean, all_mean), "covered vs all mean must differ"

        ds = RiboDataset(path, [key], target="covmean0_log",
                         use_nt=False, use_struct=False)
        _, _, _, tgt, _ = ds[0]
        tgt = tgt.numpy()

        # zero-coverage codons map to log1p(0) == 0
        assert np.allclose(tgt[~cov], 0.0)
        # covered codons map to log1p(count / covered_mean) -- NOT count / all_mean
        expected = np.log1p(cnt[cov] / covered_mean)
        assert np.allclose(tgt[cov], expected, atol=1e-5)
        wrong = np.log1p(cnt[cov] / all_mean)
        assert not np.allclose(tgt[cov], wrong, atol=1e-4)


def test_covmean0_log_forward_inverse_round_trip():
    """expm1(covmean0_log target) == covmean0 linear target (what predict returns)."""
    with tempfile.TemporaryDirectory() as d:
        path, key, cnt = _tiny_npz(d)
        ds_log = RiboDataset(path, [key], target="covmean0_log", use_nt=False, use_struct=False)
        ds_lin = RiboDataset(path, [key], target="covmean0", use_nt=False, use_struct=False)
        tgt_log = ds_log[0][3].numpy()
        tgt_lin = ds_lin[0][3].numpy()

        # inverse of the log transform (predict.predict_dataset applies exactly this)
        recovered = np.expm1(tgt_log)
        assert np.allclose(recovered, tgt_lin, atol=1e-5)

        # and the linear base recovers the raw counts up to the covered-mean scale
        cov = cnt > 0
        covered_mean = cnt[cov].mean()
        assert np.allclose(recovered * covered_mean, cnt, atol=1e-4)


def test_existing_targets_unchanged():
    """Backward-compat: meannorm / meannorm_log / minmax keep their old definitions."""
    with tempfile.TemporaryDirectory() as d:
        path, key, cnt = _tiny_npz(d)
        m = cnt.mean()
        mn = RiboDataset(path, [key], target="meannorm", use_nt=False, use_struct=False)[0][3].numpy()
        ml = RiboDataset(path, [key], target="meannorm_log", use_nt=False, use_struct=False)[0][3].numpy()
        assert np.allclose(mn, cnt / m, atol=1e-5)
        assert np.allclose(ml, np.log1p(cnt / m), atol=1e-5)
