"""Smoke test: build tiny synthetic data, train a few epochs on CPU, predict.

Runs without a GPU, without ViennaRNA, and without the real datasets, so it is safe for
CI.  It exercises the full path: RiboDataset (codon + NT features) -> model ->
early-stopping train_on_ids -> predict -> per-transcript metrics.
"""
import os
import tempfile

import numpy as np
import pytest

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def _make_synthetic(dir_path, n_tx=24, seed=0):
    """Write a tiny NPZ (cds.sequence + cds.avg_count)."""
    rng = np.random.RandomState(seed)
    npz_obj = {}
    for t in range(n_tx):
        L = int(rng.randint(30, 80))
        seq = "".join(CODONS[i] for i in rng.randint(0, 64, size=L))
        # counts loosely driven by codon identity so there is a learnable signal
        base = np.array([(idx % 7) + 1 for idx in rng.randint(0, 64, size=L)], np.float32)
        cnt = base + rng.poisson(2.0, size=L).astype(np.float32)
        key = f"ENST{t:08d}"
        npz_obj[key] = {"cds": {"sequence": seq, "avg_count": cnt,
                                 "avg_count_norm": cnt / cnt.sum()}}

    npz_path = os.path.join(dir_path, "synthetic.npz")
    np.savez(npz_path, **{k: np.array(v, dtype=object) for k, v in npz_obj.items()})
    return npz_path, list(npz_obj.keys())


def test_import_version():
    import ribopipe
    assert ribopipe.__version__


def test_train_predict_smoke():
    import ribopipe

    with tempfile.TemporaryDirectory() as d:
        npz_path, ids = _make_synthetic(d)
        tr, val, te = ids[:14], ids[14:19], ids[19:]

        # NT features on, struct off (no ViennaRNA needed for the smoke test)
        model = ribopipe.train_on_ids(
            npz_path, tr, val_ids=val,
            use_nt=True, use_struct=False,
            hidden=32, epochs=3, patience=5, batch_size=8,
            loss_name="huber", device="cpu", verbose=False,
        )
        cfg = getattr(model, "_ribopipe_config", {})
        # codon(inside model) + 120 nt (struct off)
        assert cfg.get("bio_dim") == 120

        preds = ribopipe.predict(
            model, npz_path, te,
            use_nt=True, use_struct=False, device="cpu",
        )
        assert len(preds) > 0
        for k, p in preds.items():
            assert p.ndim == 1 and len(p) > 0

        true = ribopipe.true_pause(ribopipe.load_items(npz_path, te))
        df = ribopipe.per_tx_metrics(preds, true)
        assert set(["pearson", "spearman", "recall_at5"]).issubset(df.columns)


def test_peakmse_matches_huber_when_no_peaks():
    """huber_peak_mse with tau above all targets == plain Huber (background-only)."""
    import torch
    from ribopipe.losses import huber_peak_mse, huber_mask

    torch.manual_seed(0)
    pred = torch.randn(2, 5)
    tgt = torch.rand(2, 5) * 0.5  # all below tau=1
    mask = torch.ones(2, 5, dtype=torch.bool)
    a = huber_peak_mse(pred, tgt, mask, tau=1.0, delta=1.0)
    b = huber_mask(pred, tgt, mask, delta=1.0)
    assert torch.allclose(a, b, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
