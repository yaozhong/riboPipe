"""Smoke tests for the interpretability toolkit (motif filters + ISM attribution)."""
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ribopipe.model import RiboPipeCNN
from ribopipe.interpret.motifs import filter_aa_matrices, motif_report
from ribopipe.interpret.ism import asite_codon_attribution, motif_vs_ism

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def _tiny_npz(path, n_tx=4, seed=0):
    rng = np.random.RandomState(seed)
    obj = {}
    for t in range(n_tx):
        L = int(rng.randint(30, 50))
        seq = "".join(CODONS[i] for i in rng.randint(0, 64, size=L))
        cnt = (rng.poisson(2.0, size=L) + 1).astype(np.float32)
        obj[f"tx{t}"] = {"cds": {"sequence": seq, "avg_count": cnt.tolist()}}
    np.savez(path, **{k: np.array(v, dtype=object) for k, v in obj.items()})


def test_motif_matrices_and_report():
    model = RiboPipeCNN(bio_dim=120)          # nt-only (no struct)
    mats, center = filter_aa_matrices(model)
    assert mats.shape == (128, 7, 20)         # n_filters x k x AA
    assert center == 3                        # A-site column for k=7
    with tempfile.TemporaryDirectory() as d:
        csv, png = os.path.join(d, "m.csv"), os.path.join(d, "m.png")
        order = motif_report(model, top=2, out_csv=csv, out_png=png)
        assert len(order) == 2
        assert os.path.exists(csv) and os.path.getsize(csv) > 0
        assert os.path.exists(png) and os.path.getsize(png) > 0


def test_ism_asite_attribution():
    torch.manual_seed(0)
    model = RiboPipeCNN(bio_dim=120)
    with tempfile.TemporaryDirectory() as d:
        npz = os.path.join(d, "s.npz")
        _tiny_npz(npz)
        z = np.load(npz, allow_pickle=True)
        ids = list(z.files)
        res = asite_codon_attribution(model, npz, ids, struct_npz_path=None,
                                      max_transcripts=3, max_len=60, device="cpu")
        assert res["per_aa"].shape == (20,)
        assert res["per_codon"].shape == (64,)
        assert res["n_positions"] > 0
        assert np.isfinite(res["per_aa"]).all()
        # per-codon effects are centred per position, so the grand mean is ~0
        assert abs(res["per_codon"].mean()) < 1e-6
        mv = motif_vs_ism(np.arange(20, dtype=float), res["per_aa"])
        assert set(mv) == {"pearson_r", "spearman_rho"}
