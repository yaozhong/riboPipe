"""Smoke tests for the optional magnitude head (transcript-level ribosome load).

Data-free / CPU only. Covers: the feature extractor returns the documented columns on a
tiny synthetic transcript; MagnitudeHead fit+predict round-trips (recovers a monotone
signal); and the abs-reconstruction shape/values are correct. Importing the module must
not require ViennaRNA or touch the shape model.
"""
import numpy as np

import ribopipe
from ribopipe import magnitude as MAG

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def _entry(utr_seq, cds_seq, counts):
    return {"5utr": {"sequence": utr_seq},
            "cds": {"sequence": cds_seq, "avg_count": list(counts)}}


def test_feature_columns_and_values():
    assert MAG.ALL_FEATURES == MAG.CDS_GLOBAL_FEATURES + MAG.UTR5_FEATURES
    assert len(MAG.ALL_FEATURES) == 13

    # a 5'UTR with an upstream AUG, a strong Kozak (A at -3, G at +4), pyrimidine-rich cap
    utr = "CTCTCATGCACGTACGTACGT" + "AAA"  # contains "ATG" (uAUG); s[-3]=A -> Kozak -3 purine
    cds = "ATGGCTGCTTAA"                    # +4 (0-based idx 3) = G -> Kozak +4 G; ATG start
    ent = _entry(utr, cds, np.ones(len(cds) // 3, np.float32))

    f5 = MAG.utr5_features(utr, cds)
    for c in MAG.UTR5_FEATURES:
        assert c in f5
    assert f5["kozak_m3_purine"] == 1.0
    assert f5["kozak_p4_G"] == 1.0
    assert f5["kozak_strong"] == 1.0
    assert f5["n_uAUG"] >= 1
    assert 0.0 <= f5["top_pyrimidine"] <= 1.0

    fc = MAG.cds_global_features(cds, len(cds) // 3)
    for c in MAG.CDS_GLOBAL_FEATURES:
        assert c in fc

    v = MAG.feature_vector(ent)
    assert v is not None and v.shape == (13,) and np.isfinite(v).all()

    # empty 5'UTR degrades gracefully (all-zero UTR block, no crash)
    v0 = MAG.feature_vector(_entry("", cds, np.ones(4, np.float32)))
    assert v0 is not None and v0.shape == (13,)


def _synthetic_npzd(n=200, seed=0):
    """Transcripts whose mean count is driven by CDS length -> a learnable magnitude."""
    rng = np.random.RandomState(seed)
    npzd = {}
    for t in range(n):
        L = int(rng.randint(30, 300))
        cds = "".join(CODONS[i] for i in rng.randint(0, 64, size=L))
        utr = "".join("ACGT"[i] for i in rng.randint(0, 4, size=int(rng.randint(5, 120))))
        # mean density grows strongly with CDS length -> a clearly learnable magnitude
        mean_density = 1.0 + 15.0 * (L - 30) / 270.0   # ~1 .. 16 across the length range
        cnt = np.clip(rng.poisson(mean_density, size=L), 0, None).astype(np.float32)
        if cnt.sum() == 0:
            cnt[0] = 1.0
        npzd[f"ENST{t:08d}"] = _entry(utr, cds, cnt)
    return npzd


def test_fit_predict_roundtrip_and_reconstruction():
    npzd = _synthetic_npzd()
    ids = list(npzd.keys())
    tr, te = ids[:150], ids[150:]

    Xtr, ytr, kept_tr = MAG.build_magnitude_dataset(npzd, ids=tr, min_codons=20)
    Xte, yte, kept_te = MAG.build_magnitude_dataset(npzd, ids=te, min_codons=20)
    assert Xtr.shape[1] == 13 and len(kept_tr) > 0 and len(kept_te) > 0

    ckpt = MAG.fit_magnitude_head(Xtr, ytr, epochs=150, hidden=32, seed=0, device="cpu")
    assert ckpt["in_dim"] == 13 and ckpt["features"] == MAG.ALL_FEATURES

    m = MAG.predict_log_density(ckpt, Xte, device="cpu")
    assert m.shape == (len(kept_te),) and np.isfinite(m).all()
    # the head must recover the (length-driven) magnitude signal above chance
    r = np.corrcoef(m, yte)[0, 1]
    assert r > 0.5, f"magnitude head failed to learn (Pearson {r:.3f})"

    mean_hat = MAG.predict_mean_density(ckpt, Xte, device="cpu")
    assert np.allclose(mean_hat, np.expm1(m))

    # abs reconstruction = expm1(shape_log) * expm1(m_t); shape preserved per codon
    shape_log = np.log1p(np.array([0.0, 1.0, 3.0], np.float64))  # pause = 0,1,3
    abs_prof = MAG.reconstruct_absolute(shape_log, m_t=float(m[0]))
    assert abs_prof.shape == (3,)
    assert np.allclose(abs_prof, np.array([0.0, 1.0, 3.0]) * np.expm1(m[0]))


def test_save_load_roundtrip(tmp_path):
    npzd = _synthetic_npzd(n=60)
    X, y, kept = MAG.build_magnitude_dataset(npzd, min_codons=20)
    ckpt = MAG.fit_magnitude_head(X, y, epochs=30, hidden=16, seed=1, device="cpu")
    p = tmp_path / "mag.pt"
    MAG.save_magnitude_head(ckpt, str(p))
    loaded = MAG.load_magnitude_head(str(p), device="cpu")
    a = MAG.predict_log_density(ckpt, X, device="cpu")
    b = MAG.predict_log_density(loaded, X, device="cpu")
    assert np.allclose(a, b, atol=1e-6)


def test_optional_and_exported():
    # public API surface
    for name in ("MagnitudeHead", "fit_magnitude_head", "predict_mean_density",
                 "reconstruct_absolute", "build_magnitude_dataset"):
        assert hasattr(ribopipe, name)
