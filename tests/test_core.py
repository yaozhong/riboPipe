"""Unit tests that need no paper data (CPU, < 1 min)."""
import os, json, tempfile
import numpy as np
import pytest
import torch

from ribopipe import arch as A, train as RP, ablation as AB, data as D, prepare as PR, cli


def test_parameter_counts():
    for cid, cfg in RP.CONFIGS.items():
        m = RP.build_model(cfg, -4.0, 3.0)
        assert RP.count_params(m) == cfg["expected_params"], cid
        RP.structure_gate(m, cfg)
    assert RP.count_params(RP.build_model(RP.CONFIGS["Paper_RiboPipe_MFE_H128"])) == 228321


def test_ablation_variants():
    g = AB.gate_equivalence()
    assert g["ok"], g
    for cid, cfg in AB.ABLATION_CONFIGS.items():
        AB.structure_gate(AB.build_model(cfg, -4.0, 3.0), cfg)


def _toy_inputs(B=2, L=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 64, (B, L), generator=g); bio = torch.zeros(B, L, 18)
    for b in range(B):
        for t in range(L):
            c = int(idx[b, t])
            for p in range(3): bio[b, t, 4 * p + (c >> (2 * (2 - p))) % 4] = 1.0
    bio[..., 12:15] = -torch.rand(B, L, 3, generator=g) * 20; bio[..., 15:18] = 1.0
    return idx, bio, torch.tensor([L, L - 7])


def test_invalid_mfe_windows_enter_as_zero():
    torch.manual_seed(0)
    m = RP.build_model(RP.CONFIGS["Paper_RiboPipe_MFE_H128"], -4.0, 3.0).eval()
    idx, bio, ll = _toy_inputs()
    bio[:, -10:, 15:18] = 0.0                       # last windows invalid
    a = bio.clone(); a[:, -10:, 12:15] = 0.0
    b = bio.clone(); b[:, -10:, 12:15] = 123.0      # any raw value at invalid windows
    with torch.no_grad(): assert torch.equal(m(idx, a, ll), m(idx, b, ll))


def test_features_and_loss():
    seq = "ATGGCCAAATTTGGGCCCTAA"
    idx = RP.codon_idx(seq); assert list(idx[:2]) == [RP.CIDX["ATG"], RP.CIDX["GCC"]]
    nt = RP.nt3_onehot(seq, len(seq) // 3); assert nt.shape == (7, 12) and (nt.sum(1) == 3).all()
    v, ok = RP.mfe_features(np.arange(21, dtype=float), 21 - 29, 7); assert ok.sum() == 0   # sequence shorter than a window
    t = torch.tensor(RP.targets([0, 2, 4, 2], "log"))[None]; u = torch.zeros_like(t); mk = torch.ones_like(t, dtype=torch.bool)
    assert float(RP.loss_fn("gamma_logmse", u, t, mk, 0.5)) > float(RP.loss_fn("logmse", u, t, mk, None))


def _write_toy_dataset(root, tag="toy", n=60, seed=0):
    rng = np.random.RandomState(seed); ids = [f"tx{i:03d}" for i in range(n)]
    seqs = ["ATG" + "".join(rng.choice(list("ACGT"), 3 * (rng.randint(40, 90)))) + "TAA" for _ in ids]
    seqs = [s.replace("TAA", "TAC").replace("TAG", "TAC").replace("TGA", "TAC")[:-3] + "TAA" for s in seqs]
    L = [len(s) // 3 for s in seqs]
    reps = [rng.poisson(rng.uniform(0.3, 3.0), (2, l)).astype(np.int32) for l in L]
    for r in reps: r[0, 0] += 1
    d = os.path.join(root, tag); os.makedirs(os.path.join(root, "mfe"), exist_ok=True); os.makedirs(d)
    off = np.concatenate([[0], np.cumsum(L)])
    np.savez_compressed(f"{d}/{tag}.npz", ids=np.array(ids), offsets=off, seq=np.frombuffer("".join(seqs).encode(), np.uint8),
                        reps=np.concatenate(reps, 1))
    unit = {t: i // 2 for i, t in enumerate(ids)}; uf, val = PR.fold_and_val(unit.values())
    for nm in ("partI", "partII"):
        json.dump({"ids": ids, "n": n}, open(f"{d}/universe_{nm}.json", "w"))
        json.dump({"folds": [{"fold": f, "test": sorted(t for t in ids if uf[unit[t]] == f),
                              "val": sorted(t for t in ids if unit[t] in val[f])} for f in range(5)]}, open(f"{d}/folds_{nm}.json", "w"))
    json.dump({"tx_unit": unit}, open(f"{d}/units.json", "w"))
    mfe = [np.r_[-rng.uniform(0, 12, len(s) - 29), np.zeros(29)].astype(np.float32) for s in seqs]
    np.savez_compressed(f"{root}/mfe/{tag}_mfe.npz", ids=np.array(ids), offsets=np.concatenate([[0], np.cumsum([len(x) for x in mfe])]),
                        mfe=np.concatenate(mfe))
    json.dump({"n_valid": {t: len(s) - 29 for t, s in zip(ids, seqs)}}, open(f"{d}/mfe_valid.json", "w"))
    json.dump({"human": {}, "nonhuman": {tag: [{"fold": f, "mean": -6.0, "sd": 3.4} for f in range(5)]}},
              open(f"{root}/MFE_NORM_FOLDWISE.json", "w"))
    return ids


def test_loader_and_tiny_training():
    with tempfile.TemporaryDirectory() as root:
        _write_toy_dataset(root)
        ds = D.load_dataset("toy", data=root)
        t = ds.ids[0]
        assert np.array_equal(ds.counts[t], ds.reps(t).mean(0)) and len(ds.seq[t]) == 3 * len(ds.counts[t])
        assert sorted(sum((ds.test_ids(f) for f in ds.folds), [])) == sorted(ds.ids)
        RP.CONFIGS["_toy"] = dict(RP.CONFIGS["Paper_RiboPipe_MFE_H128"], datasets=["toy"]); RP.CONFIGS["_toy"].pop("also_write")
        try:
            out = os.path.join(root, "runs")
            rc = cli._run_one("_toy", "toy", 0, 123, out, device="cpu", max_epochs=2, data=root)
            assert rc == 0
            r = json.load(open(os.path.join(out, "_toy", "toy", "seed123", "fold0", "run.json")))
            assert r["status"] == "OK" and r["params_trainable"] == 228321
            p = np.load(os.path.join(out, "_toy", "toy", "seed123", "fold0", "pred.npz"))
            assert sorted(p.files) == sorted(ds.test_ids(0))
            m, ck = RP.load_ckpt(os.path.join(out, "_toy", "toy", "seed123", "fold0", "model.pt"))
            again = RP.predict_transcripts(m, ds, ds.test_ids(0), ck["config"])
            assert all(np.allclose(again[k], p[k], atol=1e-5) for k in p.files)
        finally:
            RP.CONFIGS.pop("_toy")


def test_split_rules():
    uf, val = PR.fold_and_val([3, 1, 2, 5, 4, 1, 9, 7, 8, 6, 10, 11])
    assert sorted(set(uf.values())) == [0, 1, 2, 3, 4]
    for f in range(5): assert all(uf[u] != f for u in val[f])
    pr = PR.kmer_pairs(["ACGT" * 20, "ACGT" * 20, "TTTTGGGGCCCCAAAA" * 5], [0, 1, 2])
    assert pr[(0, 1)] == 1.0 and (0, 2) not in pr
