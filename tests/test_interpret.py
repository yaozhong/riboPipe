"""Unit tests of ribopipe.interpret on a tiny synthetic model and synthetic transcripts (CPU, no paper data).

ViennaRNA is replaced by a deterministic stand-in folding function so that the tests also run without it; one test
checks the real bindings when they are installed.
"""
import argparse
import numpy as np
import pytest
import torch

from ribopipe import train as T
from ribopipe.interpret import core as C, ism, routing, transplant
import ribopipe.interpret as I


class FakeRNA:
    """Stand-in for ViennaRNA: a deterministic 'MFE' of a 30-nt window from its base composition and order."""
    __version__ = "fake"

    @staticmethod
    def fold(s):
        gc = sum(c in "GC" for c in s); w = sum((i % 7) * (c == "G") for i, c in enumerate(s))
        return ("." * len(s), float(-0.5 * gc - 0.01 * w))


@pytest.fixture(autouse=True)
def fake_rna(monkeypatch):
    monkeypatch.setattr(C, "RNA", FakeRNA)
    nt = torch.get_num_threads(); torch.set_num_threads(1)   # the analyses run with one torch thread (core.pmap)
    yield
    torch.set_num_threads(nt)


class FakeDS:
    """Minimal dataset with the interface of ribopipe.data.PaperDataset."""

    def __init__(self, n=24, nfold=2, seed=0):
        rng = np.random.RandomState(seed)
        sense = [c for c in C.SENSE if c != "ATG"]
        self.tag = "SYNTH"; self.ids = [f"t{i:02d}" for i in range(n)]; self.folds = list(range(nfold))
        self.seq, self.counts, self.unit, self._mfe = {}, {}, {}, {}
        for i, t in enumerate(self.ids):
            L = int(rng.randint(90, 111))
            s = "ATG" + "".join(sense[k] for k in rng.randint(0, len(sense), L - 1))
            c = rng.poisson(2.0, L).astype(float); c[rng.choice(np.arange(25, L - 25), 3, replace=False)] += 25
            self.seq[t], self.counts[t], self.unit[t] = s, c, f"U{i:02d}"
            self._mfe[t] = C.mfe_track(s)
        self._test = {f: [t for i, t in enumerate(self.ids) if i % nfold == f] for f in self.folds}
        self._val = {f: [t for i, t in enumerate(self.ids) if i % nfold == (f + 1) % nfold][:2] for f in self.folds}

    def test_ids(self, f): return list(self._test[f])
    def val_ids(self, f): return list(self._val[f])
    def train_ids(self, f):
        ex = set(self._test[f]) | set(self._val[f]); return [t for t in self.ids if t not in ex]
    def truth(self, t): c = self.counts[t]; return c / c.mean()
    def mfe_raw(self, t): return np.asarray(self._mfe[t], float)
    def mfe_valid(self, t): return len(self.seq[t]) - 29


def tiny_model(seed):
    torch.manual_seed(seed)
    m = T.build_model(dict(model="mfe", hidden=16), mfe_mu=-8.0, mfe_sd=3.0)
    with torch.no_grad():                       # make the output depend visibly on the inputs
        for p in m.parameters():
            if p.requires_grad: p.mul_(3.0)
    return m.eval()


@pytest.fixture(scope="module")
def setup():
    mp = pytest.MonkeyPatch(); mp.setattr(C, "RNA", FakeRNA)
    ds = FakeDS()
    oof = C.OOF(ds, models={f: tiny_model(10 + f) for f in ds.folds})
    yield ds, oof
    mp.undo()


# ---------------------------------------------------------------------------------------------- core
def test_mfe_update_matches_full_recompute():
    rng = np.random.RandomState(1)
    s = "".join(rng.choice(list("ACGT"), 150)); tr = C.mfe_track(s)
    assert np.all(tr[-29:] == 0)
    s2 = s[:60] + "GGGCCC" + s[66:]
    tr2, starts = C.mfe_update(s2, tr, 60, 66)
    assert starts == list(range(31, 66))
    assert np.array_equal(tr2, C.mfe_track(s2))


def test_topk_and_candidate_peaks():
    v = np.array([1.0, 5, 5, 3, 0])
    assert C.topk_positions(v, 1) == {1, 2}                  # ties included
    rng = np.random.RandomState(0); y = 0.01 * rng.rand(100); yh = 0.01 * rng.rand(100)   # interior 60 codons -> k = 3
    y[[50, 60, 70]] = [9, 8, 7]; yh[[50, 61, 70]] = [9, 8, 7]
    assert C.candidate_peaks(y, yh) == [50, 70]
    assert C.representative_peak(y, yh) == 50


def test_unit_halves_frozen_and_rule():
    h = C.unit_halves()
    assert len(h) == 17977 and abs(sum(v == "A" for v in h.values()) - 17977 // 2) == 0
    assert C.unit_halves(list(h)) == h                       # the rule reproduces the frozen halves
    small = C.unit_halves(["u3", "u1", "u2", "u4"], seed=7)
    assert sorted(small.values()) == ["A", "A", "B", "B"] and small == C.unit_halves(["u4", "u2", "u1", "u3"], seed=7)


def test_bootstrap_helpers():
    x = np.arange(10.0); units = np.repeat(np.arange(5), 2)
    m, lo, hi = C.cluster_boot_ci(x, units, n_boot=200, seed=0)
    assert m == 4.5 and lo <= m <= hi
    r = C.median_ci(x, units); assert r["median"] == 4.5 and r["n"] == 10
    w = np.ones(10); assert C.wmedian(x, w) == 4.5


def test_real_vienna_if_available(monkeypatch):
    try:
        import RNA as real
    except ImportError:
        pytest.skip("ViennaRNA not installed")
    monkeypatch.setattr(C, "RNA", real)
    s = "GGGAAACCCAGGGAAACCCAAAAAAAAAAAG"
    tr = C.mfe_track(s)
    assert tr[0] == np.float32(real.fold(s[:30])[1]) and tr[1] == np.float32(real.fold(s[1:31])[1]) and tr[2] == 0


# ---------------------------------------------------------------------------------------------- ISM / maps
def test_peak_ism_effects_are_variant_minus_wildtype(setup):
    ds, oof = setup
    peaks = ism.peak_set(oof, C.halves_for(ds), [0], n_per_half=1, seed=2)
    assert peaks
    z = ism.run_peak_ism(oof, folds=[0], offsets=[-2, -1, 0], peaks=peaks, workers=1, log=lambda s: None)
    t, p = str(z["tx"][0]), int(z["p"][0]); m = oof.model_for(t); seq = ds.seq[t]; tr = oof.track(t)
    j = p - 1; w = seq[3 * j:3 * j + 3]; b = next(c for c in C.SENSE if c != w)
    sv = C.sub_codon(seq, j, b); u0 = C.forward_one(m, seq, tr)[p]
    assert np.isclose(z["dF"][0, 1, C.CIDX[b]], C.forward_one(m, sv, tr)[p] - u0, atol=1e-5)
    assert np.isclose(z["dR"][0, 1, C.CIDX[b]], C.forward_one(m, sv, C.mfe_update(sv, tr, 3 * j, 3 * j + 3)[0])[p] - u0, atol=1e-5)
    assert np.isnan(z["dF"][0, 1, C.CIDX[w]])                       # wild type is not substituted
    assert np.isfinite(z["dF"][0]).sum() == 3 * 60
    if len(set(z["half"].tolist())) < 2:                           # the selection/evaluation summary needs both halves
        z = {k: (np.concatenate([v, v]) if k not in ("offsets",) else v) for k, v in z.items()}
        z["half"] = np.array(["A"] * (len(z["tx"]) // 2) + ["B"] * (len(z["tx"]) // 2))
    s = ism.summarize_ism(z)
    assert len(s["recompute"]["mean_abs"]) == 3 and s["n_peaks"] == len(z["tx"])


def test_maps_are_centred(setup):
    ds, oof = setup
    z = ism.run_maps(oof, folds=[0], n_per_half=1, offsets=[-1, 0], workers=1, log=lambda s: None)
    c = z["cod_c_recompute"]
    assert np.nanmax(np.abs(np.nansum(c, 3))) < 1e-9
    s = ism.summarize_map(z)
    assert s["n_tx"] == len(z["tx"]) and len(s["fixed"]["importance"]) == 2


# ---------------------------------------------------------------------------------------------- routing
def test_routing_local_recompute_matches_full(setup):
    """Local recomputation == full forward within routing.GATE_TOL (bit-identical on the paper stack, see routing)."""
    ds, oof = setup
    t = ds.test_ids(0)[0]; seq = ds.seq[t]; tr = oof.track(t); net = routing.Net(oof.models[0])
    w = net.wt(seq, tr); tol = routing.GATE_TOL
    assert np.abs(w.u - C.forward_one(net.m, seq, tr)).max() <= tol
    j = 40; rows = np.array([j - r for r in C.OFFSETS])
    for b in C.SENSE:                                   # all 61 substitutions x both MFE treatments
        sv = C.sub_codon(seq, j, b); tv = C.mfe_update(sv, tr, 3 * j, 3 * j + 3)[0]
        for trk in (tv, tr):
            idx, bio = C.inputs_from(sv, trk); h = net.blocks_from_inputs(idx, bio)
            assert np.abs(net.local(w, h, rows) - C.forward_same_len(net.m, [idx], [bio])[0][rows]).max() <= tol
            rr = net.routed(w, h, rows)
            for k in routing.ROUTES:
                hr = w.h.clone(); hr[..., net.slices[k]] = h[..., net.slices[k]]
                assert np.abs(rr[k] - net.head(net.rnn_full(hr))[0].double().numpy()[rows]).max() <= tol


def test_routing_gate_reports_and_enforces_tolerance(setup):
    ds, oof = setup
    sites = routing.sample_sites(ds, folds=[0], n_per_fold=1, n_sites=1)[:1]
    r = routing.run_routing(oof, sites=sites, workers=1, gate=True, log=lambda s: None)
    g = r["gate"]
    assert g["passed"] and g["tol"] == routing.GATE_TOL and 0.0 <= g["max_abs_diff"] <= routing.GATE_TOL
    assert g["bit_exact"] == (g["max_abs_diff"] == 0.0)
    if not g["bit_exact"]:                              # a strict gate must then refuse
        with pytest.raises(RuntimeError):
            routing.run_routing(oof, sites=sites, workers=1, gate=True, gate_tol=0.0, log=lambda s: None)


def test_routing_run_and_shares(setup):
    ds, oof = setup
    sites = routing.sample_sites(ds, folds=[0, 1], n_per_fold=1, n_sites=1)
    r = routing.run_routing(oof, sites=sites[:1], workers=1, gate=False, log=lambda s: None)   # exactness: test above
    assert r["self_substitution_exact_all"]
    s = routing.summarize_routing(r["arrays"])
    assert set(s["shares"]["recompute"]["-5"]) >= {"codon", "nt", "aa", "mfe", "rho", "reading"}


def test_shares_definition():
    rng = np.random.RandomState(0); Dt = rng.normal(size=(30, 5))
    DR = np.zeros((30, 5, 4)); DR[..., 1] = Dt                      # all of the response through 'nt'
    s = routing.shares(DR, Dt, np.arange(30))
    assert np.isclose(s["nt"]["s"], 1.0) and np.isclose(s["codon"]["s"], 0.0) and s["reading"] == "main route: nt"
    DR[..., 1] = 0.5 * Dt; DR[..., 3] = 0.5 * Dt
    s = routing.shares(DR, Dt, np.arange(30))
    assert np.isclose(s["nt"]["s"], 0.5) and s["reading"] == "shared" and abs(s["rho"]["s"]) < 1e-12


# ---------------------------------------------------------------------------------------------- transplant
def test_graft():
    sa = "AAA" * 20; sd = "".join(f"{c}{c}{c}" for c in "ACGTACGTACGTACGTACGT")
    g = transplant.graft(sa, sd, 10, 5, [-1, 0, 1])
    assert g[:27] == sa[:27] and g[27:36] == sd[12:21] and g[36:] == sa[36:]


def test_transplant_and_subwindows(setup):
    ds, oof = setup
    z = transplant.run_transplant(oof, workers=1, log=lambda s: None)
    assert len(z["tx"]) > 0
    i = 0; t_d, p = str(z["tx"][i]), int(z["p"][i]); t_a, q = str(z["acc_tx"][i][0]), int(z["acc_q"][i][0])
    sa = ds.seq[t_a]; lo, hi = 3 * (q - 6), 3 * (q + 5); sn = sa[:lo] + str(z["win_T"][i]) + sa[hi:]
    m = oof.model_for(t_a); tra = oof.track(t_a)
    assert str(z["win_T"][i]) == ds.seq[t_d][3 * (p - 6):3 * (p + 5)]
    assert sorted(str(z["win_S"][i])[k:k + 3] for k in range(0, 33, 3)) == sorted(str(z["win_T"][i])[k:k + 3] for k in range(0, 33, 3))
    ref = C.forward_one(m, sn, tra)[q] - C.forward_one(m, sa, tra)[q]
    assert np.isclose(z["d_T_fixed"][i, 0], ref, atol=1e-5)
    s = transplant.summarize_transplant(z)
    assert set(s) == {"A->B", "B->A"}
    zs = transplant.run_subwindows(oof, tp=z, workers=1, log=lambda s: None)
    assert np.allclose(zs["d|W_full|fixed"], z["d_T_fixed"], atol=1e-5, equal_nan=True)
    assert np.allclose(zs["d|W_shuffled|recompute"], z["d_S_recompute"], atol=1e-5, equal_nan=True)
    ss = transplant.summarize_subwindows(zs)
    assert "W_core|recompute" in ss["windows"] and "recompute" in ss["shuffle_control"]


# ---------------------------------------------------------------------------------------------- CLI
def test_register_cli():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    I.register(sp)
    a = ap.parse_args(["transplant", "--dataset", "X", "--ckpt-dir", "d", "--out", "o", "--subwindows"])
    assert a.func is I._cmd_transplant and a.subwindows
    for cmd in ("ism", "ism-map", "routing"):
        assert ap.parse_args([cmd, "--dataset", "X", "--ckpt-dir", "d", "--out", "o"]).func
