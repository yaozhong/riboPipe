"""Unit tests of ribopipe.lowcov on small synthetic inputs (no paper data, CPU, well under a minute)."""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ribopipe.lowcov import crossover as X, libraries as LB, posterior as PO, protocol, ranking, score  # noqa: E402
from ribopipe.lowcov.preds import dummy_noise, load_predictions  # noqa: E402


# ---------------------------------------------------------------- synthetic dataset (same interface as ribopipe.data)
class FakeDS:
    def __init__(self, n_tx=400, seed=0, depth_range=(-2.5, 0.5), min_L=80, max_L=160, tag="SYNTH", part="partII"):
        rng = np.random.default_rng(seed); self.tag = tag; self.ids = [f"T{i:04d}" for i in range(n_tx)]
        self.unit = {t: f"U{i // 2:04d}" for i, t in enumerate(self.ids)}      # two transcripts per unit
        units = sorted(set(self.unit.values())); fu = {u: i % 5 for i, u in enumerate(units)}
        self.folds = list(range(5)); self._test = {f: [t for t in self.ids if fu[self.unit[t]] == f] for f in self.folds}
        self.seq, self._reps, self.lam = {}, {}, {}
        for t in self.ids:
            L = int(rng.integers(min_L, max_L)); self.seq[t] = "".join(rng.choice(list("ACGT"), 3 * L))
            lam = rng.gamma(0.6, 1.0, L); lam /= lam.mean(); self.lam[t] = lam
            d = 10 ** rng.uniform(*depth_range)
            self._reps[t] = np.stack([rng.poisson(lam * d), rng.poisson(lam * d)]).astype(np.int64)
        self.strata = {"low": list(self.ids), "high": []}

    def test_ids(self, f): return list(self._test[f])
    def val_ids(self, f): return []
    def train_ids(self, f): return [t for t in self.ids if t not in set(self._test[f])]
    def reps(self, t): return self._reps[t]


def informative_pred(ds, noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    return {t: (ds.lam[t] * np.exp(rng.normal(0, noise, len(ds.lam[t])))).astype(np.float32) for t in ds.ids}


# ---------------------------------------------------------------- protocol
def test_protocol_frozen():
    m5 = hashlib.md5(open(protocol.PROTOCOL_JSON, "rb").read()).hexdigest()
    assert m5 == protocol.PROTOCOL_MD5
    assert json.load(open(protocol.PROTOCOL_JSON))["edges_reads_per_codon"] == protocol.EDGES_V2 == protocol.edges()
    e = protocol.EDGES_V2; r = [e[i + 1] / e[i] for i in range(len(e) - 1)]
    assert max(r) / min(r) < 1.002                     # one log ratio throughout (edges rounded to 7 digits)


# ---------------------------------------------------------------- crossover definition
def _synthetic_records(true_cross=0.30, n=1500, seed=1):
    edges = protocol.EDGES_V2[12:]                      # 0.01 .. 1.5 (the v1 grid)
    rng = np.random.default_rng(seed); recs = []
    for u in range(n):
        d = float(10 ** rng.uniform(math.log10(edges[0]), math.log10(edges[-1]))) if u % 10 else 0.0
        adv = -0.12 * math.log10(d / true_cross) if d > 0 else 0.2
        for role in (0, 1):
            rm = 0.5 + rng.normal(0, 0.05); recs.append((f"U{u}", role, d, rm, rm - adv + rng.normal(0, 0.05)))
    return recs, edges


def test_dstar_recovers_known_crossover():
    recs, edges = _synthetic_records()
    res = X.dstar(recs, edges, B=60)
    assert res["status"] == "OK" and abs(res["D_star"] - 0.30) / 0.30 < 0.15
    assert res["ci95"][0] < res["D_star"] < res["ci95"][1]
    assert res["zero_depth"]["N_records"] == 2 * 150


def test_locate_rules():
    rows = [{"valid": True, "y": 0.1, "x_log10": -2, "lo": 0.01, "hi": 0.02},
            {"valid": False, "y": None, "x_log10": -1.5, "lo": 0.02, "hi": 0.05},
            {"valid": True, "y": -0.1, "x_log10": -1, "lo": 0.05, "hi": 0.1}]
    assert X.locate(rows)[0] == "NOT_LOCATABLE"            # no interpolation across an invalid bin
    assert X.locate([dict(rows[0]), dict(rows[0], y=0.05)])[0] == "NO_SIGN_CHANGE"
    two = [dict(rows[0]), dict(rows[2], valid=True), dict(rows[0], x_log10=-0.5), dict(rows[2], x_log10=0)]
    assert X.locate(two)[0] == "NOT_LOCATABLE"             # two crossings
    st, d = X.locate([dict(rows[0]), dict(rows[2])])
    assert st == "OK" and abs(math.log10(d) - (-1.5)) < 1e-12   # y = +0.1 / -0.1 -> midpoint in log10


def test_dstar_with_curves_equals_reference():
    recs, edges = _synthetic_records(n=600)
    ok, diff = X.check_equivalence(recs, edges, B=30)
    assert ok, diff
    out = X.dstar_with_curves(recs, edges, B=30)
    assert out["n_boot_ok"] + round(out["bootstrap_fail_fraction"] * 30) == 30 and len(out["bin_boot"]) == len(edges) - 1


# ---------------------------------------------------------------- libraries
def test_grid_levels_match_bruteforce():
    for L in (37, 64, 101, 250, 999, 4000):
        assert LB.grid_levels(L, protocol.EDGES_V2) == LB.grid_levels_bruteforce(L, protocol.EDGES_V2)


def test_match_and_split():
    rng = np.random.default_rng(3); a = rng.poisson(2.0, 200); b = rng.poisson(0.5, 200)
    a2, b2, m = LB.match_reps(a, b, 0, "SYNTH", "T1")
    assert m == min(a.sum(), b.sum()) and (a2 <= a).all() and np.array_equal(b2, b)
    a3, _, _ = LB.match_reps(a, b, 0, "SYNTH", "T1"); assert np.array_equal(a2, a3)          # seeded per transcript
    h1, h2, m2, pt = LB.binom_split(a, b, 0, "SYNTH", "T1")
    assert np.array_equal(h1 + h2, pt) and (pt <= a + b).all() and m2 == m
    th = LB.nested_thin(a, [5, 20, 60], 0, 1, "SYNTH", "T1")
    assert all(int(th[k].sum()) == k for k in th) and (th[5] <= th[20]).all() and (th[20] <= th[60]).all()


def test_pearson_interior():
    y = np.zeros(100); y[50] = 3.0
    assert LB.pearson_interior(np.arange(100.0), np.ones(100)) == (None, "truth_constant")
    assert LB.pearson_interior(np.ones(100), y) == (None, "pred_constant")
    assert LB.pearson_interior(np.ones(42), np.arange(42.0))[1] == "interior<3"
    r, _ = LB.pearson_interior(y * 2 + 1, y); assert abs(r - 1) < 1e-12
    p = np.r_[np.full(20, 9e9), np.arange(60.0), np.full(20, -9e9)]           # values outside the interior are ignored
    assert abs(LB.pearson_interior(p, np.r_[np.zeros(20), np.arange(60.0), np.zeros(20)])[0] - 1) < 1e-12


# ---------------------------------------------------------------- crossover evaluation end to end
def test_evaluate_dataset_end_to_end(tmp_path):
    ds = FakeDS(n_tx=600, depth_range=(-1.2, -0.4))
    preds = {"good": informative_pred(ds)}
    res = score.evaluate_dataset(ds, preds, protocol.EDGES_V2, out_dir=str(tmp_path), variants=["v2_primary", "v2_sens_binom"],
                                 B=10, workers=1, log=lambda s: None)
    assert res["predictors"] == ["good", score.DUMMY]
    v = res["variants"]["v2_primary"]
    for role in ("0", "1"):
        assert v["per_role"][role]["check"]["equals_T_low"]
    assert "v2_sens_binom" not in res["variants"]                              # not a UMI-deduplicated dataset
    d = v["dstar"]["good"]; valid = [b for b in d["bins"] if b["valid"]]
    assert valid and all(b["y"] > 0 for b in valid[:2])                         # informative prediction beats sparse reads at low depth
    assert os.path.exists(tmp_path / "eval_SYNTH.json") and os.path.exists(tmp_path / "records_natural.tsv.gz")
    assert res["opcrit"]["good"]["N_eligible"] == 600
    s = score.summarize({"SYNTH": res}); assert "SYNTH" in s["datasets"] and score.summary_markdown(s)


def test_dummy_noise_is_sequence_independent():
    a = dummy_noise("SYNTH", {"T1": "ACG" * 50, "T2": "TTT" * 50}); b = dummy_noise("SYNTH", {"T1": "GGG" * 50})
    assert np.array_equal(a["T1"], b["T1"]) and len(a["T1"]) == 50 and not np.array_equal(a["T1"], a["T2"][:50])


def test_load_predictions_layouts(tmp_path):
    p = {"A": np.ones(5, np.float32), "B": np.zeros(4, np.float32)}
    np.savez(tmp_path / "x.npz", **p)
    assert set(load_predictions(str(tmp_path / "x.npz"))[0]) == {"A", "B"}
    for f, k in ((0, "A"), (1, "B")):
        d = tmp_path / "run" / "seed123" / f"fold{f}"; d.mkdir(parents=True)
        np.savez(d / "pred.npz", **{k: p[k]}); json.dump({"status": "OK"}, open(d / "run.json", "w"))
    got, info = load_predictions(str(tmp_path / "run"), ["A", "B"])
    assert info["kind"] == "run_dir" and np.array_equal(got["B"], p["B"])
    with pytest.raises(KeyError): load_predictions(str(tmp_path / "x.npz"), ["C"])


# ---------------------------------------------------------------- posterior
def test_posterior_limits():
    rng = np.random.default_rng(5); A = rng.poisson(0.2, 140); pred = rng.gamma(1.0, 1.0, 140) - 0.2
    post0, ps = PO.posterior_full(A, pred, 0.0); sl = slice(20, 120)
    assert abs(post0[sl].sum() - 1) < 1e-12 and np.allclose(post0[sl], A[sl] / A[sl].sum())
    pinf, _ = PO.posterior_full(A, pred, 1e12)
    assert np.allclose(pinf[sl], ps[sl], atol=1e-12) and (ps[sl] > 0).all()   # negative predictions are clipped
    g = PO.posterior_grid(A[sl], ps[sl]); assert g.shape == (33, 100) and np.allclose(g.sum(1), 1)


def test_kappa_selection_prefers_informative_prior():
    ds = FakeDS(n_tx=120, depth_range=(0.3, 0.8), min_L=90, max_L=140, part="partI")
    good = informative_pred(ds, noise=0.2); rng = np.random.default_rng(9)
    bad = {t: rng.gamma(1.0, 1.0, len(ds.lam[t])).astype(np.float32) for t in ds.ids}
    kg, chg = PO.select_kappa(ds, good, log=lambda s: None); kb, _ = PO.select_kappa(ds, bad, log=lambda s: None)
    assert set(kg) == {(r, f) for r in (0, 1) for f in range(5)}
    assert all(v in set(PO.GRID.tolist()) for v in kg.values())
    assert np.median(list(kg.values())) > np.median(list(kb.values()))
    j = PO.kappa_to_json(chg, ds.tag); assert len(j["chosen"]) == 10


def test_score_posterior_and_aggregate():
    ds = FakeDS(n_tx=600, depth_range=(-1.2, -0.4))
    pred = informative_pred(ds); kappa = {(r, f): 1.0 for r in (0, 1) for f in range(5)}
    res = PO.score_posterior(ds, pred, kappa, protocol.EDGES_V2, B=10, workers=1, identity_check=True, log=lambda s: None)
    assert res["identity_check"]["pass_"]
    m, c = res["comparisons"]["main_RiboPipe-reads"], res["comparisons"]["RiboPipe-reads"]
    assert m["bins"] == c["bins"] or any(x != [0, 0] for x in res["n_common_minus_main_per_bin"])
    res.pop("_records"); agg = PO.aggregate({"SYNTH": res})
    assert agg["datasets"]["SYNTH"]["summary"]["n_valid_bins"] == agg["pooled"]["n_valid_bins"]


def test_ranking_runs():
    ds = FakeDS(n_tx=60, depth_range=(0.3, 0.8), min_L=90, max_L=130, part="partI")
    res = ranking.run(ds, informative_pred(ds), log=lambda s: None)
    assert "Post_RiboPipe" in res["estimators"] and "CodonMean" not in res["estimators"]
    c = res["cells"]["main_Bge1|role0|d0.1"]
    assert c["est"]["Post_RiboPipe"]["prec"]["n"] == c["n_tx"] and "Post_RiboPipe-Raw|prec" in c["diff"]


# ---------------------------------------------------------------- CLI
def test_cli_registration():
    from ribopipe import lowcov
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd"); lowcov.register(sp)
    for argv in (["crossover", "--dataset", "X", "--pred", "a=b", "--out", "o"], ["posterior", "--dataset", "X", "--pred", "p", "--out", "o"],
                 ["posterior-kappa", "--dataset", "X", "--oof", "r", "--out", "k.json"], ["peak-ranking", "--dataset", "X", "--oof", "r", "--out", "o"],
                 ["crossover-summary", "--eval", "e.json", "--out", "s"], ["posterior-summary", "--score", "s.json", "--out", "s"],
                 ["lowcov-predict", "--runs", "r", "--dataset", "X", "--out", "o"]):
        assert callable(ap.parse_args(argv).func)
