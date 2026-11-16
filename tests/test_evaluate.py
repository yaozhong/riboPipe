"""Synthetic tests of the Part I scorer (port of the paper's paper_eval/test_synthetic_scorer.py).

Every expected value is computed by an independent brute-force implementation (np.corrcoef, explicit loops, enumeration
of tie subsets) and compared with ribopipe.evaluate.scorer to 1e-12. CPU only, a few seconds.
"""
import os, json, zlib, hashlib, itertools, argparse
import numpy as np
import pytest

from ribopipe.evaluate import scorer as SP
from ribopipe.evaluate import partI as E
import ribopipe.evaluate as EV

SEEDS = [123, 124, 125]


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    rng = np.random.default_rng(7)
    fold_sizes = [3, 4, 5, 6, 7]; tx_fold, tx_unit, counts = {}, {}, {}
    k = 0
    for f, n in enumerate(fold_sizes):
        for _ in range(n):
            t = f"T{k:03d}"; L = int(rng.integers(60, 120)); tx_fold[t] = f; tx_unit[t] = f"U{k:03d}"
            counts[t] = rng.poisson(rng.uniform(0.5, 5), L).astype(float); counts[t][rng.integers(0, L)] += 3; k += 1
    for name, L, f in (("L42", 42, 2), ("L43", 43, 2)):          # L = 42: interior has 2 positions (unscorable); L = 43: 3
        tx_fold[name] = f; tx_unit[name] = "U_" + name; c = rng.poisson(3, L).astype(float); c[25] += 5; counts[name] = c
    tx_fold["SAMEU_a"] = 1; tx_fold["SAMEU_b"] = 1; tx_unit["SAMEU_a"] = tx_unit["SAMEU_b"] = "U_shared"
    for t in ("SAMEU_a", "SAMEU_b"): counts[t] = rng.poisson(2, 80).astype(float) + (np.arange(80) % 7 == 0) * 4
    ds = SP.Dataset(tx_fold, tx_unit, counts)
    root = str(tmp_path_factory.mktemp("runs"))

    def write(cfg, seed, f, pred, prov, status="OK"):
        d = os.path.join(root, cfg, "SYN", f"seed{seed}", f"fold{f}"); os.makedirs(d, exist_ok=True)
        json.dump({"status": status}, open(os.path.join(d, "run.json"), "w"))
        if status == "OK":
            np.savez(os.path.join(d, "pred.npz"), **pred); np.savez(os.path.join(d, "prov.npz"), **prov)

    def noisy(t, s, scale):
        y = ds.y[t]; return y + np.random.default_rng(zlib.crc32(f"{t}|{s}".encode())).normal(0, scale, len(y))

    zeros = lambda txs: {t: np.zeros(len(ds.y[t]), np.int8) for t in txs}
    for f in range(5):
        txs = [t for t in tx_fold if tx_fold[t] == f]
        for s in SEEDS:
            pa = {t: noisy(t, s, 0.6) for t in txs}; qa = zeros(txs)
            if f == 0 and s == 124: pa[txs[0]] = np.ones(len(ds.y[txs[0]]))          # constant prediction in one seed
            if f == 3: qa[txs[1]][:3] = SP.ADAPTER                                     # adapter at the end (outside interior)
            if f == 4: qa[txs[2]][30:33] = SP.ADAPTER                                  # adapter inside the interior -> unscorable
            write("A", s, f, pa, qa)
            write("C", s, f, {t: noisy(t, s + 7, 0.5) for t in txs}, zeros(txs), status=("MODEL_FAILED" if (s == 125 and f == 2) else "OK"))
        write("B", "det", f, {t: noisy(t, 999, 0.8) for t in txs}, zeros(txs))
        for s in SEEDS:   # D: one non-finite value inside the interior of one transcript in one seed
            pd_ = {t: noisy(t, s + 31, 0.7) for t in txs}
            if f == 1 and s == 125: pd_[txs[1]] = pd_[txs[1]].copy(); pd_[txs[1]][len(pd_[txs[1]]) // 2] = np.nan
            write("D", s, f, pd_, zeros(txs))
        write("E", "det", f, {t: noisy(t, 4242, 0.9) for t in txs}, zeros(txs))
        var = {t: float(np.var(ds.y[t])) for t in txs}; hard = set(sorted(txs, key=lambda t: -var[t])[:max(1, len(txs) // 3)])
        for s in SEEDS:   # J, K: both train; on fold 2, J is non-constant only on g[0], K constant only on g[0] -> empty paired fold
            g = sorted(t for t in txs if t.startswith("T"))
            write("J", s, f, {t: (np.ones(len(ds.y[t])) if (f == 2 and t != g[0]) else noisy(t, s + 55, 0.7)) for t in txs}, zeros(txs))
            write("K", s, f, {t: (np.ones(len(ds.y[t])) if (f == 2 and t == g[0]) else noisy(t, s + 66, 0.7)) for t in txs}, zeros(txs))
        for s in SEEDS:   # H: NaN exactly on the hardest transcripts -> main-table difference != paired difference
            write("H", s, f, {t: (np.full(len(ds.y[t]), np.nan) if t in hard else noisy(t, s + 77, 0.6 + 0.6 * (var[t] > np.median(list(var.values())))))
                              for t in txs}, zeros(txs))
    seeds = {m: (["det"] if m in "BE" else SEEDS) for m in "ABCDEHJK"}
    runs = {m: SP.load_method_runs(root, m, "SYN", seeds[m], range(5)) for m in "ABCDEHJK"}
    return dict(ds=ds, runs=runs, tx_fold=tx_fold, root=root)


def _dom(L, d): return list(range(20, L - 20)) if d == "interior" else list(range(L))


def _ok(syn, m, s, f, t, domain):
    st, pred, prov = syn["runs"][m][s][f]; y = syn["ds"].y[t]; idx = _dom(len(y), domain)
    if st != "OK" or len(idx) < 3 or np.std(y[idx]) == 0: return None
    p = np.asarray(pred[t], float)[idx]; q = np.asarray(prov[t])[idx]
    allowed = (0,) if domain == "interior" else (0, 1)
    if any(int(x) not in allowed for x in q) or not np.all(np.isfinite(p)) or np.std(p) == 0: return None
    return float(np.corrcoef(p, y[idx])[0, 1])


def _set(syn, ms, domains):
    return {f: sorted(t for t in syn["tx_fold"] if syn["tx_fold"][t] == f and
                      all(_ok(syn, m, s, f, t, d) is not None for m, ss in ms for s in ss for d in domains)) for f in range(5)}


def _S(syn, m, seeds, T, domain):
    return float(np.mean([np.mean([np.median([_ok(syn, m, s, f, t, domain) for t in T[f]]) for f in range(5)]) for s in seeds]))


@pytest.mark.parametrize("domain", ["interior", "full_deployed"])
def test_absolute_and_paired_against_brute_force(syn, domain):
    res = SP.score_dataset(syn["ds"], syn["runs"], domain=domain, pairs=[("A", "B"), ("A", "C")], B=300)
    TA = _set(syn, [("A", SEEDS)], [domain]); TB = _set(syn, [("B", ["det"])], [domain]); TAB = _set(syn, [("A", SEEDS), ("B", ["det"])], [domain])
    assert abs(res["absolute"]["A"]["S"] - _S(syn, "A", SEEDS, TA, domain)) < 1e-12
    assert abs(res["absolute"]["B"]["S"] - _S(syn, "B", ["det"], TB, domain)) < 1e-12
    assert abs(res["pairs"]["A|B"]["delta_hat"] - (_S(syn, "A", SEEDS, TAB, domain) - _S(syn, "B", ["det"], TAB, domain))) < 1e-12
    assert res["pairs"]["A|B"]["N_paired"] == sum(len(v) for v in TAB.values())
    assert res["absolute"]["C"]["status"] == "FAILED"          # one failed seed x fold -> FAILED on the dataset
    assert res["pairs"]["A|C"]["status"] == "FAILED"
    pr = res["pairs"]["A|B"]; lo, hi = pr["ci95"]
    assert pr["class"] == SP.classify(lo, hi)
    inTA = {t for v in TA.values() for t in v}
    if domain == "interior":
        assert "L42" not in inTA and "L43" in inTA             # interior needs >= 3 positions
        assert [t for t in syn["tx_fold"] if syn["tx_fold"][t] == 0][0] not in inTA      # constant in one seed
        assert [t for t in syn["tx_fold"] if syn["tx_fold"][t] == 4][2] not in inTA      # adapter inside the interior
    else:
        assert res["absolute"]["A"]["adapter_fraction"] > 0


def test_four_arm_common_set(syn):
    r4 = SP.score_common_arms(syn["ds"], {k: syn["runs"][k] for k in "ABDE"}, "interior", contrast=(("A", "B"), ("D", "E")), B=200)
    T4 = _set(syn, [("A", SEEDS), ("B", ["det"]), ("D", SEEDS), ("E", ["det"])], ["interior"])
    inter = (_S(syn, "A", SEEDS, T4, "interior") - _S(syn, "B", ["det"], T4, "interior")) - (_S(syn, "D", SEEDS, T4, "interior") - _S(syn, "E", ["det"], T4, "interior"))
    assert r4["N_common"] == sum(len(v) for v in T4.values()) and abs(r4["interaction"] - inter) < 1e-12
    assert [t for t in syn["tx_fold"] if syn["tx_fold"][t] == 1][1] not in {t for v in T4.values() for t in v}


def test_within_method_interior_vs_full(syn):
    TC = _set(syn, [("A", SEEDS)], ["interior", "full_deployed"])
    rb = SP.score_interior_vs_full(syn["ds"], {"A": syn["runs"]["A"], "B": syn["runs"]["B"]}, B=50)
    a = rb["A"]
    assert a["N_common"] == sum(len(v) for v in TC.values())
    assert abs(a["S_interior"] - _S(syn, "A", SEEDS, TC, "interior")) < 1e-12 and abs(a["S_full"] - _S(syn, "A", SEEDS, TC, "full_deployed")) < 1e-12
    assert "L42" not in {t for v in TC.values() for t in v}
    assert str(a["full_native_status"]).startswith("NA (1 ")    # one transcript has adapter end positions
    assert a["ids_md5"] == hashlib.md5("\n".join(sorted(t for v in TC.values() for t in v)).encode()).hexdigest()
    assert rb["B"]["full_native_status"] == "OK" and abs(rb["B"]["S_full_native"] - rb["B"]["S_full_deployed"]) < 1e-12


def test_pair_interior_vs_full(syn):
    rp = SP.score_pair_interior_vs_full(syn["ds"], syn["runs"]["A"], syn["runs"]["B"], B=50)
    TP = _set(syn, [("A", SEEDS), ("B", ["det"])], ["interior", "full_deployed"])
    di = _S(syn, "A", SEEDS, TP, "interior") - _S(syn, "B", ["det"], TP, "interior")
    dfu = _S(syn, "A", SEEDS, TP, "full_deployed") - _S(syn, "B", ["det"], TP, "full_deployed")
    assert rp["status"] == "OK" and rp["N_common"] == sum(len(v) for v in TP.values())
    assert abs(rp["delta_interior"] - di) < 1e-12 and abs(rp["delta_full_deployed"] - dfu) < 1e-12


def test_main_table_difference_differs_from_paired(syn):
    rh = SP.score_dataset(syn["ds"], {"H": syn["runs"]["H"], "A": syn["runs"]["A"]}, "interior", pairs=[("H", "A")], B=200)
    T = _set(syn, [("H", SEEDS), ("A", SEEDS)], ["interior"])
    fb = _S(syn, "H", SEEDS, T, "interior") - _S(syn, "A", SEEDS, T, "interior")
    forest = rh["pairs"]["H|A"]["delta_hat"]; main = rh["absolute"]["H"]["S"] - rh["absolute"]["A"]["S"]
    assert abs(forest - fb) < 1e-12 and abs(forest - main) >= 1e-3


def test_empty_paired_fold(syn):
    rj = SP.score_dataset(syn["ds"], {"J": syn["runs"]["J"], "K": syn["runs"]["K"]}, "interior", pairs=[("J", "K")], B=50)
    assert rj["absolute"]["J"]["status"] == "OK" and rj["absolute"]["K"]["status"] == "OK"
    assert rj["pairs"]["J|K"]["status"] == "NA: empty paired fold" and rj["pairs"]["J|K"]["empty_folds"] == [2]


def test_recall5_expected_hits_against_enumeration():
    def brute(p, y):
        q = np.quantile(y, 0.95); T = set(np.where(y >= q)[0]); k = len(T)
        if k in (0, len(y)): return None
        order = sorted(range(len(p)), key=lambda i: -p[i]); v = p[order[k - 1]]
        gt = [i for i in range(len(p)) if p[i] > v]; eq = [i for i in range(len(p)) if p[i] == v]; need = k - len(gt)
        combos = list(itertools.combinations(eq, need)) if need > 0 else [()]
        return float(np.mean([len(T & (set(gt) | set(c))) for c in combos])) / k
    r = np.random.default_rng(3)
    for _ in range(300):
        n = int(r.integers(8, 40)); y = r.integers(0, 4, n).astype(float); p = r.integers(0, 3, n).astype(float)
        a, b = SP.recall5(p, y), brute(p, y)
        assert (a is None) == (b is None) and (a is None or abs(a - b) < 1e-12)


def test_spearman_average_ranks():
    from scipy.stats import spearmanr
    r = np.random.default_rng(5)
    for _ in range(50):
        a = r.integers(0, 5, 30).astype(float); b = r.normal(size=30)
        assert abs(SP.spearman(a, b) - spearmanr(a, b)[0]) < 1e-12


def test_unit_cohesion_in_bootstrap(syn):
    ds = syn["ds"]; r2 = np.random.default_rng(1)
    sets = {1: ["SAMEU_a", "SAMEU_b", [t for t in syn["tx_fold"] if syn["tx_fold"][t] == 1][0]]}
    for _ in range(200):
        b = SP.bootstrap_units(ds, sets, r2)[1]
        assert b.count("SAMEU_a") == b.count("SAMEU_b")


def test_bootstrap_is_reproducible(syn):
    a = SP.score_dataset(syn["ds"], syn["runs"], "interior", pairs=[("A", "B")], B=100)["pairs"]["A|B"]
    b = SP.score_dataset(syn["ds"], syn["runs"], "interior", pairs=[("A", "B")], B=100)["pairs"]["A|B"]
    assert a["ci95"] == b["ci95"] and a["se"] == b["se"]


def test_holm_and_p_values():
    rej = SP.holm({"x": 0.001, "y": None, "z": 0.02}, 0.05)
    assert rej["x"] and not rej["y"] and rej["z"]                 # 0.001 <= 0.05/3, 0.02 <= 0.05/2; None stays with p = 1
    rej = SP.holm({"x": 0.001, "y": None, "z": 0.03}, 0.05)
    assert rej["x"] and not rej["z"]                             # 0.03 > 0.05/2
    rej = SP.holm({f"c{i}": (0.001 if i < 23 else None) for i in range(24)})
    assert sum(rej.values()) == 23                               # failed cell keeps m = 24
    ps, pe = SP.p_values(0.05, 0.01)
    assert abs(ps - (1 - SP._phi(4.0))) < 1e-12 and abs(pe - SP._phi(4.0)) < 1e-12
    assert SP.p_values(0.02, 0.0) == (0.0, 1.0)                   # SE = 0: p by the sign of the boundary comparison
    assert SP.classify(0.011, 0.02) == "superior" and SP.classify(-0.02, -0.011) == "inferior"
    assert SP.classify(-0.009, 0.009) == "equivalent" and SP.classify(0.005, 0.02) == "undetermined"


def test_protocol_shipped_and_frozen():
    P = SP.load_protocol()
    assert P["version"] and P["decision"]["delta"] == SP.DELTA and P["decision"]["families"]["superiority"]["m"] == 24
    assert P["seeds"]["preregistered"] == [123, 124, 125]


def test_seed_detection_and_completeness(syn):
    root = syn["root"]
    assert E.seeds_of(root, "B", "SYN", ["123"]) == ["det"] and E.seeds_of(root, "A", "SYN", ["123"]) == ["123"]
    assert E.complete(root, "A", "SYN", ["123", "124", "125"])[0] and not E.complete(root, "A", "SYN", ["126"])[0]


def test_cli_registration():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd"); EV.register(sp)
    a = ap.parse_args(["compare", "--runs", "r", "--dataset", "GSE133393_WT", "--a", "X", "--b", "Y", "--B", "10"])
    assert a.func is EV._cmd_compare and a.B == 10 and a.seeds == ["123"]
    a = ap.parse_args(["evaluate-partI", "--runs", "r", "--out", "o", "--what", "all", "--seeds", "123", "124", "125"])
    assert a.func is EV._cmd_evaluate and a.seeds == ["123", "124", "125"]
    a = ap.parse_args(["score", "--runs", "r", "--dataset", "D", "--config", "X", "Y", "--metric", "recall5"])
    assert a.func is EV._cmd_score and a.config == ["X", "Y"]
