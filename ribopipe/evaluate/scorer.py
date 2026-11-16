"""Part I scorer (frozen protocol PROTOCOL_PartI v1.3, shipped as PROTOCOL_PartI_v1.3.json next to this file).

Faithful port of the paper's ``paper_eval/score_partI.py``: every rule, every set definition, the order of the
bootstrap draws and the random seed are unchanged, so rescoring the paper's stored predictions reproduces the
published numbers (see tests/verification/evaluate_verification.md).

Input (the common output format of every runner, written by ``ribopipe.data.write_run``):
  <runs>/<config_id>/<dataset>/seed<s>/fold<f>/pred.npz   tx_id -> float array (length L codons, A site; NaN = no output)
  <runs>/<config_id>/<dataset>/seed<s>/fold<f>/prov.npz   tx_id -> int8 array (0 = model, 1 = adapter, 2 = undefined)
  <runs>/<config_id>/<dataset>/seed<s>/fold<f>/run.json   {"status": "OK" | "MODEL_FAILED", ...}
  Deterministic methods (trained once) use the seed directory name "seeddet".
Truth: y = c / mean_full(c) (denominator over the whole CDS, zeros included). Scoring domains: interior = [20, L-20),
full = [0, L) (model positions only) and full_deployed = [0, L) (model and adapter positions).

Definitions
  Truth-scorable: >= 3 domain positions and y non-constant on the domain (otherwise NO_SCORABLE_TRUTH).
  Method-scorable (one seed): run status OK; every domain position has an allowed provenance (interior / full: model only;
      full_deployed: model or adapter) and a finite value; prediction non-constant on the domain. A transcript that is not
      method-scorable only leaves that method's set.
  MODEL_FAILED: run.json says failed, or every prediction of a fold is non-finite or constant. Any failed pre-registered
      seed x fold -> the method is FAILED on that dataset.
  T_A,f: truth-scorable and A scorable in every pre-registered seed. T_AB,f: truth-scorable and A and B each scorable in
      every pre-registered seed (one set for all seeds of both methods).
  M(s,f;T) = median over T of the per-transcript metric; S(T) = mean over seeds of [mean over the five folds of M].
      An empty fold set -> NA (never re-averaged over the remaining folds).
  Delta(A,B) = S_A(T_AB) - S_B(T_AB). Bootstrap: within each fold, resample with replacement the similarity units present in
      T_AB,f (same number of units); one draw is used for both methods and all seeds; B = 2000; 95% percentile interval;
      SE = bootstrap standard deviation (ddof 1); numpy default_rng(0).
  p values (boundary-shifted Wald): p_sup = 1 - Phi((Delta - delta)/SE); TOST: p_lo = 1 - Phi((Delta + delta)/SE),
      p_hi = Phi((Delta - delta)/SE), p_eq = max(p_lo, p_hi). delta = 0.01.
  Class: interval entirely > +delta superior; entirely < -delta inferior; entirely inside (-delta, +delta) equivalent;
      otherwise undetermined. FAILED / NA are reported separately.
"""
import os, json, math, hashlib
import numpy as np

MODEL, ADAPTER, UNDEFINED = 0, 1, 2
DELTA = 0.01
B_DEFAULT = 2000
PROTOCOL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROTOCOL_PartI_v1.3.json")
PROTOCOL_MD5 = "697395008f5c3c05fd1eb0f22eae3eca"


def load_protocol():
    """The frozen Part I protocol (md5-checked)."""
    raw = open(PROTOCOL_FILE, "rb").read()
    assert hashlib.md5(raw).hexdigest() == PROTOCOL_MD5, "PROTOCOL_PartI_v1.3.json has been modified"
    return json.loads(raw)


def domain_index(L, domain):
    if domain == "interior":
        return np.arange(20, L - 20) if L > 40 else np.arange(0)
    if domain in ("full", "full_deployed"):
        return np.arange(L)
    raise ValueError(domain)


def truth_scorable(y, idx):
    return len(idx) >= 3 and float(np.std(y[idx])) > 0


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def _rank(a):
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[order] = np.arange(len(a), dtype=float)
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals)); np.add.at(sums, inv, r); return (sums / cnt)[inv]


def spearman(a, b): return pearson(_rank(np.asarray(a, float)), _rank(np.asarray(b, float)))


def recall5(pred, y):
    """Top-5% peak recall (protocol v1.3 metrics.peak_recall_rule). True peaks: domain positions with
    y >= numpy.quantile(y, 0.95) (linear interpolation); all truth ties at the cut are included; k = number of true peaks.
    Predicted top-k: positions strictly above the k-th largest prediction are taken; positions tied at the k-th value
    contribute expected hits = (k - n_strictly_above) * (true peaks among the tied) / (number tied).
    Returns expected hits / k; None (unscorable) when k = 0 or k = domain length."""
    y = np.asarray(y, float); p = np.asarray(pred, float)
    true = y >= np.quantile(y, 0.95); k = int(true.sum())
    if k == 0 or k == len(y): return None
    v = np.sort(p)[::-1][k - 1]; gt = p > v; eq = p == v
    need = k - int(gt.sum())
    hits = float((gt & true).sum()) + (need * float((eq & true).sum()) / float(eq.sum()) if need > 0 else 0.0)
    return hits / k


METRICS = {"pearson": pearson, "spearman": spearman, "recall5": recall5}


class Dataset:
    """Scoring view of a dataset: tx -> test fold, similarity unit and truth y (full-CDS denominator)."""
    def __init__(self, tx_fold, tx_unit, tx_counts):
        self.fold = dict(tx_fold); self.unit = dict(tx_unit)
        self.y = {k: np.asarray(c, float) / np.asarray(c, float).mean() for k, c in tx_counts.items()}
        self.folds = sorted(set(self.fold.values()))

    @classmethod
    def from_release(cls, ds):
        """Build from a ``ribopipe.data.PaperDataset`` (test folds of the Part I universe)."""
        fold = {t: f for f in ds.folds for t in ds.test_ids(f)}
        return cls(fold, ds.unit, ds.counts)


def load_method_runs(root, config_id, dataset, seeds, folds):
    """runs[seed][fold] = (status, pred_dict, prov_dict); a missing run.json counts as MODEL_FAILED."""
    runs = {}
    for s in seeds:
        runs[s] = {}
        for f in folds:
            d = os.path.join(root, config_id, dataset, f"seed{s}", f"fold{f}")
            if not os.path.exists(os.path.join(d, "run.json")):
                runs[s][f] = ("MODEL_FAILED", {}, {}); continue
            st = json.load(open(os.path.join(d, "run.json"))).get("status", "MODEL_FAILED")
            if st != "OK":
                runs[s][f] = ("MODEL_FAILED", {}, {}); continue
            p = np.load(os.path.join(d, "pred.npz")); q = np.load(os.path.join(d, "prov.npz"))
            runs[s][f] = ("OK", {k: p[k] for k in p.files}, {k: q[k] for k in q.files})
    return runs


def method_table(ds, runs, domain, metric="pearson"):
    """Returns (failed, per_tx, adapter_fraction); per_tx[seed][tx] = metric value or None (not scorable)."""
    allowed = {MODEL} if domain == "interior" or domain == "full" else {MODEL, ADAPTER}
    failed = False; per = {}; adapter_frac = []
    for s, by_fold in runs.items():
        per[s] = {}
        for f, (st, pred, prov) in by_fold.items():
            txs = [t for t in ds.fold if ds.fold[t] == f]
            if st != "OK":
                failed = True; continue
            ok_any = False
            for t in txs:
                y = ds.y[t]; L = len(y); idx = domain_index(L, domain)
                p = pred.get(t); q = prov.get(t)
                if p is None or q is None or len(p) != L or len(q) != L:
                    per[s][t] = None; continue
                if len(idx) == 0:
                    pall = np.asarray(p, float); fin = np.isfinite(pall)
                    if fin.any() and float(np.std(pall[fin])) > 0: ok_any = True
                    per[s][t] = None; continue
                pall = np.asarray(p, float); fin = np.isfinite(pall)
                if fin.any() and float(np.std(pall[fin])) > 0:
                    ok_any = True
                pv = pall[idx]; qv = np.asarray(q)[idx]
                if not set(np.unique(qv).tolist()) <= allowed or not np.isfinite(pv).all() or float(np.std(pv)) == 0:
                    per[s][t] = None; continue
                if domain == "full_deployed": adapter_frac.append(float((qv == ADAPTER).mean()))
                per[s][t] = METRICS[metric](pv, y[idx])
            if not ok_any:
                failed = True   # every prediction of the fold non-finite or constant
    return failed, per, (float(np.mean(adapter_frac)) if adapter_frac else 0.0)


def scorable_set(ds, per_list, domain, f):
    """Truth-scorable transcripts of fold f scored by every method of per_list in all of its seeds."""
    out = []
    for t in ds.fold:
        if ds.fold[t] != f: continue
        y = ds.y[t]
        if not truth_scorable(y, domain_index(len(y), domain)): continue
        if all(per[s].get(t) is not None for per in per_list for s in per):
            out.append(t)
    return sorted(out)


def empty_folds(sets): return sorted(int(f) for f, v in sets.items() if len(v) == 0)


def S_on(per, sets):
    """S = mean over seeds of [mean over folds of the median]; sets[f] = transcript list (may contain repeats, as in a
    bootstrap draw). Any empty fold -> None (the five-fold quantity is NA; never re-averaged over four folds).
    Returns (S, per-seed scores)."""
    seeds = sorted(per)
    vals = []
    for s in seeds:
        fm = []
        for f in sorted(sets):
            if len(sets[f]) == 0: return None
            fm.append(float(np.median([per[s][t] for t in sets[f]])))
        vals.append(float(np.mean(fm)))
    return float(np.mean(vals)), vals


def _phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bootstrap_units(ds, sets, rng):
    """One draw: within each fold, resample with replacement the units present in the set; returns {f: transcript multiset}."""
    out = {}
    for f, txs in sets.items():
        units = {}
        for t in txs: units.setdefault(ds.unit[t], []).append(t)
        keys = sorted(units)
        draw = rng.integers(0, len(keys), len(keys))
        out[f] = [t for i in draw for t in units[keys[i]]]
    return out


def classify(lo, hi, delta=DELTA):
    return ("superior" if lo > delta else "inferior" if hi < -delta else
            "equivalent" if (lo > -delta and hi < delta) else "undetermined")


def p_values(d, se, delta=DELTA):
    """Boundary-shifted Wald p values: (p_superiority, p_equivalence_tost)."""
    p_sup = 1 - _phi((d - delta) / se) if se > 0 else float(d <= delta)
    p_lo = 1 - _phi((d + delta) / se) if se > 0 else float(d <= -delta)
    p_hi = _phi((d - delta) / se) if se > 0 else float(d >= delta)
    return p_sup, max(p_lo, p_hi)


def score_dataset(ds, methods, domain="interior", pairs=(), B=B_DEFAULT, seed=0, delta=DELTA, metric="pearson"):
    """methods: {name: runs} (from load_method_runs). Returns absolute scores and paired results."""
    tabs = {m: method_table(ds, r, domain, metric) for m, r in methods.items()}
    res = {"domain": domain, "metric": metric, "absolute": {}, "pairs": {}}
    n_domain = sum(1 for t in ds.fold if len(domain_index(len(ds.y[t]), domain)) > 0)
    n_truth = sum(1 for t in ds.fold if truth_scorable(ds.y[t], domain_index(len(ds.y[t]), domain)))
    for m, (failed, per, af) in tabs.items():
        if failed:
            res["absolute"][m] = {"status": "FAILED"}; continue
        sets = {f: scorable_set(ds, [per], domain, f) for f in ds.folds}
        sv = S_on(per, sets)
        if sv is None:
            res["absolute"][m] = {"status": "NA: empty fold", "empty_folds": empty_folds(sets)}; continue
        ns = sum(len(v) for v in sets.values())
        res["absolute"][m] = {"status": "OK", "S": sv[0], "seed_scores": sv[1], "min": min(sv[1]), "max": max(sv[1]),
                              "N_domain": n_domain, "N_truth_scorable": n_truth, "N_scored": ns,
                              "scored_over_domain": ns / n_domain, "scored_over_truth_scorable": ns / n_truth,
                              "report": f"{sv[0]:.3f} [{min(sv[1]):.3f}, {max(sv[1]):.3f}]  N {ns}/{n_domain}",
                              "adapter_fraction": af}
    for a, b in pairs:
        fa, pa, _ = tabs[a]; fb, pb, _ = tabs[b]
        if fa or fb:
            res["pairs"][f"{a}|{b}"] = {"status": "FAILED"}; continue
        sets = {f: scorable_set(ds, [pa, pb], domain, f) for f in ds.folds}
        A = S_on(pa, sets); Bv = S_on(pb, sets)
        if A is None or Bv is None:
            ef = empty_folds(sets)
            res["pairs"][f"{a}|{b}"] = {"status": "NA: empty paired fold" if len(ef) < len(sets) else "NO_PAIRED_SUPPORT",
                                        "empty_folds": ef, "N_paired_per_fold": {int(f): len(v) for f, v in sets.items()}}; continue
        d = A[0] - Bv[0]
        rng = np.random.default_rng(seed); boots = []
        for _ in range(B):
            bs = bootstrap_units(ds, sets, rng)
            boots.append(S_on(pa, bs)[0] - S_on(pb, bs)[0])
        boots = np.array(boots); lo, hi = np.percentile(boots, [2.5, 97.5]); se = float(boots.std(ddof=1))
        p_sup, p_eq = p_values(d, se, delta)
        res["pairs"][f"{a}|{b}"] = {"status": "OK", "delta_hat": d, "ci95": [float(lo), float(hi)], "se": se,
                                    "p_superiority": p_sup, "p_equivalence_tost": p_eq, "class": classify(lo, hi, delta),
                                    "N_paired": sum(len(v) for v in sets.values()), "N_truth_scorable": n_truth,
                                    "N_paired_per_fold": {int(f): len(v) for f, v in sets.items()},
                                    "seed_deltas": [x - y for x, y in zip(A[1], Bv[1])] if len(A[1]) == len(Bv[1]) else None}
    return res


def _ci(boots):
    boots = np.asarray(boots); lo, hi = np.percentile(boots, [2.5, 97.5]); return [float(lo), float(hi)], float(boots.std(ddof=1))


def _ids_md5(sets):
    return hashlib.md5("\n".join(sorted(t for v in sets.values() for t in v)).encode()).hexdigest()


def score_interior_vs_full(ds, methods, B=B_DEFAULT, seed=0):
    """Within-method endpoint sensitivity (protocol v1.3 fig_B.secondary_within_method). One common population per method:
    truth-scorable on interior and full, and scorable by the method on interior (model only) and full_deployed (model +
    adapter) in every seed. interior, full_deployed (with adapter fraction) and full_native (model only; NA if any
    transcript of the population has no native full-length output) are computed on the same IDs (md5 returned)."""
    out = {}
    for m, r in methods.items():
        fi, pi, _ = method_table(ds, r, "interior"); ff, pf, af = method_table(ds, r, "full_deployed")
        fn, pn, _ = method_table(ds, r, "full")
        if fi or ff: out[m] = {"status": "FAILED"}; continue
        sets = {f: sorted(set(scorable_set(ds, [pi], "interior", f)) & set(scorable_set(ds, [pf], "full_deployed", f))) for f in ds.folds}
        a = S_on(pi, sets); b = S_on(pf, sets)
        if a is None or b is None: out[m] = {"status": "NA"}; continue
        missing_native = sum(1 for f, v in sets.items() for t in v if any(pn[s].get(t) is None for s in pn)) if not fn else None
        c = S_on(pn, sets) if (not fn and missing_native == 0) else None
        rng = np.random.default_rng(seed); boots = []
        for _ in range(B):
            bs = bootstrap_units(ds, sets, rng); boots.append(S_on(pf, bs)[0] - S_on(pi, bs)[0])
        ci, se = _ci(boots)
        out[m] = {"status": "OK", "S_interior": a[0], "S_full_deployed": b[0], "S_full": b[0],
                  "S_full_native": (c[0] if c else None),
                  "full_native_status": ("OK" if c else ("FAILED" if fn else f"NA ({missing_native} transcripts without native full-length output)")),
                  "full_minus_interior": b[0] - a[0], "ci95": ci, "se": se,
                  "N_common": sum(len(v) for v in sets.values()), "ids_md5": _ids_md5(sets), "adapter_fraction_full": af}
    return out


def score_pair_interior_vs_full(ds, runs_a, runs_b, B=B_DEFAULT, seed=0, delta=DELTA):
    """Pair-level interior vs full-length comparison (protocol v1.3 fig_B.primary_pair_level). One common set per pair:
    truth-scorable on interior and full, A and B scorable on interior (model only) and full_deployed in every seed.
    Delta_interior and Delta_full_deployed are computed on the same IDs with the same unit draws; Delta_full_native only
    when A and B both have native full-length output on the whole set (otherwise NA, never filled)."""
    T = {}
    for nm, r in (("A", runs_a), ("B", runs_b)):
        fi, pi, _ = method_table(ds, r, "interior"); fd, pd_, ad = method_table(ds, r, "full_deployed"); fn, pn, _ = method_table(ds, r, "full")
        if fi or fd: return {"status": "FAILED", "failed": nm}
        T[nm] = (pi, pd_, pn, fn, ad)
    sets = {f: sorted(set(scorable_set(ds, [T["A"][0], T["B"][0]], "interior", f)) & set(scorable_set(ds, [T["A"][1], T["B"][1]], "full_deployed", f))) for f in ds.folds}
    ef = empty_folds(sets)
    if ef: return {"status": "NA: empty paired fold" if len(ef) < len(sets) else "NO_PAIRED_SUPPORT", "empty_folds": ef}
    d_int = lambda st: S_on(T["A"][0], st)[0] - S_on(T["B"][0], st)[0]
    d_full = lambda st: S_on(T["A"][1], st)[0] - S_on(T["B"][1], st)[0]
    rng = np.random.default_rng(seed); bi, bf = [], []
    for _ in range(B):
        bs = bootstrap_units(ds, sets, rng); bi.append(d_int(bs)); bf.append(d_full(bs))
    cls = lambda lo, hi: classify(lo, hi, delta)
    ci_i, se_i = _ci(bi); ci_f, se_f = _ci(bf)
    native_ok = all(not T[n][3] and all(T[n][2][s].get(t) is not None for s in T[n][2] for v in sets.values() for t in v) for n in ("A", "B"))
    d_nat = (S_on(T["A"][2], sets)[0] - S_on(T["B"][2], sets)[0]) if native_ok else None
    return {"status": "OK", "N_common": sum(len(v) for v in sets.values()), "ids_md5": _ids_md5(sets),
            "delta_interior": d_int(sets), "ci95_interior": ci_i, "class_interior": cls(*ci_i),
            "delta_full_deployed": d_full(sets), "ci95_full_deployed": ci_f, "class_full_deployed": cls(*ci_f),
            "class_changed": cls(*ci_i) != cls(*ci_f), "delta_full_native": d_nat,
            "full_native_status": "OK" if native_ok else "NA (no native full-length output for A or B)",
            "adapter_fraction_full": {"A": T["A"][4], "B": T["B"][4]}}


def score_common_arms(ds, arms, domain="interior", contrast=None, B=B_DEFAULT, seed=0):
    """Common-set scoring of several arms (e.g. the four MFE-interaction arms): truth-scorable and every arm scorable in
    all of its seeds. contrast = ((a1, a0), (b1, b0)) adds the interaction (S_a1 - S_a0) - (S_b1 - S_b0) with a unit
    bootstrap interval."""
    tabs = {m: method_table(ds, r, domain) for m, r in arms.items()}
    if any(t[0] for t in tabs.values()): return {"status": "FAILED", "failed": [m for m, t in tabs.items() if t[0]]}
    pers = [t[1] for t in tabs.values()]
    sets = {f: scorable_set(ds, pers, domain, f) for f in ds.folds}
    S = {m: S_on(t[1], sets) for m, t in tabs.items()}
    if any(v is None for v in S.values()): return {"status": "NO_PAIRED_SUPPORT"}
    res = {"status": "OK", "domain": domain, "N_common": sum(len(v) for v in sets.values()), "S": {m: v[0] for m, v in S.items()}}
    if contrast:
        (a1, a0), (b1, b0) = contrast
        f_ = lambda st: (S_on(tabs[a1][1], st)[0] - S_on(tabs[a0][1], st)[0]) - (S_on(tabs[b1][1], st)[0] - S_on(tabs[b0][1], st)[0])
        rng = np.random.default_rng(seed); boots = [f_(bootstrap_units(ds, sets, rng)) for _ in range(B)]
        ci, se = _ci(boots)
        res.update(increment_a=S[a1][0] - S[a0][0], increment_b=S[b1][0] - S[b0][0], interaction=f_(sets), ci95=ci, se=se)
    return res


def holm(pvals, alpha=0.05):
    """Holm step-down. pvals: {key: p or None}; failed / NA cells (None) stay in the family with p = 1 (never rejected),
    so the family size does not shrink. Returns {key: rejected}."""
    items = sorted(((p if p is not None else 1.0), k) for k, p in pvals.items())
    m = len(items); rej = {k: False for k in pvals}
    for i, (p, k) in enumerate(items):
        if p <= alpha / (m - i): rej[k] = True
        else: break
    return rej
