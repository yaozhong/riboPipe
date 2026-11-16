"""Posterior estimate of a low-coverage profile: Dirichlet-multinomial posterior mean of the sparse reads with the
sequence prediction as prior, and its comparison with sparse reads and with the prediction in the crossover bins.

Posterior (interior codons [20, L-20), N = number of input reads there, L = number of interior codons):
    p_seq   = (max(yhat, 0) + 0.01 * mean(max(yhat, 0))) normalised to sum 1   (prediction as a distribution)
    post_i  = (A_i + kappa * L * p_seq_i) / (N + kappa * L)                    (kappa in reads/codon)
kappa -> 0 gives the sparse reads, kappa -> infinity gives p_seq.

Prior strength. kappa is chosen per (replicate assignment, fold) on the high-coverage transcripts (Part I universe)
of the OTHER folds: their input replicate is thinned to 0.02-1.0 reads/codon (three thinning seeds), the
posterior is computed for every kappa on a 33-point log grid (1e-3..10), and the kappa maximising the mean
log-likelihood gain of the other replicate's reads is kept. The low-coverage transcripts of Part II fold f then
use the kappa chosen for fold f.
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np

from . import crossover as X
from . import libraries as LB

EDGE = 20                       # interior [20, L-20), same as libraries.INTERIOR
GRID = 10.0 ** np.linspace(-3, 1, 33)
UNIF_MIX = 0.01
DEPTHS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
THIN_SEEDS = [0, 1, 2]
MIN_INTERIOR = 60               # calibration transcripts need >= 60 interior codons
REF_MIN_DEPTH = 1.0             # calibration records need >= 1 read/codon in the reference replicate (interior)
assert EDGE == LB.INTERIOR

C1, C2, C3 = "posterior-reads", "posterior-RiboPipe", "RiboPipe-reads"
COMP = [("posterior", "reads"), ("posterior", "RiboPipe"), ("RiboPipe", "reads")]
MARGIN = -0.01


# ---------------------------------------------------------------- estimators
def mn(x):
    x = np.asarray(x, float); s = x.sum()
    return np.ones_like(x) if s <= 0 else x * (len(x) / s)


def p_seq_of(yhat):
    y = np.clip(np.asarray(yhat, float), 0, None); m = y.mean()
    y = y + 0.01 * (m if m > 0 else 1.0); return y / y.sum()


def posterior_grid(A, pseq, grid=GRID):
    """(G, L) posterior means; kappa in reads/codon, prior weight = kappa * L."""
    L = len(A); N = A.sum(); a = grid[:, None] * L
    return (A[None, :] + a * pseq[None, :]) / (N + a)


def hybrid_grid(A, yhat, grid=GRID, slope=0.5):
    """Depth-weighted mixture of normalised reads and normalised prediction (comparison estimator of the ranking
    analysis; weight = logistic in log depth around the grid value)."""
    L = len(A); N = A.sum(); my = mn(np.clip(yhat, 0, None))
    if N <= 0: return np.repeat(my[None, :], len(grid), 0)
    d = N / L; w = 1.0 / (1.0 + np.exp(-(np.log(d) - np.log(grid)) / slope))
    return w[:, None] * mn(A)[None, :] + (1 - w)[:, None] * my[None, :]


def ll_grid(P, B):
    """P: (G, L) non-negative profiles; returns per row the log2-likelihood gain per read of B over uniform,
    after mixing 1% uniform."""
    n = P.shape[1]; s = P.sum(1, keepdims=True)
    p = np.where(s > 0, P / np.where(s > 0, s, 1), 1.0 / n)
    p = (1 - UNIF_MIX) * p + UNIF_MIX / n
    return (np.log2(p * n) @ B) / B.sum()


def posterior_full(inp, pred, kappa):
    """Posterior on the interior, placed back at length L (zeros outside the interior; scoring uses the interior
    only). Returns (posterior, p_seq), both length L."""
    L = len(inp); sl = slice(EDGE, L - EDGE)
    A = np.asarray(inp, np.int64)[sl]; ps = p_seq_of(np.clip(np.asarray(pred, float)[sl], 0, None))
    post = posterior_grid(A, ps, np.array([float(kappa)]))[0]
    full = np.zeros(L); full[sl] = post
    pfull = np.zeros(L); pfull[sl] = ps
    return full, pfull


# ---------------------------------------------------------------- prior strength (calibration on high coverage)
def calibration_transcripts(ds1, priors):
    """ds1: Part I dataset (high-coverage universe). priors: {name: {tx: out-of-fold prediction}}.
    Returns the calibration list (sorted ids, interior >= 60 codons); its index ti seeds the thinning."""
    fold_of = {t: f for f in ds1.folds for t in ds1.test_ids(f)}; txs = []
    for t in sorted(ds1.ids):
        r = np.asarray(ds1.reps(t)).astype(np.int64); n = r.shape[1]
        if n - 2 * EDGE < MIN_INTERIOR: continue
        T = dict(tx=t, unit=ds1.unit[t], fold=fold_of[t], reps=r)
        for k, p in priors.items(): T[k] = np.asarray(p[t], np.float64)
        txs.append(T)
    return txs


def thin(A, prob, key):
    rng = np.random.default_rng(key); return rng.binomial(A, prob)


def iter_calibration_records(txs):
    """(ti, role, depth index, seed, thinned input (full length), reference interior, interior slice) for every
    eligible record: reference interior depth >= 1 read/codon and input interior total >= d * L_int."""
    for ti, T in enumerate(txs):
        r = T["reps"]; n = r.shape[1]; sl = slice(EDGE, n - EDGE); Lint = n - 2 * EDGE
        for role in (0, 1):
            A0, B = (r[0], r[1]) if role == 0 else (r[1], r[0])
            Bint = B[sl]
            if Bint.sum() / Lint < REF_MIN_DEPTH: continue
            Aint_tot = int(A0[sl].sum())
            for di, d in enumerate(DEPTHS):
                if Aint_tot < d * Lint: continue
                prob = d * Lint / Aint_tot
                for s in THIN_SEEDS:
                    yield ti, role, di, s, thin(A0, prob, [s, ti, role, di]), Bint, sl


def grid_likelihoods(txs, posterior_priors=("rp",), hybrid_prior=None):
    """Sum over calibration records of the per-kappa log-likelihood gain, per (role, fold).
    posterior_priors: prior keys of txs used as posterior priors; hybrid_prior: key for the hybrid estimator."""
    G = {}
    for ti, role, di, s, Af, Bint, sl in iter_calibration_records(txs):
        T = txs[ti]; A = Af[sl]
        g = G.setdefault((role, T["fold"]), {**{k: np.zeros(len(GRID)) for k in posterior_priors}, "n": 0,
                                             **({"hyb": np.zeros(len(GRID))} if hybrid_prior else {})})
        for k in posterior_priors:
            g[k] += ll_grid(posterior_grid(A, p_seq_of(T[k][sl])), Bint)
        if hybrid_prior: g["hyb"] += ll_grid(hybrid_grid(A, T[hybrid_prior][sl]), Bint)
        g["n"] += 1
    return G


def choose_kappa(G, keys):
    """Cross-fitted choice: for (role, f) the grid value maximising the mean gain over the other folds."""
    folds = sorted({f for _, f in G}); chosen = {}
    for role in (0, 1):
        for f in folds:
            other = [G[(role, g)] for g in folds if g != f and (role, g) in G]
            n = sum(o["n"] for o in other); ch = {}
            for key in keys:
                tot = sum(o[key] for o in other) / n; j = int(np.argmax(tot))
                ch[key] = dict(value=float(GRID[j]), grid_index=j, at_edge=bool(j in (0, len(GRID) - 1)), mean_ll=float(tot[j]))
            ch["n_records_fit"] = n; chosen[(role, f)] = ch
    return chosen


def select_kappa(ds1, oof_pred, log=print):
    """Prior strength for the posterior with the given out-of-fold predictions (Part I universe).
    Returns {(role, fold): kappa} and the full choice table."""
    t0 = time.time()
    txs = calibration_transcripts(ds1, {"rp": oof_pred})
    G = grid_likelihoods(txs, ("rp",))
    ch = choose_kappa(G, ("rp",))
    log(f"[kappa] {ds1.tag} {len(txs)} calibration transcripts {time.time() - t0:.0f}s")
    return {k: v["rp"]["value"] for k, v in ch.items()}, ch


def kappa_to_json(ch, tag, key="rp"):
    return dict(dataset=tag, grid=[float(x) for x in GRID], depths=DEPTHS, thin_seeds=THIN_SEEDS,
                chosen={f"role{r}|fold{f}": dict(value=v[key]["value"], grid_index=v[key]["grid_index"], at_edge=v[key]["at_edge"],
                                                 mean_ll=v[key]["mean_ll"], n_records_fit=v["n_records_fit"])
                        for (r, f), v in sorted(ch.items())})


def kappa_from_json(fn):
    j = json.load(open(fn)); out = {}
    for k, v in j["chosen"].items():
        r = int(k.split("|")[0][4:]); f = int(k.split("|")[1][4:])
        val = v["value"] if "value" in v else v["prp"]["value"]       # also accepts the ranking analysis output
        out[(r, f)] = float(val)
    return out


# ---------------------------------------------------------------- posterior in the crossover bins
def cat(d, E):
    if d == 0: return "zero"
    if d < E[0]: return "below"
    if d >= E[-1]: return "above"
    return int(np.searchsorted(np.asarray(E), d, side="right") - 1)


def build_records(ds, pred, kappa, draw, E, identity_check=False):
    """One record per (T_low transcript, role) with the correlations of reads, prediction and posterior with the
    reference replicate (same libraries as the crossover main analysis, matching seed = draw)."""
    tag = ds.tag; fold_of = {t: f for f in ds.folds for t in ds.test_ids(f)}; recs = []
    for t in sorted(ds.strata["low"]):
        r2 = np.asarray(ds.reps(t)); A, B = r2[0].astype(np.int64), r2[1].astype(np.int64); L = len(A); m = min(int(A.sum()), int(B.sum()))
        libs, info = LB.libraries("native", A, B, draw, tag, t); assert info["m"] == m
        f = fold_of[t]; v = pred[t]; assert len(v) == L
        for role in (0, 1):
            inp, ref, lin, lref = libs[role]; assert lin != lref
            n_in, n_ref = int(inp.sum()), int(ref.sum())
            t_ok, t_why = LB.truth_status(ref); refy = ref / ref.mean() if n_ref > 0 else ref.astype(float)
            if not t_ok: rr, rwhy = None, t_why
            elif n_in == 0: rr, rwhy = None, "input_zero"
            else: rr, rwhy = LB.pearson_interior(inp / inp.mean(), refy)
            rm, mwhy = LB.pearson_interior(v, refy) if t_ok else (None, t_why)
            k = kappa[(role, f)]
            rec = dict(tx=t, unit=ds.unit[t], fold=f, role=role, L=L, R1=int(A.sum()), R2=int(B.sum()), d_axis=m / L, input_total=n_in,
                       input_interior=int(inp[EDGE:L - EDGE].sum()) if L > 2 * EDGE else 0, reference_total=n_ref,
                       truth_scorable=t_ok, kappa=k, reads_r=rr, reads_reason=rwhy, rp_r=rm, rp_reason=mwhy, post_r=None, post_reason=None)
            if not t_ok: rec["post_reason"] = t_why
            elif n_in == 0: rec["post_reason"] = "input_zero"
            else:
                post, ps = posterior_full(inp, v, k)
                rec["post_r"], rec["post_reason"] = LB.pearson_interior(post, refy)
                if identity_check and rr is not None:
                    p0, _ = posterior_full(inp, v, 0.0); pinf, _ = posterior_full(inp, v, 1e12)
                    rec["g2_post_k0_r"] = LB.pearson_interior(p0, refy)[0]
                    rec["g2_post_kinf_r"] = LB.pearson_interior(pinf, refy)[0]
                    rec["g2_pseq_r"] = LB.pearson_interior(ps, refy)[0]
            rec["_cat"] = cat(m / L, E)
            recs.append(rec)
    return recs


def tuples(recs, a, b, common):
    """Records entering D* (below / above the axis excluded; zero depth d = 0). common=True: a record counts only
    if reads, prediction and posterior are all scorable (the three-way common set)."""
    out = []
    for r in recs:
        c = r["_cat"]
        if c in ("below", "above"): continue
        ok3 = r["reads_r"] is not None and r["rp_r"] is not None and r["post_r"] is not None
        va = {"posterior": r["post_r"], "RiboPipe": r["rp_r"], "reads": r["reads_r"]}
        x, y = va[a], va[b]
        if common and not ok3: x, y = None, None
        out.append((r["unit"], r["role"], 0.0 if c == "zero" else r["d_axis"], x, y))
    return out


_G = {}


def _job(args):
    name, B = args; t0 = time.time()
    out = X.dstar_with_curves(_G["tup"][name], _G["E"], B=B, seed=0)
    out["_sec"] = round(time.time() - t0, 1); return name, out


def score_posterior(ds, pred, kappa, edges, draw=0, B=2000, workers=4, identity_check=False, log=print):
    """Paired comparisons posterior - reads, posterior - prediction and prediction - reads (three-way common set),
    plus prediction - reads on the main set (must equal the crossover main analysis). Returns a dict."""
    t0 = time.time(); E = list(edges); _G["E"] = E
    recs = build_records(ds, pred, kappa, draw, E, identity_check)
    _G["tup"] = {"main_RiboPipe-reads": tuples(recs, "RiboPipe", "reads", common=False)}
    for x, y in COMP: _G["tup"][f"{x}-{y}"] = tuples(recs, x, y, common=True)
    jobs = [(n, B) for n in _G["tup"]]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool: results = list(pool.imap_unordered(_job, jobs))
    else: results = [_job(j) for j in jobs]
    comps = dict(results); comps = {n: comps[n] for n in _G["tup"]}
    for n, o in comps.items(): log(f"[dstar] {ds.tag} draw {draw} {n} {o['status']} {o['D_star']} {o['ci95']} {o['_sec']}s")
    res = dict(dataset=ds.tag, draw=draw, T_low=len(ds.strata["low"]), edges=E,
               kappa={f"role{r}|fold{f}": v for (r, f), v in sorted(kappa.items())}, comparisons=comps)
    res["n_common_minus_main_per_bin"] = [[a_ - b_ for a_, b_ in zip(r1["N_scored_role"], r2["N_scored_role"])]
                                          for r1, r2 in zip(comps["RiboPipe-reads"]["bins"], comps["main_RiboPipe-reads"]["bins"])]
    if identity_check:
        g = [r for r in recs if "g2_post_k0_r" in r]
        d0 = [abs(r["g2_post_k0_r"] - r["reads_r"]) for r in g if r["g2_post_k0_r"] is not None]
        dinf = [abs(r["g2_post_kinf_r"] - r["g2_pseq_r"]) for r in g if r["g2_post_kinf_r"] is not None and r["g2_pseq_r"] is not None]
        res["identity_check"] = dict(n_records=len(g), max_abs_k0_minus_reads=max(d0) if d0 else None,
                                     max_abs_kinf_minus_pseq=max(dinf) if dinf else None,
                                     n_k0_undefined=sum(r["g2_post_k0_r"] is None for r in g),
                                     pass_=bool(d0 and max(d0) <= 1e-9 and (not dinf or max(dinf) <= 1e-9)
                                                and not any(r["g2_post_k0_r"] is None for r in g)))
    res["_records"] = recs
    res["wall_sec"] = round(time.time() - t0, 1)
    return res


def bin_rows(sc, boot=True):
    """Valid bins of the three-way common set (the three comparisons share validity)."""
    b1, b2, b3 = (sc["comparisons"][c]["bins"] for c in (C1, C2, C3))
    bb = {c: sc["comparisons"][c].get("bin_boot") for c in (C1, C2, C3)}
    rows = []
    for j, (x1, x2, x3) in enumerate(zip(b1, b2, b3)):
        assert x1["valid"] == x2["valid"] == x3["valid"] and x1["N_scored_role"] == x2["N_scored_role"] == x3["N_scored_role"]
        if not x1["valid"]: continue
        r = dict(bin=j, lo=x1["lo"], hi=x1["hi"], N_scored_role=x1["N_scored_role"])
        for c, x in ((C1, x1), (C2, x2), (C3, x3)):
            r[c] = dict(y=x["y"], y_role=x["y_role"], ci95=(bb[c][j]["ci95"] if boot and bb[c] else None),
                        n_valid_draws=(bb[c][j]["n_valid_draws"] if boot and bb[c] else None))
        rows.append(r)
    return rows


def summarize_bins(rows):
    n = len(rows)
    out = dict(n_valid_bins=n,
               n_post_ge_both=sum(r[C1]["y"] >= 0 and r[C2]["y"] >= 0 for r in rows),
               n_post_gt_both=sum(r[C1]["y"] > 0 and r[C2]["y"] > 0 for r in rows),
               n_both_ge_margin=sum(r[C1]["y"] >= MARGIN and r[C2]["y"] >= MARGIN for r in rows),
               bins_not_ge_margin=[r["bin"] for r in rows if not (r[C1]["y"] >= MARGIN and r[C2]["y"] >= MARGIN)],
               bins_not_gt_both=[r["bin"] for r in rows if not (r[C1]["y"] > 0 and r[C2]["y"] > 0)])
    for c in (C1, C2, C3):
        m = min(rows, key=lambda r: r[c]["y"]) if rows else None
        out[f"min|{c}"] = dict(y=m[c]["y"], bin=m["bin"], lo=m["lo"], hi=m["hi"], ci95=m[c]["ci95"], N_scored_role=m["N_scored_role"]) if m else None
    if rows and rows[0][C1]["ci95"] is not None:
        s1 = [r["bin"] for r in rows if r[C1]["ci95"] is not None and r[C1]["ci95"][1] < 0]
        s2 = [r["bin"] for r in rows if r[C2]["ci95"] is not None and r[C2]["ci95"][1] < 0]
        out.update(sig_neg_bins_post_minus_reads=s1, sig_neg_bins_post_minus_ribopipe=s2, n_sig_neg_either=len(set(s1) | set(s2)),
                   n_ci_missing=sum(r[c]["ci95"] is None for r in rows for c in (C1, C2)))
        for c, nm in ((C1, "post_minus_reads"), (C2, "post_minus_ribopipe")):
            out[f"n_ci_above_zero|{nm}"] = sum(r[c]["ci95"] is not None and r[c]["ci95"][0] > 0 for r in rows)
            out[f"n_point_above_zero|{nm}"] = sum(r[c]["y"] > 0 for r in rows)
    return out


def reading(per):
    """Pre-set decision rule: posterior recommended for all transcripts if, in every valid bin of every dataset,
    both posterior - reads and posterior - prediction are >= -0.01 and in >= 80% of the valid bins both are > 0."""
    allm = all(p["n_both_ge_margin"] == p["n_valid_bins"] for p in per.values())
    nv = sum(p["n_valid_bins"] for p in per.values()); ngt = sum(p["n_post_gt_both"] for p in per.values())
    frac = ngt / nv if nv else None
    return dict(all_bins_both_ge_margin=allm, pooled_n_valid=nv, pooled_n_gt_both=ngt, pooled_frac_gt_both=frac,
                per_dataset_frac_gt_both={t: (p["n_post_gt_both"] / p["n_valid_bins"] if p["n_valid_bins"] else None) for t, p in per.items()},
                holds=bool(allm and frac is not None and frac >= 0.8))


def aggregate(scores):
    """scores: {tag: score_posterior() output (draw 0)} -> results dict (per-bin rows, per-dataset summary, rule)."""
    res = dict(datasets={}); per = {}
    for t, s in scores.items():
        rows = bin_rows(s); per[t] = summarize_bins(rows)
        res["datasets"][t] = dict(kappa=s["kappa"], T_low=s["T_low"], bins=rows, summary=per[t],
                                  dstar={c: {k: s["comparisons"][c].get(k) for k in ("status", "D_star", "ci95", "bootstrap_fail_fraction")}
                                         for c in (C1, C2, C3)},
                                  zero_depth=s["comparisons"][C3]["zero_depth"])
    res["reading"] = reading(per)
    res["pooled"] = {k: sum(p.get(k, 0) for p in per.values()) for k in
                     ("n_valid_bins", "n_ci_above_zero|post_minus_reads", "n_ci_above_zero|post_minus_ribopipe",
                      "n_point_above_zero|post_minus_reads", "n_point_above_zero|post_minus_ribopipe", "n_post_gt_both")}
    return res
