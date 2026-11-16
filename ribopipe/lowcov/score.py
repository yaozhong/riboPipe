"""Crossover evaluation of one dataset (the paper's Part II, protocol PARTII_CROSSOVER_v2).

Variants
  v2_primary          main analysis: x axis = min(R1, R2) / L (pre-thinning depth of the shallower replicate,
                      identical for both roles); input = the replicate at its natural depth after expected-depth
                      matching (no per-bin thinning)
  v2_sens_inputdepth  sensitivity (a): same records, x axis = realised input depth after matching
  v2_sens_gridthin    sensitivity (b): nested per-bin thinning of the input library (same transcripts at
                      controlled depths)
  v2_sens_binom       sensitivity (c): binomial-split reference (only for UMI-deduplicated datasets), x axis
                      min(R1, R2) / L
Records are formed for the low-coverage stratum (T_low) of the Part II universe; every transcript is scored with
the prediction of the Part I model whose training / validation set excluded it (see predict.py).
"""
import csv
import gzip
import json
import multiprocessing as mp
import os
import platform
import socket
import time

import numpy as np

from . import crossover as X
from . import libraries as LB
from .preds import dummy_noise

UMI_DEDUP = ("DMSO_TX9P1", "GSE233886_WT", "PRJNA_iPS")   # GSE133393_WT is not deduplicated: no binomial split
VARIANTS = ["v2_primary", "v2_sens_inputdepth", "v2_sens_gridthin", "v2_sens_binom"]
DUMMY = "Dummy_noise"


def cat(d, e):
    if d == 0: return "zero"
    if d < e[0]: return "below"
    if d >= e[-1]: return "above"
    return int(np.searchsorted(np.asarray(e), d, side="right") - 1)


def natural_records(tag, t, A, B, preds, design, meta, seed):
    """Main analysis / (a) / (c): one record per (transcript, role); the input library is not thinned per bin."""
    L = len(A); R1, R2 = int(A.sum()), int(B.sum()); m = min(R1, R2)
    libs, info = LB.libraries("native" if design == "rep" else "binom_grid", A, B, seed, tag, t)
    assert info["m"] == m
    out = []
    for role in (0, 1):
        inp, ref, lin, lref = libs[role]; assert lin != lref
        n_in, n_ref = int(inp.sum()), int(ref.sum())
        t_ok, t_why = LB.truth_status(ref); refy = ref / ref.mean() if n_ref > 0 else ref.astype(float)
        if not t_ok: rr, rwhy = None, t_why
        elif n_in == 0: rr, rwhy = None, "input_zero"
        else: rr, rwhy = LB.pearson_interior(inp / inp.mean(), refy)
        r = dict(dataset=tag, tx=t, unit=meta["unit"], fold=meta["fold"], role=role, input_lib=lin, reference_lib=lref, L=L, R1=R1, R2=R2,
                 m_shallow=m, d_axis=m / L, input_total=n_in, d_input=n_in / L, reference_total=n_ref, has_reference=m > 0,
                 reference_library_positive=n_ref > 0, truth_scorable=t_ok, truth_reason=t_why, reads_r=rr, reads_reason=rwhy,
                 level_kind="natural", thin_seed=seed)
        for p, v in preds.items():
            rm, why = LB.pearson_interior(v, refy) if t_ok else (None, t_why)
            r[f"r__{p}"] = rm; r[f"why__{p}"] = why
        out.append(r)
    return out


def grid_records(tag, t, A, B, preds, meta, edges, seed):
    rs = LB.score_transcript(tag, t, A, B, preds, "grid", seed, edges, meta)
    R1, R2 = int(A.sum()), int(B.sum()); m = min(R1, R2); L = len(A)
    for r in rs:
        r.update(R1=R1, R2=R2, d_axis=m / L, has_reference=m > 0, reference_library_positive=r["reference_total"] > 0,
                 input_total=r["input_total_after"], d_input=(0.0 if r["level_kind"] == "zero" else r["depth_after"]))
    return rs


def xval(v, r):
    if v in ("v2_primary", "v2_sens_binom"): return r["d_axis"]
    if v == "v2_sens_inputdepth": return r["d_input"]
    if v == "v2_sens_gridthin": return 0.0 if r["level_kind"] == "zero" else (r["depth_after"] if r["level_kind"] == "grid" else -1.0)
    raise ValueError(v)


def fmt(x):
    if x is None: return ""
    if isinstance(x, bool): return "1" if x else "0"
    if isinstance(x, float): return repr(x)
    return str(x)


def write_records(fn, recs):
    cols = [c for c in recs[0].keys() if not c.startswith("_")]
    with gzip.open(fn, "wt", newline="") as fh:
        w = csv.writer(fh, delimiter="\t"); w.writerow(cols)
        for r in recs: w.writerow([fmt(r.get(c)) for c in cols])


_G = {}


def _job(args):
    kind, v, p, B = args; t0 = time.time()
    if kind == "dstar": out = X.dstar_with_curves(_G["recs"][(v, p)], _G["edges"], B=B, seed=0)
    else:
        ok, diff = X.check_equivalence(_G["recs"][(v, p)], _G["edges"], B=B, seed=0); out = dict(identical=ok, differing_keys=diff, B=B)
    out["_sec"] = round(time.time() - t0, 1); return kind, v, p, out


def _counts(lst, P):
    d = dict(N_eligible=len(lst), N_with_reference=sum(r["has_reference"] for r in lst),
             N_reference_library_positive=sum(r["reference_library_positive"] for r in lst),
             N_truth_scorable=sum(r["truth_scorable"] for r in lst), N_reads_scorable=sum(r["reads_r"] is not None for r in lst),
             N_scored_common=sum(r["reads_r"] is not None and all(r[f"r__{p}"] is not None for p in P) for r in lst), per_predictor={})
    for p in P:
        sc = [r for r in lst if r["reads_r"] is not None and r[f"r__{p}"] is not None]
        d["per_predictor"][p] = dict(N_scored=len(sc),
                                     median_r_model=float(np.median([r[f"r__{p}"] for r in sc])) if sc else None,
                                     median_r_reads=float(np.median([r["reads_r"] for r in sc])) if sc else None,
                                     median_paired_diff=float(np.median([r[f"r__{p}"] - r["reads_r"] for r in sc])) if sc else None)
    return d


def evaluate_dataset(ds, preds, edges, out_dir=None, variants=None, B=2000, workers=4, seed=0, dummy=True,
                     equiv=False, pred_info=None, log=print):
    """ds: Part II dataset (ribopipe.data.load_dataset(tag, "partII")); preds: {name: {tx: vector}} covering T_low.
    seed: replicate-matching thinning seed (0 = main result). Returns the evaluation dict (and writes
    eval_<tag>.json plus the per-record TSV files to out_dir if given)."""
    t0 = time.time(); tag = ds.tag; E = list(edges); nb = len(E) - 1; _G["edges"] = E
    vs = [v for v in (variants or VARIANTS) if not (v == "v2_sens_binom" and tag not in UMI_DEDUP)]
    fold_of = {t: f for f in ds.folds for t in ds.test_ids(f)}
    low = sorted(ds.strata["low"])
    P = list(preds); preds = dict(preds)
    for p in P:
        miss = [t for t in low if t not in preds[p]]
        assert not miss, f"{p}: {len(miss)} T_low transcripts without prediction"
    if dummy:
        preds[DUMMY] = dummy_noise(tag, {t: ds.seq[t] for t in ds.ids}); P = P + [DUMMY]
    log(f"[load] {tag} T_low {len(low)} predictors {P} variants {vs} {time.time() - t0:.0f}s")
    nat, binom, grid = [], [], []
    for t in low:
        r2 = np.asarray(ds.reps(t)); A, Bc = r2[0].astype(np.int64), r2[1].astype(np.int64)
        mt = dict(unit=ds.unit[t], fold=fold_of[t], stratum="low"); pv = {p: preds[p][t] for p in P}
        if "v2_primary" in vs or "v2_sens_inputdepth" in vs: nat += natural_records(tag, t, A, Bc, pv, "rep", mt, seed)
        if "v2_sens_binom" in vs: binom += natural_records(tag, t, A, Bc, pv, "binom", mt, seed)
        if "v2_sens_gridthin" in vs: grid += grid_records(tag, t, A, Bc, pv, mt, E, seed)
    REC = {v: (binom if v == "v2_sens_binom" else (grid if v == "v2_sens_gridthin" else nat)) for v in vs}
    log(f"[records] natural {len(nat)} binom {len(binom)} grid {len(grid)} {time.time() - t0:.0f}s")
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    res = dict(dataset=tag, predictors=P, T_low=len(low), edges=E, variants={})
    _G["recs"] = {}
    for v in vs:
        recs = REC[v]; K = f"_cat_{v}"
        for r in recs: r[K] = cat(xval(v, r), E) if xval(v, r) >= 0 else "below"
        rec_file = None
        if out_dir:
            rec_file = os.path.join(out_dir, f"records_{'natural' if v in ('v2_primary', 'v2_sens_inputdepth') else v}.tsv.gz")
            if not os.path.exists(rec_file) or v != "v2_sens_inputdepth": write_records(rec_file, recs)
        per_role = {}
        for role in (0, 1):
            rr = [r for r in recs if r["role"] == role]; by = {}
            for r in rr: by.setdefault(r[K], []).append(r)
            bins = [dict(bin=j, lo=E[j], hi=E[j + 1], **_counts(by.get(j, []), P)) for j in range(nb)]
            sep = {k: _counts(by.get(k, []), P) for k in ("zero", "below", "above")}
            ntx = len({r["tx"] for r in rr})
            chk = dict(n_tx=ntx, tx_set_equals_T_low=({r["tx"] for r in rr} == set(low)))
            if v != "v2_sens_gridthin":
                tot = sum(b["N_eligible"] for b in bins) + sum(sep[k]["N_eligible"] for k in sep)
                chk.update(sum_all_categories=tot, equals_T_low=(tot == len(low)), one_record_per_tx=(len(rr) == ntx))
                assert tot == len(low) and len(rr) == ntx
            assert chk["tx_set_equals_T_low"]
            per_role[str(role)] = dict(bins=bins, zero=sep["zero"], below_first_edge=sep["below"], above_last_edge=sep["above"], check=chk)
        res["variants"][v] = dict(records_file=(os.path.basename(rec_file) if rec_file else None), per_role=per_role, dstar={})
        if v in ("v2_primary", "v2_sens_binom"):
            res["variants"][v]["role_invariant_bins"] = all(r[K] == recs[i ^ 1][K] for i, r in enumerate(recs))
            assert res["variants"][v]["role_invariant_bins"]
        for p in P:
            _G["recs"][(v, p)] = [(r["unit"], r["role"], (0.0 if r[K] == "zero" else xval(v, r)), r[f"r__{p}"], r["reads_r"])
                                  for r in recs if r[K] not in ("below", "above")]
    jobs = [("dstar", v, p, B) for v in vs for p in P]
    if equiv: jobs.insert(0, ("equiv", vs[0], P[0], B))
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            results = list(pool.imap_unordered(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    for kind, v, p, out in results:
        if kind == "equiv": res["equivalence_check"] = dict(variant=v, predictor=p, **out)
        else: res["variants"][v]["dstar"][p] = out
        log(f"[{kind}] {v} {p} {out.get('status', out.get('identical'))} {out.get('D_star')} {out.get('ci95')} {out['_sec']}s")
    for v in vs:   # own N_scored must equal the reference bin_table's N_scored_role
        res["variants"][v]["dstar"] = {p: res["variants"][v]["dstar"][p] for p in P}
        for p in P:
            for j, row in enumerate(res["variants"][v]["dstar"][p]["bins"]):
                mine = [res["variants"][v]["per_role"][str(k)]["bins"][j]["per_predictor"][p]["N_scored"] for k in (0, 1)]
                assert mine == row["N_scored_role"], (v, p, j, mine, row["N_scored_role"])
    if "v2_primary" in vs: res["opcrit"] = opcrit(REC["v2_primary"], low, res["variants"]["v2_primary"]["dstar"], P)
    res.update(B=B, thin_seed=seed, pred_info=pred_info or {}, host=socket.gethostname(), python=platform.python_version(),
               numpy=np.__version__, wall_sec=round(time.time() - t0, 1), written=time.strftime("%F %T"))
    if out_dir:
        json.dump(res, open(os.path.join(out_dir, f"eval_{tag}.json"), "w"), indent=1)
    log(f"[done] {tag} {time.time() - t0:.0f}s")
    return res


def opcrit(recs, low, dst, P):
    """Fraction of the eligible population below the estimated threshold (D* and its CI ends), with three
    denominators: all T_low / with a reference (shallower replicate > 0) / scorable in the main analysis."""
    dax = {r["tx"]: r["d_axis"] for r in recs}; op = {}
    for p in P:
        sct = {r["tx"] for r in recs if r["_cat_v2_primary"] not in ("zero", "below", "above") and r["reads_r"] is not None and r[f"r__{p}"] is not None}
        d_ = dst[p]
        o = dict(N_eligible=len(low), N_with_reference=sum(dax[t] > 0 for t in low), N_scorable=len(sct), status=d_["status"])
        if d_["status"] == "OK":
            for nm, thr in (("at_D_star", d_["D_star"]), ("at_ci_lo", d_["ci95"][0]), ("at_ci_hi", d_["ci95"][1])):
                o[nm] = dict(threshold=thr, frac_eligible_below=float(np.mean([dax[t] < thr for t in low])),
                             frac_with_reference_below=float(np.mean([dax[t] < thr for t in low if dax[t] > 0])),
                             frac_scorable_below=float(np.mean([dax[t] < thr for t in sct])) if sct else None)
        op[p] = o
    return op


# ---------------------------------------------------------------- summaries (port of the aggregation step)
def g3(x): return "-" if x is None else f"{x:.3g}"


def wording(d):
    st = d["status"]
    if st == "OK":
        return (f"D* = {g3(d['D_star'])} reads/codon (95% CI [{g3(d['ci95'][0])}, {g3(d['ci95'][1])}]; "
                f"localisation success {1 - d['bootstrap_fail_fraction']:.3f})")
    if st == "NO_SIGN_CHANGE":
        v = [b for b in d["bins"] if b["valid"]]
        return f"the paired difference did not change sign within [{g3(v[0]['lo'])}, {g3(v[-1]['hi'])}] reads/codon" if v else "no valid bin"
    return f"no crossover could be located under the pre-registered rule (bootstrap failure fraction {d['bootstrap_fail_fraction']:.3f})"


def dummy_positive_bins(ev, variant="v2_primary"):
    """Valid bins in which the sequence-independent noise predictor also has a positive paired difference
    (in these bins a positive difference of a model is not evidence for the model). [(lo, hi, y)]."""
    V = ev["variants"][variant]["dstar"]
    if DUMMY not in V: return None
    return [(b["lo"], b["hi"], b["y"]) for b in V[DUMMY]["bins"] if b["valid"] and b["y"] > 0]


def last_bin_note(d):
    """Note for NO_SIGN_CHANGE: the last valid bin's bootstrap interval spans zero."""
    vb = [(j, b) for j, b in enumerate(d["bins"]) if b["valid"]]
    if d["status"] == "NO_SIGN_CHANGE" and vb:
        j, b = vb[-1]; ci = d["bin_boot"][j]["ci95"]
        if ci and ci[0] <= 0 <= ci[1]:
            return (f"difference approaches zero at the top of the analysable range (last valid bin [{g3(b['lo'])}, {g3(b['hi'])}) "
                    f"y = {b['y']:+.3f}, 95% interval [{ci[0]:+.3f}, {ci[1]:+.3f}] spans 0)")
    return None


def summarize(evals):
    """evals: {tag: evaluation dict}. Returns a JSON-able summary (main analysis)."""
    out = dict(datasets={}, dummy_positive_bins={})
    for t, e in evals.items():
        V = e["variants"].get("v2_primary")
        if V is None: continue
        out["datasets"][t] = {p: dict(status=d["status"], D_star=d["D_star"], ci95=d["ci95"],
                                      bootstrap_fail_fraction=d["bootstrap_fail_fraction"], wording=wording(d),
                                      note=last_bin_note(d), opcrit=e.get("opcrit", {}).get(p))
                              for p, d in V["dstar"].items()}
        out["dummy_positive_bins"][t] = dummy_positive_bins(e)
    return out


def summary_markdown(summ):
    L = ["# Low-coverage crossover (main analysis: x = min(R1, R2) / L)", "",
         "| Dataset | Predictor | Status | Result | Below D* (eligible / with reference / scorable) |", "|---|---|---|---|---|"]
    for t, ps in summ["datasets"].items():
        for p, d in ps.items():
            o = d["opcrit"]; fr = "-"
            if o and o["status"] == "OK":
                a = o["at_D_star"]; fr = f"{a['frac_eligible_below']:.3f} / {a['frac_with_reference_below']:.3f} / {a['frac_scorable_below']:.3f}"
            L.append(f"| {t} | {p} | {d['status']} | {d['wording']}{'; ' + d['note'] if d['note'] else ''} | {fr} |")
    L += ["", "Noise-predictor positive bins (a positive model difference in these bins is not evidence for the model):"]
    for t, bb in summ["dummy_positive_bins"].items():
        L.append(f"- {t}: " + ("; ".join(f"[{g3(lo)}, {g3(hi)}) y = {y:+.4f}" for lo, hi, y in bb) if bb else "none"))
    return "\n".join(L) + "\n"
