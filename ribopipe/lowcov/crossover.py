"""Crossover depth D* (frozen definition of the paper's Part II; protocol PARTII_CROSSOVER_v2.json).

Input: one record per (transcript, replicate assignment) = (unit, role, depth, r_model, r_reads), where
role is 0 or 1, depth is the depth on the x axis in reads/codon, and r_model / r_reads are the per-transcript
Pearson correlations of the prediction and of the sparse reads with the reference library (None = not scorable).

Rules (unchanged from the paper's reference implementation):
  * depth 0 is only counted (never placed on the log axis);
  * positive depths are binned with the frozen edges; abscissa = log10 of the geometric centre of the bin;
  * per bin and role: y_role = median(r_model) - median(r_reads) over the same scorable transcripts;
  * a bin is valid only if each role has N_scored >= 30; bin value = equal-weight mean of the two roles;
  * interpolation (linear in log10 depth) only between adjacent valid bins (an invalid bin breaks the curve);
  * y > 0 is positive, y <= 0 non-positive; exactly one positive -> non-positive crossing is required;
  * bootstrap: resample similarity units with replacement (both roles and all transcripts of a unit together),
    B = 2000, numpy default_rng(seed=0); localisation failure fraction > 0.10 -> NOT_LOCATABLE.
"""
import json
import math

import numpy as np


def bin_table(recs, edges, min_n=30):
    edges = np.asarray(edges, float); nb = len(edges) - 1
    zero = {"N_records": 0, "N_scored": 0}
    per = [[{"m": [], "r": []} for _ in range(2)] for _ in range(nb)]
    for u, role, d, rm, rr in recs:
        if d == 0:
            zero["N_records"] += 1; zero["N_scored"] += int(rm is not None and rr is not None); continue
        b = int(np.searchsorted(edges, d, side="right") - 1)
        if b < 0 or b >= nb: continue
        if rm is not None and rr is not None: per[b][role]["m"].append(rm); per[b][role]["r"].append(rr)
    rows = []
    for b in range(nb):
        n = [len(per[b][k]["m"]) for k in (0, 1)]
        valid = all(x >= min_n for x in n)
        yr = [float(np.median(per[b][k]["m"]) - np.median(per[b][k]["r"])) if n[k] > 0 else None for k in (0, 1)]
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "x_log10": 0.5 * (math.log10(edges[b]) + math.log10(edges[b + 1])),
                     "N_scored_role": n, "valid": valid, "y_role": yr, "y": (0.5 * (yr[0] + yr[1]) if valid else None)})
    return zero, rows


def locate(rows):
    """Return (status, D*); status is OK, NO_SIGN_CHANGE (never changes sign over the valid bins) or
    NOT_LOCATABLE (several crossings, or a crossing only across an invalid bin)."""
    cands = []; signs = []
    for a, b in zip(rows[:-1], rows[1:]):
        if a["valid"]: signs.append(a["y"] > 0)
        if a["valid"] and b["valid"] and a["y"] > 0 and b["y"] <= 0:
            t = (0 - a["y"]) / (b["y"] - a["y"]); cands.append(10 ** (a["x_log10"] + t * (b["x_log10"] - a["x_log10"])))
    if rows and rows[-1]["valid"]: signs.append(rows[-1]["y"] > 0)
    if len(cands) == 1: return "OK", cands[0]
    if not cands and (all(signs) or not any(signs)): return "NO_SIGN_CHANGE", None
    return "NOT_LOCATABLE", None


def _wording(st, d, rows):
    obs = [r for r in rows if r["valid"]]
    return {"OK": f"D* = {d:.3g} reads/codon" if d else "",
            "NO_SIGN_CHANGE": (f"the paired difference did not change sign within [{obs[0]['lo']:.3g}, {obs[-1]['hi']:.3g}] reads/codon"
                               if obs else "no valid bin"),
            "NOT_LOCATABLE": "no crossover could be located under the pre-registered rule"}[st]


def dstar(recs, edges, B=2000, seed=0, min_n=30, fail_frac=0.10):
    """Reference implementation (point estimate + similarity-unit bootstrap). Kept for the equivalence check."""
    zero, rows = bin_table(recs, edges, min_n); st, d = locate(rows)
    units = {}
    for r in recs: units.setdefault(r[0], []).append(r)
    keys = sorted(units); rng = np.random.default_rng(seed); boots = []; fails = 0
    for _ in range(B):
        draw = rng.integers(0, len(keys), len(keys)); rr = [x for i in draw for x in units[keys[i]]]
        s2, d2 = locate(bin_table(rr, edges, min_n)[1])
        if s2 == "OK": boots.append(d2)
        else: fails += 1
    ff = fails / B
    if st == "OK" and ff > fail_frac: st = "NOT_LOCATABLE"
    out = {"status": st, "D_star": d if st == "OK" else None, "bootstrap_fail_fraction": ff,
           "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])] if (st == "OK" and boots) else None,
           "zero_depth": zero, "bins": rows}
    out["wording"] = _wording(st, d, rows)
    return out


def dstar_with_curves(recs, edges, B=2000, seed=0, min_n=30, fail_frac=0.10):
    """Same point estimate and the same resampling sequence as dstar() (same rng, same order of unit keys, same
    locate); additionally records the bootstrap distribution of every bin value (descriptive 95% intervals).
    For speed only records scorable on both sides with positive depth enter bin_table inside the bootstrap
    (the other records do not change the rows); the unit keys are still built from all records, so the
    resampling sequence is identical to dstar(). check_equivalence() verifies this."""
    zero, rows = bin_table(recs, edges, min_n); st, d = locate(rows)
    units = {}
    for r in recs: units.setdefault(r[0], []).append(r)
    keys = sorted(units)
    sc = {u: [x for x in units[u] if x[2] != 0 and x[3] is not None and x[4] is not None] for u in keys}
    rng = np.random.default_rng(seed); boots = []; fails = 0; nb = len(edges) - 1
    ybs = [[] for _ in range(nb)]
    for _ in range(B):
        draw = rng.integers(0, len(keys), len(keys)); rr = [x for i in draw for x in sc[keys[i]]]
        rows2 = bin_table(rr, edges, min_n)[1]; s2, d2 = locate(rows2)
        for b_, row in enumerate(rows2):
            if row["valid"]: ybs[b_].append(row["y"])
        if s2 == "OK": boots.append(d2)
        else: fails += 1
    ff = fails / B
    if st == "OK" and ff > fail_frac: st = "NOT_LOCATABLE"
    out = {"status": st, "D_star": d if st == "OK" else None, "bootstrap_fail_fraction": ff,
           "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])] if (st == "OK" and boots) else None,
           "zero_depth": zero, "bins": rows}
    out["wording"] = _wording(st, d, rows)
    out["bin_boot"] = [{"n_valid_draws": len(v), "ci95": ([float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None)}
                       for v in ybs]
    out["n_boot_ok"] = len(boots); out["B"] = B
    return out


def check_equivalence(recs, edges, B=200, seed=0):
    """dstar_with_curves() vs the reference dstar(): status / D* / ci95 / failure fraction / zero / rows / wording
    must be identical. Returns (identical, differing_keys)."""
    a = dstar(recs, edges, B=B, seed=seed); b = dstar_with_curves(recs, edges, B=B, seed=seed)
    keys = ("status", "D_star", "ci95", "bootstrap_fail_fraction", "zero_depth", "bins", "wording")
    diff = [k for k in keys if json.dumps(a[k], sort_keys=True) != json.dumps(b[k], sort_keys=True)]
    return (not diff), diff
