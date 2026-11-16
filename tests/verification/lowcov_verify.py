#!/usr/bin/env python3
"""Compare the outputs of ribopipe.lowcov with the paper's stored Part II results, field by field.

Run by the authors after the full reproduction (needs the paper's analysis directory):
  python tests/verification/lowcov_verify.py --paper <Paper_2026-11-11> --app <Application_2026-11-13> \
         --ours <dir with crossover/, posterior/, ranking/> --out tests/verification/lowcov_verification
Every compared quantity is listed with its reference value, reproduced value and PASS/FAIL. Floating-point values
are compared exactly (the JSON files store full repr precision); any difference is a FAIL and the maximum
absolute difference is reported.
"""
import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from ribopipe.lowcov import posterior as PO, score as SC  # noqa: E402

DS = ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]
DISPLAY = {"DMSO_TX9P1": "TX9_WT"}
RP = "Paper_RiboPipe_MFE_H128"
ROWS = []


def _norm(x):
    """JSON round trip (tuples -> lists, numpy floats -> float)."""
    return json.loads(json.dumps(x, default=float))


def deep_diff(a, b, path="", out=None, maxabs=None):
    """List of (path, ref, ours) for every differing leaf; maxabs[0] = max |difference| over numeric leaves."""
    out = [] if out is None else out; maxabs = [0.0] if maxabs is None else maxabs
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b: out.append((f"{path}/{k}", a.get(k, "<missing>"), b.get(k, "<missing>"))); continue
            deep_diff(a[k], b[k], f"{path}/{k}", out, maxabs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out.append((path + "/len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)): deep_diff(x, y, f"{path}[{i}]", out, maxabs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if a != b and not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            out.append((path, a, b)); maxabs[0] = max(maxabs[0], abs(a - b))
    elif a != b:
        out.append((path, a, b))
    return out, maxabs[0]


def add(group, quantity, ref, ours, note=None):
    ref, ours = _norm(ref), _norm(ours)
    diff, mx = deep_diff(ref, ours)
    row = dict(group=group, quantity=quantity, status="PASS" if not diff else "FAIL")
    if isinstance(ref, (dict, list)) and len(json.dumps(ref)) > 400:
        row.update(reference=f"<structure, {len(json.dumps(ref))} chars>", reproduced=f"<structure, {len(json.dumps(ours))} chars>")
    else:
        row.update(reference=ref, reproduced=ours)
    if diff:
        row.update(n_differing_fields=len(diff), max_abs_diff=mx, first_differences=[list(d) for d in diff[:5]])
    if note: row["note"] = note
    ROWS.append(row); return not diff


# ---------------------------------------------------------------- crossover
def verify_crossover(paper, ours):
    ref = json.load(open(os.path.join(paper, "report/partII_results/rerun76/partII_eval.json")))
    ev = {}
    for t in DS:
        fn = os.path.join(ours, "crossover", t, f"eval_{t}.json")
        if not os.path.exists(fn): add("crossover", f"{t}: evaluation file", "present", "missing"); continue
        e = json.load(open(fn)); ev[t] = e; R = ref["datasets"][t]; nm = DISPLAY.get(t, t)
        add("crossover", f"{nm}: predictors", R["predictors"], e["predictors"])
        add("crossover", f"{nm}: T_low", R["T_low"], e["T_low"])
        add("crossover", f"{nm}: edges", R["edges"], e["edges"])
        # headline: RiboPipe, main analysis
        d, m = R["variants"]["v2_primary"]["dstar"][RP], e["variants"]["v2_primary"]["dstar"][RP]
        add("headline", f"{nm}: RiboPipe D* (reads/codon)", d["D_star"], m["D_star"])
        add("headline", f"{nm}: RiboPipe D* status", d["status"], m["status"])
        add("headline", f"{nm}: RiboPipe D* 95% bootstrap CI", d["ci95"], m["ci95"])
        add("headline", f"{nm}: RiboPipe bootstrap failure fraction", d["bootstrap_fail_fraction"], m["bootstrap_fail_fraction"])
        for b0, b1 in zip(d["bins"], m["bins"]):
            if b0["valid"] or b1["valid"]:
                add("headline_bins", f"{nm}: RiboPipe y, bin [{b0['lo']:.4g}, {b0['hi']:.4g})", b0["y"], b1["y"])
        o0, o1 = R["opcrit"][RP], e["opcrit"][RP]
        add("headline", f"{nm}: opcrit status", o0["status"], o1["status"])
        for k in [x for x in ("at_D_star", "at_ci_lo", "at_ci_hi") if x in o0 or x in o1]:
            for f in ("frac_eligible_below", "frac_with_reference_below", "frac_scorable_below"):
                add("headline", f"{nm}: {f} {k}", o0[k][f], o1[k][f])
        for k in ("N_eligible", "N_with_reference", "N_scorable"):
            add("headline", f"{nm}: opcrit {k}", o0[k], o1[k])
        add("headline", f"{nm}: noise-predictor positive bins (lo, hi, y)", ref["dummy_positive_bins"][t], SC.dummy_positive_bins(e))
        # everything else, per (variant, predictor): the whole dstar object and the per-role tables
        for v in R["variants"]:
            if v not in e["variants"]: add("crossover_full", f"{nm} {v}", "present", "missing"); continue
            for p in R["variants"][v]["dstar"]:
                add("crossover_full", f"{nm} {v} {p}: dstar object (status, D*, CI, failure fraction, zero depth, all bins, bin bootstrap, wording)",
                    {k: x for k, x in R["variants"][v]["dstar"][p].items() if k != "_sec"},
                    {k: x for k, x in e["variants"][v]["dstar"][p].items() if k != "_sec"})
            add("crossover_full", f"{nm} {v}: per-role counts and medians (all bins, all predictors)",
                R["variants"][v]["per_role"], e["variants"][v]["per_role"])
        add("crossover_full", f"{nm}: opcrit (all predictors)", R["opcrit"], e["opcrit"])
        eq = e.get("equivalence_check")
        if eq: add("self_check", f"{nm}: dstar_with_curves vs reference dstar() ({eq['variant']}, B={eq['B']})", True, eq["identical"])
    return ev


# ---------------------------------------------------------------- posterior
C1, C2, C3 = PO.C1, PO.C2, PO.C3


def verify_posterior(paper, app, ours, ev):
    ref = json.load(open(os.path.join(paper, "report/partII_results/posterior_partII/posterior_partII_results.json")))
    sc0, scd = {}, {}
    for t in DS:
        nm = DISPLAY.get(t, t)
        kj = os.path.join(ours, "posterior", f"kappa_{t}.json")
        a1 = json.load(open(os.path.join(app, "results", f"a1_{t}.json")))
        if os.path.exists(kj):
            k = json.load(open(kj))["chosen"]
            add("posterior_kappa", f"{nm}: chosen prior strength per (role, fold) [value, grid index, mean ll, n records]",
                {r: [v["prp"]["value"], v["prp"]["grid_index"], v["prp"]["mean_ll"], v["n_records_fit"]] for r, v in a1["chosen"].items()},
                {r: [v["value"], v["grid_index"], v["mean_ll"], v["n_records_fit"]] for r, v in k.items()})
        fn = os.path.join(ours, "posterior", f"posterior_{t}.json")
        if not os.path.exists(fn): add("posterior", f"{nm}: draw 0", "present", "missing"); continue
        sc0[t] = json.load(open(fn))
        add("posterior", f"{nm}: kappa used", ref["datasets"][t]["kappa"], sc0[t]["kappa"])
        for d in range(1, 10):
            f2 = os.path.join(ours, "posterior", f"posterior_{t}_draw{d}.json")
            if os.path.exists(f2): scd[(d, t)] = json.load(open(f2))
        # gate i-b of the paper: prediction - reads on the main set equals the crossover main analysis
        if t in ev:
            keys = ("status", "D_star", "ci95", "bootstrap_fail_fraction", "zero_depth", "bins", "bin_boot", "n_boot_ok", "B", "wording")
            add("self_check", f"{nm}: posterior run, RiboPipe - reads (main set) == crossover main analysis",
                {k: ev[t]["variants"]["v2_primary"]["dstar"][RP][k] for k in keys}, {k: sc0[t]["comparisons"]["main_RiboPipe-reads"][k] for k in keys})
        ic = sc0[t].get("identity_check"); g2 = ref["gate_ii_detail"][t]
        if ic:
            add("posterior", f"{nm}: identity check max |r(kappa=0) - r(reads)|", g2["max_abs_k0_minus_reads"], ic["max_abs_k0_minus_reads"])
            add("posterior", f"{nm}: identity check max |r(kappa=1e12) - r(p_seq)|", g2["max_abs_kinf_minus_pseq"], ic["max_abs_kinf_minus_pseq"])
    if len(sc0) < len(DS): return
    agg = PO.aggregate(sc0)
    for t in DS:
        nm = DISPLAY.get(t, t); R = ref["datasets"][t]; M = agg["datasets"][t]
        for r0, r1 in zip(R["bins"], M["bins"]):
            for c in (C1, C2, C3):
                add("posterior_bins", f"{nm}: {c} bin [{r0['lo']:.4g}, {r0['hi']:.4g}) y and 95% CI",
                    [r0[c]["y"], r0[c]["ci95"]], [r1[c]["y"], r1[c]["ci95"]])
        add("posterior_full", f"{nm}: all valid-bin rows (y, y_role, CI, valid draws, N_scored)", R["bins"], M["bins"])
        add("posterior_full", f"{nm}: summary", R["summary"], {k: v for k, v in M["summary"].items() if not k.startswith("n_ci_above") and not k.startswith("n_point_above")})
        add("posterior_full", f"{nm}: crossing status of the three comparisons", R["dstar"], M["dstar"])
        add("posterior_full", f"{nm}: zero-depth counts", R["zero_depth"], M["zero_depth"])
    add("posterior", "decision rule (draw 0)", ref["reading_draw0"], agg["reading"])
    # the interval statements (reference counts computed from the reference per-bin intervals)
    cnt = lambda bins, c, side: sum((b[c]["ci95"][0] > 0) if side == "above" else (b[c]["ci95"][1] < 0) for b in bins)
    for c, lab in ((C1, "posterior - reads"), (C2, "posterior - RiboPipe")):
        for side in ("above", "below"):
            add("headline", f"pooled bins with 95% interval {side} zero, {lab}",
                [sum(cnt(ref["datasets"][t]["bins"], c, side) for t in DS), sum(len(ref["datasets"][t]["bins"]) for t in DS)],
                [sum(cnt(agg["datasets"][t]["bins"], c, side) for t in DS), sum(len(agg["datasets"][t]["bins"]) for t in DS)])
        add("headline", f"pooled bins with point estimate > 0, {lab}",
            sum(sum(b[c]["y"] > 0 for b in ref["datasets"][t]["bins"]) for t in DS),
            sum(sum(b[c]["y"] > 0 for b in agg["datasets"][t]["bins"]) for t in DS))
    add("headline", "pooled bins with both differences > 0 (draw 0)", [ref["reading_draw0"]["pooled_n_gt_both"], ref["reading_draw0"]["pooled_n_valid"]],
        [agg["reading"]["pooled_n_gt_both"], agg["reading"]["pooled_n_valid"]])
    # draws 1-9
    for d in range(1, 10):
        if not all((d, t) in scd for t in DS): add("posterior_draws", f"draw {d}", "present", "missing"); continue
        pd = {t: PO.summarize_bins(PO.bin_rows(scd[(d, t)])) for t in DS}
        Rd = ref["draws_1_9"][str(d)]
        keys = ("n_valid_bins", "n_post_ge_both", "n_post_gt_both", "n_both_ge_margin", f"min|{C1}", f"min|{C2}", "n_sig_neg_either")
        add("posterior_draws", f"draw {d}: per-dataset summary", Rd["per_dataset"], {t: {k: pd[t][k] for k in keys} for t in DS})
        add("posterior_draws", f"draw {d}: decision rule", Rd["reading"], PO.reading(pd))


# ---------------------------------------------------------------- controlled-thinning ranking
def verify_ranking(app, ours):
    for t in DS:
        nm = DISPLAY.get(t, t); fn = os.path.join(ours, "ranking", f"a1_{t}.json")
        if not os.path.exists(fn): add("ranking", f"{nm}: output", "present", "missing"); continue
        R = json.load(open(os.path.join(app, "results", f"a1_{t}.json"))); M = json.load(open(fn))
        for k in ("n_tx_universe", "estimators", "metrics", "depths", "seeds", "grid", "chosen", "crossover_prec"):
            add("ranking", f"{nm}: {k}", R[k], M[k])
        for c in ("main_Bge1|role0|d0.1", "main_Bge1|role1|d0.1"):
            for e in ("Post_RiboPipe", "RiboPipe", "Raw"):
                add("ranking", f"{nm}: top-5% precision {e} ({c})", R["cells"][c]["est"][e]["prec"], M["cells"][c]["est"][e]["prec"])
        add("ranking_full", f"{nm}: all cells (8 estimators x 6 metrics, 10 comparisons, 2 strata x 2 roles x 6 depths)", R["cells"], M["cells"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True); ap.add_argument("--app", required=True); ap.add_argument("--ours", required=True)
    ap.add_argument("--out", required=True, help="output prefix (.json and .md)"); ap.add_argument("--runtime", default=None)
    a = ap.parse_args()
    ev = verify_crossover(a.paper, a.ours)
    verify_posterior(a.paper, a.app, a.ours, ev)
    verify_ranking(a.app, a.ours)
    groups = {}
    for r in ROWS: g = groups.setdefault(r["group"], [0, 0]); g[0 if r["status"] == "PASS" else 1] += 1
    res = dict(written=time.strftime("%F %T %Z"), references=dict(
                   crossover="report/partII_results/rerun76/partII_eval.json",
                   posterior="report/partII_results/posterior_partII/posterior_partII_results.json",
                   ranking_and_kappa="Application_2026-11-13/results/a1_<dataset>.json"),
               n_pass=sum(r["status"] == "PASS" for r in ROWS), n_fail=sum(r["status"] == "FAIL" for r in ROWS),
               per_group={g: dict(PASS=v[0], FAIL=v[1]) for g, v in groups.items()}, runtime=a.runtime, comparisons=ROWS)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    fmt = lambda x: x if isinstance(x, str) else json.dumps(x)
    L = ["# Low-coverage (Part II) port: verification against the paper's stored results", "",
         f"Written {res['written']}. {res['n_pass']} PASS, {res['n_fail']} FAIL. Floating-point values compared exactly "
         "(full stored precision). Reproduction from the paper's stored predictions (no retraining) on machine 76." +
         (f" Runtime: {a.runtime}." if a.runtime else ""), "",
         "| Group | PASS | FAIL |", "|---|---|---|"] + [f"| {g} | {v[0]} | {v[1]} |" for g, v in groups.items()]
    L += ["", "## Headline quantities", "", "| Quantity | Reference | Reproduced | Status |", "|---|---|---|---|"]
    for r in ROWS:
        if r["group"] == "headline": L.append(f"| {r['quantity']} | {fmt(r['reference'])} | {fmt(r['reproduced'])} | {r['status']} |")
    fails = [r for r in ROWS if r["status"] == "FAIL"]
    L += ["", "## Failures", ""] + ([f"- [{r['group']}] {r['quantity']}: {r.get('n_differing_fields')} fields, max |diff| {r.get('max_abs_diff')}; "
                                    f"first {r.get('first_differences')}" for r in fails] or ["none"])
    L += ["", "## Scope and notes", "",
          "- Crossover: `ribopipe crossover` on the four human datasets with the paper's stored Part II predictions "
          "(runs/partII/preds/<config>/<dataset>, md5 identical to those recorded in the reference), all five model predictors plus the "
          "noise predictor, all four variants (main analysis, input-depth axis, per-bin grid thinning, binomial split), B = 2000, matching seed 0. "
          "Reference: report/partII_results/rerun76/partII_eval.json.",
          "- Posterior: `ribopipe posterior-kappa` (prior strength chosen on Part I transcripts of the other folds from the stored Part I "
          "out-of-fold predictions) and `ribopipe posterior` for matching draws 0-9 (B = 2000 each, as in the paper's final run). "
          "Reference: report/partII_results/posterior_partII/posterior_partII_results.json; kappa reference: Application_2026-11-13/results/a1_<dataset>.json.",
          "- Ranking: `ribopipe peak-ranking` with the stored Part I out-of-fold predictions of RiboPipe, CodonMean and Ridge. "
          "Reference: Application_2026-11-13/results/a1_<dataset>.json (all cells, chosen prior strengths, descriptive crossover).",
          "- Not reproduced here: the crossover matching-seed sensitivity (seeds 1-9; supported via `ribopipe crossover --seed k`), "
          "the descriptive p_seq-versus-raw-prediction table of the posterior analysis, and Part II inference itself "
          "(`ribopipe lowcov-predict` was only smoke-tested: 40 transcripts per fold of PRJNA_iPS, max |difference| to the stored "
          "predictions 3.8e-6, 140/200 bitwise equal; the remaining differences are float32 rounding from a different batch composition)."]
    L += ["", f"All {len(ROWS)} comparisons (including per-bin values, full dstar objects, per-role tables, posterior draws 1-9 and the "
          "ranking cells) are listed in lowcov_verification.json."]
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print(f"PASS {res['n_pass']} FAIL {res['n_fail']}", json.dumps(res["per_group"]))
    return 0 if res["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
