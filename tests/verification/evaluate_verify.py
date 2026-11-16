#!/usr/bin/env python3
"""Verification of ribopipe.evaluate against the paper (run on the paper machine; not part of the unit tests).

1. Rescores the paper's stored predictions (runs/partI) with the ported evaluation:
   setting 1 seed 123, setting 1 seeds 123-125, setting 2, RiboPipe across seeds.
2. Compares every leaf of the paper's reference JSONs field by field (numbers: |diff| <= 1e-12; strings / booleans: equal).
3. Rebuilds every cell of the published Supplementary tables of Part I (setting 1 Pearson / Spearman / recall, paired
   differences + Holm, setting 2, three-seed absolute and 24-comparison table, seed range) from the reproduced numbers
   with the generator's formatting and compares the text with the final supplement.
Writes tests/verification/evaluate_verification.json and .md.

usage: PAPER=<paper dir> SUPP=<supplement .tex> RIBOPIPE_DATA=<data_full> python tests/verification/evaluate_verify.py [scratch dir]
"""
import os, sys, json, re, time, socket, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, REPO)
from ribopipe.evaluate import partI as E

PAPER = os.environ["PAPER"]; SUPP = os.environ["SUPP"]
RUNS = f"{PAPER}/runs/partI"; PR = f"{PAPER}/report/partI_results"
SCR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_scratch"); os.makedirs(SCR, exist_ok=True)
WORKERS = int(os.environ.get("WORKERS", "8"))
REF = {"setting1_seed123": f"{PR}/partI_eval_elig.json", "setting1_3seed": f"{PR}/partI_eval_elig_3seed.json",
       "setting2_seed123": f"{PR}/T2_matched_eval_elig.json", "ribopipe_seeds": f"{PR}/ribopipe_seed_noise_1114.json"}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
J = {k: json.load(open(p)) for k, p in REF.items()}
SKIP_TOP = {"provenance", "_provenance", "generated", "seed_decision", "protocol_version", "protocol_md5", "data_tag", "manifest_md5"}


def run_or_load(name, fn):
    p = os.path.join(SCR, f"{name}.json")
    if os.path.exists(p) and os.environ.get("REUSE"): return json.load(open(p))
    t = time.time(); r = fn(); r["_wall_sec"] = round(time.time() - t, 1)
    json.dump(r, open(p, "w"), indent=1, default=str); print(f"[{name}] {r['_wall_sec']} s", flush=True)
    return json.load(open(p))   # same JSON round trip as the reference files


def ref_configs(ref):
    return sorted({c for d in ref["datasets"].values() for c in d["run_status"]})


R = {}
R["setting1_seed123"] = run_or_load("setting1_seed123", lambda: E.evaluate_setting1(RUNS, ["123"], configs=ref_configs(J["setting1_seed123"]), workers=WORKERS))
R["setting1_3seed"] = run_or_load("setting1_3seed", lambda: E.evaluate_setting1(RUNS, ["123", "124", "125"], configs=ref_configs(J["setting1_3seed"]), workers=WORKERS))
R["setting2_seed123"] = run_or_load("setting2_seed123", lambda: E.evaluate_setting2(RUNS, ["123"], workers=WORKERS))
R["ribopipe_seeds"] = run_or_load("ribopipe_seeds", lambda: E.seed_noise(RUNS, workers=WORKERS))


# ---------------------------------------------------------------- 2. field-by-field
def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, path + (str(k),))
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, path + (i,))
    else:
        yield path, x


def get(x, path):
    for k in path:
        if isinstance(x, dict):
            if k not in x: return KeyError
            x = x[k]
        else:
            if not isinstance(x, list) or k >= len(x): return KeyError
            x = x[k]
    return x


field = {}; mismatches = []; not_compared = []
for name, ref in J.items():
    rep = R[name]; n = n_ok = 0; maxdiff = 0.0
    for path, rv in leaves(ref):
        if path[0] in SKIP_TOP: continue
        if len(path) >= 3 and path[0] == "datasets" and path[2] == "run_status": continue   # file-presence bookkeeping, checked below
        got = get(rep, path); key = ".".join(map(str, path))
        # a reference cell whose method was not yet complete when the reference was written is not compared
        st4 = get(ref, path[:4] + ("status",)) if (name.startswith("setting1") and len(path) >= 4 and path[0] == "datasets"
                                                    and path[2] in ("pairs", "extra_pairs", "figB_pair")) else None
        if isinstance(st4, str) and st4.startswith(("NA (method", "PENDING")) and get(rep, path[:4] + ("status",)) != st4:
            if path[-1] == "status": not_compared.append({"file": name, "key": key, "reference": rv, "reproduced": get(rep, path)})
            continue
        n += 1
        if isinstance(rv, bool) or isinstance(rv, str) or rv is None:
            ok = (got == rv)
        elif isinstance(rv, (int, float)):
            ok = isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - rv) <= 1e-12
            if isinstance(got, (int, float)): maxdiff = max(maxdiff, abs(got - rv))
        else:
            ok = got == rv
        n_ok += ok
        if not ok: mismatches.append({"file": name, "key": key, "reference": rv, "reproduced": None if got is KeyError else got})
    # run status: every configuration COMPLETE in the reference must be COMPLETE now
    rs_bad = []
    if "datasets" in ref and name.startswith("setting1"):
        for d, Dd in ref["datasets"].items():
            for c, st in Dd["run_status"].items():
                if st == "COMPLETE" and rep["datasets"][d]["run_status"].get(c) != "COMPLETE": rs_bad.append(f"{d}|{c}")
    field[name] = {"reference": REF[name], "reference_md5": md5(REF[name]), "leaves_compared": n, "leaves_equal": n_ok,
                   "max_abs_numeric_diff": maxdiff, "complete_in_reference_but_not_now": rs_bad,
                   "status": "PASS" if n_ok == n and not rs_bad else "FAIL"}
    print(name, field[name], flush=True)

# ---------------------------------------------------------------- 3. published tables
tex = open(SUPP).read()
fx = lambda x, n=3: "0.000" if abs(x) < 0.5 * 10 ** -n else (f"{x:+.{n}f}").replace("-", "$-$").replace("+", "$+$")
f3 = lambda x: f"{x:.3f}"
ALL8 = E.DATASETS
LAB = {"DMSO_TX9P1": "TX9\\_WT", "GSE233886_WT": "GSE233886\\_WT", "GSE133393_WT": "GSE133393\\_WT", "PRJNA_iPS": "PRJNA\\_iPS",
       "Subtelny14_ribomimo": "Subtelny14", "Mohammad16": "Mohammad16", "Mohammad19-1": "Mohammad19-1", "Mohammad19-2": "Mohammad19-2"}
OURS = E.OURS


def table(label):
    i = tex.index("\\label{" + label + "}"); j = tex.index("\\bottomrule", i); body = tex[tex.index("\\midrule", i) + len("\\midrule"):j]
    rows = [r.strip() for r in body.replace("\\addlinespace[2pt]", "").replace("\\midrule", "").split("\\\\\n")]
    return [[c.strip() for c in re.split(r"&", r.rstrip("\\"))] for r in rows if r.strip()]


unbold = lambda s: re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
TBL = []
def check(tab, quantity, ref_text, rep_text):
    TBL.append({"table": tab, "quantity": quantity, "reference": ref_text, "reproduced": rep_text, "status": "PASS" if ref_text == rep_text else "FAIL"})

S1, S3, S2, SN = R["setting1_seed123"], R["setting1_3seed"], R["setting2_seed123"], R["ribopipe_seeds"]
TEXNAME = {"Codon-mean": "Paper_CodonMean", "Tri-codon": "Paper_TriCodon", "Ridge": "Paper_Ridge", "RUST": "Paper_RUST", "iXnos": "Paper_iXnos_seq_lr005_elig",
           "iXnos (MFE)": "Paper_iXnos_MFE_lr005_elig", "\\quad 500 densest genes$^{b}$": "Paper_iXnos_MFE_lr005", "RiboMIMO": "Paper_RiboMIMO",
           "RiboGL": "Paper_RiboGL_heldout", "RiboExp": "Paper_RiboExp", "BiLSTM-97$^{a}$": "Paper_BiLSTM97", "RiboPipe ($-$MFE)": "Paper_RiboPipe_noMFE_H128", "RiboPipe": OURS}
for label, metric in (("tab:s2l1pearson", "pearson_interior"), ("tab:s2l1spearman", "spearman_interior"), ("tab:s2l1recall", "recall5_interior")):
    for row in table(label):
        c = TEXNAME[row[0]]
        for d, cell in zip(ALL8, row[1:]):
            v = S1["datasets"][d]["absolute"][c][metric]
            check(label, f"{metric} {d} {c}", unbold(cell), f3(v["S"]) if v["status"] == "OK" else v["status"])
PAIRNAME = {"RiboMIMO": ("pairs", "Paper_RiboMIMO"), "iXnos (MFE)": ("pairs", "Paper_iXnos_MFE_lr005_elig"), "RiboGL": ("pairs", "Paper_RiboGL_heldout"),
            "RiboExp": ("extra_pairs", "Paper_RiboExp"), "iXnos": ("extra_pairs", "Paper_iXnos_seq_lr005_elig"), "BiLSTM-97": ("extra_pairs", "Paper_BiLSTM97"),
            "iXnos (MFE), 500 genes": ("extra_pairs", "Paper_iXnos_MFE_lr005")}
INV = {v: k for k, v in LAB.items()}; CLS = E.CLS
d = None
for row in table("tab:s2l1pairs"):
    d = INV.get(row[0], d); sec, c = PAIRNAME[row[1]]; p = S1["datasets"][d][sec][f"{OURS}|{c}"]
    holm = "--" if sec == "extra_pairs" else ("significant" if S1["holm"]["superiority"][f"{d}|{c}"] else "n.s.")
    check("tab:s2l1pairs", f"paired difference {d} RiboPipe-{c}", " & ".join(row[2:]), f"{fx(p['delta_hat'])} & [{fx(p['ci95'][0])}, {fx(p['ci95'][1])}] & {CLS[p['class']]} & {holm}")
L2N = {"RiboMIMO": "Paper_RiboMIMO_L2", "iXnos": "Paper_iXnos_nocut_lr005_elig", "RiboGL": "Paper_RiboGL_L2_heldout"}
for row in table("tab:s2l2"):
    d = INV.get(row[0], d); c = L2N[row[1]]; p = S2["datasets"][d]["ours_vs_matched"][c]; q = S2["datasets"][d]["matched_vs_native"][c]
    check("tab:s2l2", f"setting 2 {d} RiboPipe-{c}", " & ".join(row[2:]), f"{fx(p['delta_hat'])} & [{fx(p['ci95'][0])}, {fx(p['ci95'][1])}] & {fx(q['delta_hat'])}")
CORE4 = [OURS, "Paper_RiboMIMO", "Paper_iXnos_MFE_lr005_elig", "Paper_RiboGL_heldout"]
for row in table("tab:s2threeseedabs"):
    d = INV[row[0]]
    for c, cell in zip(CORE4, row[1:]):
        v = S3["datasets"][d]["absolute"][c]["pearson_interior"]
        check("tab:s2threeseedabs", f"3-seed {d} {c}", cell, f"\\makecell[l]{{{f3(v['S'])} [{f3(v['min'])}, {f3(v['max'])}]\\\\{v['N_scored']:,}/{v['N_truth_scorable']:,}}}")
TS = {"RiboMIMO": "Paper_RiboMIMO", "iXnos (MFE)": "Paper_iXnos_MFE_lr005_elig", "RiboGL": "Paper_RiboGL_heldout"}
for row in table("tab:s2threeseed"):
    d = INV.get(row[0], d); c = TS[row[1]]; p = S3["datasets"][d]["pairs"][f"{OURS}|{c}"]; p1 = S1["datasets"][d]["pairs"][f"{OURS}|{c}"]
    h = S3["holm"]["superiority"][f"{d}|{c}"]
    check("tab:s2threeseed", f"3-seed paired {d} RiboPipe-{c}", " & ".join(row[2:]),
          f"{fx(p['delta_hat'])} & [{fx(p['ci95'][0])}, {fx(p['ci95'][1])}] & {fx(p1['delta_hat'])} & [{fx(min(p['seed_deltas']))}, {fx(max(p['seed_deltas']))}] & {CLS[p['class']]} & {'significant' if h else 'n.s.'}")
row = table("tab:s2seeds")[0]
for d, cell in zip(ALL8, row[1:]):
    check("tab:s2seeds", f"seed range {d}", cell, f3(SN["datasets"][d]["range_S"]))
for fam in ("superiority", "equivalence"):
    for nm, src, ref in (("seed 123", S1, J["setting1_seed123"]), ("3 seeds", S3, J["setting1_3seed"])):
        check("holm", f"{fam} family {nm}: rejected / m", f"{sum(ref['holm'][fam].values())}/{ref['holm']['m_each']}",
              f"{sum(src['holm'][fam].values())}/{src['holm']['m_each']}")

tab_pass = sum(t["status"] == "PASS" for t in TBL)
out = {"generated": time.strftime("%F %T %Z"), "host": socket.gethostname(), "runs": RUNS, "data": os.environ.get("RIBOPIPE_DATA"),
       "supplement": SUPP, "supplement_md5": md5(SUPP), "wall_sec": {k: R[k].get("_wall_sec") for k in R},
       "field_by_field": field, "field_mismatches": mismatches, "not_compared_reference_incomplete": not_compared,
       "published_tables": {"n": len(TBL), "pass": tab_pass, "fail": len(TBL) - tab_pass, "cells": TBL},
       "overall": "PASS" if all(v["status"] == "PASS" for v in field.values()) and tab_pass == len(TBL) else "FAIL"}
json.dump(out, open(os.path.join(HERE, "evaluate_verification.json"), "w"), indent=1, default=str)

M = ["# Verification of `ribopipe.evaluate` (Part I evaluation)", "",
     f"Generated {out['generated']} on {out['host']}. Runs: `{RUNS}`. Data: `{out['data']}`. Supplement: `{SUPP}` (md5 {out['supplement_md5']}).", "",
     f"**Overall: {out['overall']}**", "", "## 1. Field-by-field comparison with the paper's evaluation JSONs", "",
     "| Quantity | Reference file | Leaves compared | Equal | Max abs numeric diff | Status |", "|---|---|---|---|---|---|"]
for k, v in field.items():
    M.append(f"| {k} | `{os.path.basename(v['reference'])}` (md5 {v['reference_md5'][:10]}) | {v['leaves_compared']} | {v['leaves_equal']} | {v['max_abs_numeric_diff']:.3g} | {v['status']} |")
M += ["", "Numbers must agree within 1e-12; strings, classes, Holm decisions and set sizes exactly. Run-status bookkeeping is not compared leaf by leaf, "
      "but every configuration that was COMPLETE in the reference is required to be COMPLETE again.", ""]
if mismatches:
    M += ["Mismatches:", ""] + [f"- {m['file']} `{m['key']}`: reference {m['reference']}, reproduced {m['reproduced']}" for m in mismatches[:50]] + [""]
if not_compared:
    M += [f"Not compared ({len(not_compared)} cells): the comparator's runs were not complete when the reference file was written "
          "(reference status 'NA (method not complete or absent)' / 'PENDING'); the reproduction scores the runs present now.", ""]
    M += [f"- {m['file']} `{m['key']}`: reference '{m['reference']}', now '{m['reproduced']}'" for m in not_compared] + [""]
M += ["## 2. Published Supplementary tables of Part I rebuilt from the reproduced numbers", "",
      f"{tab_pass}/{len(TBL)} cells identical to the text of the supplement.", "",
      "| Table | Cells | PASS | FAIL |", "|---|---|---|---|"]
for t in dict.fromkeys(x["table"] for x in TBL):
    xs = [x for x in TBL if x["table"] == t]; M.append(f"| {t} | {len(xs)} | {sum(x['status'] == 'PASS' for x in xs)} | {sum(x['status'] != 'PASS' for x in xs)} |")
fails = [x for x in TBL if x["status"] != "PASS"]
if fails:
    M += ["", "Failed cells:", ""] + [f"- {x['table']} {x['quantity']}: supplement `{x['reference']}`, reproduced `{x['reproduced']}`" for x in fails]
M += ["", "Selected cells (Table of setting-1 Pearson, RiboPipe and RiboMIMO rows; setting-2 RiboGL):", "", "| Table | Quantity | Supplement | Reproduced | Status |", "|---|---|---|---|---|"]
for x in TBL:
    if (x["table"] == "tab:s2l1pearson" and (x["quantity"].endswith(OURS) or x["quantity"].endswith("Paper_RiboMIMO"))) or (x["table"] == "tab:s2l2" and "RiboGL" in x["quantity"]) or x["table"] == "holm":
        M.append(f"| {x['table']} | {x['quantity']} | {x['reference']} | {x['reproduced']} | {x['status']} |")
open(os.path.join(HERE, "evaluate_verification.md"), "w").write("\n".join(M) + "\n")
print("OVERALL", out["overall"], "tables", tab_pass, "/", len(TBL))
sys.exit(0 if out["overall"] == "PASS" else 1)
