"""Part I evaluation drivers (ports of the paper's evaluation scripts; scoring rules in ``scorer.py``).

  evaluate_setting1   evaluation setting 1 (each method with its published training target): absolute scores of every
                      method (Pearson, Spearman, top-5% peak recall on interior codons; Pearson on full_deployed), the
                      paired differences RiboPipe minus each core comparator (B = 2000), the pair-level interior vs
                      full-length comparison, descriptive paired differences against the other methods and the two
                      24-comparison Holm families (superiority beyond +0.01; TOST equivalence within +/-0.01).
                      seeds = ["123"]               -> paper file partI_eval_elig.json        (run_partI_eval_elig.py)
                      seeds = ["123", "124", "125"] -> paper file partI_eval_elig_3seed.json  (run_partI_eval_elig_3seed.py)
  evaluate_setting2   evaluation setting 2 (comparators retrained on the scored target y): RiboPipe minus each retrained
                      comparator, and each retrained comparator minus its setting-1 version (descriptive, outside Holm)
                      -> paper file T2_matched_eval_elig.json (eval_T2_matched_elig.py)
  seed_noise          RiboPipe five-fold score per seed and seeds 124/125 minus 123 (descriptive)
                      -> paper file ribopipe_seed_noise_1114.json (eval_ribopipe_seeds_1114.py)

Datasets are evaluated in parallel (one process per dataset); every number is identical to a serial run because each
bootstrap uses its own numpy default_rng(0).
"""
import os, sys, json, time, socket, platform
import numpy as np
from . import scorer as SP

HUMAN = ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]
NONHUMAN = ["Subtelny14_ribomimo", "Mohammad16", "Mohammad19-1", "Mohammad19-2"]
DATASETS = HUMAN + NONHUMAN

# Configuration IDs of the paper (paper names in comments).
OURS = "Paper_RiboPipe_MFE_H128"                                   # RiboPipe
CORE = ["Paper_RiboMIMO",                                          # RiboMIMO
        "Paper_iXnos_MFE_lr005_elig",                              # iXnos (MFE)
        "Paper_RiboGL_heldout"]                                    # RiboGL
EXTRA = ["Paper_iXnos_MFE_lr005",                                  # iXnos (MFE), 500 densest genes
         "Paper_iXnos_seq_lr005", "Paper_iXnos_nocut_lr005",
         "Paper_RiboExp",                                          # RiboExp
         "Paper_BiLSTM97",                                         # BiLSTM-97
         "Paper_iXnos_seq_lr005_elig",                             # iXnos
         "Paper_iXnos_seq_elig", "Paper_iXnos_MFE_elig", "Paper_RiboMIMO_adapt_MFE",
         "Paper_RiboGL_L2_heldout", "Paper_iXnos_nocut_lr005_elig"]
SETTING2 = {"Paper_RiboMIMO_L2": "Paper_RiboMIMO",                 # retrained -> setting-1 version
            "Paper_iXnos_nocut_lr005_elig": "Paper_iXnos_seq_lr005_elig",
            "Paper_RiboGL_L2_heldout": "Paper_RiboGL_heldout"}
TABLE_METHODS = [("Paper_CodonMean", "Codon-mean"), ("Paper_TriCodon", "Tri-codon"), ("Paper_Ridge", "Ridge"),
                 ("Paper_RUST", "RUST"), ("Paper_iXnos_seq_lr005_elig", "iXnos"), ("Paper_iXnos_MFE_lr005_elig", "iXnos (MFE)"),
                 ("Paper_iXnos_MFE_lr005", "iXnos (MFE), 500 densest genes"), ("Paper_RiboMIMO", "RiboMIMO"),
                 ("Paper_RiboGL_heldout", "RiboGL"), ("Paper_RiboExp", "RiboExp"), ("Paper_BiLSTM97", "BiLSTM-97"),
                 ("Paper_RiboPipe_noMFE_H128", "RiboPipe (-MFE)"), (OURS, "RiboPipe")]
PAPER_SEEDS = ["123", "124", "125"]
ABS_METRICS = (("pearson", "interior"), ("spearman", "interior"), ("recall5", "interior"), ("pearson", "full_deployed"))


def _stamp(extra=None):
    d = {"host": socket.gethostname(), "time": time.strftime("%F %T %Z"), "python": platform.python_version(),
         "numpy": np.__version__, "module": "ribopipe.evaluate", "protocol_file": os.path.basename(SP.PROTOCOL_FILE),
         "protocol_md5": SP.PROTOCOL_MD5, "argv": sys.argv}
    d.update(extra or {}); return d


def list_configs(runs):
    return sorted(d for d in os.listdir(runs) if os.path.isdir(os.path.join(runs, d)) and not d.startswith("_")) if os.path.isdir(runs) else []


def seeds_of(runs, cfg, d, seeds):
    """Deterministic methods (only a 'seeddet' directory) use seed 'det'; every other method the requested seeds."""
    p = os.path.join(runs, cfg, d)
    ss = sorted(x[4:] for x in os.listdir(p) if x.startswith("seed")) if os.path.isdir(p) else []
    return ["det"] if ss == ["det"] else list(seeds)


def complete(runs, cfg, d, seeds, folds=range(5)):
    ss = seeds_of(runs, cfg, d, seeds)
    return all(os.path.exists(os.path.join(runs, cfg, d, f"seed{s}", f"fold{f}", "run.json")) for s in ss for f in folds), ss


def load_scoring_dataset(tag, data=None):
    from .. import data as D
    return SP.Dataset.from_release(D.load_dataset(tag, data=data, load_mfe=False))


def _pmap(fn, items, workers):
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(workers, len(items))) as pool:
        return pool.map(fn, items)


def _data_available(tag, data):
    from .. import data as D
    return os.path.isdir(os.path.join(D.data_dir(data), tag))


# ---------------------------------------------------------------- evaluation setting 1
def _setting1_one(args):
    d, runs_root, seeds, configs, ours, core, extra, B, data = args
    ds = load_scoring_dataset(d, data)
    runs, status = {}, {}
    for c in configs:
        if not os.path.isdir(os.path.join(runs_root, c, d)): continue
        ok, ss = complete(runs_root, c, d, seeds)
        status[c] = "COMPLETE" if ok else "INCOMPLETE"
        if ok: runs[c] = SP.load_method_runs(runs_root, c, d, ss, range(5))
    R = {"run_status": status, "absolute": {}, "pairs": {}, "figB_pair": {}}
    for metric, dom in ABS_METRICS:
        r = SP.score_dataset(ds, runs, dom, pairs=(), B=0, metric=metric)
        for m, v in r["absolute"].items(): R["absolute"].setdefault(m, {})[f"{metric}_{dom}"] = v
    sup, eq = {}, {}
    for c in core:
        key = f"{ours}|{c}"
        if ours in runs and c in runs:
            pr = SP.score_dataset(ds, {ours: runs[ours], c: runs[c]}, "interior", pairs=[(ours, c)], B=B)["pairs"][key]
            R["pairs"][key] = pr; sup[c] = pr.get("p_superiority"); eq[c] = pr.get("p_equivalence_tost")
            R["figB_pair"][key] = SP.score_pair_interior_vs_full(ds, runs[ours], runs[c], B=B)
        else:
            R["pairs"][key] = {"status": "PENDING (method not complete)"}; sup[c] = None; eq[c] = None
    R["extra_pairs"] = {}
    for c in extra:
        key = f"{ours}|{c}"
        if ours in runs and c in runs:
            R["extra_pairs"][key] = SP.score_dataset(ds, {ours: runs[ours], c: runs[c]}, "interior", pairs=[(ours, c)], B=B)["pairs"][key]
        else:
            R["extra_pairs"][key] = {"status": "NA (method not complete or absent)"}
    print(f"[setting1] {d} complete: {sorted(runs)}", flush=True)
    return d, R, sup, eq


def evaluate_setting1(runs_root, seeds=("123",), datasets=DATASETS, configs=None, ours=OURS, core=CORE, extra=EXTRA,
                      B=SP.B_DEFAULT, data=None, workers=8):
    """Evaluation setting 1 with the frozen-protocol aggregation over ``seeds``. Cells whose method is missing or whose
    data are unavailable enter both Holm families with p = 1 (family size stays 3 x number of datasets)."""
    seeds = [str(s) for s in seeds]; configs = list(configs) if configs else list_configs(runs_root)
    todo = [d for d in datasets if _data_available(d, data)]
    out = _pmap(_setting1_one, [(d, runs_root, seeds, configs, ours, list(core), list(extra), B, data) for d in todo], workers)
    res = {"provenance": _stamp({"runs": os.path.abspath(runs_root)}), "seeds": seeds, "B": B, "ours": ours, "core": list(core),
           "extra": list(extra), "datasets": {}, "datasets_unavailable": [d for d in datasets if d not in todo]}
    sup_p, eq_p = {}, {}
    got = {d: (R, s, e) for d, R, s, e in out}
    for d in datasets:
        if d in got:
            R, s, e = got[d]; res["datasets"][d] = R
            for c in core: sup_p[(d, c)] = s[c]; eq_p[(d, c)] = e[c]
        else:
            for c in core: sup_p[(d, c)] = None; eq_p[(d, c)] = None
    hs = SP.holm({f"{d}|{c}": p for (d, c), p in sup_p.items()}); he = SP.holm({f"{d}|{c}": p for (d, c), p in eq_p.items()})
    res["holm"] = {"superiority": hs, "equivalence": he, "m_each": len(sup_p), "note": "pending/failed cells enter with p = 1"}
    return res


# ---------------------------------------------------------------- evaluation setting 2
def _setting2_one(args):
    d, runs_root, seeds, ours, pairs, B, data = args
    ds = load_scoring_dataset(d, data)
    cfgs = [ours] + list(pairs) + list(pairs.values())
    runs = {c: SP.load_method_runs(runs_root, c, d, seeds_of(runs_root, c, d, seeds), range(5)) for c in cfgs}
    R = {"ours_vs_matched": {}, "matched_vs_native": {}}
    for c, nat in pairs.items():
        R["ours_vs_matched"][c] = SP.score_dataset(ds, {ours: runs[ours], c: runs[c]}, "interior", pairs=[(ours, c)], B=B)["pairs"][f"{ours}|{c}"]
        R["matched_vs_native"][c] = SP.score_dataset(ds, {c: runs[c], nat: runs[nat]}, "interior", pairs=[(c, nat)], B=B)["pairs"][f"{c}|{nat}"]
    print(f"[setting2] {d} " + str({c: round(R['ours_vs_matched'][c].get('delta_hat', float('nan')), 4) for c in pairs}), flush=True)
    return d, R


def evaluate_setting2(runs_root, seeds=("123",), datasets=DATASETS, ours=OURS, pairs=None, B=SP.B_DEFAULT, data=None, workers=8):
    pairs = dict(pairs or SETTING2); seeds = [str(s) for s in seeds]
    todo = [d for d in datasets if _data_available(d, data)]
    out = _pmap(_setting2_one, [(d, runs_root, seeds, ours, pairs, B, data) for d in todo], workers)
    res = {"provenance": _stamp({"runs": os.path.abspath(runs_root)}), "B": B, "seeds": seeds, "domain": "interior", "metric": "pearson",
           "note": "descriptive; not part of the pre-registered Holm families", "pairs": pairs, "datasets": {},
           "datasets_unavailable": [d for d in datasets if d not in todo]}
    got = dict(out)
    for d in datasets:
        if d in got: res["datasets"][d] = got[d]
    return res


# ---------------------------------------------------------------- RiboPipe across seeds
def _seed_one(args):
    d, runs_root, config, seeds, B, data = args
    ds = load_scoring_dataset(d, data); C = config; base = seeds[0]
    runs = {s: SP.load_method_runs(runs_root, C, d, [s], range(5)) for s in seeds}
    R = {"absolute": {}, f"paired_vs_{base}": {}}
    for s in runs:
        a = SP.score_dataset(ds, {f"{C}@{s}": runs[s]}, "interior", pairs=(), B=0)["absolute"][f"{C}@{s}"]
        R["absolute"][s] = {k: a[k] for k in ("status", "S", "N_scored") if k in a}
    for s in seeds[1:]:
        pr = SP.score_dataset(ds, {f"{C}@{s}": runs[s], f"{C}@{base}": runs[base]}, "interior",
                              pairs=[(f"{C}@{s}", f"{C}@{base}")], B=B)["pairs"][f"{C}@{s}|{C}@{base}"]
        R[f"paired_vs_{base}"][s] = {k: pr.get(k) for k in ("status", "delta_hat", "ci95", "N_paired")}
    S = [R["absolute"][s]["S"] for s in runs if R["absolute"][s]["status"] == "OK"]
    R["range_S"] = max(S) - min(S) if len(S) == len(seeds) else None
    return d, R


def seed_noise(runs_root, config=OURS, seeds=PAPER_SEEDS, datasets=DATASETS, B=200, data=None, workers=8):
    """Five-fold score of one configuration per seed and the paired difference of each later seed minus the first
    (descriptive; B = 200 as in the paper, only to satisfy the scoring interface)."""
    seeds = [str(s) for s in seeds]
    todo = [d for d in datasets if _data_available(d, data)]
    out = dict(_pmap(_seed_one, [(d, runs_root, config, seeds, B, data) for d in todo], workers))
    return {"provenance": _stamp({"runs": os.path.abspath(runs_root)}), "config": config, "domain": "interior", "metric": "pearson",
            "seeds": seeds, "datasets": {d: out[d] for d in datasets if d in out}}


# ---------------------------------------------------------------- reports
CLS = {"superior": "above", "equivalent": "within", "undetermined": "undetermined", "inferior": "below"}
SHOW = {"DMSO_TX9P1": "TX9_WT", "Subtelny14_ribomimo": "Subtelny14"}


def _f(x, p=3): return "--" if x is None else f"{x:+.{p}f}"


def markdown_setting1(res, methods=TABLE_METHODS):
    ds = list(res["datasets"]); ours = res["ours"]
    L = [f"# Part I, evaluation setting 1 (seeds {', '.join(res['seeds'])}; B = {res['B']})", ""]
    for key, title in (("pearson_interior", "Per-transcript Pearson, interior codons"), ("spearman_interior", "Per-transcript Spearman, interior codons"),
                       ("recall5_interior", "Top-5% peak recall, interior codons"), ("pearson_full_deployed", "Per-transcript Pearson, full length (deployed)")):
        L += [f"## {title}", "", "| Method | " + " | ".join(SHOW.get(d, d) for d in ds) + " |", "|---" * (len(ds) + 1) + "|"]
        for c, nm in methods:
            row = []
            for d in ds:
                v = res["datasets"][d]["absolute"].get(c, {}).get(key)
                row.append("--" if not v else (f"{v['S']:.3f}" if v.get("status") == "OK" else v.get("status")))
            L.append(f"| {nm} | " + " | ".join(row) + " |")
        L.append("")
    L += ["## Paired differences, RiboPipe minus each comparator (interior Pearson)", "",
          "| Dataset | Comparator | Delta | 95% interval | Class | N paired | p sup | Holm sup | p eq | Holm eq |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d in ds:
        for c in res["core"]:
            pr = res["datasets"][d]["pairs"][f"{ours}|{c}"]; k = f"{d}|{c}"
            if pr.get("status") == "OK":
                L.append(f"| {SHOW.get(d, d)} | {c} | {_f(pr['delta_hat'], 4)} | [{_f(pr['ci95'][0], 4)}, {_f(pr['ci95'][1], 4)}] | {CLS[pr['class']]} | {pr['N_paired']} | "
                         f"{pr['p_superiority']:.3g} | {'yes' if res['holm']['superiority'][k] else 'no'} | {pr['p_equivalence_tost']:.3g} | {'yes' if res['holm']['equivalence'][k] else 'no'} |")
            else: L.append(f"| {SHOW.get(d, d)} | {c} | -- | -- | {pr.get('status')} | | | | | |")
        for c in res["extra"]:
            pr = res["datasets"][d]["extra_pairs"].get(f"{ours}|{c}", {})
            if pr.get("status") == "OK":
                L.append(f"| {SHOW.get(d, d)} | {c} (descriptive) | {_f(pr['delta_hat'], 4)} | [{_f(pr['ci95'][0], 4)}, {_f(pr['ci95'][1], 4)}] | {CLS[pr['class']]} | {pr['N_paired']} | | | | |")
    n = sum(res["holm"]["superiority"].values())
    L += ["", f"Holm superiority family: {n}/{res['holm']['m_each']} rejected; equivalence family: {sum(res['holm']['equivalence'].values())}/{res['holm']['m_each']} rejected.", ""]
    return "\n".join(L)


def markdown_setting2(res):
    L = [f"# Part I, evaluation setting 2 (seeds {', '.join(res['seeds'])}; B = {res['B']}; descriptive)", "",
         "| Dataset | Retrained comparator | RiboPipe - method | 95% interval | Change from setting 1 |", "|---|---|---|---|---|"]
    for d, R in res["datasets"].items():
        for c in res["pairs"]:
            p, q = R["ours_vs_matched"][c], R["matched_vs_native"][c]
            if p.get("status") == "OK":
                L.append(f"| {SHOW.get(d, d)} | {c} | {_f(p['delta_hat'])} | [{_f(p['ci95'][0])}, {_f(p['ci95'][1])}] | {_f(q.get('delta_hat'))} |")
            else: L.append(f"| {SHOW.get(d, d)} | {c} | {p.get('status')} | | |")
    return "\n".join(L) + "\n"
