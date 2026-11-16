"""Part I evaluation: per-transcript scores, paired similarity-unit bootstrap, margin classes and Holm families.

Commands registered on the ``ribopipe`` CLI:

  ribopipe score    --runs RUNS --dataset GSE133393_WT --config Paper_RiboPipe_MFE_H128 [Paper_RiboMIMO ...]
                    [--seeds 123] [--metric pearson|spearman|recall5] [--domain interior|full|full_deployed] [--json OUT]
      absolute five-fold score S of each configuration (median per fold, mean over folds, mean over seeds)
  ribopipe compare  --runs RUNS --dataset GSE133393_WT --a Paper_RiboPipe_MFE_H128 --b Paper_RiboMIMO
                    [--seeds 123] [--B 2000] [--metric ...] [--domain ...] [--json OUT]
      paired difference S_A - S_B on the common set, 95% similarity-unit bootstrap interval, margin class (+/-0.01) and
      the boundary-shifted Wald p values (superiority beyond +0.01; TOST equivalence)
  ribopipe evaluate-partI --runs RUNS --out DIR [--what setting1|setting2|seeds|all] [--seeds 123 [124 125]]
                    [--datasets ...] [--configs ...] [--B 2000] [--workers 8]
      the paper's Part I evaluation (all methods, 24-comparison Holm families, setting 2, RiboPipe across seeds);
      writes JSON + Markdown

Runs follow the layout written by ``ribopipe.data.write_run``; data are read with ``ribopipe.data.load_dataset``
(``--data`` or RIBOPIPE_DATA). See ``scorer.py`` for every rule.
"""
import os, json
from . import scorer
from .scorer import (Dataset, load_method_runs, method_table, scorable_set, S_on, bootstrap_units, score_dataset,
                     score_interior_vs_full, score_pair_interior_vs_full, score_common_arms, holm, pearson, spearman,
                     recall5, load_protocol, DELTA)
from . import partI


def _dump(obj, path):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        json.dump(obj, open(path, "w"), indent=1, default=str); print(f"wrote {path}")


def _cmd_score(a):
    ds = partI.load_scoring_dataset(a.dataset, a.data)
    runs = {c: load_method_runs(a.runs, c, a.dataset, partI.seeds_of(a.runs, c, a.dataset, a.seeds), range(5)) for c in a.config}
    r = score_dataset(ds, runs, a.domain, pairs=(), B=0, metric=a.metric)
    for c in a.config:
        v = r["absolute"][c]
        if v["status"] == "OK":
            print(f"{c:36s} {a.metric}_{a.domain}: S = {v['S']:.4f}  seeds {['%.4f' % x for x in v['seed_scores']]}  "
                  f"N scored {v['N_scored']}/{v['N_truth_scorable']} truth-scorable ({v['N_domain']} in domain)")
        else:
            print(f"{c:36s} {v['status']}")
    _dump(r, a.json); return 0


def _cmd_compare(a):
    ds = partI.load_scoring_dataset(a.dataset, a.data)
    runs = {c: load_method_runs(a.runs, c, a.dataset, partI.seeds_of(a.runs, c, a.dataset, a.seeds), range(5)) for c in (a.a, a.b)}
    r = score_dataset(ds, runs, a.domain, pairs=[(a.a, a.b)], B=a.B, metric=a.metric)
    p = r["pairs"][f"{a.a}|{a.b}"]
    if p["status"] == "OK":
        print(f"{a.a} - {a.b} on {a.dataset} ({a.metric}, {a.domain}): Delta = {p['delta_hat']:+.4f} "
              f"95% [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  class {p['class']}  N paired {p['N_paired']}  "
              f"p_sup {p['p_superiority']:.3g}  p_eq {p['p_equivalence_tost']:.3g}")
    else:
        print(f"{a.a} - {a.b}: {p['status']}")
    _dump(r, a.json); return 0


def _cmd_evaluate(a):
    os.makedirs(a.out, exist_ok=True)
    kw = dict(datasets=a.datasets or partI.DATASETS, B=a.B, data=a.data, workers=a.workers)
    tag = "_".join(a.seeds)
    if a.what in ("setting1", "all"):
        r = partI.evaluate_setting1(a.runs, seeds=a.seeds, configs=a.configs, **kw)
        _dump(r, os.path.join(a.out, f"partI_setting1_seed{tag}.json"))
        open(os.path.join(a.out, f"partI_setting1_seed{tag}.md"), "w").write(partI.markdown_setting1(r))
    if a.what in ("setting2", "all"):
        r = partI.evaluate_setting2(a.runs, seeds=a.seeds, **kw)
        _dump(r, os.path.join(a.out, f"partI_setting2_seed{tag}.json"))
        open(os.path.join(a.out, f"partI_setting2_seed{tag}.md"), "w").write(partI.markdown_setting2(r))
    if a.what in ("seeds", "all"):
        kw2 = dict(kw); kw2["B"] = 200
        r = partI.seed_noise(a.runs, seeds=a.seeds if len(a.seeds) > 1 else partI.PAPER_SEEDS, **kw2)
        _dump(r, os.path.join(a.out, "partI_ribopipe_seeds.json"))
    return 0


def register(sp):
    def common(p, seeds_default=("123",)):
        p.add_argument("--runs", required=True, help="runs root: <runs>/<config>/<dataset>/seed<k>/fold<f>/")
        p.add_argument("--seeds", nargs="+", default=list(seeds_default), help="pre-registered seeds to aggregate (deterministic methods use 'det')")
        p.add_argument("--data", default=None, help="data directory (default: ./data or $RIBOPIPE_DATA)")
    p = sp.add_parser("score", help="Part I absolute score of one or more configurations")
    common(p); p.add_argument("--dataset", required=True); p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--metric", default="pearson", choices=sorted(scorer.METRICS))
    p.add_argument("--domain", default="interior", choices=("interior", "full", "full_deployed"))
    p.add_argument("--json", default=None); p.set_defaults(func=_cmd_score)
    p = sp.add_parser("compare", help="Part I paired comparison A - B with the similarity-unit bootstrap")
    common(p); p.add_argument("--dataset", required=True); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--B", type=int, default=scorer.B_DEFAULT)
    p.add_argument("--metric", default="pearson", choices=sorted(scorer.METRICS))
    p.add_argument("--domain", default="interior", choices=("interior", "full", "full_deployed"))
    p.add_argument("--json", default=None); p.set_defaults(func=_cmd_compare)
    p = sp.add_parser("evaluate-partI", help="the paper's Part I evaluation (settings 1 and 2, Holm families, seeds)")
    common(p); p.add_argument("--out", required=True)
    p.add_argument("--what", default="setting1", choices=("setting1", "setting2", "seeds", "all"))
    p.add_argument("--datasets", nargs="+", default=None); p.add_argument("--configs", nargs="+", default=None,
                   help="configurations to score in setting 1 (default: every directory under --runs)")
    p.add_argument("--B", type=int, default=scorer.B_DEFAULT); p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_cmd_evaluate)
