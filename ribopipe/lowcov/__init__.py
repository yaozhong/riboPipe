"""Low-coverage analysis (the paper's Part II).

  protocol.py   frozen bin edges and rules (PARTII_CROSSOVER_v2.json)
  crossover.py  per-bin paired differences and the crossover depth D* with similarity-unit bootstrap
  libraries.py  replicate matching, binomial split, nested thinning, interior Pearson
  score.py      crossover evaluation of one dataset (main analysis and three sensitivities, noise control)
  predict.py    out-of-fold inference on the Part II universe with the Part I fold checkpoints
  posterior.py  Dirichlet-multinomial posterior with the prediction as prior; prior strength chosen on
                high-coverage transcripts of the other folds; comparison in the crossover bins
  ranking.py    controlled thinning on high-coverage transcripts: top-5% codon ranking

Commands (registered into `ribopipe`):
  ribopipe lowcov-predict     --runs RUNS --dataset TAG --out PREDS [--config C --seed 123]
  ribopipe crossover          --dataset TAG --pred NAME=PATH [--pred ...] --out DIR [--variants v2_primary]
  ribopipe crossover-summary  --eval DIR/eval_TAG.json [...] --out PREFIX
  ribopipe posterior-kappa    --dataset TAG --oof RUN_DIR --out kappa.json
  ribopipe posterior          --dataset TAG --pred PATH --kappa kappa.json --out DIR
  ribopipe posterior-summary  --score DIR/posterior_TAG.json [...] --out PREFIX
  ribopipe peak-ranking       --dataset TAG --oof RUN_DIR [--oof-codonmean RUN_DIR --oof-ridge RUN_DIR] --out a1.json
"""
import json
import os
import time


def _common(p):
    p.add_argument("--data", default=None, help="data directory (default: ./data or $RIBOPIPE_DATA)")
    p.add_argument("--mfe", default=None, help="MFE track directory (default: <data>/mfe or $RIBOPIPE_MFE)")


def _dump(obj, fn):
    d = os.path.dirname(os.path.abspath(fn)); os.makedirs(d, exist_ok=True)
    json.dump(obj, open(fn, "w"), indent=1, default=float)
    return fn


def _cmd_predict(a):
    from .predict import predict_partII
    predict_partII(a.runs, a.dataset, a.out, config=a.config, seed=a.seed, device=a.device, data=a.data, mfe=a.mfe,
                   limit_per_fold=a.limit_per_fold)
    return 0


def _cmd_crossover(a):
    from .. import data as D
    from . import protocol, score
    from .preds import load_predictions, parse_pred_args
    ds = D.load_dataset(a.dataset, "partII", data=a.data, mfe=a.mfe, load_mfe=False)
    low = sorted(ds.strata["low"]); preds, info = {}, {}
    for name, path in parse_pred_args(a.pred):
        preds[name], info[name] = load_predictions(path, low)
    variants = a.variants.split(",") if a.variants else None
    out_dir = os.path.join(a.out, ds.tag)
    score.evaluate_dataset(ds, preds, protocol.edges(a.protocol), out_dir=out_dir, variants=variants, B=a.B, workers=a.workers,
                           seed=a.seed, dummy=not a.no_dummy, equiv=a.equiv, pred_info=info)
    fn = os.path.join(out_dir, f"eval_{ds.tag}.json")
    print(f"[crossover] wrote {fn}")
    return 0 if os.path.exists(fn) else 1


def _cmd_crossover_summary(a):
    from . import score
    ev = {}
    for fn in a.eval:
        e = json.load(open(fn)); ev[e["dataset"]] = e
    s = score.summarize(ev)
    _dump(s, a.out + ".json"); open(a.out + ".md", "w").write(score.summary_markdown(s))
    print(score.summary_markdown(s)); return 0


def _cmd_kappa(a):
    from .. import data as D
    from . import posterior as PO
    from .preds import load_predictions
    ds1 = D.load_dataset(a.dataset, "partI", data=a.data, mfe=a.mfe, load_mfe=False)
    oof, info = load_predictions(a.oof, ds1.ids)
    _, ch = PO.select_kappa(ds1, oof)
    j = PO.kappa_to_json(ch, ds1.tag); j["oof"] = info; j["written"] = time.strftime("%F %T")
    _dump(j, a.out); print(f"[posterior-kappa] wrote {a.out}")
    for k, v in j["chosen"].items(): print(f"  {k}: kappa = {v['value']:.6g} reads/codon (grid index {v['grid_index']})")
    return 0


def _cmd_posterior(a):
    from .. import data as D
    from . import posterior as PO, protocol
    from .preds import load_predictions
    from .score import write_records
    ds = D.load_dataset(a.dataset, "partII", data=a.data, mfe=a.mfe, load_mfe=False)
    low = sorted(ds.strata["low"])
    pred, info = load_predictions(a.pred, low)
    if a.kappa:
        kappa = PO.kappa_from_json(a.kappa)
    else:
        if not a.oof: raise SystemExit("give --kappa (from `ribopipe posterior-kappa`) or --oof (Part I out-of-fold predictions)")
        ds1 = D.load_dataset(a.dataset, "partI", data=a.data, mfe=a.mfe, load_mfe=False)
        kappa, _ = PO.select_kappa(ds1, load_predictions(a.oof, ds1.ids)[0])
    res = PO.score_posterior(ds, pred, kappa, protocol.edges(a.protocol), draw=a.draw, B=a.B, workers=a.workers,
                             identity_check=a.identity_check)
    recs = res.pop("_records")
    os.makedirs(a.out, exist_ok=True)
    write_records(os.path.join(a.out, f"posterior_records_{ds.tag}_draw{a.draw}.tsv.gz"), recs)
    res["pred_info"] = info
    fn = _dump(res, os.path.join(a.out, f"posterior_{ds.tag}" + ("" if a.draw == 0 else f"_draw{a.draw}") + ".json"))
    print(f"[posterior] wrote {fn}")
    return 0


def _cmd_posterior_summary(a):
    from . import posterior as PO
    sc = {}
    for fn in a.score:
        s = json.load(open(fn)); sc[s["dataset"]] = s
    res = PO.aggregate(sc)
    _dump(res, a.out + ".json")
    r = res["reading"]; p = res["pooled"]
    txt = (f"valid bins {p['n_valid_bins']}; posterior - reads: point > 0 in {p['n_point_above_zero|post_minus_reads']}, "
           f"95% interval > 0 in {p['n_ci_above_zero|post_minus_reads']}; posterior - prediction: point > 0 in "
           f"{p['n_point_above_zero|post_minus_ribopipe']}, interval > 0 in {p['n_ci_above_zero|post_minus_ribopipe']}; "
           f"both > 0 in {r['pooled_n_gt_both']} ({r['pooled_frac_gt_both']:.3f}); decision rule holds: {r['holds']}")
    open(a.out + ".md", "w").write("# Posterior in the crossover bins\n\n" + txt + "\n"); print(txt)
    return 0


def _cmd_ranking(a):
    from .. import data as D
    from . import ranking
    from .preds import load_predictions
    ds1 = D.load_dataset(a.dataset, "partI", data=a.data, mfe=a.mfe, load_mfe=False)
    ld = lambda p: (load_predictions(p, ds1.ids, dtype="float64")[0] if p else None)
    res = ranking.run(ds1, ld(a.oof), ld(a.oof_codonmean), ld(a.oof_ridge))
    _dump(res, a.out); print(f"[peak-ranking] wrote {a.out}")
    return 0


def register(sp):
    p = sp.add_parser("lowcov-predict", help="out-of-fold predictions on the Part II universe with the Part I fold checkpoints")
    p.add_argument("--runs", required=True, help="run root holding <config>/<dataset>/seed<s>/fold<f>/model.pt")
    p.add_argument("--dataset", required=True); p.add_argument("--out", required=True)
    p.add_argument("--config", default="Paper_RiboPipe_MFE_H128"); p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu"); p.add_argument("--limit-per-fold", type=int, default=0, help="smoke runs only")
    _common(p); p.set_defaults(func=_cmd_predict)

    p = sp.add_parser("crossover", help="paired comparison of prediction and sparse reads against the other replicate; D*")
    p.add_argument("--dataset", required=True)
    p.add_argument("--pred", action="append", required=True, metavar="NAME=PATH",
                   help="predictions on the Part II universe (npz, pred directory or write_run directory); repeatable")
    p.add_argument("--out", required=True, help="output root; files go to <out>/<dataset>/")
    p.add_argument("--variants", default="", help="comma-separated subset of v2_primary,v2_sens_inputdepth,v2_sens_gridthin,v2_sens_binom")
    p.add_argument("--B", type=int, default=2000); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="replicate-matching thinning seed (0 = main result)")
    p.add_argument("--no-dummy", action="store_true", help="omit the sequence-independent noise predictor")
    p.add_argument("--equiv", action="store_true", help="also check dstar_with_curves against the reference dstar()")
    p.add_argument("--protocol", default=None, help="protocol JSON with other edges (default: frozen v2)")
    _common(p); p.set_defaults(func=_cmd_crossover)

    p = sp.add_parser("crossover-summary", help="summarise crossover evaluations (D*, fractions below D*, noise-control bins)")
    p.add_argument("--eval", nargs="+", required=True); p.add_argument("--out", required=True, help="output prefix (.json, .md)")
    p.set_defaults(func=_cmd_crossover_summary)

    p = sp.add_parser("posterior-kappa", help="choose the posterior prior strength on high-coverage transcripts of other folds")
    p.add_argument("--dataset", required=True); p.add_argument("--oof", required=True, help="Part I out-of-fold predictions")
    p.add_argument("--out", required=True); _common(p); p.set_defaults(func=_cmd_kappa)

    p = sp.add_parser("posterior", help="posterior (reads + prediction prior) vs reads and prediction in the crossover bins")
    p.add_argument("--dataset", required=True); p.add_argument("--pred", required=True, help="predictions on the Part II universe")
    p.add_argument("--kappa", default=None, help="output of posterior-kappa"); p.add_argument("--oof", default=None)
    p.add_argument("--out", required=True); p.add_argument("--draw", type=int, default=0, help="matching thinning seed")
    p.add_argument("--B", type=int, default=2000); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--identity-check", action="store_true", help="check kappa=0 -> reads and kappa->inf -> prediction")
    p.add_argument("--protocol", default=None); _common(p); p.set_defaults(func=_cmd_posterior)

    p = sp.add_parser("posterior-summary", help="aggregate posterior results over datasets (per-bin table, decision rule)")
    p.add_argument("--score", nargs="+", required=True); p.add_argument("--out", required=True, help="output prefix")
    p.set_defaults(func=_cmd_posterior_summary)

    p = sp.add_parser("peak-ranking", help="controlled thinning: top-5%% codon ranking on high-coverage transcripts")
    p.add_argument("--dataset", required=True); p.add_argument("--oof", required=True, help="RiboPipe Part I out-of-fold predictions")
    p.add_argument("--oof-codonmean", default=None); p.add_argument("--oof-ridge", default=None)
    p.add_argument("--out", required=True); _common(p); p.set_defaults(func=_cmd_ranking)
