"""Model analyses of Part III: in silico substitutions, branch routing and window transplantation.

  ribopipe ism        --dataset D --ckpt-dir RUNS/<config>/D/seed123 --out OUT [--controls] [--workers N]
                      peak ISM at offsets -12..+10 (and the far-offset controls +-15, +-20 with --controls)
  ribopipe ism-map    --dataset D --ckpt-dir ... --out OUT      centred codon / nucleotide sensitivity maps
  ribopipe routing    --dataset D --ckpt-dir ... --out OUT      branch routing at 1,600 sampled interior codons
  ribopipe transplant --dataset D --ckpt-dir ... --out OUT [--subwindows]
                      whole-window transplant (ordered, shuffled, neutral) and optionally the sub-window grafts
  ribopipe logos      --dataset D --ckpt-dir ... --out OUT [--fold 0]   activation logos of the first-layer filters

Every command writes the raw arrays (<OUT>/<dataset>/<name>.npz) and a summary (<name>_summary.json). Folding energies
are recomputed with ViennaRNA (optional dependency ``ViennaRNA``). Models are analysed out of fold: --ckpt-dir must
contain fold<f>/model.pt for every fold (and, for the paper's peak selection, fold<f>/pred.npz).
"""
import os, json, time, socket
import numpy as np

from . import core, ism, routing, transplant, logos   # noqa: F401

__all__ = ["core", "ism", "routing", "transplant", "logos", "register"]


def _ctx(a):
    from .. import data as D
    ds = D.load_dataset(a.dataset, data=a.data, mfe=a.mfe)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    return core.OOF(ds, a.ckpt_dir, preds=a.preds, folds=folds)


def _outdir(a):
    d = os.path.join(a.out, a.dataset); os.makedirs(d, exist_ok=True); return d


def _save(d, name, arrays, summary, oof, extra=None):
    np.savez_compressed(os.path.join(d, f"{name}.npz"), **{k: v for k, v in arrays.items()})
    meta = dict(provenance=oof.provenance(), host=socket.gethostname(), finished=time.strftime("%F %T"), **(extra or {}))
    core.jdump(dict(meta=meta, summary=summary), os.path.join(d, f"{name}_summary.json"))
    print(f"[write] {os.path.join(d, name)}.npz / _summary.json", flush=True)


def _cmd_ism(a):
    oof = _ctx(a); d = _outdir(a)
    z = ism.run_peak_ism(oof, oof.folds, core.OFFSETS, workers=a.workers)
    _save(d, "ism", z, ism.summarize_ism(z), oof)
    if a.controls:
        peaks = list(zip(z["tx"], z["p"], z["fold"], z["half"], z["unit"]))
        ze = ism.run_peak_ism(oof, oof.folds, core.EXT_OFFSETS, peaks=[(str(t), int(p), int(f), str(h), str(u)) for t, p, f, h, u in peaks],
                              workers=a.workers)
        _save(d, "ism_controls", ze, ism.summarize_response_range(z, ze), oof)
    return 0


def _cmd_map(a):
    oof = _ctx(a); d = _outdir(a)
    z = ism.run_maps(oof, oof.folds, workers=a.workers)
    _save(d, "map", z, ism.summarize_map(z), oof)
    return 0


def _cmd_routing(a):
    oof = _ctx(a); d = _outdir(a)
    r = routing.run_routing(oof, workers=a.workers, gate=not a.no_gate, gate_tol=a.gate_tol)
    _save(d, "routing", r["arrays"], routing.summarize_routing(r["arrays"]), oof,
          dict(gate=r.get("gate"), self_substitution_exact_all=r["self_substitution_exact_all"], wt_dev_vs_pred_max=r["wt_dev_vs_pred_max"]))
    return 0 if r["self_substitution_exact_all"] else 3


def _cmd_transplant(a):
    oof = _ctx(a); d = _outdir(a)
    z = transplant.run_transplant(oof, oof.folds, workers=a.workers)
    _save(d, "transplant", z, transplant.summarize_transplant(z), oof)
    if a.subwindows:
        zs = transplant.run_subwindows(oof, tp=z, workers=a.workers)
        _save(d, "subwindow", zs, transplant.summarize_subwindows(zs), oof)
    return 0


def _cmd_logos(a):
    from .. import data as D
    ds = D.load_dataset(a.dataset, data=a.data, mfe=a.mfe); d = _outdir(a)
    m, info = core.load_model(os.path.join(a.ckpt_dir, f"fold{a.fold}", "model.pt"), dataset=ds.tag, fold=a.fold)
    lg = logos.activation_logos(m, ds, ds.test_ids(a.fold), a.device)
    np.savez_compressed(os.path.join(d, f"logos_fold{a.fold}.npz"), **{f"logo|{b}|{j}": v[j] for b, v in lg.items() for j in range(len(v))},
                        **{f"letters|{b}": np.array(logos.letter_labels(b)) for b in lg})
    core.jdump(dict(checkpoint={k: v for k, v in info.items() if k != "mfe_norm"}, fold=a.fold, device=a.device,
                    definition=logos.__doc__.strip()), os.path.join(d, f"logos_fold{a.fold}.json"))
    print(f"[write] {d}/logos_fold{a.fold}.npz", flush=True); return 0


def register(sp):
    def common(p):
        p.add_argument("--dataset", required=True)
        p.add_argument("--ckpt-dir", required=True, help="directory with fold<f>/model.pt (e.g. runs/<config>/<dataset>/seed123)")
        p.add_argument("--out", required=True)
        p.add_argument("--folds", default="", help="comma-separated folds (default: all)")
        p.add_argument("--preds", default="stored", choices=("stored", "compute"),
                       help="out-of-fold predictions for peak/neutral-site selection: fold<f>/pred.npz (stored) or computed on CPU")
        p.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 1)))
        p.add_argument("--data", default=None); p.add_argument("--mfe", default=None)
    p = sp.add_parser("ism", help="peak in silico mutagenesis (single-codon edits around predicted peaks)")
    common(p); p.add_argument("--controls", action="store_true", help="also edit offsets -20, -15, +15, +20 of the same peaks")
    p.set_defaults(func=_cmd_ism)
    p = sp.add_parser("ism-map", help="centred codon and nucleotide sensitivity maps at sampled interior sites")
    common(p); p.set_defaults(func=_cmd_map)
    p = sp.add_parser("routing", help="branch routing of single edits at sampled interior codons")
    common(p); p.add_argument("--no-gate", action="store_true", help="skip the local-recomputation gate")
    p.add_argument("--gate-tol", type=float, default=routing.GATE_TOL,
                   help="largest allowed |local - full forward| in the gate (default %(default)g; 0 = bit-identical)")
    p.set_defaults(func=_cmd_routing)
    p = sp.add_parser("transplant", help="window transplantation (whole window, shuffled, neutral; optional sub-windows)")
    common(p); p.add_argument("--subwindows", action="store_true", help="also graft parts of the window, a wider window and the shuffled window")
    p.set_defaults(func=_cmd_transplant)
    p = sp.add_parser("logos", help="activation logos of the first-layer filters (inspection only)")
    p.add_argument("--dataset", required=True); p.add_argument("--ckpt-dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--fold", type=int, default=0); p.add_argument("--device", default="cpu")
    p.add_argument("--data", default=None); p.add_argument("--mfe", default=None); p.set_defaults(func=_cmd_logos)
