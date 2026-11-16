"""Command-line entry point for the paper's baseline configurations.

  python -m baselines.run --list
  python -m baselines.run --config Paper_Ridge --dataset GSE133393_WT --out runs/                       (all 5 folds, seed det)
  python -m baselines.run --config Paper_RiboMIMO --dataset Mohammad16 --out runs/ --folds 0 --seeds 123
  python -m baselines.run --config Paper_iXnos_MFE_lr005_elig --dataset PRJNA_iPS --out runs/ --device cpu

Smoke runs (tiny subsets; never used for results): --smoke NTRAIN NVAL NTEST --max-epochs E
Data: RIBOPIPE_DATA / RIBOPIPE_MFE (or --data / --mfe).  RiboMIMO, RiboExp and RiboGL need the official code fetched and
patched by baselines/fetch_upstream.sh (location: baselines/upstream or RIBOPIPE_BASELINE_UPSTREAM).
"""
import argparse, sys

# config id -> (module, kind, deterministic, description)
CONFIGS = {
    "Paper_CodonMean": ("inhouse", None, True, "codon-mean lookup (in-house reference)"),
    "Paper_TriCodon": ("inhouse", None, True, "tri-codon (P, A, A+1) lookup (in-house reference)"),
    "Paper_Ridge": ("inhouse", None, True, "ridge, one-hot codons -5..+5, alpha 10 (in-house reference)"),
    "Paper_RUST": ("inhouse", None, True, "RUST enrichment-ratio estimator (our re-implementation)"),
    "Paper_iXnos_seq_lr005_elig": ("ixnos", None, False, "iXnos, sequence only (table row 'iXnos')"),
    "Paper_iXnos_MFE_lr005_elig": ("ixnos", None, False, "iXnos + 3 raw MFE windows (row 'iXnos (MFE)'; core comparator)"),
    "Paper_iXnos_MFE_lr005": ("ixnos", None, False, "iXnos (MFE), 500 densest genes of the training fold (sensitivity row)"),
    "Paper_iXnos_nocut_lr005_elig": ("ixnos", None, False, "iXnos sequence model, target c/mean(c), no end trim (setting 2)"),
    "Paper_iXnos_seq_lr005": ("ixnos", None, False, "iXnos sequence only, 500 densest genes (reference for setting 2 of the 500-gene family)"),
    "Paper_RiboMIMO": ("ribomimo", None, False, "RiboMIMO, official architecture and target, adapted training driver (core comparator)"),
    "Paper_RiboMIMO_L2": ("ribomimo", None, False, "RiboMIMO retrained on y = c/mean(c) (setting 2)"),
    "Paper_RiboMIMO_impl": ("ribomimo", None, False, "RiboMIMO task-adapted control (target c/mean(c[c>0]), no filter)"),
    "Paper_RiboMIMO_adapt_noMFE": ("ribomimo", None, False, "RiboMIMO-adapt backbone without MFE (setting 3 / components figure)"),
    "Paper_RiboMIMO_adapt_MFE": ("ribomimo", None, False, "RiboMIMO-adapt backbone + RiboPipe's 3 MFE inputs (components figure)"),
    "Paper_iXnos_seq": ("ixnos", None, False, "iXnos sequence only, official executed lr 0.01, 500 densest genes"),
    "Paper_iXnos_MFE": ("ixnos", None, False, "iXnos + MFE, official executed lr 0.01, 500 densest genes (collapses on 5/8)"),
    "Paper_iXnos_nocut": ("ixnos", None, False, "iXnos nocut, lr 0.01, 500 densest genes"),
    "Paper_iXnos_nocut_lr005": ("ixnos", None, False, "iXnos nocut, lr 0.005, 500 densest genes"),
    "Paper_RiboExp_noStruct_port": ("riboexp", None, False, "RiboExp --no_structure (non-default; extended control)"),
    "Paper_RiboExp": ("riboexp", None, False, "RiboExp, official default with structure features"),
    "Paper_RiboGL_heldout": ("ribogl", None, False, "RiboGL, checkpoint on frozen validation set (core comparator)"),
    "Paper_RiboGL_L2_heldout": ("ribogl", None, False, "RiboGL retrained on y = c/mean(c) (setting 2)"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m baselines.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list configurations")
    ap.add_argument("--config", choices=sorted(CONFIGS))
    ap.add_argument("--dataset")
    ap.add_argument("--out", help="runs root")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="123", help="comma list; pre-registered seeds 123,124,125 (ignored by deterministic methods)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N (iXnos paper runs used cpu)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", type=int, nargs=3, metavar=("NTRAIN", "NVAL", "NTEST"), default=None,
                    help="SMOKE ONLY: first n sorted ids of each split")
    ap.add_argument("--max-epochs", type=int, default=0, help="SMOKE ONLY: cap the number of epochs")
    ap.add_argument("--allow-any-seed", action="store_true", help="allow seeds outside 123/124/125 (smoke)")
    ap.add_argument("--data", default=None); ap.add_argument("--mfe", default=None)
    ap.add_argument("--nproc", type=int, default=8, help="RiboGL: processes for RNAfold graph construction")
    a = ap.parse_args(argv)
    if a.list or not a.config:
        for k, (m, _, det, desc) in CONFIGS.items(): print(f"{k:32s} {m:9s} {'det' if det else 'seeded':7s} {desc}")
        return 0
    if not (a.dataset and a.out): ap.error("--dataset and --out are required")
    smoke = bool(a.smoke or a.max_epochs)
    folds = [int(x) for x in a.folds.split(",") if x != ""]
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    from . import common as C
    bad = [s for s in seeds if s not in C.PREREG_SEEDS]
    if bad and not (a.allow_any_seed or smoke) and not CONFIGS[a.config][2]:
        ap.error(f"seeds {bad} are not pre-registered {C.PREREG_SEEDS} (use --allow-any-seed)")
    mod = CONFIGS[a.config][0]
    if mod == "inhouse":
        from . import inhouse
        return inhouse.run(a.config, a.dataset, a.out, folds, a.smoke, a.overwrite, a.data, a.mfe)
    if mod == "ixnos":
        from . import ixnos
        return ixnos.run(a.config, a.dataset, a.out, folds, seeds, a.device, a.smoke, a.max_epochs, a.overwrite, a.data, a.mfe)
    rc = 0
    for f in folds:
        for s in seeds:
            if mod == "ribomimo":
                from . import ribomimo
                rc |= ribomimo.run(a.config, a.dataset, f, s, a.out, a.device, a.smoke, a.max_epochs, a.overwrite, a.data, a.mfe)
            elif mod == "riboexp":
                from . import riboexp
                rc |= riboexp.run(a.config, a.dataset, f, s, a.out, a.device, a.smoke, a.max_epochs, a.overwrite, a.data, a.mfe)
            elif mod == "ribogl":
                from . import ribogl
                rc |= ribogl.run(a.config, a.dataset, f, s, a.out, a.smoke, a.max_epochs, a.overwrite, a.data, a.mfe, a.nproc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
