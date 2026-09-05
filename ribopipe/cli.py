"""RiboPipe command-line interface."""
from __future__ import annotations

import argparse
import sys


def _add_feature_flags(ap):
    """Shared feature toggles (headline defaults: all on)."""
    ap.add_argument("--no-nt", dest="use_nt", action="store_false",
                    help="Drop the +/-15 nt one-hot window (ablation)")
    ap.add_argument("--no-struct", dest="use_struct", action="store_false",
                    help="Drop the ViennaRNA MFE features (ablation / no struct cache)")
    ap.add_argument("--backbone", default="cnn", choices=["cnn", "bilstm"],
                    help="cnn = headline exp-motif CNN + BiGRU-128 (default); bilstm = legacy")
    ap.set_defaults(use_nt=True, use_struct=True)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="ribopipe",
        description=(
            "RiboPipe: within-sample Ribo-seq imputation.\n"
            "Learn from high-coverage transcripts, predict pause profiles for sparse ones.\n"
            "Headline model: motif-CNN (k=7) + BiGRU-128 (codon + NT +/-15 + MFE, unweighted Huber; ~0.35M params)."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- raw2csv ----
    ap_raw = sub.add_parser(
        "raw2csv",
        help="Convert CDS-aligned BAM(s) to a per-codon P-site count CSV (input to preprocess)")
    ap_raw.add_argument("--bam", required=True, nargs="+",
                        help="One or more CDS-aligned BAM files (one per sample/replicate)")
    ap_raw.add_argument("--sample", required=True, nargs="+",
                        help="Output count-column name per BAM, e.g. 'HEK293T|WT|DMSO|rep1|mono|x' "
                             "(parallel to --bam)")
    ap_raw.add_argument("--fasta", required=True, help="CDS FASTA the BAMs are aligned to")
    ap_raw.add_argument("--out-csv", required=True, help="Output codon-count CSV")
    ap_raw.add_argument("--offset", type=int, default=12,
                        help="Fixed P-site offset in nt (default 12; ignored if --auto-offset)")
    ap_raw.add_argument("--auto-offset", action="store_true",
                        help="Derive per-read-length P-site offsets from the start-codon metagene")
    ap_raw.add_argument("--min-len", type=int, default=25, help="Min footprint length (default 25)")
    ap_raw.add_argument("--max-len", type=int, default=35, help="Max footprint length (default 35)")

    # ---- preprocess ----
    ap_p = sub.add_parser("preprocess", help="Convert raw Ribo-seq CSV to per-condition NPZ")
    ap_p.add_argument("--csv", required=True, help="Input codon-level counts CSV")
    ap_p.add_argument("--fasta", required=True, help="Transcript FASTA (CDS sequences)")
    ap_p.add_argument("--out-dir", required=True, help="Output directory for NPZ files")
    ap_p.add_argument("--fasta-cache", default=None, help="Optional pickle cache for FASTA dict")

    # ---- matrix ----
    ap_m = sub.add_parser("matrix", help="Build transcript × sample coverage matrix from NPZ directory")
    ap_m.add_argument("--npz-dir", required=True)
    ap_m.add_argument("--out-csv", required=True)

    # ---- struct ----
    ap_s = sub.add_parser(
        "struct",
        help="Precompute the ViennaRNA local-structure (MFE) cache for a dataset NPZ")
    ap_s.add_argument("--npz", required=True, help="Per-transcript NPZ (from preprocess)")
    ap_s.add_argument("--out-npz", default=None,
                      help="Cache output path (default: <npz_dir>/struct_cache/<base>_struct.npz)")
    ap_s.add_argument("--window", type=int, default=30, help="Fold window length in nt (paper: 30)")
    ap_s.add_argument("--workers", type=int, default=None, help="Process-pool size (default: cores-2)")

    # ---- train ----
    ap_t = sub.add_parser("train", help="Train the headline RiboPipe model on a single Ribo-seq sample")
    ap_t.add_argument("--npz", required=True, help="Per-transcript NPZ (from preprocess)")
    ap_t.add_argument("--coverage-csv", required=True, help="Coverage matrix CSV (from matrix)")
    ap_t.add_argument("--sample", required=True, help="Column name in coverage CSV for this sample")
    ap_t.add_argument("--struct-npz", default=None,
                      help="Struct MFE cache (from `ribopipe struct`); required unless --no-struct")
    ap_t.add_argument("--enst2ensg", default=None,
                      help="ENST→ENSG JSON(.gz) for a leak-free gene-level validation hold-out")
    ap_t.add_argument("--hidden", type=int, default=256, choices=[128, 256, 512],
                      help="BiLSTM hidden units per direction (only used with --backbone bilstm)")
    ap_t.add_argument("--epochs", type=int, default=200)
    ap_t.add_argument("--patience", type=int, default=20)
    ap_t.add_argument("--batch-size", type=int, default=64)
    ap_t.add_argument("--lr", type=float, default=1e-3)
    ap_t.add_argument("--loss", default="huber",
                      choices=["huber", "peakmse", "wmse", "peakw"],
                      help="Training loss (headline: unweighted Huber, delta=1)")
    ap_t.add_argument("--target", default="meannorm",
                      choices=["meannorm", "meannorm_log", "minmax"])
    ap_t.add_argument("--max-codons", type=int, default=1000)
    ap_t.add_argument("--out-dir", required=True, help="Output directory for checkpoint + log")
    ap_t.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto)")
    ap_t.add_argument("--seed", type=int, default=123)
    _add_feature_flags(ap_t)

    # ---- predict ----
    ap_r = sub.add_parser("predict", help="Predict pause profiles from a trained checkpoint")
    ap_r.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap_r.add_argument("--npz", required=True, help="Per-transcript NPZ")
    ap_r.add_argument("--struct-npz", default=None, help="Struct MFE cache (if the model uses struct)")
    ap_r.add_argument("--ids", default=None,
                      help="Text file with one transcript ID per line (default: all in NPZ)")
    ap_r.add_argument("--max-codons", type=int, default=1000)
    ap_r.add_argument("--out-csv", required=True, help="Output CSV with per-transcript Pearson scores")
    ap_r.add_argument("--device", default=None)

    # ---- cv5 ----
    ap_c = sub.add_parser(
        "cv5",
        help="Gene-level 5-fold cross-validation (the leak-free headline benchmark protocol)")
    ap_c.add_argument("--npz", required=True, help="Per-transcript NPZ")
    ap_c.add_argument("--enst2ensg", required=True, help="ENST→ENSG JSON(.gz) for gene-level folds")
    ap_c.add_argument("--struct-npz", default=None, help="Struct MFE cache (for the headline model)")
    ap_c.add_argument("--methods", default="ribopipe",
                      help="Comma-separated: ribopipe,bilstm_base,codon_mean,tricodon,ridge")
    ap_c.add_argument("--ids", default=None,
                      help="Text file of transcript IDs to use (default: all in NPZ)")
    ap_c.add_argument("--folds", default=None,
                      help="Frozen gene-level fold file (reproduce/folds/cv5_folds_<TAG>.json). "
                           "Reproduces the paper's evaluation universe (T_high INTERSECT gene-longest "
                           "REP) and fixed fold assignments; overrides --ids and the on-the-fly split.")
    ap_c.add_argument("--n-folds", type=int, default=5)
    ap_c.add_argument("--epochs", type=int, default=200)
    ap_c.add_argument("--patience", type=int, default=20)
    ap_c.add_argument("--hidden", type=int, default=256)
    ap_c.add_argument("--loss", default="huber")
    ap_c.add_argument("--target", default="covmean0_log",
                      choices=["covmean0_log", "covmean0", "meannorm", "meannorm_log", "minmax"],
                      help="Regression target for the neural methods (headline: covmean0_log)")
    ap_c.add_argument("--out-json", default=None, help="Write the summary dict to this JSON")
    ap_c.add_argument("--device", default=None)
    _add_feature_flags(ap_c)  # --backbone {cnn,bilstm}, --no-nt, --no-struct (headline: all on)

    # ---- motifs (interpretability) ----
    ap_mo = sub.add_parser(
        "motifs", help="Read the first-layer exp-motif CNN filters as E/P/A amino-acid logos")
    ap_mo.add_argument("--checkpoint", required=True, help="Headline .pt checkpoint")
    ap_mo.add_argument("--top", type=int, default=3, help="Number of top filters to report")
    ap_mo.add_argument("--rank-aa", default="P", help="Amino acid to rank filters by (default P)")
    ap_mo.add_argument("--rank-register", default="P", choices=["E", "P", "A"],
                       help="Register for ranking (default P-site)")
    ap_mo.add_argument("--out-csv", default=None, help="Per-position AA weight table CSV")
    ap_mo.add_argument("--out-png", default=None, help="E/P/A sequence-logo figure")
    ap_mo.add_argument("--device", default="cpu")

    # ---- ism (interpretability) ----
    ap_is = sub.add_parser(
        "ism", help="End-to-end in-silico-mutagenesis A-site codon/AA attribution")
    ap_is.add_argument("--checkpoint", required=True, help="Headline .pt checkpoint")
    ap_is.add_argument("--npz", required=True, help="Per-transcript NPZ")
    ap_is.add_argument("--struct-npz", default=None, help="Struct MFE cache (if the model uses struct)")
    ap_is.add_argument("--ids", default=None, help="Text file of transcript IDs (default: all)")
    ap_is.add_argument("--max-transcripts", type=int, default=100)
    ap_is.add_argument("--max-len", type=int, default=200, help="Skip transcripts longer than this")
    ap_is.add_argument("--out-csv", default=None, help="Per-codon / per-AA A-site attribution CSV")
    ap_is.add_argument("--device", default="cpu")

    # ---- crossover (reliability) ----
    ap_x = sub.add_parser(
        "crossover",
        help="Estimate the read-split crossover depth D* and the model-favoured fraction")
    ap_x.add_argument("--checkpoint", required=True, help="Headline .pt checkpoint")
    ap_x.add_argument("--npz", required=True, help="Per-transcript NPZ (predictions + pooled counts)")
    ap_x.add_argument("--struct-npz", default=None, help="Struct MFE cache (if the model uses struct)")
    ap_x.add_argument("--ids", default=None, help="Text file of transcript IDs (default: all)")
    ap_x.add_argument("--seeds", type=int, default=10, help="Binomial read-split seeds to average")
    ap_x.add_argument("--max-codons", type=int, default=1000)
    ap_x.add_argument("--out-json", default=None, help="Write D*, fraction and curves to JSON")
    ap_x.add_argument("--device", default="cpu")

    # ---- impute (reliability) ----
    ap_im = sub.add_parser(
        "impute",
        help="Depth-weighted hybrid of prediction and reads (low-coverage estimator)")
    ap_im.add_argument("--checkpoint", required=True, help="Headline .pt checkpoint")
    ap_im.add_argument("--npz", required=True, help="Per-transcript NPZ")
    ap_im.add_argument("--struct-npz", default=None, help="Struct MFE cache (if the model uses struct)")
    ap_im.add_argument("--ids", default=None, help="Text file of transcript IDs (default: all)")
    ap_im.add_argument("--dstar", type=float, default=None,
                       help="Crossover reads/codon (default: estimate from the read split)")
    ap_im.add_argument("--slope", type=float, default=0.5, help="Logistic gate slope in log-depth")
    ap_im.add_argument("--max-codons", type=int, default=1000)
    ap_im.add_argument("--out-csv", required=True, help="Output imputed per-codon profiles CSV")
    ap_im.add_argument("--device", default="cpu")

    args = p.parse_args(argv)

    # ---- dispatch ----
    if args.cmd == "raw2csv":
        if len(args.bam) != len(args.sample):
            p.error(f"--bam ({len(args.bam)}) and --sample ({len(args.sample)}) must match in count")
        from .rawcount import bams_to_codon_csv
        bams_to_codon_csv(
            args.bam, args.sample, args.fasta, args.out_csv,
            auto_offset=args.auto_offset, default_offset=args.offset,
            min_len=args.min_len, max_len=args.max_len,
        )
        return 0

    if args.cmd == "preprocess":
        from .preprocess.builder import csv_to_npz_dir
        written = csv_to_npz_dir(args.csv, args.fasta, args.out_dir, fasta_cache=args.fasta_cache)
        for fp in written:
            print(fp)
        return 0

    if args.cmd == "matrix":
        from .preprocess.matrix import build_coverage_matrix
        out = build_coverage_matrix(args.npz_dir, args.out_csv)
        print(out)
        return 0

    if args.cmd == "struct":
        from .struct import compute_struct_cache
        out = compute_struct_cache(args.npz, cache_path=args.out_npz,
                                   n_workers=args.workers, win=args.window)
        print(out)
        return 0

    if args.cmd == "train":
        from .train import train
        if args.use_struct and not args.struct_npz:
            p.error("--struct-npz is required for the headline model; "
                    "run `ribopipe struct` first, or pass --no-struct for the ablation.")
        train(
            npz_path=args.npz,
            coverage_csv=args.coverage_csv,
            sample_col=args.sample,
            struct_npz_path=args.struct_npz,
            enst2ensg_path=args.enst2ensg,
            hidden=args.hidden,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            max_codons=args.max_codons,
            use_nt=args.use_nt,
            use_struct=args.use_struct,
            loss_name=args.loss,
            target=args.target,
            device=args.device,
            seed=args.seed,
            out_dir=args.out_dir,
            verbose=True,
        )
        return 0

    if args.cmd == "predict":
        import numpy as np
        from .predict import predict_from_checkpoint

        if args.ids:
            with open(args.ids) as f:
                ids = [line.strip() for line in f if line.strip()]
        else:
            z = np.load(args.npz, allow_pickle=True)
            ids = list(z.files)

        _, scores = predict_from_checkpoint(
            pt_path=args.checkpoint,
            npz_path=args.npz,
            transcript_ids=ids,
            struct_npz_path=args.struct_npz,
            out_csv=args.out_csv,
            device=args.device,
            max_codons=args.max_codons,
        )
        print(f"Wrote {len(scores)} transcripts → {args.out_csv}")
        print(scores.head(10).to_string(index=False))
        return 0

    if args.cmd == "cv5":
        import json
        import numpy as np
        from .cv5 import run_cv5

        folds = None
        if args.folds:
            from .folds import load_frozen_folds
            folds, ids, meta = load_frozen_folds(args.folds)
            print(f"[folds] {args.folds}: {meta.get('tag','?')} "
                  f"{len(ids)} tx, {len(folds)} folds (seed {meta.get('seed', 0)})")
        elif args.ids:
            with open(args.ids) as f:
                ids = [line.strip() for line in f if line.strip()]
        else:
            z = np.load(args.npz, allow_pickle=True)
            ids = list(z.files)

        summary = run_cv5(
            npz_path=args.npz,
            all_ids=ids,
            enst2ensg_path=args.enst2ensg,
            methods=[m.strip() for m in args.methods.split(",") if m.strip()],
            struct_npz_path=args.struct_npz,
            folds=folds,
            n_folds=args.n_folds,
            epochs=args.epochs,
            patience=args.patience,
            loss_name=args.loss,
            hidden=args.hidden,
            backbone=args.backbone,
            use_nt=args.use_nt,
            use_struct=args.use_struct,
            target=args.target,
            device=args.device,
            out_json=args.out_json,
            verbose=True,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "motifs":
        from .interpret.motifs import motif_report
        order = motif_report(args.checkpoint, top=args.top, rank_aa=args.rank_aa,
                             rank_register=args.rank_register, out_csv=args.out_csv,
                             out_png=args.out_png, device=args.device)
        print("Top motif filters (by %s-site %s): %s"
              % (args.rank_register, args.rank_aa, ", ".join(str(int(f)) for f in order)))
        return 0

    if args.cmd == "ism":
        import csv
        import numpy as np
        from .interpret.aa import AA20, CODONS
        from .interpret.ism import asite_codon_attribution
        from .model import load_cnn_from_paper_checkpoint

        if args.ids:
            with open(args.ids) as f:
                ids = [line.strip() for line in f if line.strip()]
        else:
            z = np.load(args.npz, allow_pickle=True)
            ids = list(z.files)

        model = load_cnn_from_paper_checkpoint(args.checkpoint, device=args.device)
        res = asite_codon_attribution(
            model, args.npz, ids, struct_npz_path=args.struct_npz,
            max_transcripts=args.max_transcripts, max_len=args.max_len, device=args.device)
        print(f"A-site ISM attribution over {res['n_positions']} positions")
        aa_order = np.argsort(-res["per_aa"])
        print("  top A-site amino acids: " +
              ", ".join(f"{AA20[j]}({res['per_aa'][j]:+.3f})" for j in aa_order[:6]))
        if args.out_csv:
            with open(args.out_csv, "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["level", "symbol", "asite_attribution"])
                for j in range(20):
                    wr.writerow(["aa", AA20[j], f"{res['per_aa'][j]:.5f}"])
                for c in range(64):
                    wr.writerow(["codon", CODONS[c], f"{res['per_codon'][c]:.5f}"])
            print(f"  wrote {args.out_csv}")
        return 0

    if args.cmd in ("crossover", "impute"):
        import numpy as np
        from .model import load_cnn_from_paper_checkpoint
        from .predict import predict
        from .reliability.readsplit import pooled_counts_from_npz

        if args.ids:
            with open(args.ids) as f:
                ids = [line.strip() for line in f if line.strip()]
        else:
            z = np.load(args.npz, allow_pickle=True)
            ids = list(z.files)

        model = load_cnn_from_paper_checkpoint(args.checkpoint, device=args.device)
        use_struct = int(model.config.get("bio_dim", 123)) >= 123
        pred = predict(model, args.npz, ids, use_nt=True, use_struct=use_struct,
                       struct_npz_path=args.struct_npz, device=args.device, max_codons=args.max_codons)
        counts = {k: v for k, v in pooled_counts_from_npz(args.npz).items() if k in pred}

        if args.cmd == "crossover":
            import json
            from .reliability.crossover import estimate_dstar, invert_dstar, fraction_below
            res = estimate_dstar(pred, counts, seeds=args.seeds)
            dstar = res["dstar_codon"]
            inv = invert_dstar(res["rc"], res["raw"], res["model"])
            print(f"model accuracy level m = {res['model']:.3f}  (n={res['n']} tx-splits)")
            print(f"crossover D* = {dstar:.3f} reads/codon" if dstar else
                  "crossover D*: model does not cross reads in the grid")
            print(f"D* via r^-1(m) inversion (no sweep) = {inv:.3f} reads/codon" if inv else "")
            if dstar:
                fb = fraction_below(counts, dstar)
                print(f"model-favoured: {fb['frac_expressed']*100:.0f}% of {fb['n_expressed']} "
                      f"expressed transcripts below D*")
            if args.out_json:
                out = {"dstar_codon": dstar, "dstar_inverted": inv, "model_level": res["model"],
                       "rc_centres": res["rc"].tolist(), "raw_curve": res["raw"].tolist(),
                       "n_tx_splits": res["n"]}
                if dstar:
                    out.update(fraction_below(counts, dstar))
                with open(args.out_json, "w") as fh:
                    json.dump(out, fh, indent=2)
                print(f"wrote {args.out_json}")
            return 0

        # impute
        import csv
        from .reliability.crossover import estimate_dstar
        from .reliability.hybrid import impute as hybrid_impute
        dstar = args.dstar if args.dstar is not None else estimate_dstar(pred, counts)["dstar_codon"]
        if dstar is None:
            p.error("could not estimate D*; pass --dstar explicitly")
        imp = hybrid_impute(pred, counts, dstar, slope=args.slope)
        with open(args.out_csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["transcript", "codon", "imputed_meannorm"])
            for tid, prof in imp.items():
                for i, v in enumerate(prof):
                    wr.writerow([tid, i, f"{v:.5f}"])
        print(f"imputed {len(imp)} transcripts at D*={dstar:.3f} (slope {args.slope}) -> {args.out_csv}")
        return 0

    p.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
