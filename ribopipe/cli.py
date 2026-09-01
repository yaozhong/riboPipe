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
    ap.add_argument("--no-bio", dest="use_bio", action="store_false",
                    help="Drop the 6 biological features (ablation)")
    ap.add_argument("--with-bio", dest="use_bio", action="store_true",
                    help="add 6 hand-crafted biological features (ablation; headline is use_bio=False)")
    ap.add_argument("--backbone", default="cnn", choices=["cnn", "bilstm"],
                    help="cnn = headline exp-motif CNN + BiGRU-128 (default); bilstm = legacy")
    ap.set_defaults(use_nt=True, use_struct=True, use_bio=False)


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

    # ---- biofeat ----
    ap_b = sub.add_parser("biofeat", help="Generate per-codon biological features from an NPZ")
    ap_b.add_argument("--cds-npz", required=True, help="NPZ with cds.sequence entries")
    ap_b.add_argument("--trna-json", required=True, help="tRNA abundance JSON")
    ap_b.add_argument("--out-npz", required=True)

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
    ap_t.add_argument("--bio-npz", required=True, help="Biological features NPZ (from biofeat)")
    ap_t.add_argument("--coverage-csv", required=True, help="Coverage matrix CSV (from matrix)")
    ap_t.add_argument("--sample", required=True, help="Column name in coverage CSV for this sample")
    ap_t.add_argument("--struct-npz", default=None,
                      help="Struct MFE cache (from `ribopipe struct`); required unless --no-struct")
    ap_t.add_argument("--enst2ensg", default=None,
                      help="ENST→ENSG JSON(.gz) for a leak-free gene-level validation hold-out")
    ap_t.add_argument("--hidden", type=int, default=256, choices=[128, 256, 512],
                      help="BiLSTM hidden units per direction (256=headline)")
    ap_t.add_argument("--epochs", type=int, default=200)
    ap_t.add_argument("--patience", type=int, default=20)
    ap_t.add_argument("--batch-size", type=int, default=64)
    ap_t.add_argument("--lr", type=float, default=1e-3)
    ap_t.add_argument("--loss", default="peakmse",
                      choices=["peakmse", "huber", "wmse", "peakw"],
                      help="Training loss (headline: peakmse)")
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
    ap_r.add_argument("--bio-npz", required=True, help="Biological features NPZ")
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
    ap_c.add_argument("--bio-npz", required=True, help="Biological features NPZ")
    ap_c.add_argument("--enst2ensg", required=True, help="ENST→ENSG JSON(.gz) for gene-level folds")
    ap_c.add_argument("--struct-npz", default=None, help="Struct MFE cache (for the headline model)")
    ap_c.add_argument("--methods", default="ribopipe_nt_struct_h256",
                      help="Comma-separated: ribopipe_nt_struct_h256,bilstm_base,codon_mean,tricodon,ridge")
    ap_c.add_argument("--ids", default=None,
                      help="Text file of transcript IDs to use (default: all in NPZ)")
    ap_c.add_argument("--n-folds", type=int, default=5)
    ap_c.add_argument("--epochs", type=int, default=200)
    ap_c.add_argument("--patience", type=int, default=20)
    ap_c.add_argument("--hidden", type=int, default=256)
    ap_c.add_argument("--loss", default="peakmse")
    ap_c.add_argument("--out-json", default=None, help="Write the summary dict to this JSON")
    ap_c.add_argument("--device", default=None)

    args = p.parse_args(argv)

    # ---- dispatch ----
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

    if args.cmd == "biofeat":
        from .preprocess.biofeat import build_bio_features
        out = build_bio_features(args.cds_npz, args.trna_json, args.out_npz)
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
            bio_npz_path=args.bio_npz,
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
            use_bio=args.use_bio,
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
            bio_npz_path=args.bio_npz,
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

        if args.ids:
            with open(args.ids) as f:
                ids = [line.strip() for line in f if line.strip()]
        else:
            z = np.load(args.npz, allow_pickle=True)
            ids = list(z.files)

        summary = run_cv5(
            npz_path=args.npz,
            bio_npz_path=args.bio_npz,
            all_ids=ids,
            enst2ensg_path=args.enst2ensg,
            methods=[m.strip() for m in args.methods.split(",") if m.strip()],
            struct_npz_path=args.struct_npz,
            n_folds=args.n_folds,
            epochs=args.epochs,
            patience=args.patience,
            loss_name=args.loss,
            hidden=args.hidden,
            device=args.device,
            out_json=args.out_json,
            verbose=True,
        )
        print(json.dumps(summary, indent=2))
        return 0

    p.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
