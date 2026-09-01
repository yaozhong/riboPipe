#!/usr/bin/env python3
"""Visualise observed vs predicted pause profiles for a set of transcripts.

Usage:
    python visualise_predictions.py \
        --npz path/to/sample.npz \
        --bio-npz path/to/bio_features.npz \
        --checkpoint path/to/ribopipe_model.pt \
        --ids transcript_ids.txt \
        --out predictions.pdf
"""
import argparse, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, "..")
from ribopipe.model import load_model
from ribopipe.predict import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--bio-npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ids", required=True, help="Text file, one transcript ID per line")
    ap.add_argument("--out", default="predictions.pdf")
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    with open(args.ids) as f:
        ids = [l.strip() for l in f if l.strip()]

    model = load_model(args.checkpoint, hidden=args.hidden)
    preds = predict(model, args.npz, args.bio_npz, ids)

    z = np.load(args.npz, allow_pickle=True)

    with PdfPages(args.out) as pdf:
        for tid, p in preds.items():
            entry = z[tid].item()
            cnt = np.asarray(entry["cds"]["avg_count"], np.float32)
            mu = cnt.mean()
            obs = cnt / mu if mu > 0 else cnt
            x = np.arange(len(obs))

            fig, ax = plt.subplots(figsize=(10, 2.8))
            ax.fill_between(x, obs, step="mid", color="#aaaaaa", alpha=0.7, lw=0, label="observed")
            ax.plot(x, p, color="#d62728", lw=0.9, label="RiboPipe")
            ax.set_title(tid, fontsize=8)
            ax.set_xlabel("codon position", fontsize=7)
            ax.set_ylabel("pause score", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=6, loc="upper right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved {len(preds)} panels → {args.out}")


if __name__ == "__main__":
    main()
