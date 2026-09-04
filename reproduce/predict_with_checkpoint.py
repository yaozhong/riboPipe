#!/usr/bin/env python3
r"""Reproduce the paper's per-transcript accuracy from a released headline checkpoint,
using only the installed ``ribopipe`` package (no training).

Example
-------
    python reproduce/predict_with_checkpoint.py \
        --checkpoint checkpoints/ribopipe_headline_TX9_WT.pt \
        --npz    TX9_WT.npz \
        --struct-npz struct_cache/TX9_WT_struct.npz \
        --ids    test_ids.txt

``--ids`` is a text file with one held-out transcript id per line (the gene-longest
test set). It prints the median per-transcript Pearson / Spearman / top-5% peak
recall / peak Jaccard over those transcripts.

The four headline checkpoints ship in ``checkpoints/``; they are the exact weights
behind the paper's benchmark (motif-CNN + BiGRU-128, ~0.35M params, covered-mean-norm
log target). On the held-out gene-longest test sets they reproduce the paper's Table:
TX9_WT 0.598, GSE233886_WT 0.690, GSE133393_WT 0.529, PRJNA_iPS 0.518 (Pearson).
"""
import argparse
from ribopipe.predict import predict_from_checkpoint, load_items
from ribopipe.metrics import true_pause, per_tx_medians


def main():
    ap = argparse.ArgumentParser(description="Score a headline checkpoint on held-out transcripts.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--npz", required=True, help="per-transcript NPZ (ribopipe preprocess)")
    ap.add_argument("--struct-npz", default=None, help="ViennaRNA MFE cache (ribopipe struct)")
    ap.add_argument("--ids", required=True, help="text file, one held-out transcript id per line")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    ids = [x.strip() for x in open(a.ids) if x.strip()]
    preds, _ = predict_from_checkpoint(
        a.checkpoint, a.npz, ids,
        struct_npz_path=a.struct_npz, device=a.device)
    true = true_pause(load_items(a.npz, ids))
    P, S, REC, JAC, n = per_tx_medians(preds, true)
    print(f"n_scored_transcripts       {n}")
    print(f"per-transcript Pearson     {P:.4f}")
    print(f"per-transcript Spearman    {S:.4f}")
    print(f"top-5% peak recall         {REC:.4f}")
    print(f"peak-region Jaccard        {JAC:.4f}")


if __name__ == "__main__":
    main()
