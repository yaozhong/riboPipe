#!/usr/bin/env python3
r"""Extract the readable E/P/A motif filters from a RiboPipe headline checkpoint.

The first-layer k=7 convolution (``c1``) has 187 input channels; the first 64 are the
codon one-hot. Averaging a filter's codon-channel weights to the amino-acid level gives a
position x amino-acid weight matrix that reads out as a sequence logo over the
E(-2)/P(-1)/A(0) register -- the paper's interpretable first layer.

    python experiments/motif_filters.py --checkpoint checkpoints/ribopipe_headline_TX9_WT.pt --top 3

Prints, for the filters with the largest P-site proline weight, the per-position
amino-acid weight table. No data is required (weights only). Feed the matrices to a logo
library (e.g. logomaker) to draw the panels.
"""
import argparse, numpy as np, torch
from ribopipe.model import RiboPipeCNN, load_cnn_from_paper_checkpoint

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
GC = {  # standard genetic code (sense codons)
 'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L','ATT':'I','ATC':'I',
 'ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V','TCT':'S','TCC':'S','TCA':'S','TCG':'S',
 'CCT':'P','CCC':'P','CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A',
 'GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N',
 'AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGG':'W','CGT':'R',
 'CGC':'R','CGA':'R','CGG':'R','AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def codon_channels_to_aa(w64):
    out, cnt = np.zeros(20), np.zeros(20)
    for ci, c in enumerate(CODONS):
        a = GC.get(c)
        if a in AA20:
            j = AA20.index(a); out[j] += w64[ci]; cnt[j] += 1
    return out / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser(description="Read out the interpretable motif filters.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--top", type=int, default=3, help="filters with the largest P-site proline weight")
    a = ap.parse_args()
    m = load_cnn_from_paper_checkpoint(a.checkpoint, device="cpu")
    w = m.c1.weight.detach().numpy()               # (128, 187, 7)
    k = w.shape[2]; cen = k // 2                    # centre index; A-site = column cen
    pro = [i for i, c in enumerate(CODONS) if GC.get(c) == 'P']
    pro_ps = w[:, pro, cen - 1].mean(axis=1)        # P-site (offset -1) proline weight per filter
    order = np.argsort(-pro_ps)[:a.top]
    reg = {cen - 2: "E", cen - 1: "P", cen: "A"}
    for f in order:
        print(f"\n# filter #{int(f)}  (mean P-site Pro weight = {pro_ps[f]:+.3f})")
        print("pos  " + "  ".join(f"{x:>5s}" for x in AA20))
        for p in range(k):
            aa = codon_channels_to_aa(w[f, :64, p])
            tag = reg.get(p, str(p - cen))
            print(f"{tag:>3s}  " + "  ".join(f"{v:+.2f}" for v in aa))


if __name__ == "__main__":
    main()
