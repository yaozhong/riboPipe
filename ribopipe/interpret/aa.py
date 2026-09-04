"""Shared codon / amino-acid constants and helpers for interpretability."""
from __future__ import annotations

import numpy as np

# 64 sense+stop codons in the model's channel order (A,C,G,T x3), matching model.CODONS.
CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]

# Standard genetic code (DNA codons). Stops map to '*'.
GC = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a: i for i, a in enumerate(AA20)}

# codon index -> amino-acid index (-1 for stops), for fast aggregation
CODON_AA_IDX = np.array([AA_INDEX.get(GC[c], -1) for c in CODONS], dtype=np.int64)


def codon_vec_to_aa(w64: np.ndarray) -> np.ndarray:
    """Average a 64-codon weight vector to a 20-dim amino-acid vector (synonymous mean)."""
    out = np.zeros(20, dtype=np.float64)
    cnt = np.zeros(20, dtype=np.float64)
    for ci in range(64):
        j = CODON_AA_IDX[ci]
        if j >= 0:
            out[j] += w64[ci]
            cnt[j] += 1
    return out / np.maximum(cnt, 1)
