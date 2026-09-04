"""Independent read-split primitives for the low-coverage reliability analysis.

The whole metric chain is **covered-mean-normalised Pearson** (divide each profile by its
own mean; never log1p, never raw counts). Reads are split into two disjoint halves so the
scored ("input") reads and the "reference" reads share nothing:

* binomial mode: ``ref = Binomial(n, 0.5)``, ``input = n - ref`` on the pooled integer
  counts (``n = round(avg_count * 2)`` from the rep-averaged NPZ), averaged over seeds;
* repsplit mode: input = replicate 1, reference = replicate 2 (symmetrised) where genuine
  per-replicate counts exist.
"""
from __future__ import annotations

import numpy as np

MIN_CODONS = 20     # transcripts shorter than this are skipped
MIN_NONZERO = 5     # the reference half must have at least this many nonzero codons


def meannorm(c) -> np.ndarray:
    """Divide a profile by its own mean (covered-mean normalisation)."""
    c = np.asarray(c, dtype=float)
    m = c.mean()
    return c / m if m > 0 else c


def pear(a, b) -> float:
    """Pearson r with the analysis guards (>= MIN_CODONS codons, nonzero variance)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape or a.size < MIN_CODONS or a.std() == 0 or b.std() == 0:
        return np.nan
    from scipy.stats import pearsonr
    return float(pearsonr(a, b)[0])


def binomial_split(counts, rng: np.random.RandomState):
    """Split pooled integer counts into disjoint (input, reference) halves."""
    n = np.asarray(counts).astype(np.int64)
    ref = rng.binomial(n, 0.5)
    return n - ref, ref


def downsample(pool, D: int, rng: np.random.RandomState):
    """Multinomially thin ``pool`` to a total of ``D`` reads (None if too shallow)."""
    pool = np.asarray(pool, float)
    s = int(pool.sum())
    if s < D or s == 0:
        return None
    return rng.multinomial(D, pool / s).astype(float)


def pooled_counts_from_npz(npz_path):
    """``{transcript_id: int per-codon counts}`` = ``round(cds.avg_count * 2)`` per NPZ entry."""
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in z.files:
        e = z[k].item()
        if "cds" not in e:
            continue
        a = np.asarray(e["cds"].get("avg_count", []), float)
        if a.size:
            out[k] = np.round(a * 2.0).astype(np.int64)
    return out
