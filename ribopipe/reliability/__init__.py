"""Low-coverage reliability: the crossover D* and the depth-weighted hybrid estimator.

- :mod:`ribopipe.reliability.readsplit` -- independent read-split primitives (binomial /
  repsplit, covered-mean-norm Pearson, multinomial down-sampling).
- :mod:`ribopipe.reliability.crossover` -- estimate the crossover depth
  ``D* ~ 0.22-0.57 reads/codon`` below which model predictions beat read-derived profiles,
  the model-favoured fraction of expressed genes, and the sweep-free ``r^{-1}(m)`` inversion.
- :mod:`ribopipe.reliability.hybrid` -- the depth-weighted blend of prediction and reads
  (the drop-in low-coverage imputation estimator).
"""
from __future__ import annotations

from .readsplit import meannorm, binomial_split, downsample, pooled_counts_from_npz
from .crossover import (read_split_curves, crossover_dstar, estimate_dstar,
                        invert_dstar, fraction_below, RC_CENT)
from .hybrid import depth_weight, hybrid_profile, impute, SLOPE

__all__ = [
    "meannorm", "binomial_split", "downsample", "pooled_counts_from_npz",
    "read_split_curves", "crossover_dstar", "estimate_dstar", "invert_dstar",
    "fraction_below", "RC_CENT",
    "depth_weight", "hybrid_profile", "impute", "SLOPE",
]
