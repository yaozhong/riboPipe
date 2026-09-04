"""Interpretability: readable E/P/A motif filters and end-to-end ISM attribution.

- :mod:`ribopipe.interpret.motifs` -- read the first-layer exp-motif CNN filters as E/P/A
  amino-acid sequence logos (local, per-codon-window interpretation).
- :mod:`ribopipe.interpret.ism` -- perturbation in-silico mutagenesis: A-site codon/AA
  attribution through the full non-local network, and its agreement with the motif readout.
- :mod:`ribopipe.interpret.logo` -- minimal matplotlib amino-acid logo rendering.
"""
from __future__ import annotations

from .aa import AA20, CODONS, GC, codon_vec_to_aa
from .motifs import filter_aa_matrices, rank_filters, motif_report
from .ism import asite_codon_attribution, ism_saliency_track, motif_vs_ism

__all__ = [
    "AA20", "CODONS", "GC", "codon_vec_to_aa",
    "filter_aa_matrices", "rank_filters", "motif_report",
    "asite_codon_attribution", "ism_saliency_track", "motif_vs_ism",
]
