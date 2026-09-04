"""End-to-end in-silico mutagenesis (ISM) attribution for the headline CNN.

Unlike the readable first-layer motif filters (:mod:`ribopipe.interpret.motifs`), ISM
explains occupancy through the *whole* non-local network. For each codon position we
substitute the A-site codon-identity input with each of the 64 codons, re-run the full
forward pass, and read the predicted occupancy at that position. Centring the 64 responses
(subtracting their mean) gives the marginal A-site effect of each codon; averaging over all
positions and transcripts yields a per-codon (and, by synonymous mean, per-amino-acid)
A-site attribution -- the model-derived counterpart of an empirical A-site dwell profile.

Note: this perturbs the codon-identity channel (the model also sees the +/-15 nt one-hot,
held fixed), so it isolates the codon-identity contribution as the trained network uses it.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from .aa import codon_vec_to_aa


def _dataset(npz_path, ids, struct_npz_path, target, max_len):
    from ..dataset import RiboDataset
    return RiboDataset(npz_path, ids, target=target,
                       use_nt=True, use_struct=bool(struct_npz_path),
                       struct_npz_path=struct_npz_path, max_codons=max_len)


def asite_codon_attribution(
    model,
    npz_path: str,
    ids: List[str],
    struct_npz_path: Optional[str] = None,
    max_transcripts: int = 100,
    max_len: int = 200,
    target: str = "meannorm",
    device: str = "cpu",
) -> dict:
    """Per-codon / per-amino-acid A-site ISM attribution.

    Returns ``{"per_codon": (64,), "per_aa": (20,), "n_positions": int}``.  Cost is
    O(n_positions) forward passes (batch 64); cap ``max_transcripts`` / ``max_len`` for speed.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    ds = _dataset(npz_path, ids, struct_npz_path, target, max_len)
    effect = np.zeros(64, dtype=np.float64)
    n = 0
    codon_grid = torch.arange(64, dtype=torch.long, device=dev)
    with torch.no_grad():
        for i in range(min(len(ds), max_transcripts)):
            _, idx, ext, _, L = ds[i]
            idx = idx.to(dev)
            ext = ext.to(dev)
            lengths = torch.full((64,), L, dtype=torch.long)
            ext_b = ext.unsqueeze(0).expand(64, -1, -1)
            for p in range(L):
                var = idx.unsqueeze(0).repeat(64, 1)
                var[:, p] = codon_grid
                pred = model(var, ext_b, lengths)          # (64, L)
                col = pred[:, p].detach().cpu().numpy()     # response to each codon at p
                effect += col - col.mean()
                n += 1
    per_codon = effect / max(n, 1)
    return {"per_codon": per_codon, "per_aa": codon_vec_to_aa(per_codon), "n_positions": n}


def ism_saliency_track(model, idx: torch.Tensor, ext: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """Per-position ISM saliency for one transcript: how much the A-site codon matters.

    For each position, the spread (max-min) of the 64-codon-substitution responses at that
    position -- a per-codon 'context sensitivity' track (length L).
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    idx = idx.to(dev)
    ext = ext.to(dev)
    L = idx.shape[0]
    codon_grid = torch.arange(64, dtype=torch.long, device=dev)
    ext_b = ext.unsqueeze(0).expand(64, -1, -1)
    lengths = torch.full((64,), L, dtype=torch.long)
    out = np.zeros(L, dtype=np.float64)
    with torch.no_grad():
        for p in range(L):
            var = idx.unsqueeze(0).repeat(64, 1)
            var[:, p] = codon_grid
            col = model(var, ext_b, lengths)[:, p].detach().cpu().numpy()
            out[p] = col.max() - col.min()
    return out


def motif_vs_ism(motif_aa_asite: np.ndarray, ism_aa: np.ndarray) -> dict:
    """Agreement between the local motif A-site AA weights and the ISM A-site AA attribution.

    Both are length-20 vectors. Returns Pearson r and Spearman rho; low agreement flags
    occupancy the local codon window cannot explain (context-driven).
    """
    from scipy.stats import pearsonr, spearmanr
    a = np.asarray(motif_aa_asite, float)
    b = np.asarray(ism_aa, float)
    return {"pearson_r": float(pearsonr(a, b)[0]), "spearman_rho": float(spearmanr(a, b)[0])}
