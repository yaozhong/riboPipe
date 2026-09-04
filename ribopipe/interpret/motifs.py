"""Readable E/P/A motif filters from a headline RiboPipe checkpoint.

The first-layer k=7 exp-motif convolution (``c1``) has ``64 + feat`` input channels; the
first 64 are the codon one-hot. Averaging a filter's 64 codon-channel weights to the
amino-acid level (synonymous mean) gives, per filter and per kernel position, a 20-dim
amino-acid weight vector -- a sequence-logo matrix over the E(-2)/P(-1)/A(0) register.
This is the paper's interpretable first layer: filters read out directly as E/P/A
elongation motifs (e.g. P-site proline, acidic/aromatic A-site residues).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from .aa import AA20, AA_INDEX, codon_vec_to_aa


def _load(model_or_ckpt, device: str = "cpu"):
    if isinstance(model_or_ckpt, str):
        from ..model import load_cnn_from_paper_checkpoint
        return load_cnn_from_paper_checkpoint(model_or_ckpt, device=device)
    return model_or_ckpt


def filter_aa_matrices(model_or_ckpt, device: str = "cpu") -> Tuple[np.ndarray, int]:
    """Per-filter amino-acid weight matrices from the first conv layer.

    Returns ``(mats, center)`` where ``mats`` has shape ``(n_filters, k, 20)`` -- the
    synonymous-mean amino-acid weight at each kernel position -- and ``center = k // 2``
    is the A-site column (P = center-1, E = center-2).
    """
    model = _load(model_or_ckpt, device)
    w = model.c1.weight.detach().cpu().numpy()      # (n_filters, 64+feat, k)
    n_filters, _, k = w.shape
    mats = np.zeros((n_filters, k, 20), dtype=np.float64)
    for f in range(n_filters):
        for p in range(k):
            mats[f, p] = codon_vec_to_aa(w[f, :64, p])
    return mats, k // 2


def rank_filters(mats: np.ndarray, center: int, aa: str = "P", register: str = "P") -> np.ndarray:
    """Order filters by the weight of amino acid ``aa`` at register E/P/A (default P-site Pro)."""
    off = {"E": center - 2, "P": center - 1, "A": center}[register]
    return np.argsort(-mats[:, off, AA_INDEX[aa]])


def motif_report(
    model_or_ckpt,
    top: int = 3,
    rank_aa: str = "P",
    rank_register: str = "P",
    out_csv: Optional[str] = None,
    out_png: Optional[str] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Extract, rank and (optionally) render the top interpretable motif filters.

    Ranks filters by ``rank_aa`` at ``rank_register`` (default P-site proline), writes a
    tidy CSV of the per-position amino-acid weights, and -- if ``out_png`` -- draws an
    E/P/A sequence logo per top filter. Returns the indices of the selected filters.
    """
    mats, center = filter_aa_matrices(model_or_ckpt, device)
    order = rank_filters(mats, center, aa=rank_aa, register=rank_register)[:top]
    k = mats.shape[1]
    reg = {center - 2: "E", center - 1: "P", center: "A"}
    positions = list(range(k))

    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["filter", "position", "register"] + AA20)
            for f in order:
                for p in positions:
                    tag = reg.get(p, str(p - center))
                    wr.writerow([int(f), p, tag] + [f"{mats[f, p, j]:.4f}" for j in range(20)])
        print(f"[motifs] wrote {out_csv}: top-{top} filters x {k} positions x 20 AA")

    if out_png:
        from .logo import draw_logo_grid
        # letter heights read straight from the (signed) AA weights, centred at 0
        draw_logo_grid(
            [mats[f] for f in order],
            titles=[f"filter {int(f)}" for f in order],
            center=center, out_png=out_png,
            ylabel="motif weight",
        )
        print(f"[motifs] wrote {out_png}")

    return order
