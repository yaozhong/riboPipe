"""Dataset and collation utilities for RiboPipe.

The per-codon input concatenates, in order:

    codon one-hot   64   (built on-device from a fixed embedding; see model.py)
    bio features     6   (tAI, codon frequency, wobble, AA physico-chemical)   [use_bio]
    NT one-hot     120   (+/-15 nt around the A-site, 30 positions x 4)         [use_nt]
    struct MFE       3   (positions 3i-17, 3i-16, 3i-15 relative to A-site)     [use_struct]
    ----------------------
    total          187   for the headline ``ribopipe`` config (use_bio=False: codon+NT+struct)

The codon one-hot is added inside the model, so the dataset returns the remaining
``bio_dim = 6/126/129/...`` feature block.  Toggling the flags reproduces the paper's
feature-ablation rows (e.g. ``use_nt=False, use_struct=False`` = BiLSTM-base, 6 dims).
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import seq_to_idx, PAD_IDX

NT_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def nt_onehot(cds_nt: str, L: int, win: int = 15) -> np.ndarray:
    """NT one-hot for all L codons: a +/-``win`` nt window around each A-site.

    Returns ``(L, 2*win*4)`` float32 (120 dims for win=15).
    """
    W = 2 * win  # 30-nt window around each A-site
    seq = np.array([NT_MAP.get(c, -1) for c in cds_nt], dtype=np.int8)
    out = np.zeros((L, W * 4), dtype=np.float32)
    for j in range(W):
        offset = j - win  # -15 ... +14 relative to A-site first nt
        ps = np.arange(L, dtype=np.int32) * 3 + offset
        valid = (ps >= 0) & (ps < len(seq))
        b = seq[ps[valid]]
        ok = b >= 0
        i_ok = np.where(valid)[0][ok]
        b_ok = b[ok]
        out[i_ok, j * 4 + b_ok] = 1.0
    return out


def struct_features(mfe_arr: np.ndarray, L: int, rel_idxs=(-17, -16, -15)) -> np.ndarray:
    """Struct MFE at 3 positions relative to each A-site. Returns ``(L, 3)`` float32."""
    out = np.zeros((L, 3), dtype=np.float32)
    mfe_len = len(mfe_arr)
    for k, rel in enumerate(rel_idxs):
        ps = np.arange(L, dtype=np.int32) * 3 + rel
        valid = (ps >= 0) & (ps < mfe_len)
        out[valid, k] = mfe_arr[ps[valid]]
    return out


def load_struct_cache(struct_npz_path: Optional[str]) -> dict:
    """Load a struct MFE cache (transcript_id -> per-nucleotide MFE array)."""
    if not struct_npz_path:
        return {}
    if os.path.isfile(struct_npz_path):
        c = np.load(struct_npz_path, allow_pickle=True)
        return {k: c[k] for k in c.files}
    print(f"  [WARN] struct cache not found at {struct_npz_path}; struct features zeroed", flush=True)
    return {}


class RiboDataset(Dataset):
    """Codon-indexed, feature-augmented Ribo-seq dataset.

    Parameters
    ----------
    npz_path : str
        Per-transcript NPZ (each key -> dict with ``cds.sequence`` and ``cds.avg_count``).
    bio_npz_path : str
        Biological-features NPZ (per transcript, shape (L, 6)).
    transcript_ids : list[str]
        Transcripts to include; invalid/missing ones are silently dropped.
    target : {"meannorm", "meannorm_log", "minmax"}
        Regression target space. ``meannorm`` (count/mean) is the paper default and
        preserves peak amplitude.
    use_nt, use_struct, use_bio : bool
        Feature toggles (headline = all True). ``use_nt`` adds the +/-15 nt one-hot;
        ``use_struct`` adds the 3 struct-MFE dims (requires ``struct_npz_path``).
    struct_npz_path : str or None
        Path to the struct MFE cache (from ``ribopipe struct``). Required if
        ``use_struct``.
    max_codons : int
        Exclude transcripts longer than this (paper default: 1000).
    """

    VALID_TARGETS = ("meannorm", "meannorm_log", "minmax")

    def __init__(
        self,
        npz_path: str,
        bio_npz_path: str,
        transcript_ids: List[str],
        target: str = "meannorm",
        use_nt: bool = True,
        use_struct: bool = True,
        use_bio: bool = True,
        struct_npz_path: Optional[str] = None,
        max_codons: int = 1000,
    ):
        if target not in self.VALID_TARGETS:
            raise ValueError(f"target must be one of {self.VALID_TARGETS}")
        self.target = target
        self.items: List[Tuple] = []  # (key, idx, ext, raw_count, L)

        z = np.load(npz_path, allow_pickle=True)
        b = np.load(bio_npz_path, allow_pickle=True)
        b_keys = set(b.files)
        struct_cache = load_struct_cache(struct_npz_path) if use_struct else {}

        for key in transcript_ids:
            if key not in z.files:
                continue
            entry = z[key].item()
            if "cds" not in entry:
                continue
            seq = entry["cds"].get("sequence", "")
            cnt = np.asarray(entry["cds"].get("avg_count", []), dtype=np.float32)
            L = len(cnt)
            if len(seq) != L * 3 or L == 0 or L > max_codons or cnt.sum() == 0:
                continue

            idx = seq_to_idx(seq, L)

            parts = []
            if use_bio:
                bio = b[key].astype(np.float32) if key in b_keys else np.zeros((L, 6), np.float32)
                if bio.shape[0] != L:
                    bio = np.zeros((L, 6), np.float32)
                parts.append(bio)
            if use_nt:
                parts.append(nt_onehot(seq, L))
            if use_struct:
                mfe_arr = struct_cache.get(key)
                if mfe_arr is not None:
                    parts.append(struct_features(mfe_arr, L))
                else:
                    parts.append(np.zeros((L, 3), np.float32))
            ext = np.concatenate(parts, axis=1) if parts else np.zeros((L, 0), np.float32)

            self.items.append((key, idx, ext, cnt, L))

        if self.items:
            self.bio_dim = self.items[0][2].shape[1]
        else:
            self.bio_dim = (6 if use_bio else 0) + (120 if use_nt else 0) + (3 if use_struct else 0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        key, idx, ext, cnt, L = self.items[i]
        c = cnt.astype(np.float32)
        if self.target == "meannorm":
            m = c.mean()
            tgt = (c / m).astype(np.float32) if m > 0 else np.zeros_like(c)
        elif self.target == "meannorm_log":
            m = c.mean()
            ps = c / m if m > 0 else c
            tgt = np.log1p(ps).astype(np.float32)
        else:  # minmax
            mn, mx = c.min(), c.max()
            tgt = (c - mn) / (mx - mn) if mx > mn else np.zeros_like(c)
        return (
            key,
            torch.from_numpy(idx),
            torch.from_numpy(ext),
            torch.from_numpy(tgt),
            L,
        )


def collate_fn(batch):
    """Sort by length (descending) and pad to the batch maximum."""
    batch = sorted(batch, key=lambda x: x[4], reverse=True)
    keys, idxs, exts, tgts, Ls = zip(*batch)
    B, L_max, D = len(batch), Ls[0], exts[0].shape[1]

    idx_t = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    ext_t = torch.zeros(B, L_max, D)
    tgt_t = torch.zeros(B, L_max)
    mask_t = torch.zeros(B, L_max, dtype=torch.bool)

    for i, (ii, ef, tg, l) in enumerate(zip(idxs, exts, tgts, Ls)):
        ii = ii.clone()
        ii[ii < 0] = PAD_IDX
        idx_t[i, :l] = ii
        ext_t[i, :l] = ef
        tgt_t[i, :l] = tg
        mask_t[i, :l] = True

    return idx_t, ext_t, tgt_t, mask_t, torch.tensor(Ls), list(keys)


def build_split(
    npz_path: str,
    coverage_csv: str,
    sample_col: str,
    train_frac: float = 0.8,
    top_frac: float = 0.25,
    max_codons: int = 1000,
    seed: int = 123,
) -> Tuple[List[str], List[str]]:
    """Return ``(train_ids, test_ids)`` over the high-coverage T_high transcripts.

    T_high = transcripts whose coverage is in the top ``top_frac`` (default top-25%);
    they are split ``train_frac`` / (1 - ``train_frac``) at the transcript level.  For
    the leak-free gene-level split used in the paper, use :mod:`ribopipe.folds` with an
    ENST->ENSG map instead.
    """
    import pandas as pd

    z = np.load(npz_path, allow_pickle=True)
    mat = pd.read_csv(coverage_csv, index_col=0)
    cov_series = mat[sample_col]
    threshold = float(cov_series.quantile(1.0 - top_frac))

    high: List[str] = []
    for key in z.files:
        entry = z[key].item()
        if "cds" not in entry:
            continue
        cnt = np.asarray(entry["cds"].get("avg_count", []), np.float32)
        L = len(cnt)
        if L == 0 or L > max_codons or cnt.sum() == 0:
            continue
        val = float(cov_series[key]) if key in cov_series.index else -np.inf
        if val >= threshold:
            high.append(key)

    rng = np.random.RandomState(seed)
    rng.shuffle(high)
    n_train = int(len(high) * train_frac)
    return high[:n_train], high[n_train:]
