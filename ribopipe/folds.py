"""Gene-level partitioning helpers (leak-free evaluation).

All isoforms of a gene are kept in the same partition, so every held-out set has
0% isoform overlap with training.  This is the honest protocol used for the paper's
headline results; transcript-level splits (isoforms of one gene spread across
folds) leak and are *not* used here.

* :func:`gene_folds`   — partition genes into N round-robin folds (5CV headline).
* :func:`split_val`    — carve a gene-level validation hold-out from a train set
                         (for early stopping).

Both need an Ensembl transcript->gene map (``ENST -> ENSG``); pass its path via
``enst2ensg_path`` or a preloaded dict via ``e2g``.  Unmapped transcripts always go
to TRAIN, never to a held-out fold.
"""
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict

import numpy as np


def load_enst2ensg(path: str) -> dict:
    """Load an ENST->ENSG map from a .json or .json.gz file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def _resolve_map(e2g, enst2ensg_path):
    if e2g is not None:
        return e2g
    if enst2ensg_path is None:
        raise ValueError("provide either e2g (dict) or enst2ensg_path (file)")
    return load_enst2ensg(enst2ensg_path)


def gene_folds(all_ids, e2g=None, n_folds: int = 5, seed: int = 0, enst2ensg_path: str = None):
    """Partition genes (not transcripts) into ``n_folds`` folds after a fixed shuffle.

    Returns ``(folds, n_genes, n_unmapped)`` where ``folds`` is a list of
    ``(train_ids, test_ids)`` tuples.  All isoforms of a gene share a fold; unmapped
    transcripts always join TRAIN.
    """
    e2g = _resolve_map(e2g, enst2ensg_path)
    gene2tx = defaultdict(list)
    unmapped = []
    for k in all_ids:
        ensg = e2g.get(k.split(".")[0])
        (gene2tx[ensg].append(k) if ensg else unmapped.append(k))
    genes = sorted(gene2tx.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(genes)
    gene_fold = {g: i % n_folds for i, g in enumerate(genes)}
    folds = []
    for f in range(n_folds):
        te = [tx for g in genes if gene_fold[g] == f for tx in gene2tx[g]]
        tr = [tx for g in genes if gene_fold[g] != f for tx in gene2tx[g]] + unmapped
        folds.append((tr, te))
    return folds, len(genes), len(unmapped)


def split_val(tr_ids, e2g=None, val_frac: float = 0.1, seed: int = 42, enst2ensg_path: str = None):
    """Carve ``val_frac`` of GENES from ``tr_ids`` (gene-level, no isoform leakage).

    Returns ``(train90_ids, val_ids)``.
    """
    e2g = _resolve_map(e2g, enst2ensg_path)
    gene2tx = defaultdict(list)
    unmapped = []
    for k in tr_ids:
        ensg = e2g.get(k.split(".")[0])
        if ensg:
            gene2tx[ensg].append(k)
        else:
            unmapped.append(k)
    genes = sorted(gene2tx.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(genes)
    cut = int(val_frac * len(genes))
    tr90 = [k for g in genes[cut:] for k in gene2tx[g]] + unmapped
    val = [k for g in genes[:cut] for k in gene2tx[g]]
    return tr90, val
