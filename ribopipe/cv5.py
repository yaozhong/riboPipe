"""Gene-level 5-fold cross-validation harness (paper headline table).

Genes are partitioned into ``n_folds`` folds (seed 0); every isoform of a gene stays in
one fold, so each held-out fold has 0% isoform overlap with training -- the same
leak-free protocol as the 80/20 gene split.  The *same* folds are used for every method.
Per fold we report the median per-transcript Pearson / Spearman / top-5% peak recall /
Jaccard; the summary is their mean +/- SD across folds.

Supported ``methods`` (self-contained): ``codon_mean``, ``tricodon``, ``ridge``,
``bilstm_base`` (codon+bio) and ``ribopipe_nt_struct_h256`` (codon+bio+NT+struct, the
headline).  Published neural baselines are external code; run them separately under the
same folds (see the paper supplement).
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import numpy as np

from .folds import gene_folds, split_val
from .predict import load_items, predict_dataset
from .metrics import per_tx_medians, true_pause
from .dataset import RiboDataset
from .train import train_on_ids


def _our_method(npz_path, bio_npz_path, tr_ids, te_ids, *, use_nt, use_struct,
                struct_npz_path, enst2ensg_path, epochs, patience, loss_name,
                hidden, device, e2g, backbone="cnn"):
    tr90, val = split_val(tr_ids, e2g=e2g, enst2ensg_path=enst2ensg_path)
    model = train_on_ids(
        npz_path, bio_npz_path, tr90, val_ids=val,
        struct_npz_path=struct_npz_path, hidden=hidden, backbone=backbone, epochs=epochs, patience=patience,
        use_nt=use_nt, use_struct=use_struct, use_bio=False,
        loss_name=loss_name, device=device, verbose=False,
    )
    ds = RiboDataset(npz_path, bio_npz_path, te_ids, target="meannorm",
                     use_nt=use_nt, use_struct=use_struct, use_bio=False,
                     struct_npz_path=struct_npz_path)
    dev = device or ("cuda" if _has_cuda() else "cpu")
    return predict_dataset(model, ds, dev)


def _has_cuda():
    import torch
    return torch.cuda.is_available()


def run_cv5(
    npz_path: str,
    bio_npz_path: str,
    all_ids: List[str],
    enst2ensg_path: str,
    *,
    methods: Optional[List[str]] = None,
    struct_npz_path: Optional[str] = None,
    n_folds: int = 5,
    epochs: int = 200,
    patience: int = 20,
    loss_name: str = "huber",
    hidden: int = 256,
    backbone: str = "cnn",
    device: Optional[str] = None,
    out_json: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Run gene-level ``n_folds``-fold CV; return the summary dict (and optionally write JSON)."""
    if methods is None:
        methods = ["ribopipe_nt_struct_h256"]
    from .baselines import codon_mean_lookup, ridge_window

    e2g = json.load(open(enst2ensg_path)) if enst2ensg_path.endswith(".json") else None
    if e2g is None:
        from .folds import load_enst2ensg
        e2g = load_enst2ensg(enst2ensg_path)

    folds, n_genes, n_unmapped = gene_folds(all_ids, e2g=e2g, n_folds=n_folds, seed=0)
    if verbose:
        print(f"{len(all_ids)} tx, {n_genes} genes, {n_unmapped} unmapped -> {n_folds} gene-folds "
              f"(test sizes: {[len(te) for _, te in folds]})", flush=True)

    lookup = {"codon_mean", "tricodon", "ridge"}
    need_tr_items = bool(lookup & set(methods))
    results = {m: [] for m in methods}

    for f, (tr_ids, te_ids) in enumerate(folds):
        te_items = load_items(npz_path, te_ids)
        true = true_pause(te_items)
        tr_items = load_items(npz_path, tr_ids) if need_tr_items else None
        if verbose:
            print(f"\n== fold {f+1}/{n_folds}: train={len(tr_ids)} test={len(te_ids)} ==", flush=True)
        for m in methods:
            t0 = time.time()
            if m == "codon_mean":
                pred = codon_mean_lookup(tr_items, te_items, order=1)
            elif m == "tricodon":
                pred = codon_mean_lookup(tr_items, te_items, order=3)
            elif m == "ridge":
                pred = ridge_window(tr_items, te_items)
            elif m == "bilstm_base":
                pred = _our_method(npz_path, bio_npz_path, tr_ids, te_ids,
                                   use_nt=False, use_struct=False, struct_npz_path=None,
                                   enst2ensg_path=enst2ensg_path, epochs=epochs, patience=patience,
                                   loss_name=loss_name, hidden=hidden, device=device, e2g=e2g, backbone=backbone)
            elif m == "ribopipe_nt_struct_h256":
                pred = _our_method(npz_path, bio_npz_path, tr_ids, te_ids,
                                   use_nt=True, use_struct=True, struct_npz_path=struct_npz_path,
                                   enst2ensg_path=enst2ensg_path, epochs=epochs, patience=patience,
                                   loss_name=loss_name, hidden=hidden, device=device, e2g=e2g, backbone=backbone)
            else:
                raise SystemExit(f"unsupported method {m}")
            P, S, REC, JAC, n = per_tx_medians(pred, true)
            results[m].append((P, S, REC, JAC, n))
            if verbose:
                print(f"   [{m}] fold{f+1}: P={P:.4f} S={S:.4f} Rec={REC:.4f} Jac={JAC:.4f} "
                      f"n={n} ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for m in methods:
        arr = np.array([[r[0], r[1], r[2], r[3]] for r in results[m]], float)
        ns = [r[4] for r in results[m]]
        mean = arr.mean(0)
        std = arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(4)
        summary[m] = {
            "P_mean": round(float(mean[0]), 4), "P_std": round(float(std[0]), 4),
            "S_mean": round(float(mean[1]), 4), "S_std": round(float(std[1]), 4),
            "Rec_mean": round(float(mean[2]), 4), "Rec_std": round(float(std[2]), 4),
            "Jac_mean": round(float(mean[3]), 4), "Jac_std": round(float(std[3]), 4),
            "n_per_fold": ns,
            "folds": [list(map(float, r[:4])) for r in results[m]],
        }
        if verbose:
            print(f"  {m:28s} P={mean[0]:.4f}+/-{std[0]:.4f}  S={mean[1]:.4f}+/-{std[1]:.4f}  "
                  f"Rec={mean[2]:.4f}+/-{std[2]:.4f}  Jac={mean[3]:.4f}+/-{std[3]:.4f}", flush=True)

    blob = {"n_folds": n_folds, "n_genes": n_genes, "n_unmapped": n_unmapped, "results": summary}
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        json.dump(blob, open(out_json, "w"), indent=2)
        if verbose:
            print(f"\nsaved {out_json}", flush=True)
    return blob
