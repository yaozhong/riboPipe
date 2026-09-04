"""RiboPipe training loop.

Headline configuration (``ribopipe``, motif-CNN k=7 + BiGRU-128): codon + NT(+/-15) + struct
features, covered-mean-norm log target, unweighted Huber loss, Adam(1e-3), batch 64,
up to 200 epochs with early stopping (patience 20) on the median per-transcript Pearson
of a gene-level validation hold-out.

Two entry points:

* :func:`train_on_ids` — the core loop over explicit train / validation id lists
  (used directly by the gene-level 5CV harness).
* :func:`train`        — deployment wrapper: builds the high-coverage split for one
  sample, carves a validation hold-out, then calls :func:`train_on_ids`.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np
import torch

from .model import BiLSTM, RiboPipeCNN
from .dataset import RiboDataset, collate_fn, build_split
from .predict import predict_dataset, load_items
from .metrics import per_tx_metrics, true_pause
from .losses import huber_mask, wmse_mask, huber_peak_weighted, huber_peak_mse, listwise_rank_loss


def set_seed(s: int = 123):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _val_pearson(model, npz_path, val_ids, dev, *,
                 use_nt, use_struct, struct_npz_path, target) -> float:
    ds = RiboDataset(npz_path, val_ids, target=target,
                     use_nt=use_nt, use_struct=use_struct,
                     struct_npz_path=struct_npz_path)
    pred = predict_dataset(model, ds, dev)
    true = true_pause(load_items(npz_path, val_ids))
    df = per_tx_metrics(pred, true)
    return float(df.pearson.median()) if len(df) > 0 else 0.0


def train_on_ids(
    npz_path: str,
    train_ids: List[str],
    val_ids: Optional[List[str]] = None,
    *,
    struct_npz_path: Optional[str] = None,
    hidden: int = 256,
    backbone: str = "cnn",   # 'cnn' = headline exp-motif CNN + BiGRU-128; 'bilstm' = legacy
    epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    grad_clip: float = 0.0,
    use_nt: bool = True,
    use_struct: bool = True,
    target: str = "meannorm",
    loss_name: str = "huber",
    tau: float = 1.0,
    delta: float = 1.0,
    alpha: float = 2.0,
    rank_lambda: float = 0.0,
    rank_tau: float = 0.05,
    seed: int = 123,
    device: Optional[str] = None,
    verbose: bool = True,
) -> BiLSTM:
    """Train a RiboPipe model (headline motif-CNN + BiGRU-128, or `--backbone bilstm`) over explicit train / val id lists. Returns the model.

    With ``val_ids`` provided, trains up to ``epochs`` and restores the checkpoint with
    the best median per-transcript Pearson on the validation set (early stop after
    ``patience`` epochs without improvement).  Without ``val_ids`` it simply trains for
    ``epochs``.
    """
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    ds = RiboDataset(npz_path, train_ids, target=target,
                     use_nt=use_nt, use_struct=use_struct,
                     struct_npz_path=struct_npz_path)
    if verbose:
        print(f"  [data] n={len(ds)} bio_dim={ds.bio_dim} use_nt={use_nt} "
              f"use_struct={use_struct} target={target}", flush=True)
        print(f"  [loss] {loss_name} tau={tau} delta={delta}", flush=True)

    if backbone == "cnn":
        model = RiboPipeCNN(bio_dim=ds.bio_dim).to(dev)
    else:
        model = BiLSTM(bio_dim=ds.bio_dim, hidden=hidden).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr)
    if verbose:
        print(f"  [model] backbone={backbone} hidden={hidden} params={n_params:,} lr={lr} batch={batch_size}", flush=True)

    lengths = np.array([it[4] for it in ds.items])
    best_r, best_ep, best_state, no_imp = -1.0, 0, None, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        rng = np.random.RandomState(1000 + ep)
        order = np.arange(len(ds))
        rng.shuffle(order)
        chunk = batch_size * 50
        chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
        bidx = []
        for ch in chunks:
            ch = ch[np.argsort(lengths[ch])[::-1]]
            for i in range(0, len(ch), batch_size):
                bidx.append(ch[i:i + batch_size])
        rng.shuffle(bidx)

        for bi in bidx:
            sub = [ds[i] for i in bi]
            idx, ext, tgt, mask, lens, _ = collate_fn(sub)
            idx, ext, tgt, mask, lens = (t.to(dev) for t in (idx, ext, tgt, mask, lens))
            y = model(idx, ext, lens)
            if loss_name == "peakmse":
                loss = huber_peak_mse(y, tgt, mask, tau=tau, delta=delta)
            elif loss_name == "wmse":
                loss = wmse_mask(y, tgt, mask)
            elif loss_name == "huber":
                loss = huber_mask(y, tgt, mask, delta=delta)
            else:  # peak-weighted Huber
                loss = huber_peak_weighted(y, tgt, mask, alpha=alpha, tau=tau)
                if rank_lambda > 0:
                    loss = loss + rank_lambda * listwise_rank_loss(y, tgt, mask, lens, tau=rank_tau)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        if val_ids is not None:
            vr = _val_pearson(model, npz_path, val_ids, dev,
                              use_nt=use_nt, use_struct=use_struct,
                              struct_npz_path=struct_npz_path, target=target)
            if vr > best_r + 1e-4:
                best_r, best_ep, no_imp = vr, ep, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 1
            if verbose and ep % 10 == 0:
                print(f"  ep {ep:3d} val_pearson={vr:.4f} best={best_r:.4f}@ep{best_ep} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if no_imp >= patience:
                if verbose:
                    print(f"  early-stop @ ep{ep}, best val_pearson={best_r:.4f} @ ep{best_ep}", flush=True)
                break
        elif verbose and (ep % 10 == 0 or ep == 1):
            print(f"  ep {ep:3d}/{epochs} ({time.time()-t0:.0f}s)", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        if verbose:
            print(f"  restored best model ep{best_ep}", flush=True)

    model._ribopipe_config = dict(
        backbone=backbone, hidden=hidden, bio_dim=ds.bio_dim, use_nt=use_nt,
        use_struct=use_struct, target=target,
    )
    return model


def save_checkpoint(model: BiLSTM, path: str):
    """Save a checkpoint dict (state + feature config) so predict can restore it."""
    cfg = getattr(model, "_ribopipe_config", {})
    torch.save({"state_dict": model.state_dict(), **cfg}, path)


def train(
    npz_path: str,
    coverage_csv: str,
    sample_col: str,
    *,
    struct_npz_path: Optional[str] = None,
    enst2ensg_path: Optional[str] = None,
    hidden: int = 256,
    backbone: str = "cnn",   # 'cnn' = headline exp-motif CNN + BiGRU-128; 'bilstm' = legacy
    epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_codons: int = 1000,
    use_nt: bool = True,
    use_struct: bool = True,
    loss_name: str = "huber",
    target: str = "meannorm",
    val_frac: float = 0.1,
    device: Optional[str] = None,
    seed: int = 123,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> BiLSTM:
    """Deployment wrapper: build the high-coverage split for one sample and train.

    A validation hold-out (``val_frac``) is carved gene-level when ``enst2ensg_path`` is
    given (no isoform leakage, matching the paper), otherwise transcript-level as a
    fallback.  Saves ``<out_dir>/ribopipe_model.pt`` if ``out_dir`` is set.
    """
    tr_ids, _test_ids = build_split(npz_path, coverage_csv, sample_col,
                                    max_codons=max_codons, seed=seed)

    if enst2ensg_path:
        from .folds import split_val
        tr90, val = split_val(tr_ids, val_frac=val_frac, enst2ensg_path=enst2ensg_path)
    else:
        rng = np.random.RandomState(seed)
        perm = list(tr_ids)
        rng.shuffle(perm)
        cut = int(val_frac * len(perm))
        val, tr90 = perm[:cut], perm[cut:]
    if verbose:
        print(f"[split] T_high train={len(tr90)} val={len(val)}", flush=True)

    model = train_on_ids(
        npz_path, tr90, val_ids=val,
        struct_npz_path=struct_npz_path, hidden=hidden, backbone=backbone, epochs=epochs,
        patience=patience, batch_size=batch_size, lr=lr,
        use_nt=use_nt, use_struct=use_struct,
        loss_name=loss_name, target=target, seed=seed, device=device, verbose=verbose,
    )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ckpt = os.path.join(out_dir, "ribopipe_model.pt")
        save_checkpoint(model, ckpt)
        if verbose:
            print(f"Saved checkpoint -> {ckpt}", flush=True)
    return model
