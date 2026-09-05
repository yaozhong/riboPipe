"""RiboPipe inference: predict pause scores for target transcripts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from .model import BiLSTM
from .dataset import RiboDataset, collate_fn


def load_items(npz_path: str, ids: List[str], max_codons: int = 1000):
    """Load transcripts as ``(id, codon_idx, raw_count)`` (for validation scoring)."""
    from .model import seq_to_idx

    z = np.load(npz_path, allow_pickle=True)
    idset = set(ids)
    items = []
    for k in z.files:
        if k not in idset:
            continue
        ent = z[k].item()
        if "cds" not in ent:
            continue
        seq = ent["cds"].get("sequence", "")
        cnt = np.asarray(ent["cds"].get("avg_count", []), np.float32)
        L = len(cnt)
        if len(seq) != L * 3 or L == 0 or L > max_codons or cnt.sum() == 0:
            continue
        items.append((k, seq_to_idx(seq, L), cnt))
    return items


@torch.no_grad()
def predict_dataset(model: BiLSTM, ds: RiboDataset, device, batch_size: int = 16) -> Dict[str, np.ndarray]:
    """Run inference over a pre-built :class:`RiboDataset`. Back-transforms log targets."""
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev).eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    out: Dict[str, np.ndarray] = {}
    for idx_t, ext_t, _, mask_t, lens_t, keys in loader:
        idx_t, ext_t, lens_t = idx_t.to(dev), ext_t.to(dev), lens_t.to(dev)
        y = model(idx_t, ext_t, lens_t)
        for i, key in enumerate(keys):
            m = mask_t[i].bool()
            p = y[i][m].cpu().numpy()
            if ds.target in ("meannorm_log", "covmean0_log"):
                p = np.expm1(p)  # back to (covered-)mean-normalised pause scale
            out[key] = p
    return out


@torch.no_grad()
def predict(
    model: BiLSTM,
    npz_path: str,
    transcript_ids: List[str],
    *,
    use_nt: bool = True,
    use_struct: bool = True,
    struct_npz_path: Optional[str] = None,
    target: str = "meannorm",
    batch_size: int = 16,
    device: Optional[str] = None,
    max_codons: int = 1000,
) -> Dict[str, np.ndarray]:
    """Return predicted mean-normalised pause scores for each transcript.

    The feature flags MUST match the ones the model was trained with (they are stored
    in the checkpoint; :func:`predict_from_checkpoint` restores them automatically).
    """
    if device is None:
        device = next(model.parameters()).device
    ds = RiboDataset(
        npz_path, transcript_ids, target=target,
        use_nt=use_nt, use_struct=use_struct,
        struct_npz_path=struct_npz_path, max_codons=max_codons,
    )
    return predict_dataset(model, ds, device, batch_size=batch_size)


def pearson_per_transcript(predictions: Dict[str, np.ndarray], npz_path: str) -> pd.DataFrame:
    """Per-transcript Pearson between predicted and observed (mean-normalised) pause."""
    from scipy.stats import pearsonr

    z = np.load(npz_path, allow_pickle=True)
    rows = []
    for key, p in predictions.items():
        if key not in z.files:
            continue
        entry = z[key].item()
        cnt = np.asarray(entry.get("cds", {}).get("avg_count", []), np.float32)
        mu = cnt.mean()
        if mu <= 0 or len(cnt) != len(p):
            continue
        obs = cnt / mu
        r = np.nan if (np.std(obs) == 0 or np.std(p) == 0) else pearsonr(obs, p)[0]
        rows.append({"transcript_id": key, "pearson": r, "n_codons": len(p)})
    df = pd.DataFrame(rows).dropna(subset=["pearson"])
    return df.sort_values("pearson", ascending=False).reset_index(drop=True)


def predict_from_checkpoint(
    pt_path: str,
    npz_path: str,
    transcript_ids: List[str],
    hidden: int = 256,
    bio_dim: Optional[int] = None,
    struct_npz_path: Optional[str] = None,
    out_csv: Optional[str] = None,
    device: Optional[str] = None,
    max_codons: int = 1000,
) -> Tuple[Dict[str, np.ndarray], Optional[pd.DataFrame]]:
    """Load a checkpoint (with its feature config) and run prediction in one call."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    ckpt = torch.load(pt_path, map_location=dev)

    # Checkpoints saved by this package are config dicts; bare state_dicts are also accepted.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = ckpt
        state = ckpt["state_dict"]
        hidden = cfg.get("hidden", hidden)
        bio_dim = cfg.get("bio_dim", bio_dim)
        use_nt = cfg.get("use_nt", True)
        use_struct = cfg.get("use_struct", True)
        target = cfg.get("target", "meannorm")
    else:
        state = ckpt
        use_nt = use_struct = True
        target = "meannorm"
        if bio_dim is None:
            bio_dim = 123  # headline default (codon inside model; 120 nt + 3 struct)

    # Detect the backbone from the state dict: the headline model has a first
    # conv `c1.weight` (shape (ch1, 64+bio_dim, k)); the legacy BiLSTM has `l1.*`.
    is_cnn = any(k == "c1.weight" or k.endswith(".c1.weight") for k in state)
    if is_cnn:
        from .model import RiboPipeCNN
        c1 = state.get("c1.weight")
        if c1 is None:  # research checkpoints store the backbone under a `bb.` prefix
            c1 = next(v for k, v in state.items() if k.endswith(".c1.weight"))
        bio_dim = int(c1.shape[1]) - 64          # in-channels = 64 codon one-hot + bio_dim
        # strip any backbone prefix (research checkpoints store under `bb.`)
        state = {k.split("bb.", 1)[-1]: v for k, v in state.items()
                 if not any(k.startswith(p) for p in ("bA.", "bP.", "bE.", "U.", "V.", "Wt.",
                            "wchg", "chg_", "log_"))}
        model = RiboPipeCNN(bio_dim=bio_dim).to(dev)
        model.load_state_dict(state, strict=True)
    else:
        if bio_dim is None:
            bio_dim = 123
        model = BiLSTM(bio_dim=bio_dim, hidden=hidden).to(dev)
        model.load_state_dict(state)
    model.eval()

    preds = predict(
        model, npz_path, transcript_ids,
        use_nt=use_nt, use_struct=use_struct,
        struct_npz_path=struct_npz_path, target=target,
        device=dev, max_codons=max_codons,
    )
    scores = pearson_per_transcript(preds, npz_path)
    if out_csv:
        scores.to_csv(out_csv, index=False)
    return preds, scores
