"""Optional magnitude head (transcript-level ribosome load) for RiboPipe.

The headline **shape** model (``RiboPipeCNN`` / ``BiLSTM``) predicts the *within*-transcript
mean-normalised occupancy pattern; by construction it is blind to a transcript's absolute
ribosome LOAD (the between-transcript magnitude).  This module adds a small, **optional,
stackable** MAGNITUDE head that predicts the transcript-level log mean density

    m_t = log(1 + mean_count)

from 5'UTR + CDS-global **sequence** features only (no observed counts at deployment):

    5'UTR : length, GC, cap-proximal GC, upstream-AUG / uORF counts, Kozak -3/+4 strength,
            5'TOP pyrimidine proxy                                        (10 features)
    CDS   : length, GC, GC at the 3rd codon position                     (3 features)

It is a plain MLP trained with smooth-L1 on the **high-coverage training split only**.
Absolute coverage is then reconstructed by stacking the two independently-trained heads::

    abs_i = expm1(shape_log_i) * expm1(m_t)
          =    pause_i         *   mean_hat

where ``shape_log_i = log1p(pause_i)`` is the shape head's ``log(1+.)`` variant (e.g. the
``covmean0_log`` / ``meannorm_log`` target) and ``mean_hat = expm1(m_t)`` is the predicted
mean density.  The shape benchmark is scale-invariant and completely unaffected -- the two
heads are separate and independently trainable; importing or using this module never
touches the shape model.

No heavy dependencies: features are pure string statistics (cap-proximal GC is a GC
proxy, *not* a folding-energy term), so the module imports with only NumPy + PyTorch --
both core RiboPipe dependencies.  See :func:`utr5_features` for the guarded optional
ViennaRNA cap-folding feature.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

STOPS = {"TAA", "TAG", "TGA"}

UTR5_FEATURES = ["log_utr_len", "gc", "gc_cap30", "gc_last30", "n_uAUG", "n_uORF",
                 "kozak_m3_purine", "kozak_p4_G", "kozak_strong", "top_pyrimidine"]
CDS_GLOBAL_FEATURES = ["log_cds_len", "cds_gc", "cds_gc3"]
ALL_FEATURES = CDS_GLOBAL_FEATURES + UTR5_FEATURES


# ---------------------------------------------------------------------------
# Sequence feature extraction (transcript level)
# ---------------------------------------------------------------------------
def _gc(x: str) -> float:
    return (x.count("G") + x.count("C")) / max(1, len(x))


def utr5_features(utr_seq: str, cds_seq: str, use_vienna: bool = False) -> dict:
    """5'UTR initiation features.  ``use_vienna`` optionally adds a cap-proximal MFE
    feature; if ViennaRNA is not installed it is skipped with a warning (the default
    feature set -- which the paper uses -- needs no folding and no ViennaRNA)."""
    s = (utr_seq or "").upper().replace("U", "T")
    n = len(s)
    f = {"utr_len": n, "log_utr_len": float(np.log1p(n))}
    if n == 0:
        f.update(dict(gc=0.0, gc_cap30=0.0, gc_last30=0.0, n_uAUG=0, n_uORF=0,
                      kozak_m3_purine=0.0, kozak_p4_G=0.0, kozak_strong=0.0,
                      top_pyrimidine=0.0))
    else:
        f["gc"] = _gc(s)
        f["gc_cap30"] = _gc(s[:30])       # cap-proximal GC (a folding proxy, no ViennaRNA)
        f["gc_last30"] = _gc(s[-30:])
        f["n_uAUG"] = s.count("ATG")
        uorf = 0
        for i in range(0, n - 2):
            if s[i:i + 3] == "ATG":
                for j in range(i + 3, n - 2, 3):
                    if s[j:j + 3] in STOPS:
                        uorf += 1
                        break
        f["n_uORF"] = uorf
        cds = (cds_seq or "").upper().replace("U", "T")
        m3 = s[-3] if n >= 3 else "N"
        p4 = cds[3] if len(cds) >= 4 else "N"
        f["kozak_m3_purine"] = 1.0 if m3 in ("A", "G") else 0.0
        f["kozak_p4_G"] = 1.0 if p4 == "G" else 0.0
        f["kozak_strong"] = f["kozak_m3_purine"] * f["kozak_p4_G"]
        f["top_pyrimidine"] = sum(c in "CT" for c in s[:5]) / 5.0

    if use_vienna:
        # Optional, non-default: replace the cap-proximal GC proxy with a real MFE fold
        # energy of the first 30 nt.  Guarded so the module works without ViennaRNA.
        try:
            import RNA  # type: ignore
            if n:
                f["cap_mfe"] = float(RNA.fold(s[:30])[1])
        except Exception:
            warnings.warn("ViennaRNA (RNA) not available; skipping cap-proximal MFE "
                          "feature and keeping the GC proxy (gc_cap30).", RuntimeWarning)
    return f


def cds_global_features(cds_seq: str, n_codons: int) -> dict:
    s = (cds_seq or "").upper().replace("U", "T")
    third = s[2::3]
    return {"log_cds_len": float(np.log1p(n_codons)),
            "cds_gc": _gc(s),
            "cds_gc3": _gc(third)}


def feature_vector(entry: dict) -> Optional[np.ndarray]:
    """:data:`ALL_FEATURES` vector for one transcript NPZ entry, or ``None`` if no CDS."""
    if "cds" not in entry:
        return None
    cseq = entry["cds"].get("sequence", "")
    ncod = len(entry["cds"].get("avg_count", []))
    u = entry.get("5utr")
    useq = u.get("sequence", "") if isinstance(u, dict) else ""
    f = {}
    f.update(cds_global_features(cseq, ncod))
    f.update(utr5_features(useq, cseq))
    return np.array([f[c] for c in ALL_FEATURES], dtype=np.float32)


def build_magnitude_dataset(npzd: Dict[str, dict], ids: Optional[List[str]] = None,
                            min_codons: int = 20) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build ``(X[n, F], y[n], ids)`` for the magnitude head.

    Target ``y = log1p(mean avg_count)`` per transcript.  ``ids`` restricts to a subset
    (e.g. the high-coverage TRAIN split); default = every covered transcript in ``npzd``.
    """
    keys = list(npzd.keys()) if ids is None else [k for k in ids if k in npzd]
    X, y, kept = [], [], []
    for tid in keys:
        ent = npzd[tid]
        if "cds" not in ent:
            continue
        cnt = np.asarray(ent["cds"].get("avg_count", []), np.float32)
        if cnt.size < min_codons or cnt.sum() == 0:
            continue
        v = feature_vector(ent)
        if v is None:
            continue
        X.append(v)
        y.append(float(np.log1p(cnt.mean())))
        kept.append(tid)
    return np.asarray(X, np.float32).reshape(-1, len(ALL_FEATURES)), np.asarray(y, np.float32), kept


# ---------------------------------------------------------------------------
# Magnitude head (small MLP)
# ---------------------------------------------------------------------------
class MagnitudeHead(nn.Module):
    """Transcript-level ribosome-load predictor: sequence features -> log mean density.

    A two-hidden-layer MLP (ReLU, dropout on the first layer).  Optional and stackable:
    it is trained and used independently of the shape model.
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_magnitude_head(X: np.ndarray, y: np.ndarray, *, epochs: int = 300, hidden: int = 64,
                       lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 4096,
                       seed: int = 123, device=None) -> dict:
    """Train a standardised :class:`MagnitudeHead` with smooth-L1 loss.

    ``X`` / ``y`` MUST come from the high-coverage training split only.  Returns a
    checkpoint dict (weights + standardisation stats + feature names) consumable by
    :func:`predict_log_density`.
    """
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    X = np.asarray(X, np.float32).reshape(-1, len(ALL_FEATURES))
    y = np.asarray(y, np.float32)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    net = MagnitudeHead(X.shape[1], hidden=hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
    lossf = nn.SmoothL1Loss()
    n = len(Xt)
    for _ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
    return {"state": {k: v.cpu() for k, v in net.state_dict().items()},
            "mu": mu, "sd": sd, "in_dim": int(X.shape[1]), "hidden": hidden,
            "features": list(ALL_FEATURES), "target": "log1p_mean_count"}


def _rebuild(ckpt: dict, device) -> MagnitudeHead:
    net = MagnitudeHead(ckpt["in_dim"], hidden=ckpt["hidden"]).to(device)
    net.load_state_dict(ckpt["state"])
    net.eval()
    return net


@torch.no_grad()
def predict_log_density(ckpt: dict, X: np.ndarray, device=None) -> np.ndarray:
    """Predicted log mean density ``m_t`` per transcript (the raw head output)."""
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _rebuild(ckpt, dev)
    Xs = ((np.asarray(X, np.float32).reshape(-1, ckpt["in_dim"]) - ckpt["mu"]) / ckpt["sd"]).astype(np.float32)
    return net(torch.tensor(Xs, device=dev)).cpu().numpy()


def predict_mean_density(ckpt: dict, X: np.ndarray, device=None) -> np.ndarray:
    """Predicted mean ribosome density ``expm1(m_t)`` (inverse of the log1p target)."""
    return np.expm1(predict_log_density(ckpt, X, device=device))


def reconstruct_absolute(shape_log: np.ndarray, m_t: float) -> np.ndarray:
    """Stack the two heads into an absolute per-codon coverage profile.

    ``shape_log`` is the shape head's ``log(1+pause)`` output for one transcript (per
    codon); ``m_t`` is that transcript's predicted log mean density.  Returns
    ``expm1(shape_log) * expm1(m_t)``.
    """
    return np.expm1(np.asarray(shape_log, np.float64)) * float(np.expm1(m_t))


def save_magnitude_head(ckpt: dict, path: str) -> None:
    torch.save(ckpt, path)


def load_magnitude_head(path: str, device=None) -> dict:
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(path, map_location=dev, weights_only=False)
