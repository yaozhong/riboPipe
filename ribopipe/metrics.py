"""Per-transcript evaluation metrics (paper-consistent).

Ground truth and prediction are per-transcript mean-normalised pause profiles
(count / transcript-mean).  The metrics below match the paper exactly:

* per-transcript Pearson / Spearman correlation,
* top-5% peak **recall@k**: the true peak set is the top-5% observed positions
  (t >= q95); recall@k takes the top-|peak| predicted positions, so
  ``recall == precision`` (equal set sizes),
* top-5% peak **Jaccard**: |intersection| / |union| of the two peak sets.

:func:`per_tx_metrics` returns the full per-transcript table (used by the 80/20
benchmark); :func:`per_tx_medians` returns the fold-level medians used by the
gene-level 5-fold cross-validation (Table: gene-level 5CV benchmark).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def per_tx_metrics(pred, true) -> pd.DataFrame:
    """Full per-transcript metric table (one row per scored transcript)."""
    rows = []
    for k in true:
        if k not in pred:
            continue
        t = np.asarray(true[k], np.float64)
        p = np.asarray(pred[k], np.float64)
        if len(t) != len(p) or len(t) < 5 or np.std(t) == 0 or np.std(p) == 0:
            continue
        thr = np.quantile(t, 0.95)
        peak = t >= thr
        npk = int(peak.sum())
        valid_peak = npk > 3 and np.std(p[peak]) > 0
        ps = spearmanr(t[peak], p[peak])[0] if valid_peak else np.nan
        pp = pearsonr(t[peak], p[peak])[0] if valid_peak else np.nan
        if npk >= 3 and not peak.all():
            pred_topk = set(np.argsort(p)[-npk:].tolist())
            true_pk = set(np.where(peak)[0].tolist())
            inter = len(pred_topk & true_pk)
            recall_at5 = inter / npk
            jaccard = inter / (2 * npk - inter)
        else:
            recall_at5 = jaccard = np.nan
        rows.append(dict(
            transcript=k,
            pearson=pearsonr(t, p)[0],
            spearman=spearmanr(t, p)[0],
            peak_spearman=ps,
            peak_pearson=pp,
            recall_at5=recall_at5,
            jaccard=jaccard,
            length=len(t),
        ))
    return pd.DataFrame(rows)


def per_tx_medians(pred, true):
    """Fold-level medians (P, S, recall@k, Jaccard, n) for gene-level 5CV.

    Filters and peak definition match :func:`per_tx_metrics` exactly.
    """
    P, S, REC, JAC = [], [], [], []
    for k in true:
        if k not in pred:
            continue
        t = np.asarray(true[k], float)
        p = np.asarray(pred[k], float)
        if len(t) != len(p) or len(t) < 5 or np.std(t) < 1e-9 or np.std(p) < 1e-9:
            continue
        P.append(pearsonr(t, p)[0])
        S.append(spearmanr(t, p)[0])
        thr = np.quantile(t, 0.95)
        peak = t >= thr
        npk = int(peak.sum())
        if npk >= 3 and not peak.all():
            pred_topk = set(np.argsort(p)[-npk:].tolist())
            true_pk = set(np.where(peak)[0].tolist())
            inter = len(pred_topk & true_pk)
            REC.append(inter / npk)
            JAC.append(inter / (2 * npk - inter))
    return (
        float(np.median(P)) if P else float("nan"),
        float(np.median(S)) if S else float("nan"),
        float(np.median(REC)) if REC else float("nan"),
        float(np.median(JAC)) if JAC else float("nan"),
        len(P),
    )


def true_pause(items):
    """Map (id, idx, count) items to per-transcript mean-normalised pause profiles."""
    out = {}
    for k, _idx, cnt in items:
        m = cnt.mean()
        out[k] = cnt / m if m > 0 else cnt
    return out
