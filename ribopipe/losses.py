"""RiboPipe training losses (paper Methods: training objective).

The headline RiboPipe configuration (``ribopipe_nt_struct_h256``) is trained with the
parameter-free **peak-gated** loss :func:`huber_peak_mse` (``peakmse``): squared error on
peak positions (target > tau) and robust Huber on the background (target <= tau).  The
other losses are kept for the ablation / baseline configurations reported in the paper.

All losses operate on a padded ``(B, L_max)`` prediction/target pair together with a
boolean ``mask`` selecting the real (non-padded) codon positions.  Targets are the
mean-normalised pause score (count / transcript-mean), so ``mean(target) ~= 1`` and a
threshold ``tau = 1`` marks above-mean (candidate pause) occupancy.
"""
from __future__ import annotations

import torch


def huber_mask(pred, tgt, mask, delta: float = 1.0):
    """Plain masked Huber loss (BiLSTM-base / legacy configuration)."""
    e = (pred - tgt)[mask]
    a = e.abs()
    return torch.where(a < delta, 0.5 * e * e, delta * (a - 0.5 * delta)).mean()


def wmse_mask(pred, tgt, mask):
    """Weighted MSE with a hard-threshold peak up-weight (1 + 2*[t > mean+2sd])."""
    t = tgt[mask]
    p = pred[mask]
    w = 1.0 + 2.0 * ((t > t.mean() + 2 * t.std()).float())
    return ((p - t) ** 2 * w).mean()


def huber_peak_weighted(pred, tgt, mask, delta: float = 1.0, alpha: float = 0.0, tau: float = 1.0):
    """Masked Huber, optionally up-weighting high-pause (peak) positions.

    With ``alpha == 0`` this is identical to :func:`huber_mask`.  The weight is
    ``w_i = 1 + alpha * relu(tgt_i - tau)``: peaks (tgt >> 1) are penalised harder for
    under-prediction, sharpening peaks and improving within-peak ranking.
    """
    e = (pred - tgt)[mask]
    a = e.abs()
    base = torch.where(a < delta, 0.5 * e * e, delta * (a - 0.5 * delta))
    if alpha <= 0:
        return base.mean()
    t = tgt[mask]
    w = 1.0 + alpha * torch.relu(t - tau)
    return (base * w).sum() / w.sum().clamp_min(1e-6)


def huber_peak_mse(pred, tgt, mask, tau: float = 1.0, delta: float = 1.0):
    """Parameter-free peak-gated loss (RiboPipe headline, ``peakmse``).

    Squared error (MSE) on peak positions (tgt > tau); robust Huber (delta) on the
    background (tgt <= tau).  Unlike :func:`huber_peak_weighted` there is no peak-weight
    magnitude to tune: the gradient on a peak is the raw (uncapped, quadratic) residual,
    so the harder a peak is under-predicted the harder it is pushed, while the background
    keeps the robust linear Huber tail.  Equivalent to a position-dependent Huber with
    ``delta = inf`` on peaks and ``delta`` on the background.  ``tau`` is a peak
    *threshold* (tau = 1 = above-mean occupancy under mean-normalisation), not a loss
    weight.
    """
    e = (pred - tgt)[mask]
    t = tgt[mask]
    a = e.abs()
    huber = torch.where(a < delta, 0.5 * e * e, delta * (a - 0.5 * delta))
    mse = 0.5 * e * e
    return torch.where(t > tau, mse, huber).mean()


def listwise_rank_loss(pred, tgt, mask, lens, tau: float = 0.05):
    """Per-transcript soft-Spearman surrogate: ``1 - corr(softrank(pred), softrank(tgt))``.

    Differentiable rank via a softmax-over-pairs approximation, averaged over the
    transcripts in the batch.  Off by default (headline uses ``rank_lambda = 0``).
    """
    B_, _ = pred.shape
    total = pred.new_zeros(())
    cnt = 0
    for b in range(B_):
        L = int(lens[b].item())
        if L < 5:
            continue
        p = pred[b, :L]
        t = tgt[b, :L]
        dp = (p[:, None] - p[None, :]) / tau
        dt = (t[:, None] - t[None, :]) / tau
        rp = torch.sigmoid(dp).sum(1)
        rt = torch.sigmoid(dt).sum(1)
        rp = rp - rp.mean()
        rt = rt - rt.mean()
        denom = (rp.norm() * rt.norm()).clamp_min(1e-6)
        corr = (rp * rt).sum() / denom
        total = total + (1.0 - corr)
        cnt += 1
    return total / max(cnt, 1)
