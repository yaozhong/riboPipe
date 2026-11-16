"""In-house reference predictors of the paper (deterministic, trained once per fold, seed directory "det").

  Paper_CodonMean   per-codon table of mean log1p(c/mean(c)) over training transcripts, prediction expm1(table[A-site codon])
  Paper_TriCodon    the same table keyed on the codon triple (i-1, i, i+1) = (P, A, A+1) in the A-site register
  Paper_Ridge       ridge regression (alpha = 10, intercept) on one-hot codons at offsets -5..+5 (704 features),
                    target log1p(c/mean(c)), prediction expm1
  Paper_RUST        our re-implementation of the RUST enrichment-ratio estimator (O'Connor et al. 2016): per transcript
                    b_i = 1[c_i > mean(c)], rho_j(codon) = P(b = 1 | codon at offset j) / P(b = 1) for offsets -5..+5,
                    prediction = product of the ratios in the window (offsets outside the CDS are skipped).

Training set = train_ids(f) (universe minus test minus validation); transcripts with mean(c) = 0 are skipped; validation
transcripts are not used; no length cap.  Every position is written with provenance 0 (model): the end handling
(-1 key, zero features, skipped offsets) is part of each method's own definition, not an adapter.

This file is a line-by-line port of the paper's baseline_code/ixnos_ref/lookup_ridge_paper.py (md5 9f111c69...) and
rust_paper.py (md5 a2589ab6...); comments translated, numerics unchanged.  The ridge solution is the closed form of
sklearn Ridge(alpha=10, fit_intercept=True) accumulated per transcript in float64 (same solution, bounded memory).
"""
import json, os
from collections import defaultdict
import numpy as np

from . import common as C

RIDGE_K = 5
RIDGE_ALPHA = 10.0
TRICODON_OFFSETS = (-1, 0, 1)
RUST_K = 5
RATIO_FLOOR = 1e-6
LENGTH_CAP = None
CONFIGS = {"Paper_CodonMean": "codon_mean", "Paper_TriCodon": "tricodon", "Paper_Ridge": "ridge", "Paper_RUST": "rust"}


# ------------------------------------------------------------------ codon-mean / tri-codon lookup
def _key(idx, i, order):
    if order == 1: return int(idx[i])
    return tuple(int(idx[j]) if 0 <= j < len(idx) else -1 for j in (i + o for o in TRICODON_OFFSETS))


def lookup_fit(tr_items, order=1):
    """Table value = arithmetic mean of log1p(c/mean(c)) over all training positions with that key; unseen keys fall
    back to glob = unweighted mean of the per-key means.  tr_items: [(tx, codon_idx, counts)] from the training fold."""
    s = defaultdict(lambda: [0.0, 0])
    for k, idx, cnt in tr_items:
        m = cnt.mean()
        if m <= 0: continue
        lp = np.log1p(cnt / m)
        for i in range(len(idx)):
            kk = _key(idx, i, order); s[kk][0] += lp[i]; s[kk][1] += 1
    table = {k: (v[0] / v[1]) for k, v in s.items() if v[1] > 0}
    glob = float(np.mean([v[0] / v[1] for v in s.values()])) if s else 0.0
    return table, glob


def lookup_predict(table, glob, idx, order=1):
    vals = []; unseen = 0
    for i in range(len(idx)):
        kk = _key(idx, i, order)
        if kk in table: vals.append(table[kk])
        else: vals.append(glob); unseen += 1
    return np.expm1(np.asarray(vals, float)), unseen


# ------------------------------------------------------------------ ridge
def ridge_feat(idx, K=RIDGE_K, nC=64):
    """One-hot codons at offsets -K..+K; out-of-range and non-ACGT codons are all-zero."""
    L = len(idx); W = 2 * K + 1; idx = np.asarray(idx)
    F = np.zeros((L, W * nC), np.float32); ii = np.arange(L)
    for j, off in enumerate(range(-K, K + 1)):
        p = ii + off; inb = (p >= 0) & (p < L)
        c = np.full(L, -1, np.int64); c[inb] = idx[p[inb]]
        ok = c >= 0
        F[ii[ok], j * nC + c[ok]] = 1.0
    return F


def ridge_fit(tr_items, K=RIDGE_K, alpha=RIDGE_ALPHA):
    """Closed form of sklearn Ridge(alpha, fit_intercept=True): w = (Xc'Xc + alpha I)^-1 Xc'yc with column-centred
    X and y, b = ybar - xbar.w; sufficient statistics accumulated per transcript in float64."""
    Dm = (2 * K + 1) * 64
    XtX = np.zeros((Dm, Dm)); Xty = np.zeros(Dm); sx = np.zeros(Dm); sy = 0.0; n = 0
    for k, idx, cnt in tr_items:
        m = cnt.mean()
        if m <= 0: continue
        X = ridge_feat(idx, K).astype(np.float64); y = np.log1p(cnt / m)
        XtX += X.T @ X; Xty += X.T @ y; sx += X.sum(0); sy += float(y.sum()); n += len(y)
    xbar = sx / n; ybar = sy / n
    G = XtX - n * np.outer(xbar, xbar); r = Xty - n * xbar * ybar
    w = np.linalg.solve(G + alpha * np.eye(Dm), r)
    b = ybar - float(xbar @ w)
    return w, b, n


def ridge_predict(w, b, idx, K=RIDGE_K):
    return np.expm1(ridge_feat(idx, K).astype(np.float64) @ w + b)


# ------------------------------------------------------------------ RUST (re-implementation)
def _win_slices(L, off):
    if off >= 0: i_lo, i_hi = 0, L - off
    else: i_lo, i_hi = -off, L
    if i_hi <= i_lo: return None, None
    ii = np.arange(i_lo, i_hi)
    return ii, ii + off


def rust_fit(tr_items, K=RUST_K, nC=64):
    W = 2 * K + 1
    enr = np.zeros((W, nC)); tot = np.zeros((W, nC))
    glob_enr = 0.0; glob_tot = 0.0; n_tx = 0
    for k, idx, cnt in tr_items:
        cnt = np.asarray(cnt, float); m = cnt.mean()
        if m <= 0: continue
        b = (cnt > m).astype(float)
        L = len(cnt); idx = np.asarray(idx); n_tx += 1
        glob_enr += b.sum(); glob_tot += L
        for ji, off in enumerate(range(-K, K + 1)):
            ii, pp = _win_slices(L, off)
            if ii is None: continue
            cc = idx[pp].astype(int); valid = (cc >= 0) & (cc < nC)
            cc = cc[valid]; bi = b[ii][valid]
            np.add.at(tot[ji], cc, 1.0)
            np.add.at(enr[ji], cc, bi)
    p0 = (glob_enr / glob_tot) if glob_tot > 0 else 1e-9
    ratio = np.ones((W, nC))
    nz = tot > 0
    ratio[nz] = (enr[nz] / tot[nz]) / (p0 if p0 > 0 else 1e-9)
    ratio = np.clip(ratio, RATIO_FLOOR, None)
    return ratio, float(p0), n_tx, float(glob_tot)


def rust_predict(ratio, idx, K=RUST_K, nC=64):
    L = len(idx); idx = np.asarray(idx)
    d = np.ones(L)
    for ji, off in enumerate(range(-K, K + 1)):
        ii, pp = _win_slices(L, off)
        if ii is None: continue
        cc = idx[pp].astype(int); valid = (cc >= 0) & (cc < nC)
        d[ii[valid]] *= ratio[ji][cc[valid]]
    return d


# ------------------------------------------------------------------ driver
def run(config_id, dataset, out_root, folds=range(5), smoke=None, overwrite=False, data=None, mfe=None, save_model=True):
    """Train once per fold on train_ids(f), predict test_ids(f), write <out>/<config>/<dataset>/seeddet/fold<f>."""
    kind = CONFIGS[config_id]
    log = C.log_fn(f"{config_id}/{dataset}")
    ds = C.D.load_dataset(dataset, data=data, mfe=mfe, load_mfe=False)
    rc = 0
    for f in folds:
        if C.run_exists(out_root, config_id, dataset, "det", f) and not overwrite:
            log(f"fold{f}: run.json exists, skipped"); continue
        tr, te = ds.train_ids(f), ds.test_ids(f)
        if smoke: tr, te = C.smoke_pick(tr, smoke[0]), C.smoke_pick(te, smoke[2])
        counts = {t: np.asarray(ds.counts[t], float) for t in tr}          # training counts only
        items = [(t, C.seq_to_idx(ds.seq[t], len(counts[t])), counts[t]) for t in tr]
        preds, extra, model_arrays = {}, {}, {}
        if kind in ("codon_mean", "tricodon"):
            order = 1 if kind == "codon_mean" else 3
            table, glob = lookup_fit(items, order); n_unseen = n_pos = 0
            for t in te:
                L = len(ds.seq[t]) // 3; p, u = lookup_predict(table, glob, C.seq_to_idx(ds.seq[t], L), order)
                preds[t] = p.astype(np.float32); n_unseen += u; n_pos += L
            keys = list(table)
            model_arrays = dict(keys=np.array([json.dumps(k) for k in keys]), values=np.array([table[k] for k in keys]), glob=np.array(glob))
            extra = dict(n_table_keys=len(table), glob_fallback=glob, test_positions_unseen_key=int(n_unseen), test_positions=int(n_pos),
                         target="log1p(c/mean(c)) table mean, prediction expm1")
        elif kind == "ridge":
            w, b, n = ridge_fit(items)
            for t in te:
                L = len(ds.seq[t]) // 3; preds[t] = ridge_predict(w, b, C.seq_to_idx(ds.seq[t], L)).astype(np.float32)
            model_arrays = dict(coef=w, intercept=np.array(b))
            extra = dict(n_coef=int(len(w)), intercept=float(b), alpha=RIDGE_ALPHA, K=RIDGE_K, n_train_rows=int(n),
                         solver="closed-form normal equations (== sklearn Ridge, fit_intercept=True)", target="log1p(c/mean(c)), prediction expm1")
        else:
            ratio, p0, ntx, nwin = rust_fit(items)
            for t in te:
                L = len(ds.seq[t]) // 3; preds[t] = rust_predict(ratio, C.seq_to_idx(ds.seq[t], L)).astype(np.float32)
            model_arrays = dict(ratio=ratio, p0=np.array(p0))
            extra = dict(p0=p0, K=RUST_K, ratio_floor=RATIO_FLOOR, n_ratio_cells=int(ratio.size))
        provs = {t: np.zeros(len(preds[t]), np.int8) for t in te}
        meta = dict(status="OK", method=kind, config_id_paper=config_id, n_train_tx=int(sum(1 for i in items if i[2].mean() > 0)),
                    n_optimizer_steps=0, val_metric_name="none (closed form / counting; validation not used)",
                    length_cap=LENGTH_CAP, smoke=(list(smoke) if smoke else False),
                    code_md5=C.code_md5([os.path.abspath(__file__), os.path.join(C.HERE, "common.py")]), **extra)
        d = C.write(out_root, config_id, dataset, "det", f, preds, provs, meta)
        if save_model:
            np.savez(os.path.join(d, "model_det.npz"), **model_arrays)
        log(f"fold{f}: wrote {d} ({len(preds)} transcripts)")
    return rc
