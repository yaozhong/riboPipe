"""Controlled thinning on high-coverage transcripts: ranking of the top-5% codons at low input depth.

For every high-coverage transcript (Part I universe, interior >= 60 codons) and replicate assignment, the input
replicate is thinned to 0.02, 0.05, 0.1, 0.2, 0.5 and 1.0 reads/codon (interior; three thinning seeds), the other
replicate (>= 1 read/codon) is the reference. Each estimator ranks the interior codons; the top k = ceil(0.05 L)
codons are compared with the reference's top k (ties are given fractional weights).
Estimators: Raw (sparse reads), Smooth (5-codon moving sum), CodonMean, Ridge, RiboPipe (out-of-fold
predictions), Hybrid (depth-weighted mixture), Post_RiboPipe and Post_CodonMean (posterior means, prior strength
cross-fitted on the other folds, see posterior.py).
Metrics: prec (top-k precision), prec_tol (+-1 codon tolerance), ps_err / ps_err_p05 (mean |log2 pause-score
error| at the reference's top codons, pseudocount 0.1 / 0.5), ll_gain, pearson. Per cell the transcript means
(over seeds) are averaged with similarity-unit bootstrap intervals (B = 1000, seed 123).
CodonMean / Ridge predictions are optional; without them the estimators and comparisons that need them are skipped.
"""
import time

import numpy as np

from . import posterior as PO

EDGE = PO.EDGE
NBOOT, BOOT_SEED = 1000, 123
PS_PSEUDO, PS_HALF = 0.1, 10
SMOOTH_HALF = 2
EST_ALL = ["Raw", "Smooth", "CodonMean", "Ridge", "RiboPipe", "Hybrid", "Post_RiboPipe", "Post_CodonMean"]
MET = ["prec", "prec_tol", "ps_err", "ps_err_p05", "ll_gain", "pearson"]
COMP_ALL = [("RiboPipe", "Raw"), ("Post_RiboPipe", "Raw"), ("Post_RiboPipe", "RiboPipe"), ("Post_RiboPipe", "Post_CodonMean"),
            ("RiboPipe", "Ridge"), ("Hybrid", "Raw"), ("Hybrid", "RiboPipe"), ("Post_RiboPipe", "Hybrid"), ("Smooth", "Raw"),
            ("RiboPipe", "CodonMean")]
NEEDS = {"CodonMean": "cm", "Ridge": "rg", "Post_CodonMean": "cm"}


def tie_expect(score, k):
    score = np.asarray(score); n = len(score); assert 1 <= k <= n
    thr = np.sort(score)[::-1][k - 1]
    gt = score > thr; eq = score == thr; g = int(gt.sum()); t = int(eq.sum())
    w = gt.astype(float); w[eq] = (k - g) / t
    return w


def k_of(Lint): return int(np.ceil(0.05 * Lint))
def precision(w_sel, w_ref, k): return float(np.dot(w_sel, w_ref) / k)


def precision_tol(w_sel, w_ref, k):
    d = np.convolve(w_ref, np.ones(3), mode="same")
    return float(np.dot(w_sel, np.minimum(1.0, d)) / k)


def local_mean_excl(x, h=PS_HALF):
    n = len(x); cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n); lo = np.maximum(0, i - h); hi = np.minimum(n, i + h + 1)
    return (cs[hi] - cs[lo] - x) / (hi - lo - 1)


def pause_score(x, eps=PS_PSEUDO):
    xt = PO.mn(x); return (xt + eps) / (local_mean_excl(xt) + eps)


def ps_error(x, B, w_ref, eps=PS_PSEUDO):
    a = np.log2(pause_score(x, eps)); b = np.log2(pause_score(B, eps))
    return float(np.dot(w_ref, np.abs(a - b)) / w_ref.sum())


def to_prob(x):
    x = np.asarray(x, float); n = len(x); s = x.sum()
    p = np.full(n, 1.0 / n) if s <= 0 else x / s
    return (1 - PO.UNIF_MIX) * p + PO.UNIF_MIX / n


def ll_gain(p_mixed, B):
    n = len(B); tot = B.sum(); assert tot > 0
    return float(np.dot(B, np.log2(p_mixed * n)) / tot)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0: return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def smooth(x, h=SMOOTH_HALF):
    return np.convolve(np.asarray(x, float), np.ones(2 * h + 1), mode="same")


class UnitBoot:
    """Similarity-unit bootstrap weights (B x U), drawn once and shared by all cells."""
    def __init__(self, units, B=NBOOT, seed=BOOT_SEED):
        self.keys = sorted(set(units)); self.idx = {u: i for i, u in enumerate(self.keys)}; U = len(self.keys)
        rng = np.random.default_rng(seed); self.W = np.empty((B, U), np.float32)
        for b in range(B): self.W[b] = np.bincount(rng.integers(0, U, U), minlength=U)

    def _agg(self, units, vals):
        ui = np.array([self.idx[u] for u in units]); U = len(self.keys)
        return np.bincount(ui, weights=vals, minlength=U)

    def mean_ci(self, units, vals):
        """Transcript-weighted mean; NaN dropped."""
        vals = np.asarray(vals, float); ok = ~np.isnan(vals); units = [u for u, o in zip(units, ok) if o]; vals = vals[ok]
        if len(vals) == 0: return dict(mean=None, lo=None, hi=None, n=0)
        num = self.W @ self._agg(units, vals); den = self.W @ self._agg(units, np.ones(len(vals)))
        bs = num[den > 0] / den[den > 0]; lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(mean=float(vals.mean()), lo=float(lo), hi=float(hi), n=int(len(vals)))


def run(ds1, rp, cm=None, rg=None, log=print):
    """ds1: Part I dataset; rp / cm / rg: out-of-fold predictions {tx: vector} of RiboPipe / CodonMean / Ridge."""
    t0 = time.time()
    priors = {"rp": rp}
    if cm is not None: priors["cm"] = cm
    if rg is not None: priors["rg"] = rg
    EST = [e for e in EST_ALL if NEEDS.get(e) is None or NEEDS[e] in priors]
    COMP = [(x, y) for x, y in COMP_ALL if x in EST and y in EST]
    txs = PO.calibration_transcripts(ds1, priors)
    folds = sorted({T["fold"] for T in txs})
    # pass 1: grid likelihoods for the cross-fitted prior strengths
    pkeys = ("rp", "cm") if "cm" in priors else ("rp",)
    G = PO.grid_likelihoods(txs, pkeys, hybrid_prior="rp")
    raw = PO.choose_kappa(G, pkeys + ("hyb",))
    ren = {"rp": "prp", "cm": "pcm", "hyb": "hyb"}
    chosen = {k: {**{ren[a]: v[a] for a in pkeys + ("hyb",)}, "n_records_fit": v["n_records_fit"]} for k, v in raw.items()}
    log(f"[ranking] {ds1.tag} pass1 {time.time() - t0:.0f}s")
    # pass 2: metrics
    acc, meta = {}, {}
    for ti, role, di, s, Af, Bint, sl in PO.iter_calibration_records(txs):
        T = txs[ti]; A = Af[sl]; Lint = len(A); K = k_of(Lint); ch = chosen[(role, T["fold"])]
        rpi = T["rp"][sl]
        kr = ch["prp"]["grid_index"]; kh = ch["hyb"]["grid_index"]
        X = {"Raw": A.astype(float), "Smooth": smooth(Af)[sl], "RiboPipe": rpi,
             "Hybrid": PO.hybrid_grid(A, rpi, PO.GRID[kh:kh + 1])[0],
             "Post_RiboPipe": PO.posterior_grid(A, PO.p_seq_of(rpi), PO.GRID[kr:kr + 1])[0]}
        if "cm" in priors:
            kc = ch["pcm"]["grid_index"]; X["CodonMean"] = T["cm"][sl]
            X["Post_CodonMean"] = PO.posterior_grid(A, PO.p_seq_of(T["cm"][sl]), PO.GRID[kc:kc + 1])[0]
        if "rg" in priors: X["Ridge"] = T["rg"][sl]
        X = {e: X[e] for e in EST}
        wref = tie_expect(Bint, K); Bf = Bint.astype(float)
        cell = acc.setdefault((ti, role, di), {e: {m: [] for m in MET} for e in EST})
        for e, x in X.items():
            x = np.clip(np.asarray(x, float), 0, None); w = tie_expect(x, K)
            v = dict(prec=precision(w, wref, K), prec_tol=precision_tol(w, wref, K), ps_err=ps_error(x, Bf, wref),
                     ps_err_p05=ps_error(x, Bf, wref, 0.5), ll_gain=ll_gain(to_prob(x), Bf), pearson=pearson(x, Bf))
            for m in MET: cell[e][m].append(v[m])
        meta[(ti, role, di)] = dict(K=K, Lint=Lint, rand=K / Lint, Bdepth=float(Bint.sum() / Lint), Aobs=float(A.sum() / Lint),
                                    Azero=bool(A.sum() == 0))
    log(f"[ranking] {ds1.tag} pass2 {len(acc)} cells {time.time() - t0:.0f}s")
    units_all = [T["unit"] for T in txs]; boot = UnitBoot(units_all)
    out = {}
    for stratum, bmin in (("main_Bge1", 1.0), ("sens_Bge4", 4.0)):
        for role in (0, 1):
            for di, d in enumerate(PO.DEPTHS):
                keys = [k for k in acc if k[1] == role and k[2] == di and meta[k]["Bdepth"] >= bmin]
                if not keys: continue
                units = [txs[k[0]]["unit"] for k in keys]
                V = {e: {m: np.array([np.nanmean(acc[k][e][m]) if not np.all(np.isnan(acc[k][e][m])) else np.nan for k in keys])
                         for m in MET} for e in EST}
                cell = dict(n_tx=len(keys), n_units=len(set(units)), rand_prec=float(np.mean([meta[k]["rand"] for k in keys])),
                            A_zero_frac=float(np.mean([meta[k]["Azero"] for k in keys])),
                            A_obs_median=float(np.median([meta[k]["Aobs"] for k in keys])),
                            B_depth_median=float(np.median([meta[k]["Bdepth"] for k in keys])),
                            est={e: {m: boot.mean_ci(units, V[e][m]) for m in MET} for e in EST}, diff={})
                for x, y in COMP:
                    for m in ("prec", "prec_tol", "ps_err", "ll_gain", "pearson"):
                        cell["diff"][f"{x}-{y}|{m}"] = boot.mean_ci(units, V[x][m] - V[y][m])
                out[f"{stratum}|role{role}|d{d}"] = cell
    # descriptive crossover: where the mean precision difference RiboPipe - Raw changes sign (log-linear interpolation)
    cross = {}
    for role in (0, 1):
        ds_ = [d for d in PO.DEPTHS if f"main_Bge1|role{role}|d{d}" in out]
        diff = [out[f"main_Bge1|role{role}|d{d}"]["diff"]["RiboPipe-Raw|prec"]["mean"] for d in ds_]
        c = None
        for j in range(len(ds_) - 1):
            if diff[j] > 0 >= diff[j + 1]:
                x0, x1 = np.log(ds_[j]), np.log(ds_[j + 1]); c = float(np.exp(x0 + (x1 - x0) * diff[j] / (diff[j] - diff[j + 1])))
        cross[f"role{role}"] = dict(depths=ds_, diff=diff, crossover=c)
    return dict(dataset=ds1.tag, n_tx_universe=len(txs), estimators=EST, metrics=MET, depths=PO.DEPTHS, seeds=PO.THIN_SEEDS,
                grid=[float(x) for x in PO.GRID], folds=folds,
                chosen={f"role{r}|fold{f}": v for (r, f), v in chosen.items()}, cells=out, crossover_prec=cross,
                runtime_s=time.time() - t0)
