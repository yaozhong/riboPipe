"""Shared definitions for the model analyses of Part III (in silico substitutions, branch routing, window transplants).

Coordinates (all A-site):
  * codon i of a sequence is the A-site codon when the ribosome A site is at codon i (the release data are A-site
    aligned and the model applies no register offset);
  * relative codon offset r = edited codon index j - read-out codon index i; r = 0 is the A site, r = -1 the P site,
    r = -2 the E site;
  * relative nucleotide offset d = edited nucleotide index q - 3i (first base of the A-site codon is d = 0).
Effects are reported on the model output scale u = log1p(predicted relative density).
Folding energies (MFE) are the ViennaRNA MFEs of the 30-nt windows starting at every nucleotide (the last 29 positions
are 0), the same definition as the release MFE tracks. After a sequence edit the windows overlapping the edit are either
recomputed on the edited sequence ("recompute") or kept at their original values ("fixed").

Models are analysed out of fold: each transcript is analysed only with the model of the fold in which it was a test
transcript.
"""
import os, json, gzip, math, hashlib
import numpy as np
import torch

from .. import arch as A
from .. import train as T
from .. import data as D

try:                                   # optional dependency (pip install ViennaRNA); needed only for "recompute"
    import RNA                         # noqa: F401
except ImportError:                    # pragma: no cover
    RNA = None

MFE_WIN = 30
STOPS = ("TAA", "TAG", "TGA")
CODONS = list(T.CODONS)                # 64 codons, ACGT order (index = model codon index)
CIDX = dict(T.CIDX)
SENSE = [c for c in CODONS if c not in STOPS]          # 61 sense codons (substitution targets)
SENSE_IDX = [CIDX[c] for c in SENSE]
CODON_TABLE = dict(A._CODON2AA)                        # the model's own codon -> amino-acid table
NT = "ACGT"
OFFSETS = list(range(-12, 11))       # edited-codon offsets r of the peak ISM and the sensitivity maps
EXT_OFFSETS = [-20, -15, 15, 20]     # control offsets outside the convolution windows
FF_FIELD = (-6, 4)                   # codons covered by the convolutions and the MFE windows of read-out i
INTERIOR_MARGIN = 20                 # interior domain [20, L-20)
UNIT_SPLIT_SEED = 20261112
REGISTER = dict(coordinate="A-site", offset_definition="r = edited codon index j - read-out codon index i",
                A_site=0, P_site=-1, E_site=-2, nt_offset_definition="d = edited nt index q - 3*i (A-site codon first base = 0)",
                model_register_offset_applied=0)

assert CODON_TABLE["ATG"] == "M" and CODON_TABLE["TGG"] == "W" and CODON_TABLE["CCG"] == "P" and CODON_TABLE["TAA"] == "*"
assert len(SENSE) == 61


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _js(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=1, ensure_ascii=False, default=_js)
    os.replace(tmp, path)


def lab(r):
    return {0: "A", -1: "P", -2: "E"}.get(int(r), f"{int(r):+d}")


# ---------------------------------------------------------------------------------------------- folding energies
def _need_rna():
    if RNA is None:
        raise ImportError("recomputing folding energies needs the ViennaRNA Python bindings (pip install ViennaRNA)")


def mfe_track(seq):
    """MFE of the 30-nt window starting at every nucleotide (float32, len(seq)); the last 29 positions are 0."""
    _need_rna()
    n = len(seq); out = np.zeros(n, np.float32)
    for i in range(n - MFE_WIN + 1):
        out[i] = RNA.fold(seq[i:i + MFE_WIN])[1]
    return out


def mfe_update(seq_new, track_old, nt_lo, nt_hi):
    """Recompute every 30-nt window overlapping the edited span [nt_lo, nt_hi) (including windows across both seams).
    Window s covers [s, s+30) and overlaps the edit iff nt_lo-29 <= s <= nt_hi-1; only valid starts s <= n-30 are
    recomputed. The sequence length must not change. Returns (new track float32, list of recomputed starts)."""
    _need_rna()
    n = len(seq_new); assert len(track_old) == n
    tr = np.array(track_old, np.float32, copy=True)
    s0, s1 = max(0, nt_lo - (MFE_WIN - 1)), min(nt_hi - 1, n - MFE_WIN)
    starts = list(range(s0, s1 + 1))
    for s in starts:
        tr[s] = RNA.fold(seq_new[s:s + MFE_WIN])[1]
    return tr, starts


# ---------------------------------------------------------------------------------------------- model inputs / forward
def inputs_from(seq, track):
    """(codon index (L,), bio (L, 18)) of a sequence and its MFE track (same feature functions as training)."""
    L = len(seq) // 3; assert len(seq) == 3 * L and len(track) == len(seq)
    idx = T.codon_idx(seq); nt = T.nt3_onehot(seq, L)
    nv = len(seq) - (MFE_WIN - 1)
    v, ok = T.mfe_features(np.asarray(track, float), nv, L)
    return idx, np.concatenate([nt, v, ok], 1).astype(np.float32)


@torch.no_grad()
def forward_same_len(model, idx_list, bio_list):
    """Batched forward of equal-length variants -> (B, L) float64 u (log1p scale)."""
    idx = torch.from_numpy(np.stack(idx_list)); bio = torch.from_numpy(np.stack(bio_list))
    ll = torch.full((idx.shape[0],), idx.shape[1], dtype=torch.long)
    return model(idx, bio, ll).double().numpy()


def forward_variants(model, seqs, tracks, batch=32):
    """u of many equal-length variant sequences, in batches of ``batch`` (the paper's peak ISM and transplant used 32)."""
    out = []
    for s in range(0, len(seqs), batch):
        il, bl = zip(*[inputs_from(a, b) for a, b in zip(seqs[s:s + batch], tracks[s:s + batch])])
        out.append(forward_same_len(model, list(il), list(bl)))
    return np.concatenate(out, 0) if out else np.zeros((0, 0))


def forward_one(model, seq, track):
    """Batch-1 forward of one sequence -> u (float64). Used by the sub-window transplants and the routing gates."""
    idx, bio = inputs_from(seq, track)
    return forward_same_len(model, [idx], [bio])[0]


def sub_codon(seq, j, cod):
    return seq[:3 * j] + cod + seq[3 * j + 3:]


# ---------------------------------------------------------------------------------------------- out-of-fold models
def load_model(path, expected_params=None, dataset=None, fold=None, seed=None, check_run_json=True):
    """Load a checkpoint with ribopipe.train.load_ckpt and check it (parameter count, identity, run.json md5)."""
    m, ck = T.load_ckpt(path, "cpu"); m.eval()
    n = T.count_params(m)
    exp = expected_params if expected_params is not None else T.CONFIGS.get(ck.get("config_id"), {}).get("expected_params")
    if exp is not None and n != exp:
        raise ValueError(f"[gate] {path}: {n} parameters != expected {exp}")
    if dataset is not None and ck.get("dataset") not in (None, dataset):
        raise ValueError(f"[gate] {path} was trained on {ck.get('dataset')}, not {dataset}")
    if fold is not None and ck.get("fold") is not None and int(ck["fold"]) != int(fold):
        raise ValueError(f"[gate] {path} is fold {ck['fold']}, not {fold}")
    if seed is not None and ck.get("seed") is not None and int(ck["seed"]) != int(seed):
        raise ValueError(f"[gate] {path} is seed {ck['seed']}, not {seed}")
    md5 = md5_file(path); rj = os.path.join(os.path.dirname(path), "run.json")
    if check_run_json and os.path.exists(rj):
        r = json.load(open(rj))
        if r.get("status") != "OK": raise ValueError(f"[gate] {rj}: status {r.get('status')}")
        cid = r.get("checkpoint_id", "")
        if cid and not cid.replace("md5 ", "md5:").endswith("md5:" + md5):
            raise ValueError(f"[gate] checkpoint md5 {md5} does not match {rj}")
    info = dict(path=os.path.abspath(path), md5=md5, params=n, config_id=ck.get("config_id"), dataset=ck.get("dataset"),
                fold=ck.get("fold"), seed=ck.get("seed"), selected_epoch=ck.get("selected_epoch"), mfe_norm=list(ck.get("mfe_norm", [])))
    return m, info


class OOF:
    """Out-of-fold view of one dataset: every test transcript -> the model of its test fold, its MFE track and the
    out-of-fold prediction used to select peaks and neutral sites.

    ckpt_dir: directory with fold<f>/model.pt (for example runs/<config>/<dataset>/seed123 as written by
              ``ribopipe cv5``). Predictions are read from fold<f>/pred.npz when present (preds="stored", the
              paper's setting: peaks were selected on the stored predictions); with preds="compute" (or when no
              pred.npz exists) they are computed here on the CPU.
    models:   optional {fold: model} instead of ckpt_dir (used by the tests).
    """

    def __init__(self, ds, ckpt_dir=None, models=None, preds="stored", expected_params=None, folds=None, load_models=True):
        self.ds = ds; self.dataset = ds.tag; self.ckpt_dir = ckpt_dir
        self.folds = list(folds) if folds is not None else list(ds.folds)
        self.fold_of = {t: f for f in ds.folds for t in ds.test_ids(f)}
        assert sorted(self.fold_of) == sorted(ds.ids)
        # tracks are read into memory once (float32, as in the paper); npz handles cannot be shared across forks
        self.tracks = {t: np.asarray(ds.mfe_raw(t), np.float32) for t in ds.ids}
        self.models, self.ckinfo, self.pred, self.pred_source = {}, {}, {}, {}
        for f in self.folds:
            if models is not None:
                self.models[f] = models[f]; self.models[f].eval(); self.ckinfo[f] = dict(path=None, params=T.count_params(models[f]))
            elif load_models:
                self.models[f], self.ckinfo[f] = load_model(os.path.join(ckpt_dir, f"fold{f}", "model.pt"), expected_params,
                                                            dataset=ds.tag, fold=f)
            pz = os.path.join(ckpt_dir, f"fold{f}", "pred.npz") if ckpt_dir else None
            if preds == "stored" and pz and os.path.exists(pz):
                z = np.load(pz)
                for t in ds.test_ids(f): self.pred[t] = np.asarray(z[t], np.float64)
                self.pred_source[f] = dict(kind="stored", path=os.path.abspath(pz), md5=md5_file(pz))
            elif f in self.models:
                for t in ds.test_ids(f):
                    u = forward_one(self.models[f], ds.seq[t], self.tracks[t])
                    self.pred[t] = np.expm1(u).astype(np.float32).astype(np.float64)
                self.pred_source[f] = dict(kind="computed_cpu")

    def model_for(self, tx): return self.models[self.fold_of[tx]]
    def track(self, tx): return self.tracks[tx]

    def assert_oof(self, tx, f):
        """The transcript is a test transcript of fold f and its similarity unit is not in that fold's train/val."""
        assert self.fold_of[tx] == f, (tx, f)
        tv = set(self.ds.train_ids(f)) | set(self.ds.val_ids(f))
        assert tx not in tv and self.ds.unit[tx] not in {self.ds.unit[t] for t in tv}, (tx, f)
        return True

    def provenance(self):
        return dict(dataset=self.dataset, data_tag=D.DATA_TAG, ckpt_dir=(os.path.abspath(self.ckpt_dir) if self.ckpt_dir else None),
                    checkpoints={int(f): {k: v for k, v in self.ckinfo[f].items() if k != "mfe_norm"} for f in self.ckinfo},
                    predictions={int(f): v for f, v in self.pred_source.items()},
                    rna_version=(getattr(RNA, "__version__", None) if RNA is not None else None), torch=torch.__version__,
                    register=REGISTER)


# ---------------------------------------------------------------------------------------------- similarity-unit halves
_HALVES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_unit_halves.json.gz")


def unit_halves(units=None, seed=UNIT_SPLIT_SEED):
    """{unit: 'A' | 'B'}. Without ``units``: the frozen halves of the paper's 17,977 human similarity units (shared by
    the four human datasets). With ``units`` (any iterable of unit ids): sort them, permute with
    RandomState(seed) and give the first half A (the rule that produced the frozen halves)."""
    if units is None:
        with gzip.open(_HALVES_FILE, "rt") as f: h = json.load(f)
        out = {u: "A" for u in h["A"]}; out.update({u: "B" for u in h["B"]}); return out
    units = sorted(set(units)); perm = np.random.RandomState(seed).permutation(len(units)); half = {}
    for rank, k in enumerate(perm): half[units[k]] = "A" if rank < len(units) // 2 else "B"
    return half


def halves_for(ds, halves=None):
    """Halves for a dataset: the frozen human halves when they cover every unit of ``ds``; otherwise the rule applied
    to the dataset's own units."""
    if halves is not None: return halves
    h = unit_halves()
    if all(ds.unit[t] in h for t in ds.ids): return h
    return unit_halves(ds.unit[t] for t in ds.ids)


# ---------------------------------------------------------------------------------------------- peaks
def topk_positions(v, k):
    """Positions of v not smaller than its k-th largest value (ties included)."""
    order = np.argsort(-v, kind="stable"); thr = v[order[k - 1]]
    return set(np.where(v >= thr)[0].tolist())


def candidate_peaks(y, yhat):
    """Interior [20, L-20): observed top 5% (k = ceil(0.05 n), ties included, y > 0) intersected with the predicted top
    5%. Returns absolute indices (ascending)."""
    L = len(y); lo, hi = INTERIOR_MARGIN, L - INTERIOR_MARGIN
    n = hi - lo
    if n < 20: return []
    k = int(math.ceil(0.05 * n))
    Tset = {i for i in topk_positions(y[lo:hi], k) if y[lo + i] > 0}
    P = topk_positions(yhat[lo:hi], k)
    return sorted(lo + i for i in (Tset & P))


def representative_peak(y, yhat):
    """The candidate peak with the highest observed density (lowest index on ties), or None."""
    c = candidate_peaks(y, yhat)
    return max(c, key=lambda i: (y[i], -i)) if c else None


# ---------------------------------------------------------------------------------------------- bootstrap
def cluster_boot_ci(vals, clusters, n_boot=2000, seed=0):
    """Mean with a 95% interval from a similarity-unit cluster bootstrap (resample units, pool their elements)."""
    vals = np.asarray(vals, float); clusters = np.asarray(clusters)
    ok = np.isfinite(vals); vals, clusters = vals[ok], clusters[ok]
    if len(vals) == 0: return (float("nan"),) * 3
    uq = sorted(set(clusters.tolist())); grp = [vals[clusters == u] for u in uq]
    sums = np.array([g.sum() for g in grp]); cnts = np.array([len(g) for g in grp])
    rng = np.random.RandomState(seed); n = len(uq); bs = np.empty(n_boot)
    for b in range(n_boot):
        k = rng.randint(0, n, n); bs[b] = sums[k].sum() / cnts[k].sum()
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


BOOT_SEED2, B2 = 20261115, 1000      # follow-up analyses (routing, sub-window transplants): 1,000 resamples


def unit_weights(n_units, seed=BOOT_SEED2, n=B2):
    """(n, n_units) multiplicities of a unit bootstrap."""
    rng = np.random.RandomState(seed)
    return np.stack([np.bincount(rng.randint(0, n_units, n_units), minlength=n_units) for _ in range(n)]).astype(float)


def ci95(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def cluster_boot(fn, units, n=B2, seed=BOOT_SEED2):
    """fn(weights over elements) -> statistic, for n unit-bootstrap replicates."""
    units = np.asarray(units); uq, inv = np.unique(units, return_inverse=True); rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        w = np.bincount(rng.randint(0, len(uq), len(uq)), minlength=len(uq)).astype(float)[inv]
        out.append(fn(w))
    return np.array(out, float)


def wmedian(x, w):
    ok = np.isfinite(x) & (w > 0); x, w = x[ok], w[ok]
    if len(x) == 0: return np.nan
    o = np.argsort(x, kind="mergesort"); x, w = x[o], w[o]; cs = np.cumsum(w); n = cs[-1]
    k1 = (n - 1) // 2; k2 = n // 2
    return 0.5 * (x[np.searchsorted(cs, k1, side="right")] + x[np.searchsorted(cs, k2, side="right")])


def median_ci(x, units, mask=None):
    """Median over elements with a unit-bootstrap 95% interval of the median (1,000 resamples, seed 20261115)."""
    x = np.asarray(x, float); units = np.asarray(units)
    if mask is not None: x, units = x[mask], units[mask]
    pt = float(np.nanmedian(x)); b = cluster_boot(lambda w: wmedian(x, w), units)
    return dict(median=pt, ci95=ci95(b), n=int(np.isfinite(x).sum()))


def mean_ci(x, units):
    x = np.asarray(x, float); ok = np.isfinite(x)
    b = cluster_boot(lambda w: float((w * np.where(ok, x, 0)).sum() / (w * ok).sum()), units)
    return dict(mean=float(np.nanmean(x)), ci95=ci95(b))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------------------------- parallel map
G = {}                                 # worker globals (inherited through fork)


def _init_worker():
    torch.set_num_threads(1)


def pmap(fn, tasks, workers):
    """Order-preserving map; fork pool with one torch thread per worker (serial when workers <= 1)."""
    if workers <= 1:
        nt = torch.get_num_threads(); torch.set_num_threads(1)
        try: return [fn(t) for t in tasks]
        finally: torch.set_num_threads(nt)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
        return pool.map(fn, tasks, chunksize=1)
