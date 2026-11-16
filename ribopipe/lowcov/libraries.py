"""Input / reference libraries, controlled thinning and per-record scoring for the low-coverage analysis.

No model code: this module only receives finished prediction vectors. The D* definition lives in crossover.py.
  1) expected-depth matching of the two replicates (or a binomial split, sensitivity);
  2) nested thinning of the input library to one integer read count per depth bin (grid sensitivity);
  3) Pearson correlation on the interior domain [20, L-20), with scorability and its reason.
All random streams are seeded per (seed, stream, dataset key, transcript id), so results do not depend on the
order in which transcripts are processed.
"""
import functools
import math
import zlib

import numpy as np

from . import crossover as X

INTERIOR = 20


def rng_for(seed, stream, tag, tx):
    """Per-transcript generator. `tag` is the internal dataset key (e.g. DMSO_TX9P1, not the display name)."""
    return np.random.default_rng([int(seed), int(stream), zlib.crc32(tag.encode()), zlib.crc32(tx.encode())])


# ---------------------------------------------------------------- input / reference libraries
def match_reps(a, b, seed, tag, tx):
    """Expected-depth matching: the deeper replicate is thinned position-wise with Binomial(p = m / total),
    m = the shallower replicate's total. Returns (a', b', m)."""
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64); ta, tb = int(a.sum()), int(b.sum()); m = min(ta, tb)
    g = rng_for(seed, 0, tag, tx)
    if ta > tb: a2, b2 = g.binomial(a, m / ta).astype(np.int64), b.copy()
    elif tb > ta: a2, b2 = a.copy(), g.binomial(b, m / tb).astype(np.int64)
    else: a2, b2 = a.copy(), b.copy()
    return a2, b2, m


def binom_split(a, b, seed, tag, tx):
    """Binomial-split sensitivity: the pooled counts are thinned with p = 2m / pooled total and then split with
    Binomial(0.5) into two halves. Returns (h1, h2, m, pooled_thinned)."""
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64); m = min(int(a.sum()), int(b.sum()))
    pool = a + b; tp = int(pool.sum()); g = rng_for(seed, 0, tag, tx)
    pt = g.binomial(pool, (2 * m / tp) if tp > 0 else 0.0).astype(np.int64)
    h1 = g.binomial(pt, 0.5).astype(np.int64); h2 = pt - h1
    return h1, h2, m, pt


def libraries(design, a, b, seed, tag, tx):
    """{role: (input, reference, input_label, reference_label)} and metadata.
    design: "grid" / "native" (replicate split; role 0 = replicate 1 as input, role 1 = swapped) or "binom_grid"."""
    if design in ("grid", "native"):
        a2, b2, m = match_reps(a, b, seed, tag, tx)
        assert (a2 <= a).all() and (b2 <= b).all()
        return {0: (a2, b2, "rep1", "rep2"), 1: (b2, a2, "rep2", "rep1")}, dict(m=m, matched=(int(a2.sum()), int(b2.sum())))
    if design == "binom_grid":
        h1, h2, m, pt = binom_split(a, b, seed, tag, tx)
        assert (h1 + h2 == pt).all() and (pt <= a + b).all()
        return {0: (h1, h2, "half1", "half2"), 1: (h2, h1, "half2", "half1")}, dict(m=m, matched=(int(h1.sum()), int(h2.sum())))
    raise ValueError(design)


@functools.lru_cache(maxsize=None)
def _grid_levels_cached(L, edges_t):
    e = np.asarray(edges_t, float); nb = len(e) - 1; out = []
    for j in range(nb):
        c = math.sqrt(e[j] * e[j + 1]) * L
        best = None
        for k in sorted({max(1, int(math.floor(c))), max(1, int(math.ceil(c)))}):
            if int(np.searchsorted(e, k / L, side="right") - 1) != j: continue
            s = abs(math.log(k / c))
            if best is None or s < best[0] - 1e-15: best = (s, k)
        if best is not None: out.append((j, best[1]))
    return tuple(out)


def grid_levels(L, edges):
    """For each bin j one integer read count k with k/L inside bin j, closest (in log) to the geometric centre
    c_j * L (ties -> smaller k). The closest integer is floor(c_j L) or ceil(c_j L), so only these are checked;
    grid_levels_bruteforce() is the exhaustive version used in the tests. Returns [(j, k)]."""
    return list(_grid_levels_cached(int(L), tuple(float(x) for x in edges)))


def grid_levels_bruteforce(L, edges):
    e = np.asarray(edges, float); nb = len(e) - 1; out = []
    for j in range(nb):
        c = math.sqrt(e[j] * e[j + 1]) * L; best = None
        for k in range(max(1, int(math.floor(e[j] * L)) - 1), int(math.ceil(e[j + 1] * L)) + 2):
            if int(np.searchsorted(e, k / L, side="right") - 1) != j: continue
            s = abs(math.log(k / c))
            if best is None or s < best[0] - 1e-15: best = (s, k)
        if best is not None: out.append((j, best[1]))
    return out


def nested_thin(inp, ks, seed, stream, tag, tx):
    """Permute the reads of the input library once and keep the first k for every k (nested). Returns {k: counts}."""
    inp = np.asarray(inp, np.int64); L = len(inp); n = int(inp.sum())
    reads = np.repeat(np.arange(L), inp); perm = rng_for(seed, stream, tag, tx).permutation(n)
    out = {}
    for k in ks:
        assert 1 <= k <= n
        x = np.bincount(reads[perm[:k]], minlength=L).astype(np.int64)
        assert (x <= inp).all() and int(x.sum()) == k
        out[k] = x
    return out


# ---------------------------------------------------------------- scoring
def pearson_interior(p, y):
    """Pearson on the interior [20, L-20) in float64. Returns (r, None) or (None, reason) when the interior has
    fewer than 3 codons or p / y is non-finite or constant there."""
    L = len(y)
    if L - 2 * INTERIOR < 3: return None, "interior<3"
    yi = np.asarray(y, np.float64)[INTERIOR:L - INTERIOR]; pi = np.asarray(p, np.float64)[INTERIOR:L - INTERIOR]
    if not np.isfinite(pi).all(): return None, "pred_nonfinite"
    if yi.max() == yi.min(): return None, "truth_constant"
    if pi.max() == pi.min(): return None, "pred_constant"
    a = pi - pi.mean(); b = yi - yi.mean(); d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if not d > 0: return None, "zero_variance"
    return float((a * b).sum() / d), None


def truth_status(ref):
    L = len(ref)
    if int(np.sum(ref)) == 0: return False, "reference_zero"
    if L - 2 * INTERIOR < 3: return False, "interior<3"
    yi = ref[INTERIOR:L - INTERIOR]
    if yi.max() == yi.min(): return False, "reference_constant_interior"
    return True, None


def score_transcript(tag, tx, a, b, preds, design, seed, edges, meta):
    """All records of one transcript and both roles for the grid-thinning design (or native depth).
    preds: {predictor: prediction vector (L,)}; meta: dict(unit, fold, stratum). Returns a list of dicts."""
    L = len(a); assert len(b) == L
    for k, v in preds.items(): assert len(v) == L, (tx, k)
    libs, info = libraries(design, a, b, seed, tag, tx)
    recs = []
    for role in (0, 1):
        inp, ref, lin, lref = libs[role]
        assert lin != lref
        n_in, n_ref = int(inp.sum()), int(ref.sum())
        t_ok, t_why = truth_status(ref)
        refy = ref / ref.mean() if n_ref > 0 else ref.astype(float)
        rmod = {}
        for k, v in preds.items():
            rmod[k] = pearson_interior(v, refy) if t_ok else (None, t_why)
        base = dict(dataset=tag, design=design, thin_seed=seed, tx=tx, unit=meta["unit"], fold=meta["fold"], stratum=meta["stratum"],
                    role=role, input_lib=lin, reference_lib=lref, L=L, m_shallow=info["m"], input_total_before=n_in,
                    depth_before=n_in / L, reference_total=n_ref, has_reference=n_ref > 0, truth_scorable=t_ok, truth_reason=t_why)
        if n_in == 0:
            levels = [("zero", 0, None)]
        elif design == "native":
            d = n_in / L; j = int(np.searchsorted(np.asarray(edges), d, side="right") - 1)
            levels = [("native", n_in, j if 0 <= j < len(edges) - 1 else None)]
        else:
            lv = [(j, k) for j, k in grid_levels(L, edges) if k <= n_in]
            levels = [("grid", k, j) for j, k in lv] if lv else [("below_axis", n_in, None)]
        thinned = nested_thin(inp, [k for kind, k, j in levels if kind == "grid"], seed, 1 + role, tag, tx) if design != "native" else {}
        for kind, k, j in levels:
            x = thinned[k] if kind == "grid" else inp
            if kind == "grid": assert (x <= inp).all()
            r = dict(base, level_kind=kind, input_total_after=int(k), depth_after=(k / L), bin=j)
            if kind in ("zero", "below_axis") or (kind == "native" and j is None):
                r.update(reads_r=None, reads_reason=("input_zero" if kind == "zero" else "outside_axis"))
            else:
                xr = x / x.mean()
                rr, why = pearson_interior(xr, refy) if t_ok else (None, t_why)
                r.update(reads_r=rr, reads_reason=why)
            for kname, (rm, why) in rmod.items():
                r[f"r__{kname}"] = rm; r[f"why__{kname}"] = why
            recs.append(r)
    return recs


def check_equivalence(recs, edges, B=200, seed=0):
    return X.check_equivalence(recs, edges, B=B, seed=seed)
