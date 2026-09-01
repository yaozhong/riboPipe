"""Local mRNA-structure (MFE) feature cache — ViennaRNA sliding-window fold.

The headline model (`ribopipe_nt_struct_h256`) reads a small local-structure feature
(3 dims per codon) alongside the codon/bio/nucleotide block.  That feature is derived
from a *per-nucleotide MFE track*: for every nucleotide position ``i`` we fold a fixed
30-nt window starting at ``i`` with ViennaRNA and record its minimum free energy.

Convention (verified bit-for-bit against the paper's cache files)::

    track[i] = RNA.fold(seq[i : i + 30])[1]      for i in 0 .. len(seq) - 30
    track[i] = 0.0                                for the trailing len(seq)-29 .. end

i.e. only *full* 30-nt windows are folded; the final 29 positions (which cannot host a
complete window) are left at 0.  The sequence is upper-cased and DNA ``T`` is mapped to
RNA ``U`` before folding.  Tracks are saved as ``float32`` arrays keyed by transcript id
to ``{npz_dir}/struct_cache/{npz_basename}_struct.npz`` — exactly where
:func:`ribopipe.dataset.load_struct_cache` looks for them.

ViennaRNA is an optional dependency (``pip install "ribopipe[struct]"``); it is needed
only to *generate* the cache.  Training and prediction on an existing cache do not import
it.
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np

WINDOW = 30  # local fold window length (nt); one MFE per nucleotide start position


def struct_cache_path(npz_path: str) -> str:
    """Canonical cache location for a dataset NPZ (mirrors ``load_struct_cache``)."""
    npz_dir = os.path.dirname(os.path.abspath(npz_path))
    npz_base = os.path.splitext(os.path.basename(npz_path))[0]
    return os.path.join(npz_dir, "struct_cache", f"{npz_base}_struct.npz")


def mfe_track(seq: str, win: int = WINDOW) -> np.ndarray:
    """Per-nucleotide MFE track for one CDS via a sliding ``win``-nt RNAfold.

    Returns a ``float32`` array of length ``len(seq)``; the trailing ``win-1``
    positions (no complete window) are 0.
    """
    import RNA  # imported lazily so training/prediction never require ViennaRNA

    s = seq.upper().replace("T", "U")
    n = len(s)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n - win + 1):
        _, mfe = RNA.fold(s[i:i + win])
        out[i] = mfe
    return out


def _worker(args):
    key, seq = args
    try:
        return key, mfe_track(seq)
    except Exception:
        return key, None


def compute_struct_cache(npz_path, transcript_ids=None, cache_path=None,
                         n_workers=None, win=WINDOW, verbose=True):
    """Fold every transcript's CDS and write the per-nucleotide MFE cache.

    Parameters
    ----------
    npz_path : str
        Dataset NPZ produced by ``ribopipe preprocess`` (entries carry
        ``cds.sequence``).
    transcript_ids : list[str] | None
        Which transcripts to fold; default = every key in the NPZ.
    cache_path : str | None
        Output ``.npz``; default = :func:`struct_cache_path` (the location the
        dataset loader expects).
    n_workers : int | None
        Process-pool size; default = ``cpu_count() - 2`` (min 1).
    win : int
        Fold window length in nt (default 30 — the paper setting).

    Returns
    -------
    str
        The path the cache was written to.
    """
    z = np.load(npz_path, allow_pickle=True)
    ids = list(z.files) if transcript_ids is None else list(transcript_ids)
    if cache_path is None:
        cache_path = struct_cache_path(npz_path)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 2)

    jobs = []
    for k in ids:
        if k not in z.files:
            continue
        ent = z[k].item()
        seq = ent.get("cds", {}).get("sequence", "")
        if seq:
            jobs.append((k, seq))

    if verbose:
        print(f"[struct] folding {len(jobs)} transcripts "
              f"(win={win}nt, workers={n_workers}) -> {cache_path}", flush=True)

    out = {}
    if n_workers == 1:
        for job in jobs:
            key, track = _worker(job)
            if track is not None:
                out[key] = track
    else:
        with Pool(n_workers) as pool:
            for i, (key, track) in enumerate(
                    pool.imap_unordered(_worker, jobs, chunksize=8), 1):
                if track is not None:
                    out[key] = track
                if verbose and i % 2000 == 0:
                    print(f"[struct]   {i}/{len(jobs)} folded", flush=True)

    np.savez(cache_path, **out)
    if verbose:
        print(f"[struct] done: {len(out)} tracks -> {cache_path}", flush=True)
    return cache_path
