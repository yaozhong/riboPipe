"""Reading prediction vectors for the low-coverage analysis.

Accepted sources (all hold {transcript id: float32 vector of length L}):
  * a .npz file;
  * a directory holding pred.npz (the paper's runs/partII/preds/<config>/<dataset>/ layout, written by
    `ribopipe lowcov-predict`; if a meta.json with pred_md5 is present the file's md5 is checked);
  * a run directory written by ribopipe.data.write_run, i.e. <dir>/fold<f>/pred.npz with run.json status OK
    (or <dir>/seed<s>/fold<f>/...); the folds are merged (each transcript must appear exactly once).
"""
import glob
import json
import os
import zlib

import numpy as np

from ..data import md5_file


def _load_npz(fn):
    z = np.load(fn)
    return {k: np.asarray(z[k]) for k in z.files}


def _fold_dirs(d):
    fd = sorted(glob.glob(os.path.join(d, "fold*")))
    if fd: return fd
    sd = sorted(glob.glob(os.path.join(d, "seed*")))
    if len(sd) == 1: return sorted(glob.glob(os.path.join(sd[0], "fold*")))
    if len(sd) > 1: raise ValueError(f"{d}: several seed directories {sd}; pass one of them")
    return []


def load_predictions(source, ids=None, dtype=None):
    """Return ({tx: vector}, info). If ids is given, every id must be present; only those are returned."""
    info = dict(source=os.path.abspath(source))
    if os.path.isfile(source):
        pred = _load_npz(source); info.update(kind="npz", md5=md5_file(source))
    elif os.path.isfile(os.path.join(source, "pred.npz")):
        fn = os.path.join(source, "pred.npz"); m5 = md5_file(fn); pred = _load_npz(fn)
        mj = os.path.join(source, "meta.json")
        if os.path.exists(mj):
            meta = json.load(open(mj))
            if "pred_md5" in meta: assert meta["pred_md5"] == m5, f"{fn}: md5 differs from meta.json"
            assert not meta.get("smoke"), f"{fn}: smoke-test predictions"
        info.update(kind="pred_dir", md5=m5)
    else:
        fds = _fold_dirs(source)
        if not fds: raise FileNotFoundError(f"no predictions found in {source}")
        pred = {}; info.update(kind="run_dir", folds={})
        for d in fds:
            rj = os.path.join(d, "run.json")
            if os.path.exists(rj):
                st = json.load(open(rj)).get("status")
                if st != "OK": raise ValueError(f"{d}: run status {st}")
            fn = os.path.join(d, "pred.npz"); p = _load_npz(fn)
            dup = set(p) & set(pred)
            assert not dup, f"{d}: {len(dup)} transcripts already predicted by another fold"
            pred.update(p); info["folds"][os.path.basename(d)] = md5_file(fn)
    if ids is not None:
        miss = [t for t in ids if t not in pred]
        if miss: raise KeyError(f"{source}: {len(miss)} transcripts without prediction (e.g. {miss[:3]})")
        pred = {t: pred[t] for t in ids}
    if dtype is not None: pred = {t: np.asarray(v, dtype) for t, v in pred.items()}
    return pred, info


def dummy_noise(tag, seq):
    """Sequence-independent noise predictor (diagnostic control): N(0, 1) per codon, seeded per transcript.
    tag is the internal dataset key; seq is {tx: CDS sequence}."""
    return {t: np.random.default_rng([7, zlib.crc32(tag.encode()), zlib.crc32(t.encode())]).normal(size=len(s) // 3).astype(np.float32)
            for t, s in seq.items()}


def parse_pred_args(items):
    """['NAME=PATH', ...] -> [(name, path)] (order kept)."""
    out = []
    for it in items or []:
        if "=" not in it: raise ValueError(f"--pred expects NAME=PATH, got {it}")
        n, p = it.split("=", 1); out.append((n, p))
    names = [n for n, _ in out]
    assert len(set(names)) == len(names), f"duplicate predictor names {names}"
    return out
