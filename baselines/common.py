"""Shared helpers for the baseline wrappers.

All wrappers read data only through ``ribopipe.data.load_dataset`` (honours RIBOPIPE_DATA / RIBOPIPE_MFE) and write
runs only through ``ribopipe.data.write_run`` (``<out>/<config>/<dataset>/seed<k>/fold<f>/{pred.npz,prov.npz,run.json}``).
Position provenance codes: 0 = model, 1 = adapter, 2 = undefined.  Deterministic methods use the seed directory "det".
Test-transcript counts are never passed to method code: each fold hands the method a dict with train and validation
counts only.
"""
import os, sys, json, time, hashlib, platform, socket
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from ribopipe import data as D  # noqa: E402

MODEL, ADAPTER, UNDEFINED = 0, 1, 2
PREREG_SEEDS = (123, 124, 125)   # pre-registered seeds of the frozen Part I protocol (v1.3)
HUMAN, NONHUMAN = D.HUMAN, D.NONHUMAN


def upstream_dir():
    """Directory produced by baselines/fetch_upstream.sh (override with RIBOPIPE_BASELINE_UPSTREAM)."""
    return os.environ.get("RIBOPIPE_BASELINE_UPSTREAM", os.path.join(HERE, "upstream"))


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def code_md5(files):
    return {os.path.relpath(f, REPO) if f.startswith(REPO) else f: md5_file(f) for f in files if os.path.exists(f)}


def env_info():
    d = {"python": platform.python_version(), "numpy": np.__version__, "host": socket.gethostname(),
         "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE")}
    if "torch" in sys.modules:
        t = sys.modules["torch"]
        d.update(torch=t.__version__, torch_cuda=t.version.cuda,
                 gpu=(t.cuda.get_device_name(0) if t.cuda.is_available() else None))
    return d


def log_fn(prefix):
    def _log(msg): print(f"[{time.strftime('%H:%M:%S')}] {prefix} {msg}", flush=True)
    return _log


def smoke_pick(ids, n):
    """Smoke runs only: the first n sorted ids."""
    return sorted(ids)[:n] if n else list(ids)


# codon index (64 ACGT codons in ACGT order; anything else -1)
CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CIDX = {c: i for i, c in enumerate(CODONS)}


def seq_to_idx(seq, L):
    idx = np.full(L, -1, dtype=np.int64)
    for i in range(L):
        idx[i] = CIDX.get(seq[i * 3:i * 3 + 3], -1)
    return idx


def write(out_root, config_id, dataset, seed, fold, preds, provs, meta):
    """write_run plus the bookkeeping fields used by the paper runners (flat / non-finite count, provenance counts)."""
    assert meta.get("status") in ("OK", "MODEL_FAILED")
    if meta["status"] == "OK":
        meta["n_test_tx_written"] = len(preds)
        meta["n_test_tx_flat_or_nonfinite"] = int(sum(1 for p in preds.values() if not (np.isfinite(p).all() and np.std(p) > 0)))
        meta["prov_counts"] = {str(k): int(sum(int((q == k).sum()) for q in provs.values())) for k in (0, 1, 2)}
    meta.setdefault("env", env_info())
    return D.write_run(out_root, config_id, dataset, seed, fold, preds, provs, meta)


def run_exists(out_root, config_id, dataset, seed, fold):
    return os.path.exists(os.path.join(out_root, config_id, dataset, f"seed{seed}", f"fold{fold}", "run.json"))
