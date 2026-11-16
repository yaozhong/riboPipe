"""Datasets in the RiboPipe release format (the paper's analysis input, PaperData_v1.0, reduced to the
transcripts of the paper's populations).

    ds = load_dataset("GSE133393_WT")              # Part I universe (high-coverage transcripts)
    ds = load_dataset("GSE133393_WT", "partII")    # Part II universe (high and low coverage)
    ds.ids, ds.folds, ds.test_ids(f), ds.val_ids(f), ds.train_ids(f)
    ds.seq[tx]        A-site-aligned CDS (len = 3L; codon i is the A-site codon when the A site is at codon i)
    ds.counts[tx]     per-codon A-site count (human: mean of the two replicates; non-human: released average)
    ds.truth(tx)      relative footprint density y = c / mean(c) over the full CDS, zeros included
    ds.reps(tx)       human: per-replicate integer counts (2, L); non-human: None
    ds.mfe_raw(tx)    RNAfold MFE of the 30-nt window starting at each nucleotide (len = 3L; positions >= n_valid are 0)
    ds.mfe_valid(tx)  number of valid windows, len(seq) - 29
    ds.mfe_norm(f)    (mean, sd) of the MFE over the valid windows of the training transcripts of fold f
    ds.unit[tx]       sequence-similarity unit (folds keep each unit on one side of a split)

The interface, the transcript order and every array are identical to those used for the paper
(tests/test_data_equivalence.py checks this against PaperData_v1.0 when it is available).
The data directory defaults to ./data (run from the repository) and can be set with RIBOPIPE_DATA;
the MFE tracks (release assets) default to <data dir>/mfe and can be set with RIBOPIPE_MFE.
"""
import os, json, hashlib
import numpy as np

HUMAN = ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]
NONHUMAN = ["Subtelny14_ribomimo", "Mohammad16", "Mohammad19-1", "Mohammad19-2"]
DISPLAY = {"DMSO_TX9P1": "TX9_WT", "Subtelny14_ribomimo": "Subtelny14"}
DATA_TAG = "PaperData_v1.0"
SOURCE_MANIFEST_MD5 = "63080cc9a3144c83834f938560ba6b18"


def data_dir(path=None):
    """Data directory: the argument, else $RIBOPIPE_DATA, else ./data (run from the repository), else the data/
    directory next to the package source (editable install)."""
    if path: return path
    if os.environ.get("RIBOPIPE_DATA"): return os.environ["RIBOPIPE_DATA"]
    cands = [os.path.join(os.getcwd(), "data"),
             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")]
    for c in cands:
        if os.path.exists(os.path.join(c, "MFE_NORM_FOLDWISE.json")): return c
    return cands[0]


def mfe_dir(path=None, data=None):
    if path: return path
    if os.environ.get("RIBOPIPE_MFE"): return os.environ["RIBOPIPE_MFE"]
    return os.path.join(data_dir(data), "mfe")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


class _Concat(dict):
    """Lazy {id: slice} view over a concatenated array (same values as the paper's per-transcript arrays)."""
    def __init__(s, ids, off, arr, axis=-1, cast=None):
        super().__init__(); s._i = {t: i for i, t in enumerate(ids)}; s._off, s._arr, s._axis, s._cast = off, arr, axis, cast
        for t in ids: dict.__setitem__(s, t, None)
    def __getitem__(s, t):
        i = s._i[t]; a, b = int(s._off[i]), int(s._off[i + 1])
        x = s._arr[..., a:b] if s._axis == -1 else s._arr[a:b]
        return x if s._cast is None else s._cast(x)
    def values(s): return [s[t] for t in s]
    def items(s): return [(t, s[t]) for t in s]


class PaperDataset:
    def __init__(s, tag, part="partI", data=None, mfe=None, load_mfe=True):
        if tag == "TX9_WT": tag = "DMSO_TX9P1"
        d = os.path.join(data_dir(data), tag); s.dir = d
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{d} not found" + (" (TX9_WT counts are released on publication; see data/README.md)"
                                                        if tag == "DMSO_TX9P1" else ""))
        # paper datasets: human = Part I / Part II universes with two replicates; non-human = one universe.
        # datasets made with `ribopipe prepare` use the Part I / Part II layout ("user").
        s.tag = tag
        s.group = "human" if tag in HUMAN else "nonhuman" if tag in NONHUMAN else "user"
        if s.group in ("human", "user"):
            assert part in ("partI", "partII"), part
            uj = json.load(open(f"{d}/universe_{part}.json")); fj = json.load(open(f"{d}/folds_{part}.json"))
            s.strata = uj.get("strata")
        else:
            uj = json.load(open(f"{d}/universe.json")); fj = json.load(open(f"{d}/folds.json")); s.strata = None
        s.ids = list(uj["ids"]); units = json.load(open(f"{d}/units.json"))["tx_unit"]
        s.unit = {t: units[t] for t in s.ids}
        s._folds = {fo["fold"]: fo for fo in fj["folds"]}; s.folds = sorted(s._folds)
        z = np.load(f"{d}/{tag}.npz")
        rid = [str(x) for x in z["ids"]]; off = z["offsets"]; pos = {t: i for i, t in enumerate(rid)}
        seq = z["seq"].tobytes().decode("ascii")
        s.seq, s.counts = {}, {}
        has_reps = "reps" in z.files
        s._reps = {} if has_reps else None
        reps = z["reps"] if has_reps else None; avg = None if has_reps else z["avg_count"]
        for t in s.ids:
            i = pos[t]; a, b = int(off[i]), int(off[i + 1])
            s.seq[t] = seq[3 * a:3 * b]
            if reps is not None:
                r = np.asarray(reps[:, a:b]); s._reps[t] = r; s.counts[t] = r.mean(0)
            else:
                s.counts[t] = np.asarray(avg[a:b], float)
            assert len(s.seq[t]) == 3 * len(s.counts[t]), t
        s._mfe_valid = json.load(open(f"{d}/mfe_valid.json"))["n_valid"]
        fn = json.load(open(os.path.join(data_dir(data), "MFE_NORM_FOLDWISE.json")))
        s._mfe_norm = {r["fold"]: (r["mean"], r["sd"]) for r in (fn["human"][tag]["partI"] if tag in fn.get("human", {}) else fn["nonhuman"][tag])}
        s._mfe = None
        if load_mfe:
            p = os.path.join(mfe_dir(mfe, data), f"{tag}_mfe.npz")
            if os.path.exists(p):
                m = np.load(p); mid = [str(x) for x in m["ids"]]; mp = {t: i for i, t in enumerate(mid)}
                moff, arr = m["offsets"], m["mfe"]
                s._mfe = {t: arr[int(moff[mp[t]]):int(moff[mp[t] + 1])] for t in s.ids}
        # gates: test and val disjoint, both inside the universe, each transcript tested exactly once
        allt = []
        for f in s.folds:
            te, va = set(s._folds[f]["test"]), set(s._folds[f]["val"])
            assert not (te & va) and (te | va) <= set(s.ids), (tag, f)
            allt += list(te)
        assert sorted(allt) == sorted(s.ids), tag

    def test_ids(s, f): return list(s._folds[f]["test"])
    def val_ids(s, f): return list(s._folds[f]["val"])
    def train_ids(s, f):
        ex = set(s._folds[f]["test"]) | set(s._folds[f]["val"]); return [t for t in s.ids if t not in ex]
    def truth(s, tx): c = s.counts[tx]; return c / c.mean()
    def reps(s, tx): return None if s._reps is None else s._reps[tx]
    def has_mfe(s): return s._mfe is not None
    def mfe_raw(s, tx):
        if s._mfe is None:
            raise FileNotFoundError(f"MFE tracks for {s.tag} not found; run `python tools/download_mfe.py` "
                                    "or set RIBOPIPE_MFE (see data/README.md)")
        return np.asarray(s._mfe[tx], float)
    def mfe_valid(s, tx): return int(s._mfe_valid[tx])
    def mfe_norm(s, f): return s._mfe_norm[f]
    @property
    def display_name(s): return DISPLAY.get(s.tag, s.tag)


def load_dataset(tag, part="partI", data=None, mfe=None, load_mfe=True):
    return PaperDataset(tag, part, data=data, mfe=mfe, load_mfe=load_mfe)


def write_run(out_root, config_id, dataset, seed, fold, preds, provs, meta):
    """Write one (config, dataset, seed, fold) run: pred.npz, prov.npz (0 model / 1 adapter / 2 undefined) and run.json."""
    import time, socket, sys, platform
    d = os.path.join(out_root, config_id, dataset, f"seed{seed}", f"fold{fold}"); os.makedirs(d, exist_ok=True)
    assert meta.get("status") in ("OK", "MODEL_FAILED")
    if meta["status"] == "OK":
        assert set(preds) == set(provs)
        np.savez(os.path.join(d, "pred.npz"), **{k: np.asarray(v, np.float32) for k, v in preds.items()})
        np.savez(os.path.join(d, "prov.npz"), **{k: np.asarray(v, np.int8) for k, v in provs.items()})
    paper = dataset in HUMAN + NONHUMAN + ["TX9_WT"]
    m = dict(meta); m.update(config_id=config_id, dataset=dataset, seed=seed, fold=fold, data_tag=(DATA_TAG if paper else "user"),
                            source_manifest_md5=(SOURCE_MANIFEST_MD5 if paper else None), host=socket.gethostname(),
                            python=platform.python_version(), written=time.strftime("%F %T"))
    if "torch" in sys.modules:
        t = sys.modules["torch"]; m.setdefault("torch", t.__version__); m.setdefault("cuda", getattr(t.version, "cuda", None))
    json.dump(m, open(os.path.join(d, "run.json"), "w"), indent=1)
    return d
