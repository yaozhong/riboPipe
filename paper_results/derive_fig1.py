#!/usr/bin/env python3
"""Derive paper_results/partI/fig1_derived.json, the per-transcript-free summary that reproduce/figures/fig1.py draws.

This script is NOT needed to reproduce Figure 1 from this repository; it documents how the shipped derived file was made.
It needs the internal Part I artefacts (PaperData_v1 count matrices, the out-of-fold prediction runs and the paper_eval
scoring library), which are not part of the public release. The computation is copied line for line from the published
figure generator (make_fig_partI_main_1115_v5.py, md5 9ce1e6da027ebe973d80f6b4185666c2).

What it writes (aggregate values only, no per-codon or per-transcript values):
  panel_a1   for each dataset x method x metric (interior per-transcript Pearson, top-5% recall): the five per-fold medians
             and the number of scored transcripts; fig1.py takes the mean and SD of the five fold medians.
  params     trainable parameter counts of the neural methods (legend), read from each method's run.json.
  panel_a2   TX9_WT pooled-site statistics (pooled r, median per-transcript r, n transcripts, n codons) and the 2-D
             histogram drawn as hexbins: the centre and the count of every non-empty hexagon (gridsize 32, log1p axes,
             extent [0, log1p(60)]^2). These are counts of codons per hexagon, not per-codon values.

Usage (on the machine holding the Part I artefacts, with the interpreter the figure was made with):
  python3 paper_results/derive_fig1.py --paper /path/to/Paper_2026-11-11 --out paper_results/partI/fig1_derived.json
"""
import os, sys, json, glob, argparse, hashlib, socket, time
sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument("--paper", default="/mnt/ws1/ribo_seq_work2/20261102_human_reAnalysis/2026-11-04_newSplit/Paper_2026-11-11")
ap.add_argument("--out", required=True)
a = ap.parse_args()
PAPER = a.paper
sys.path.insert(0, f"{PAPER}/paper_eval")
import numpy as np, paperdata_io as io, score_partI as SP
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RUNS = f"{PAPER}/runs/partI"
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
METH = ["Paper_CodonMean", "Paper_TriCodon", "Paper_Ridge", "Paper_RUST", "Paper_iXnos_MFE_lr005_elig", "Paper_RiboMIMO",
        "Paper_RiboGL_heldout", "Paper_RiboExp", "Paper_BiLSTM97", "Paper_RiboPipe_MFE_H128"]
OURS = "Paper_RiboPipe_MFE_H128"
DS = io.HUMAN + io.NONHUMAN
NEURAL = ("Paper_iXnos_MFE_lr005_elig", "Paper_RiboMIMO", "Paper_RiboGL_heldout", "Paper_RiboExp", "Paper_BiLSTM97", OURS)


def _nparams(cfg):   # identical to the figure generator
    fs = sorted(glob.glob(f"{RUNS}/{cfg}/**/run.json", recursive=True)); d = json.load(open(fs[0]))
    if "params_trainable" in d: return d["params_trainable"], fs[0]
    if "n_params" in d: return d["n_params"], fs[0]
    if "params" in d: return d["params"]["predictive_path_params"], fs[0]
    if "input_dim" in d: return d["input_dim"] * 200 + 200 + 200 + 1, fs[0]


NPAR, NPAR_SRC = {}, {}
for c in NEURAL: NPAR[c], NPAR_SRC[c] = _nparams(c)
assert NPAR[OURS] == 228321

A1 = {}
for d in DS:
    pd_ = io.load_dataset(d); fold = {t: f for f in pd_.folds for t in pd_.test_ids(f)}; ds = SP.Dataset(fold, pd_.unit, pd_.counts)
    R = {}
    for c in METH:
        runs = SP.load_method_runs(RUNS, c, d, ["det"] if os.path.isdir(f"{RUNS}/{c}/{d}/seeddet") else ["123"], range(5))
        out = {}
        for metric in ("pearson", "recall5"):
            failed, per, _ = SP.method_table(ds, runs, "interior", metric)
            s = list(per)[0]
            sets = {f: SP.scorable_set(ds, [per], "interior", f) for f in ds.folds}
            fm = [float(np.median([per[s][t] for t in sets[f]])) for f in ds.folds]
            out[metric] = {"folds": fm, "n": int(sum(len(v) for v in sets.values()))}
        R[c] = out
    A1[d] = R
    print(d, flush=True)

# TX9_WT pooled sites (panel a-2)
d = "DMSO_TX9P1"; pd_ = io.load_dataset(d)
runs = SP.load_method_runs(RUNS, OURS, d, ["123"], range(5)); X, Y, rr = [], [], []
for f in range(5):
    st, pred, prov = runs["123"][f]
    for t in pd_.test_ids(f):
        y = pd_.truth(t); L = len(y); I = SP.domain_index(L, "interior")
        if not SP.truth_scorable(y, I) or t not in pred: continue
        p = np.asarray(pred[t], float)[I]
        if not np.isfinite(p).all() or p.std() == 0: continue
        X.append(y[I]); Y.append(p); rr.append(SP.pearson(p, y[I]))
X = np.concatenate(X); Y = np.concatenate(Y)
pooled = float(np.corrcoef(X, Y)[0, 1]); med = float(np.median(rr)); ntx = len(rr); ncod = int(len(X))
# the hexbin exactly as the figure generator calls it; keep only the non-empty hexagons (centre, count)
fig, ax = plt.subplots()
EXT = (0, np.log1p(60), 0, np.log1p(60))
hb = ax.hexbin(np.log1p(X), np.log1p(Y), gridsize=32, bins="log", cmap="magma", mincnt=1, extent=EXT, linewidths=0)
off = np.asarray(hb.get_offsets(), float); cnt = np.asarray(hb.get_array(), float)
assert off.shape[0] == cnt.shape[0] and np.isfinite(cnt).all() and (cnt >= 1).all() and (cnt == np.round(cnt)).all()
plt.close(fig)
out = {"meta": {"description": "Aggregate values drawn in Figure 1 panels a-1 and a-2 (see derive_fig1.py docstring).",
                "derived_by": "paper_results/derive_fig1.py", "host": socket.gethostname(), "time": time.strftime("%F %T %Z"),
                "python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "protocol": {"file": os.path.basename(io.PROTOCOL), "md5": md5(io.PROTOCOL)}, "paperdata_manifest_md5": io.MANIFEST_MD5,
                "params_source": {c: os.path.relpath(p, PAPER) for c, p in NPAR_SRC.items()},
                "hexbin_codons_in_extent": int(cnt.sum())},
       "panel_a1": A1, "params_legend": NPAR,
       "panel_a2": {"pooled_r": pooled, "median_tx_r": med, "n_tx": ntx, "n_codons": ncod,
                    "hexbin": {"gridsize": 32, "extent": [float(x) for x in EXT], "centres": off.tolist(), "counts": [int(x) for x in cnt]}}}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out, "hexagons", len(cnt), "codons in extent", int(cnt.sum()), "of", ncod)
