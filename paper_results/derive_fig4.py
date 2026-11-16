#!/usr/bin/env python3
"""Derive paper_results/partIII/fig4_logos.json, the three activation logos drawn in Figure 4 panel a.

This script is NOT needed to reproduce Figure 4 from this repository; it documents how the shipped derived file was made.
The published generator (make_fig_part3_merge_1115_v5.py, md5 9dec4b57d4ed3ad7ab37965daeae799b) reads three arrays from
the internal E12 logo archive (e12_logos.npz, 2,620 arrays, 3.6 MB): keys logo|CAN|0|<branch>|<filter> for the filters
selected from xr_e12_fwd_logorep.json (nt 21, aa 19, codon 7). Each array is a positions x letters table of log2
enrichments of letters in the top-1% activating windows of one convolution filter of the full model (TX9_WT, fold 0)
relative to background; it is an aggregate, not per-codon data. The arrays are written unchanged (float32 values stored as
JSON numbers; fig4.py converts them back to float32).

Usage:
  python3 paper_results/derive_fig4.py --npz /path/to/E12_motif/e12_logos.npz --out paper_results/partIII/fig4_logos.json
"""
import os, json, hashlib, argparse, socket, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="/mnt/ws1/ribo_seq_work2/20261102_human_reAnalysis/2026-11-04_newSplit/ResponseLetter_exp_2026-11-12/analysis/E12_motif/e12_logos.npz")
ap.add_argument("--out", required=True)
a = ap.parse_args()
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
Z = np.load(a.npz)
KEYS = ["logo|CAN|0|nt|21", "logo|CAN|0|aa|19", "logo|CAN|0|codon|7"]
out = {"meta": {"description": "Activation logos drawn in Figure 4a (log2 enrichment, positions x letters); see derive_fig4.py.",
                "derived_by": "paper_results/derive_fig4.py", "host": socket.gethostname(), "time": time.strftime("%F %T %Z"),
                "source_file": os.path.basename(a.npz), "source_md5": md5(a.npz)}, "logos": {}}
for k in KEYS:
    M = Z[k]; assert M.dtype == np.float32
    L = [[float(x) for x in row] for row in M]
    assert np.array_equal(np.array(L, dtype=np.float32), M)        # exact round trip
    out["logos"][k] = {"dtype": "float32", "shape": list(M.shape), "values": L}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=0)
print("wrote", a.out)
