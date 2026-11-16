#!/usr/bin/env python3
"""Derive paper_results/partII/fig2_depth_cdf.json, the curve drawn in Figure 2 panel d.

This script is NOT needed to reproduce Figure 2 from this repository; it documents how the shipped derived file was made.
It reads the internal per-transcript Part II record tables (runs/partII/eval_v2/<dataset>/records_natural.tsv.gz, listed with
their md5 in partII_eval.json) and keeps only the depth axis of the cumulative-distribution curve, exactly as the
published generator (make_fig_part2_1115_v8.py, md5 d7255aab527f589e61ca89071941f648) computed it:
  n          number of records (transcript x role) of the main analysis, draw 0
  n_zero     number of records with depth 0 (they start the curve at n_zero / n)
  depth_pos  the sorted positive depths (reads/codon), i.e. the x positions of the steps of the empirical CDF
No transcript identifiers, no per-codon counts and no other column are written.

Usage:
  python3 paper_results/derive_fig2.py --paper /path/to/Paper_2026-11-11 --out paper_results/partII/fig2_depth_cdf.json
"""
import os, sys, json, gzip, csv, hashlib, argparse, socket, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--paper", default="/mnt/ws1/ribo_seq_work2/20261102_human_reAnalysis/2026-11-04_newSplit/Paper_2026-11-11")
ap.add_argument("--out", required=True)
a = ap.parse_args(); PAPER = a.paper
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
SRC = f"{PAPER}/report/partII_results/rerun76/partII_eval.json"
ev = json.load(open(SRC))
DS = ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]
out = {"meta": {"description": "Empirical CDF of the depth axis of the low-coverage population (Figure 2 panel d); see derive_fig2.py.",
                "derived_by": "paper_results/derive_fig2.py", "host": socket.gethostname(), "time": time.strftime("%F %T %Z"),
                "partII_eval_md5": md5(SRC), "records": {}}, "datasets": {}}
for t in DS:
    V = ev["datasets"][t]["variants"]["v2_primary"]; rec = os.path.join(PAPER, V["records_file"]); assert md5(rec) == V["records_md5"]
    dax = []
    with gzip.open(rec, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"): dax.append(float(r["d_axis"]))
    v = np.array(sorted(dax)); n = len(v); pos = np.sort(v[v > 0]); nz = n - len(pos)
    out["meta"]["records"][t] = {"file": V["records_file"], "md5": V["records_md5"]}
    out["datasets"][t] = {"n": int(n), "n_zero": int(nz), "depth_pos": [float(x) for x in pos]}
    print(t, n, nz, flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w"))
print("wrote", a.out)
