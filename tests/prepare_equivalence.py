#!/usr/bin/env python3
"""`ribopipe prepare` applies the paper's rules (authors' check on the released data):
(1) fold_and_val on the paper's similarity units gives the paper's test and validation sets;
(2) the fold-wise MFE normalisation recomputed from the released tracks equals MFE_NORM_FOLDWISE.json;
(3) mfe_track (ViennaRNA) recomputes the released MFE tracks bit for bit on sampled transcripts."""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ribopipe import data as D, prepare as PR

ap = argparse.ArgumentParser(); ap.add_argument("--report", required=True); ap.add_argument("--n-mfe", type=int, default=20)
ap.add_argument("--paperdata", help="PaperData_v1 directory (for the full human unit table)")
a = ap.parse_args(); rep = {}
fn = json.load(open(os.path.join(D.data_dir(), "MFE_NORM_FOLDWISE.json")))
GROUPS = {"Subtelny14_ribomimo": ["Subtelny14_ribomimo"], "Mohammad16": ["Mohammad16", "Mohammad19-1", "Mohammad19-2"]}
GROUPS.update({"Mohammad19-1": GROUPS["Mohammad16"], "Mohammad19-2": GROUPS["Mohammad16"]})
if a.paperdata:   # human: units were built on all v46 genes; fold_and_val must reproduce the paper's unit -> fold map
    hu = json.load(open(os.path.join(a.paperdata, "human_units.json")))
    uf, _ = PR.fold_and_val(hu["gene_unit"].values())
    rep["human_unit_fold"] = dict(n_units=len(uf), status="PASS" if {str(k): v for k, v in uf.items()} == {str(k): v for k, v in hu["unit_fold"].items()} else "FAIL")
    print("human_unit_fold", rep["human_unit_fold"], flush=True)
for tag in [t for t in D.HUMAN + D.NONHUMAN if os.path.isdir(os.path.join(D.data_dir(), t))]:
    ds = D.load_dataset(tag); r = {}
    if tag in D.HUMAN:
        r["test_by_unit"] = all(len({f for f in ds.folds if t in set(ds.test_ids(f))}) == 1 for t in ds.ids)
    else:   # non-human: folds use units built on the union of the CDS of one species' datasets (node = sha1 of the CDS),
            # rebuilt here with prepare.kmer_pairs / components / fold_and_val exactly as for the paper
        import hashlib
        seqs = {}
        for m in GROUPS[tag]:
            dm = D.load_dataset(m, load_mfe=False)
            for t in dm.ids: seqs[hashlib.sha1(dm.seq[t].encode()).hexdigest()] = dm.seq[t]
        nodes = sorted(seqs); pr = PR.kmer_pairs([seqs[x] for x in nodes], list(range(len(nodes))))
        un = PR.components(nodes, [(nodes[i], nodes[j]) for (i, j), c in pr.items() if c >= PR.TH])
        uf, val = PR.fold_and_val(un.values())
        node = {t: hashlib.sha1(ds.seq[t].encode()).hexdigest() for t in ds.ids}
        r["folds_reproduced"] = all(sorted(t for t in ds.ids if uf[un[node[t]]] == f) == sorted(ds.test_ids(f)) and
                                    sorted(t for t in ds.ids if un[node[t]] in val[f]) == sorted(ds.val_ids(f)) for f in ds.folds)
    ref = {x["fold"]: x for x in (fn["human"][tag]["partI"] if tag in D.HUMAN else fn["nonhuman"][tag])}
    dif = []
    for f in ds.folds:
        v = np.concatenate([ds.mfe_raw(t)[:ds.mfe_valid(t)] for t in ds.train_ids(f) if ds.mfe_valid(t) > 0])
        dif.append(max(abs(v.mean() - ref[f]["mean"]), abs(v.std() - ref[f]["sd"])))
    r["mfe_norm_max_abs_diff"] = float(max(dif))
    rng = np.random.RandomState(0); pick = list(rng.choice(ds.ids, min(a.n_mfe, len(ds.ids)), replace=False))
    r["mfe_track_bitwise"] = all(np.array_equal(PR.mfe_track(ds.seq[t]).astype(float), ds.mfe_raw(t)) for t in pick)
    r["status"] = "PASS" if (r.get("folds_reproduced", True) and r.get("test_by_unit", True) and r["mfe_norm_max_abs_diff"] < 1e-9
                             and r["mfe_track_bitwise"]) else "FAIL"
    rep[tag] = r; print(tag, r, flush=True)
json.dump(rep, open(a.report, "w"), indent=1); print("ALL PASS" if all(v["status"] == "PASS" for v in rep.values()) else "SOME FAIL")
