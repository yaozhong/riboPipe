#!/usr/bin/env python3
"""Build the slim, pickle-free release files of RiboPipe v2 from the frozen analysis input PaperData_v1.0
(manifest MD5 63080cc9a3144c83834f938560ba6b18). Run by the authors; users download the result.

For each dataset it keeps only the transcripts of the paper's populations (human: Part II universe, which contains
the Part I universe; non-human: the benchmark universe) and writes

  data/<TAG>/<TAG>.npz          ids, CDS sequences, per-codon A-site counts (human: two replicates, int32;
                                non-human: released average, float64), concatenated with offsets
  data/<TAG>/*.json             universes, folds (test/val ids per fold), similarity units, MFE valid-window counts,
                                and (human) the Part II reference-depth table, copied unchanged
  data/MFE_NORM_FOLDWISE.json, data/REFERENCE_PROTOCOL.json   copied unchanged
  <assets>/<TAG>_mfe.npz        RNAfold MFE tracks of the same transcripts (release asset, not in git)
  data/RELEASE_MANIFEST.json    md5 of every written file and of every PaperData_v1 source file used

Usage: build_release_data.py --paperdata DIR --supp DIR --out DATA_DIR --assets ASSET_DIR [--exclude DMSO_TX9P1]
"""
import os, json, argparse, hashlib, shutil
import numpy as np

HUMAN = ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]
NONHUMAN = ["Subtelny14_ribomimo", "Mohammad16", "Mohammad19-1", "Mohammad19-2"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paperdata", required=True); ap.add_argument("--supp", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--assets", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    assert open(f"{a.paperdata}/TAG").read().strip() == "PaperData_v1.0"
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.assets, exist_ok=True)
    man = {"source_tag": "PaperData_v1.0", "source_manifest_md5": md5(f"{a.paperdata}/MANIFEST.json"),
           "excluded": a.exclude, "sources": {}, "files": {}, "assets": {}}
    for g in ("MFE_NORM_FOLDWISE.json", "REFERENCE_PROTOCOL.json"):
        shutil.copyfile(f"{a.paperdata}/{g}", f"{a.out}/{g}"); man["sources"][g] = md5(f"{a.paperdata}/{g}")
    hu = json.load(open(f"{a.paperdata}/human_units.json")); man["sources"]["human_units.json"] = md5(f"{a.paperdata}/human_units.json")
    tx2gene = {tx: g for g, tx in hu["rep_by_gene"].items()}
    for tag in HUMAN + NONHUMAN:
        if tag in a.exclude: continue
        grp = "human" if tag in HUMAN else "nonhuman"; src = f"{a.paperdata}/{grp}/{tag}"; dst = f"{a.out}/{tag}"
        os.makedirs(dst, exist_ok=True)
        jsons = (["universe_partI.json", "universe_partII.json", "folds_partI.json", "folds_partII.json", f"{tag}_partII_reference_depth.json"]
                 if grp == "human" else ["universe.json", "folds.json"])
        for j in jsons:
            shutil.copyfile(f"{src}/{j}", f"{dst}/{j}"); man["sources"][f"{grp}/{tag}/{j}"] = md5(f"{src}/{j}")
        if grp == "human":
            u1 = json.load(open(f"{src}/universe_partI.json"))["ids"]; ids = list(json.load(open(f"{src}/universe_partII.json"))["ids"])
            assert set(u1) <= set(ids), tag
            units = {t: hu["gene_unit"][tx2gene[t]] for t in ids}
        else:
            ids = list(json.load(open(f"{src}/universe.json"))["ids"])
            sup = f"{a.supp}/nonhuman_units_{tag}.json"; units = {t: json.load(open(sup))["tx_unit"][t] for t in ids}
            man["sources"][f"supp/nonhuman_units_{tag}.json"] = md5(sup)
        json.dump({"tx_unit": units}, open(f"{dst}/units.json", "w"))
        z = np.load(f"{src}/{tag}.npz", allow_pickle=True); man["sources"][f"{grp}/{tag}/{tag}.npz"] = md5(f"{src}/{tag}.npz")
        seqs, avg, L = [], [], []
        for t in ids:
            e = z[t].item()["cds"]; s = e["sequence"].upper(); c = np.asarray(e["avg_count"], float)
            assert len(s) == 3 * len(c) and set(s) <= set("ACGT"), t
            seqs.append(s); avg.append(c); L.append(len(c))
        off = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
        out = {"ids": np.array(ids), "offsets": off, "seq": np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)}
        if grp == "human":
            r = np.load(f"{src}/{tag}_reps.npz"); man["sources"][f"{grp}/{tag}/{tag}_reps.npz"] = md5(f"{src}/{tag}_reps.npz")
            reps = [np.asarray(r[t]) for t in ids]
            for t, x, c in zip(ids, reps, avg):
                assert x.shape == (2, len(c)) and x.dtype == np.int32, t
                assert np.array_equal(x.mean(0), c), f"avg_count != mean of replicates: {t}"   # avg is derived exactly
            out["reps"] = np.concatenate(reps, axis=1)
        else:
            out["avg_count"] = np.concatenate(avg).astype(np.float64)
        np.savez_compressed(f"{dst}/{tag}.npz", **out)
        mv = json.load(open(f"{src}/{tag}_mfe_valid.json")); man["sources"][f"{grp}/{tag}/{tag}_mfe_valid.json"] = md5(f"{src}/{tag}_mfe_valid.json")
        json.dump({"n_valid": {t: mv["n_valid"][t] for t in ids}}, open(f"{dst}/mfe_valid.json", "w"))
        m = np.load(f"{src}/{tag}_mfe.npz"); man["sources"][f"{grp}/{tag}/{tag}_mfe.npz"] = md5(f"{src}/{tag}_mfe.npz")
        tr = [np.asarray(m[t]) for t in ids]; dt = {x.dtype for x in tr}; assert len(dt) == 1, dt
        for t, x, s in zip(ids, tr, seqs): assert len(x) == len(s), t
        moff = np.concatenate([[0], np.cumsum([len(x) for x in tr])]).astype(np.int64)
        np.savez_compressed(f"{a.assets}/{tag}_mfe.npz", ids=np.array(ids), offsets=moff, mfe=np.concatenate(tr))
        print(tag, "n =", len(ids), "codons =", int(off[-1]), flush=True)
    for root, _, fs in os.walk(a.out):
        for f in sorted(fs):
            p = os.path.join(root, f)
            if f != "RELEASE_MANIFEST.json": man["files"][os.path.relpath(p, a.out)] = md5(p)
    for f in sorted(os.listdir(a.assets)): man["assets"][f] = md5(os.path.join(a.assets, f))
    json.dump(man, open(f"{a.out}/RELEASE_MANIFEST.json", "w"), indent=1)


if __name__ == "__main__":
    main()
