#!/usr/bin/env python3
"""Evaluation setting 3 against the paper's runs (authors only; needs the original run directory):
parameter counts and structure of every variant, bitwise equivalence of the full variant with RiboPipe, and
reproduction of the paper's test predictions from the paper's variant checkpoints."""
import os, sys, json, argparse
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--report", required=True); a = ap.parse_args()
    import torch
    from ribopipe import ablation as AB, train as RP, data as D
    rep = {"gate_equivalence": AB.gate_equivalence(), "structure": {}, "ckpt": {}}
    for cid, cfg in AB.ABLATION_CONFIGS.items():
        m = AB.build_model(cfg, -4.0, 3.0)
        try: g = AB.structure_gate(m, cfg); rep["structure"][cid] = dict(params=g["params"], status="PASS")
        except SystemExit as e: rep["structure"][cid] = dict(error=str(e), status="FAIL")
    print("structure:", sum(v["status"] == "PASS" for v in rep["structure"].values()), "/", len(rep["structure"]), flush=True)
    cases = [("Letter_E1_A2_noNT", "GSE133393_WT", 0), ("Letter_E1_A5_noRecur_pm", "Mohammad16", 1),
             ("Letter_E1_A6_noConv", "PRJNA_iPS", 2), ("Letter_E2_BiLSTM_pm", "GSE233886_WT", 3),
             ("Letter_E2_TF2_pm", "Mohammad19-1", 4), ("Letter_E12nm_CA", "DMSO_TX9P1", 0), ("Letter_E3_frac0.2_d0", "GSE133393_WT", 1)]
    for cid, tag, fold in cases:
        d = os.path.join(a.runs, cid, tag, "seed123", f"fold{fold}")
        if not os.path.exists(os.path.join(d, "model.pt")):
            rep["ckpt"][f"{cid}/{tag}/fold{fold}"] = dict(status="MISSING"); continue
        m, cfg = AB.load_ckpt(os.path.join(d, "model.pt"))
        ds = D.load_dataset(tag); ref = np.load(os.path.join(d, "pred.npz")); ids = ds.test_ids(fold)
        assert sorted(ids) == sorted(ref.files)
        pred = RP.predict_transcripts(m, ds, ids, cfg, a.device)
        same = all(np.array_equal(pred[k], ref[k]) for k in ids); dmax = max(float(np.abs(pred[k] - ref[k]).max()) for k in ids)
        rep["ckpt"][f"{cid}/{tag}/fold{fold}"] = dict(n=len(ids), bitwise_equal=same, max_abs_diff=dmax, status="PASS" if same else "FAIL")
        print(f"[ckpt] {cid} {tag} fold{fold}: bitwise={same} max|d|={dmax:.3g}", flush=True)
    rep["all_pass"] = rep["gate_equivalence"]["ok"] and all(v["status"] == "PASS" for c in ("structure", "ckpt") for v in rep[c].values())
    json.dump(rep, open(a.report, "w"), indent=1); print("ALL PASS" if rep["all_pass"] else "SOME FAIL", rep["gate_equivalence"])
