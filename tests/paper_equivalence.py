#!/usr/bin/env python3
"""Equivalence of this release with the code and data that produced the paper (run by the authors; needs the
original analysis directory, given with --paper). Each check writes PASS/FAIL to the JSON report.

  data      every array of every dataset (ids and their order, sequences, counts, replicate counts, MFE tracks,
            valid-window counts, fold-wise MFE normalisation, similarity units, folds) equals PaperData_v1.0
  ckpt      the paper's checkpoints load into ribopipe.arch unchanged and reproduce the paper's test predictions
  retrain   retraining with ribopipe.train (same seed and machine) reproduces the paper's run bit for bit
"""
import os, sys, json, argparse, importlib.util, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load_paper_io(paper):
    spec = importlib.util.spec_from_file_location("paperdata_io", os.path.join(paper, "paper_eval", "paperdata_io.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def check_data(paper, tags):
    from ribopipe import data as D
    P = load_paper_io(paper); out = {}
    for tag in tags:
        parts = ["partI", "partII"] if tag in D.HUMAN else ["partI"]
        for part in parts:
            a = P.load_dataset(tag, part); b = D.load_dataset(tag, part)
            bad = []
            if a.ids != b.ids: bad.append("ids/order")
            if a.folds != b.folds: bad.append("folds")
            for f in a.folds:
                if a.test_ids(f) != b.test_ids(f) or a.val_ids(f) != b.val_ids(f) or a.train_ids(f) != b.train_ids(f): bad.append(f"split{f}")
                if tuple(a.mfe_norm(f)) != tuple(b.mfe_norm(f)): bad.append(f"mfe_norm{f}")
            for t in a.ids:
                if a.seq[t] != b.seq[t]: bad.append(f"seq:{t}"); break
                if not np.array_equal(a.counts[t], b.counts[t]) or a.counts[t].dtype != b.counts[t].dtype: bad.append(f"counts:{t}"); break
                if (a.reps(t) is None) != (b.reps(t) is None) or (a.reps(t) is not None and not np.array_equal(a.reps(t), b.reps(t))): bad.append(f"reps:{t}"); break
                if not np.array_equal(a.mfe_raw(t), b.mfe_raw(t)): bad.append(f"mfe:{t}"); break
                if a.mfe_valid(t) != b.mfe_valid(t): bad.append(f"n_valid:{t}"); break
                if a.unit[t] != b.unit[t]: bad.append(f"unit:{t}"); break
                if not np.array_equal(a.truth(t), b.truth(t)): bad.append(f"truth:{t}"); break
            out[f"{tag}/{part}"] = dict(n=len(a.ids), status="PASS" if not bad else "FAIL", issues=bad[:5])
            print(f"[data] {tag}/{part} n={len(a.ids)} {'PASS' if not bad else 'FAIL ' + str(bad[:5])}", flush=True)
    return out


def check_ckpt(paper, runs, cases, device):
    import torch
    from ribopipe import train as RP, data as D
    out = {}
    for config_id, tag, seed, fold in cases:
        d = os.path.join(paper, runs, config_id, tag, f"seed{seed}", f"fold{fold}")
        m, ck = RP.load_ckpt(os.path.join(d, "model.pt"))
        assert RP.count_params(m) == RP.CONFIGS[config_id.replace("_sameCkpt", "")]["expected_params"]
        ds = D.load_dataset(tag)
        ref = np.load(os.path.join(d, "pred.npz"))
        ids = ds.test_ids(fold); assert sorted(ids) == sorted(ref.files)
        pred = RP.predict_transcripts(m, ds, ids, ck["config"], device)
        dmax = max(float(np.max(np.abs(pred[k] - ref[k]))) for k in ids)
        same = all(np.array_equal(pred[k], ref[k]) for k in ids)
        # effect on the score: per-transcript Pearson with the truth over interior codons [20, L-20)
        def r(p, y):
            p, y = np.asarray(p, float), np.asarray(y, float)
            return float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 and y.std() > 0 else np.nan
        ra, rb = [], []
        for k in ids:
            L = len(ref[k])
            if L > 40:
                y = ds.truth(k)[20:L - 20]; ra.append(r(ref[k][20:L - 20], y)); rb.append(r(pred[k][20:L - 20], y))
        ra, rb = np.array(ra), np.array(rb); ok = np.isfinite(ra) & np.isfinite(rb)
        dr = float(np.max(np.abs(ra[ok] - rb[ok]))); dmed = abs(float(np.median(ra[ok]) - np.median(rb[ok])))
        status = "PASS" if (same or (device == "cpu" and dr < 1e-3 and dmed < 1e-4)) else "FAIL"
        out[f"{config_id}/{tag}/seed{seed}/fold{fold}"] = dict(device=device, n=len(ids), bitwise_equal=same, max_abs_diff=dmax,
                                                               max_abs_diff_tx_pearson=dr, abs_diff_median_pearson=dmed, status=status)
        print(f"[ckpt] {config_id} {tag} seed{seed} fold{fold} on {device}: n={len(ids)} bitwise={same} max|d|={dmax:.3g} "
              f"max|d r_tx|={dr:.2g} |d median r|={dmed:.2g}", flush=True)
    return out


def check_retrain(paper, runs, cases, device, scratch):
    from ribopipe import cli
    out = {}
    for config_id, tag, seed, fold in cases:
        t0 = time.time()
        rc = cli._run_one(config_id, tag, fold, seed, scratch, device=device)
        a = np.load(os.path.join(paper, runs, config_id, tag, f"seed{seed}", f"fold{fold}", "pred.npz"))
        b = np.load(os.path.join(scratch, config_id, tag, f"seed{seed}", f"fold{fold}", "pred.npz"))
        same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
        dmax = max(float(np.max(np.abs(a[k] - b[k]))) for k in a.files)
        ra = json.load(open(os.path.join(paper, runs, config_id, tag, f"seed{seed}", f"fold{fold}", "run.json")))
        rb = json.load(open(os.path.join(scratch, config_id, tag, f"seed{seed}", f"fold{fold}", "run.json")))
        out[f"{config_id}/{tag}/seed{seed}/fold{fold}"] = dict(rc=rc, bitwise_equal=same, max_abs_diff=dmax,
            selected_epoch=[ra.get("selected_epoch"), rb.get("selected_epoch")], val_best=[ra.get("val_metric_best"), rb.get("val_metric_best")],
            n_steps=[ra.get("n_optimizer_steps"), rb.get("n_optimizer_steps")], sec=round(time.time() - t0, 1),
            status="PASS" if same else "FAIL")
        print(f"[retrain] {config_id} {tag} seed{seed} fold{fold}: bitwise={same} max|d|={dmax:.3g} "
              f"epoch {ra.get('selected_epoch')}/{rb.get('selected_epoch')}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True); ap.add_argument("--runs", default="runs/partI")
    ap.add_argument("--checks", nargs="+", default=["data", "ckpt", "retrain"])
    ap.add_argument("--tags", nargs="+", default=["GSE233886_WT", "GSE133393_WT", "PRJNA_iPS", "Subtelny14_ribomimo",
                                                  "Mohammad16", "Mohammad19-1", "Mohammad19-2"])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--scratch", default="/tmp/ribopipe_equiv")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    rep = {"checks": {}}
    ck_cases = [("Paper_RiboPipe_MFE_H128", t, 123, f) for t in ("GSE133393_WT", "Mohammad16") for f in (0, 3)] + \
               [("Paper_RiboPipe_noMFE_H128", "PRJNA_iPS", 123, 1), ("Paper_BiLSTM97", "GSE233886_WT", 123, 2)]
    if "data" in a.checks: rep["checks"]["data"] = check_data(a.paper, a.tags)
    if "ckpt" in a.checks:
        rep["checks"]["ckpt_cpu"] = check_ckpt(a.paper, a.runs, ck_cases, "cpu")
        rep["checks"]["ckpt_gpu"] = check_ckpt(a.paper, a.runs, ck_cases, a.device)
    if "retrain" in a.checks:
        rep["checks"]["retrain"] = check_retrain(a.paper, a.runs, [("Paper_RiboPipe_MFE_H128", "Mohammad16", 123, 0),
                                                                    ("Paper_RiboPipe_MFE_H128", "GSE133393_WT", 123, 0)], a.device, a.scratch)
    rep["all_pass"] = all(v["status"] == "PASS" for c in rep["checks"].values() for v in c.values())
    json.dump(rep, open(a.report, "w"), indent=1); print("ALL PASS" if rep["all_pass"] else "SOME FAIL")
