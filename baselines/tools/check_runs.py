"""Structural check of written runs (no scoring): run.json status, pred/prov keys = the fold's test ids (or the smoke
subset), lengths = L, provenance codes in {0, 1, 2}, interior [20, L-20) all model, finite predictions on model positions.

  python baselines/tools/check_runs.py --runs <root> --config <ID> --dataset <TAG> --seed <k|det> --folds 0 [--subset]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ribopipe import data as D  # noqa: E402


def check(root, config, dataset, seed, fold, subset=False, data=None):
    d = os.path.join(root, config, dataset, f"seed{seed}", f"fold{fold}")
    r = {"config": config, "dataset": dataset, "seed": seed, "fold": fold, "dir": d}
    if not os.path.exists(os.path.join(d, "run.json")):
        r["error"] = "no run.json"; r["pass"] = False; return r
    m = json.load(open(os.path.join(d, "run.json"))); r["status"] = m.get("status")
    if r["status"] != "OK":
        r["pass"] = False; r["failure_reason"] = m.get("failure_reason"); return r
    ds = D.load_dataset(dataset, data=data, load_mfe=False)
    p, q = np.load(os.path.join(d, "pred.npz")), np.load(os.path.join(d, "prov.npz"))
    te = set(ds.test_ids(fold)); keys = set(p.files)
    r["n_tx"] = len(keys); r["keys_ok"] = (keys <= te) if subset else (keys == te)
    bad_len = bad_code = bad_interior = nonfinite_model = flat = 0
    for t in keys:
        L = len(ds.seq[t]) // 3; a, b = p[t], q[t]
        if len(a) != L or len(b) != L: bad_len += 1; continue
        if not set(np.unique(b).tolist()) <= {0, 1, 2}: bad_code += 1
        if L > 40 and (b[20:L - 20] != 0).any(): bad_interior += 1
        if not np.isfinite(a[b == 0]).all(): nonfinite_model += 1
        if L > 40 and np.std(a[20:L - 20]) == 0: flat += 1
    r.update(bad_length=bad_len, bad_prov_code=bad_code, interior_not_model=bad_interior, nonfinite_on_model=nonfinite_model,
             flat_interior=flat, prov_counts={str(k): int(sum(int((q[t] == k).sum()) for t in keys)) for k in (0, 1, 2)})
    r["pass"] = bool(r["keys_ok"] and bad_len == 0 and bad_code == 0 and bad_interior == 0 and nonfinite_model == 0 and len(keys) > 0)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--config", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", default="123"); ap.add_argument("--folds", default="0"); ap.add_argument("--subset", action="store_true")
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv); ok = True
    for f in [int(x) for x in a.folds.split(",")]:
        r = check(a.runs, a.config, a.dataset, a.seed, f, a.subset, a.data); ok &= r["pass"]; print(json.dumps(r), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
