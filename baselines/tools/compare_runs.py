"""Compare reproduced runs with reference runs, array by array.

  python baselines/tools/compare_runs.py --ref <paper runs root> --new <reproduced runs root> --config Paper_Ridge \
      --datasets GSE133393_WT Mohammad16 --seed det [--json out.json]

For every (dataset, fold): same transcript set, same lengths, max |pred_new - pred_ref|, number of bitwise-identical
transcripts, identical provenance arrays.  Exit 0 when every fold is within --atol (default 0: bitwise).
"""
import argparse, json, os, sys
import numpy as np


def compare(ref, new, config, dataset, seed, fold):
    dr = os.path.join(ref, config, dataset, f"seed{seed}", f"fold{fold}"); dn = os.path.join(new, config, dataset, f"seed{seed}", f"fold{fold}")
    out = {"dataset": dataset, "fold": fold}
    for d, k in ((dr, "ref"), (dn, "new")):
        if not os.path.exists(os.path.join(d, "pred.npz")):
            out["error"] = f"missing {k} pred.npz: {d}"; return out
    pr, pn = np.load(os.path.join(dr, "pred.npz")), np.load(os.path.join(dn, "pred.npz"))
    qr, qn = np.load(os.path.join(dr, "prov.npz")), np.load(os.path.join(dn, "prov.npz"))
    kr, kn = set(pr.files), set(pn.files)
    out.update(n_ref=len(kr), n_new=len(kn), same_ids=(kr == kn))
    common = sorted(kr & kn)
    maxabs, maxrel, n_bit, n_len_mismatch, n_prov_diff = 0.0, 0.0, 0, 0, 0
    for t in common:
        a, b = pr[t], pn[t]
        if a.shape != b.shape: n_len_mismatch += 1; continue
        if np.array_equal(a, b, equal_nan=True): n_bit += 1
        fin = np.isfinite(a) & np.isfinite(b)
        if (np.isfinite(a) != np.isfinite(b)).any(): maxabs = float("inf")
        if fin.any():
            d = np.abs(a[fin].astype(np.float64) - b[fin].astype(np.float64))
            maxabs = max(maxabs, float(d.max()))
            maxrel = max(maxrel, float((d / np.maximum(np.abs(a[fin].astype(np.float64)), 1e-12)).max()))
        if not np.array_equal(qr[t], qn[t]): n_prov_diff += 1
    out.update(n_common=len(common), n_bitwise_identical=n_bit, n_length_mismatch=n_len_mismatch,
               n_prov_diff=n_prov_diff, max_abs_diff=maxabs, max_rel_diff=maxrel)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--new", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", nargs="+", required=True); ap.add_argument("--seed", default="det")
    ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res, ok = [], True
    for d in a.datasets:
        for f in [int(x) for x in a.folds.split(",")]:
            r = compare(a.ref, a.new, a.config, d, a.seed, f)
            r["pass"] = ("error" not in r and r["same_ids"] and r["n_length_mismatch"] == 0 and r["n_prov_diff"] == 0
                         and r["max_abs_diff"] <= a.atol)
            ok &= r["pass"]; res.append(r); print(json.dumps(r), flush=True)
    if a.json:
        json.dump({"config": a.config, "atol": a.atol, "results": res, "all_pass": ok}, open(a.json, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
