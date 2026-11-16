"""Out-of-fold inference on the Part II universe (low- and high-coverage transcripts) with the Part I checkpoints.

A transcript in Part II test fold f is predicted by the Part I fold-f model. Before predicting, the rule is
checked: no such transcript, and no transcript sharing its similarity unit, is in that model's training or
validation set. Inference uses only sequence and MFE (the prediction functions never read counts).
Output (paper layout): <out>/<config>/<dataset>/pred.npz and meta.json (with pred_md5).
"""
import json
import os
import platform
import socket
import time

import numpy as np

from .. import data as D


def predict_partII(run_root, dataset, out_root, config="Paper_RiboPipe_MFE_H128", seed=123, device="cpu",
                   chunk=512, data=None, mfe=None, limit_per_fold=0, log=print):
    """run_root/<config>/<dataset>/seed<seed>/fold<f>/model.pt -> out_root/<config>/<dataset>/pred.npz."""
    import torch
    from .. import train as RP
    t0 = time.time()
    ds = D.load_dataset(dataset, "partII", data=data, mfe=mfe)
    ds1 = D.load_dataset(dataset, "partI", data=data, mfe=mfe, load_mfe=False)
    fold_of = {t: f for f in ds.folds for t in ds.test_ids(f)}
    assert sorted(fold_of) == sorted(ds.ids)
    # out-of-fold rule: transcript and its similarity unit outside the fold-f model's training / validation set
    rule = {}
    for f in ds1.folds:
        trval = (set(ds1.ids) - set(ds1.test_ids(f))) | set(ds1.val_ids(f))
        trval_units = {ds1.unit[t] for t in trval}
        mine = [t for t, ff in fold_of.items() if ff == f]
        bad_tx = [t for t in mine if t in trval]; bad_unit = [t for t in mine if ds.unit[t] in trval_units]
        assert not bad_tx and not bad_unit, (dataset, f, len(bad_tx), len(bad_unit))
        rule[f] = dict(n_assigned=len(mine), n_in_model_train_or_val=0, n_unit_shared_with_train_or_val=0,
                       n_model_train_plus_val=len(trval))
    preds, ck_md5, per_fold = {}, {}, {}
    for f in ds.folds:
        ck = os.path.join(run_root, config, ds.tag, f"seed{seed}", f"fold{f}", "model.pt")
        rj = os.path.join(os.path.dirname(ck), "run.json")
        if os.path.exists(rj): assert json.load(open(rj)).get("status") == "OK", rj
        m, cki = RP.load_ckpt(ck, map_location="cpu"); m.eval(); ck_md5[str(f)] = D.md5_file(ck)
        if "fold" in cki: assert int(cki["fold"]) == f, (ck, cki["fold"])
        if "dataset" in cki: assert cki["dataset"] == ds.tag, (ck, cki["dataset"])
        ids = sorted(ds.test_ids(f), key=lambda t: len(ds.seq[t]))
        if limit_per_fold: ids = ids[:limit_per_fold]
        t1 = time.time()
        for s in range(0, len(ids), chunk):
            preds.update(RP.predict_transcripts(m, ds, ids[s:s + chunk], cki["config"], device))
        per_fold[str(f)] = dict(n=len(ids), sec=round(time.time() - t1, 1))
        log(f"[fold {f}] {len(ids)} transcripts {time.time() - t1:.0f}s")
    for t, v in preds.items(): assert len(v) == len(ds.seq[t]) // 3, t
    od = os.path.join(out_root, config, ds.tag); os.makedirs(od, exist_ok=True)
    smoke = bool(limit_per_fold)
    fn = os.path.join(od, "pred_smoke.npz" if smoke else "pred.npz")
    np.savez(fn, **preds)
    low = set(ds.strata["low"]) if ds.strata else set()
    meta = dict(config=config, dataset=ds.tag, seed=seed, n_tx=len(preds), smoke=smoke, n_low=sum(t in low for t in preds),
                checkpoint_md5=ck_md5, fold_rule=rule,
                fold_rule_text="transcript in Part II test fold f -> Part I fold-f model; not in its train/val, no shared similarity unit",
                per_fold=per_fold, pred_file=os.path.basename(fn), pred_md5=D.md5_file(fn), data_tag=D.DATA_TAG,
                host=socket.gethostname(), python=platform.python_version(), torch=torch.__version__, device=device,
                wall_sec=round(time.time() - t0, 1), written=time.strftime("%F %T"))
    json.dump(meta, open(os.path.join(od, "meta_smoke.json" if smoke else "meta.json"), "w"), indent=1)
    log(f"[done] {fn} {len(preds)} transcripts {time.time() - t0:.0f}s")
    return fn
