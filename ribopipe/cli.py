"""Command-line interface: ``ribopipe <command> ...``

  ribopipe train    --config Paper_RiboPipe_MFE_H128 --dataset GSE133393_WT --fold 0 --seed 123 --out runs/
                    one (config, dataset, seed, fold) run: train on train_ids, early-stop on val_ids, predict test_ids
  ribopipe cv5      same, all five folds
  ribopipe predict  --ckpt runs/.../model.pt --dataset GSE133393_WT [--part partII] [--ids file] --out pred.npz
  ribopipe configs  list the model configurations and their parameter counts
Further commands (score, crossover, posterior, ism, transplant, ablation) are registered by their modules.
"""
import os, sys, json, time, argparse, traceback
import numpy as np


def _run_one(config_id, dataset, fold, seed, out_root, device="cuda:0", deterministic=True, max_epochs=0,
             smoke_n=(0, 0, 0), allow_any_seed=False, data=None, mfe=None, configs=None, builder=None, gate=None, inputs=None,
             train_transform=None):
    import torch
    from . import train as RP, data as D
    cfgs = configs or RP.CONFIGS
    cfg = cfgs[config_id]
    if seed not in RP.PREREG_SEEDS and not allow_any_seed:
        raise SystemExit(f"seed {seed} is not one of the pre-registered seeds {RP.PREREG_SEEDS} (use --allow-any-seed)")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ds = D.load_dataset(dataset, data=data, mfe=mfe)
    te, va, tr = ds.test_ids(fold), ds.val_ids(fold), ds.train_ids(fold)
    assert not (set(te) & set(tr)) and not (set(va) & set(tr)) and not (set(te) & set(va))
    sub_info = None
    if train_transform is not None:
        tr, sub_info = train_transform(ds, tr)
    smoke = any(list(smoke_n) + [max_epochs])
    if smoke:
        pick = lambda ids, n: sorted(ids)[:n] if n else ids
        tr, va, te = pick(tr, smoke_n[0]), pick(va, smoke_n[1]), pick(te, smoke_n[2])
    print(f"[run] {config_id} {dataset} fold {fold} seed {seed} | train {len(tr)} val {len(va)} test {len(te)} | "
          f"device {dev} | deterministic {deterministic} | smoke {smoke} | data {D.DATA_TAG if ds.group != 'user' else ds.dir}", flush=True)
    base_meta = dict(method=cfg["method"], tier=cfg.get("tier"), env=RP.env_info(), code_md5=RP.code_md5(),
                     deterministic=deterministic, smoke=({"n": list(smoke_n), "max_epochs": max_epochs} if smoke else False),
                     batch_window=RP.BATCH_WINDOW, batch_stride=RP.BATCH_STRIDE, loss=cfg["loss"], gamma=cfg["gamma"],
                     target=cfg["target"], hidden=cfg["hidden"], expected_params=cfg["expected_params"],
                     length_cap=None, register="A-site (no offset applied)", train_subsample=sub_info)
    kw = {}
    if builder: kw["model_builder"] = builder
    if gate: kw["gate"] = gate
    if inputs: kw["inputs"] = inputs
    try:
        model, best, info = RP.train_one(cfg, ds, fold, seed, dev, tr, va, max_epochs=(max_epochs or None),
                                         deterministic=deterministic, log=lambda s: print(s, flush=True), **kw)
    except RP.TrainingFailure as e:
        D.write_run(out_root, config_id, dataset, seed, fold, {}, {}, dict(base_meta, status="MODEL_FAILED", failure_reason=str(e),
                    traceback=traceback.format_exc()))
        print(f"[FAILED] {e}", flush=True); return 3
    run_dir = os.path.join(out_root, config_id, dataset, f"seed{seed}", f"fold{fold}"); os.makedirs(run_dir, exist_ok=True)
    json.dump(info, open(os.path.join(run_dir, "train_log.json"), "w"), indent=1)
    te_items, _ = RP.make_items(ds, te, cfg["model"], cfg["target"], with_target=False, **({"inputs": inputs} if inputs else {}))
    outputs = {config_id: ("native", "val_median_tx_pearson_linear_full")}
    for cid, crit in cfg.get("also_write", {}).items():
        outputs[cid] = (crit, "val_median_tx_pearson_linear_interior")
    rc = 0
    for cid, (crit, vname) in outputs.items():
        vbest, ep_sel, state = best[crit]
        d = os.path.join(out_root, cid, dataset, f"seed{seed}", f"fold{fold}"); os.makedirs(d, exist_ok=True)
        if state is None:
            D.write_run(out_root, cid, dataset, seed, fold, {}, {}, dict(base_meta, status="MODEL_FAILED",
                        failure_reason=f"validation criterion {crit} was never valid")); rc = 3; continue
        model.load_state_dict(state)
        u = RP.predict_raw(model, te_items, dev)
        preds = {k: RP.back_transform(u[k], cfg["target"]).astype(np.float32) for k in te}
        provs = {k: np.zeros(len(preds[k]), np.int8) for k in te}
        for k in te: assert len(preds[k]) == len(ds.seq[k]) // 3, k
        ck = os.path.join(d, "model.pt")
        ck_md5 = RP.save_ckpt(ck, model, cfg, cid, dataset, seed, fold, ep_sel, info["mfe_norm"])
        n_bad = int(sum((~np.isfinite(p)).any() or float(np.nanstd(p)) == 0 for p in preds.values()))
        status = "MODEL_FAILED" if (preds and n_bad >= len(preds)) else "OK"
        meta = dict(base_meta, status=status, checkpoint_id=f"{os.path.relpath(ck, out_root)} md5:{ck_md5}",
                    n_train_tx=info["n_train_tx"], n_val_tx=info["n_val_tx"], n_test_tx=len(te),
                    n_optimizer_steps=info["n_optimizer_steps"], epochs_run=info["epochs_run"], stopped_early=info["stopped_early"],
                    val_metric_name=vname, val_metric_best=vbest, selected_epoch=ep_sel, selection_criterion=crit,
                    params_trainable=info["structure"]["params"], mfe_norm=info["mfe_norm"], wall_sec=round(time.time() - t0, 1))
        out = D.write_run(out_root, cid, dataset, seed, fold, preds if status == "OK" else {}, provs if status == "OK" else {}, meta)
        print(f"[write] {cid} -> {out} status={status} selected_epoch={ep_sel} val={vbest:.5f} ({crit})", flush=True)
        if status != "OK": rc = 3
    return rc


def _add_common(p):
    p.add_argument("--data", default=None, help="data directory (default: ./data or $RIBOPIPE_DATA)")
    p.add_argument("--mfe", default=None, help="MFE track directory (default: <data>/mfe or $RIBOPIPE_MFE)")


def main(argv=None):
    from . import train as RP
    ap = argparse.ArgumentParser(prog="ribopipe", description="RiboPipe: within-sample imputation of relative footprint density")
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "cv5"):
        p = sp.add_parser(name, help="train one fold" if name == "train" else "train all five folds")
        p.add_argument("--config", default="Paper_RiboPipe_MFE_H128", choices=sorted(RP.CONFIGS))
        p.add_argument("--dataset", required=True)
        if name == "train": p.add_argument("--fold", type=int, required=True)
        p.add_argument("--seed", type=int, default=RP.DEFAULT_SEED)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--no-deterministic", action="store_true")
        p.add_argument("--max-epochs", type=int, default=0, help="smoke runs only")
        p.add_argument("--smoke-n", type=int, nargs=3, default=(0, 0, 0), metavar=("TRAIN", "VAL", "TEST"),
                       help="smoke runs only: first n sorted ids of each split")
        p.add_argument("--allow-any-seed", action="store_true")
        _add_common(p)
    p = sp.add_parser("predict", help="predict relative footprint density with a checkpoint")
    p.add_argument("--ckpt", required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--part", default="partI", choices=("partI", "partII"))
    p.add_argument("--ids", default=None, help="text file with one transcript id per line (default: whole universe)")
    p.add_argument("--out", required=True); p.add_argument("--device", default="cpu"); _add_common(p)
    sp.add_parser("configs", help="list model configurations")
    import importlib
    for mod in ("evaluate", "lowcov", "interpret", "ablation", "prepare"):
        try:
            m = importlib.import_module(f"ribopipe.{mod}")
        except ImportError:
            continue
        if hasattr(m, "register"): m.register(sp)
    a = ap.parse_args(argv)
    if a.cmd == "configs":
        for k, c in RP.CONFIGS.items(): print(f"{k:32s} {c['method']:18s} {c['expected_params']:>8,d}  {c['note']}")
        return 0
    if a.cmd in ("train", "cv5"):
        folds = [a.fold] if a.cmd == "train" else range(5)
        rc = 0
        for f in folds:
            rc |= _run_one(a.config, a.dataset, f, a.seed, a.out, a.device, not a.no_deterministic, a.max_epochs,
                           tuple(a.smoke_n), a.allow_any_seed, a.data, a.mfe)
        return rc
    if a.cmd == "predict":
        from . import data as D
        m, ck = RP.load_ckpt(a.ckpt)
        ds = D.load_dataset(a.dataset, a.part, data=a.data, mfe=a.mfe)
        ids = [l.strip() for l in open(a.ids)] if a.ids else ds.ids
        preds = RP.predict_transcripts(m, ds, ids, ck["config"], a.device)
        np.savez(a.out, **preds); print(f"[predict] {len(preds)} transcripts -> {a.out}"); return 0
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
