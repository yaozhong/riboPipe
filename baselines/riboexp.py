"""RiboExp (Hu et al. 2019 "Rationalizing Translation Elongation by Reinforcement Learning"; github.com/Liuxg16/Riboexp
@ 8edf66b) -- official default with RNA-structure features (Paper_RiboExp).

The official code is Python 2.7 / PyTorch 0.3.1.  fetch_upstream.sh downloads it and applies patches/riboexp_*.diff,
which produce the py3/torch2 ports used for the paper (md5-identical): models.py (API-level edits; the structure branch
encoder_fold BiGRU(13->256) of the official default restored), nntools.py (torch2 one-hot / repackage), dataio.py
(py3 edits; codon vocabulary in the py2 dict iteration order A,C,T,G as measured with python 2.7.18; PaperData adapter
paper_fold_track / paper_docs), runRiboexp.py -> riboexp_run_py3.py (TrainingModel line by line: Adadelta lr 1.0,
gradient clip 1.0, fc max-norm 3, sum-loss backward, validation every interval*4 = 80 batches, save only when
corr > best AND rationale length < 9, stop when 50 validations pass without update, python random.shuffle of the batch
order, <= 200 epochs, batch 600, n_hids 512, lambda1 0.0083, drate 0.4, window 10; best state kept in memory;
batched full-length inference predict_full equivalent to the official testSeq).
This file is a port of the paper's baseline_code/riboexp_ribogl/src/riboexp_paper.py (md5 5389754c...).

Adaptations: data through ribopipe.data; frozen folds; training windows from train_ids(f) with the official
process_structure (drops transcripts with sum(c) < 200) / process_fragment (no-structure path, no gate); validation on
val_ids(f) with the same window rule but without the sum(c) gate; structure track fold[j] = MFE of the 30-nt window
starting at j-14 (official alignment) where defined inside the CDS, else the official missing marker -1 (the official
files extend into 13 nt of 5'UTR / 10 nt of 3'UTR); seed s passed to random/numpy/torch/cuda, to the window shuffle
(official np seed 1234) and to the model constructor (official re-seeds torch there); no length cap; for the structure
model the end windows (codon i < 5 or i > L-5) are padded like the official testSeq and written with provenance 1
(adapter); deterministic kernels at inference only.  The paper runs used CUDA (RTX 4090) in a separate environment
(python 3.10, torch 2.6.0+cu124, numpy 2.2.6).
"""
import os, sys, json, time, random, traceback
import numpy as np

from . import common as C

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CONFIGS = {
    "Paper_RiboExp": dict(no_structure=False, tier="T1", label="official default (with RNA-structure features)"),
    "Paper_RiboExp_noStruct_port": dict(no_structure=True, tier="extended", label="NON-OFFICIAL-DEFAULT: --no_structure variant"),
}
LENGTH_CAP = None
PATCHED_MD5 = {"riboexp_models_py3.py": "60aea21eaeef05cb1bb56eaf687612e7", "riboexp_dataio_py3.py": "03bc80e7814960b6d79812e468bff8bd",
               "riboexp_nntools_py3.py": "b38776a12efca1dab3d3922aa543ae64", "riboexp_run_py3.py": "14526114e8965279ec18c6127409c6c4"}


def _official():
    d = os.path.join(C.upstream_dir(), "patched", "riboexp")
    for fn, m in PATCHED_MD5.items():
        p = os.path.join(d, fn)
        if not os.path.exists(p): raise SystemExit(f"{p} not found: run `bash baselines/fetch_upstream.sh` first")
        assert C.md5_file(p) == m, f"{p} differs from the paper's version"
    if d not in sys.path: sys.path.insert(0, d)
    import riboexp_dataio_py3 as DIO, riboexp_run_py3 as RUN
    from riboexp_models_py3 import RiboexpModel
    return DIO, RUN, RiboexpModel, d


def seed_all(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def process_structure_valid(docs_condons, docs_nt, docs_fold, docs_counts, window_condon):
    """Official dataio.process_structure without the transcript gate sum(c) < 200 (validation set only).
    Transcripts without any codon > 0.5 (official normaliser undefined) are skipped and counted."""
    offset_relative = int((window_condon - 1) / 2)
    offset_l = offset_relative + int(window_condon - 1) % 2
    offset_r = offset_relative
    data, nts, folds, labels = [], [], [], []
    skipped = 0
    for condon_seq, nt_seq, fold_seq, asite_seq1 in zip(docs_condons, docs_nt, docs_fold, docs_counts):
        n_seq = len(condon_seq)
        asite_seq = np.array(asite_seq1).astype(float)
        asite_sum = np.sum(asite_seq)
        n_valid_asite = np.sum(asite_seq > 0.5)
        if n_valid_asite == 0:
            skipped += 1; continue
        asite_seq = asite_seq / (asite_sum / n_valid_asite)
        for i in range(n_seq):
            footprints = asite_seq[i]
            if footprints > 1e-6:
                start = (i - offset_l); end = i + offset_r + 1
                if start < 0 or end > n_seq:
                    continue
                data.append([x for x in condon_seq[start:end]]); nts.append(nt_seq[3 * start:3 * end])
                if fold_seq is not None: folds.append(fold_seq[3 * start:3 * end])
                labels.append(footprints)
    out = [np.array(data).astype(int), np.array(nts).astype(int)]
    if any(f is not None for f in docs_fold):
        out.append(np.array(folds).astype(float))
    out.append(np.array(labels))
    return out, skipped


def build_windows(DIO, ds, ids, structure, train):
    c, nt, fold, lab = DIO.paper_docs(ds, ids, with_counts=True, structure=structure)
    ci = DIO.to_index(c, DIO.CODON_VOCAB); ni = DIO.to_index(nt, DIO.CHAR_VOCAB)
    info = dict(n_tx_in=len(ids))
    if train:
        if structure:
            d, n, f, y = DIO.process_structure(ci, ni, fold, lab, 10)
            info["n_tx_dropped_sum_lt_200"] = int(sum(1 for x in lab if np.sum(x) < 200))
            arrs = [d, n, f, y]
        else:
            d, n, y = DIO.process_fragment(ci, ni, lab, 10)
            info["n_tx_dropped_sum_lt_200"] = 0
            arrs = [d, n, y]
    else:
        arrs, sk = process_structure_valid(ci, ni, fold if structure else [None] * len(ci), lab, 10)
        info["n_tx_skipped_no_codon_gt_0p5"] = sk
    info["n_windows"] = int(len(arrs[0]))
    return arrs, info


def param_report(model):
    """Parameters on the full predictive path (forward hooks during an eval forward)."""
    import torch
    tot = sum(p.numel() for p in model.parameters())
    per = {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children() if sum(p.numel() for p in m.parameters())}
    called = set()
    hooks = [m.register_forward_hook(lambda mod, i, o, n=n: called.add(n)) for n, m in model.named_children()]
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.randint(1, 65, (4, 10), device=dev); nt = torch.randint(1, 5, (4, 30), device=dev)
        fold = None if model.kwargs['no_structure'] else torch.randn(4, 30, device=dev)
        model(x, nt, fold, torch.zeros(4, 1, device=dev))
    for h in hooks: h.remove()
    path = sum(v for n, v in per.items() if n in called)
    return dict(total=int(tot), per_module=per, predictive_path_params=int(path), unused_in_prediction=sorted(set(per) - called))


def run(config_id, dataset, fold, seed, out_root, device="auto", smoke=None, max_epochs=0, overwrite=False, data=None, mfe=None,
        val_every_batches=None, save_ckpt=True):
    """smoke=(ntrain, nval, ntest): first n sorted ids (test transcripts with L >= 45); max_epochs caps the official 200."""
    import torch
    DIO, RUN, RiboexpModel, updir = _official()
    Cf = CONFIGS[config_id]; structure = not Cf["no_structure"]
    if C.run_exists(out_root, config_id, dataset, seed, fold) and not overwrite:
        print(f"[skip] run.json exists for {config_id} {dataset} seed{seed} fold{fold}", flush=True); return 0
    t0 = time.time(); log = C.log_fn(f"{config_id}/{dataset}/s{seed}/f{fold}")
    meta = dict(status="MODEL_FAILED", method="RiboExp", config_label=Cf["label"], tier=Cf["tier"], config_id_paper=config_id,
                val_metric_name="pooled Pearson over validation windows (official runRiboexp.validation)",
                failure_reason=None, smoke=(list(smoke) if smoke else False), length_cap=LENGTH_CAP,
                code_md5=C.code_md5([os.path.abspath(__file__), os.path.join(C.HERE, "common.py")] +
                                    [os.path.join(updir, f) for f in PATCHED_MD5]))
    preds, provs = {}, {}
    try:
        dev = torch.device("cuda" if device == "auto" else device)
        if dev.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA not available")
        ds = C.D.load_dataset(dataset, data=data, mfe=mfe, load_mfe=structure)
        tr, va, te = ds.train_ids(fold), ds.val_ids(fold), ds.test_ids(fold)
        if smoke:
            tr, va = C.smoke_pick(tr, smoke[0]), C.smoke_pick(va, smoke[1])
            te = [t for t in sorted(te) if len(ds.seq[t]) // 3 >= 45][:smoke[2]]
            if val_every_batches is None: val_every_batches = 1   # smoke: validate every batch (official 80)
        for k in tr + va + te:
            assert set(ds.seq[k]) <= set("ACGT"), k
        seed_all(seed)
        trw, tri = build_windows(DIO, ds, tr, structure, train=True)
        vaw, vai = build_windows(DIO, ds, va, structure, train=False)
        for arr in trw:                      # official Dataset: np.random.seed(rnd_seed); np.random.shuffle(each array)
            np.random.seed(seed); np.random.shuffle(arr)
        log(f"train tx {len(tr)} windows {tri['n_windows']} (sum<200 dropped {tri['n_tx_dropped_sum_lt_200']}); val tx {len(va)} windows {vai['n_windows']}; test {len(te)}")
        cfg = dict(RUN.OFFICIAL_CFG); cfg.update(seed=seed, no_structure=not structure)
        if max_epochs: cfg["epochs"] = max_epochs
        if val_every_batches: cfg["val_every_batches"] = val_every_batches
        seed_all(seed)
        tm = RUN.TrainingModel(RiboexpModel, cfg, dev)
        meta["params"] = param_report(tm.model)
        tm.model.train()
        best = tm.loop_train(trw, vaw, log)
        seqs = {k: ds.seq[k] for k in te}
        folds_tr = {k: DIO.paper_fold_track(ds.mfe_raw(k), ds.mfe_valid(k), len(ds.seq[k])) for k in te} if structure else None
        out, padded = tm.predict_full(seqs, DIO.CODON_VOCAB, folds_tr)
        for k in te:
            p = out[k].astype(np.float32); q = np.zeros(len(p), np.int8)
            if structure: q[padded[k]] = C.ADAPTER
            preds[k] = p; provs[k] = q
        fin = all(np.isfinite(v).all() for v in preds.values()); nonconst = any(np.std(v) > 0 for v in preds.values())
        meta.update(status="OK" if (fin and nonconst) else "MODEL_FAILED",
                    failure_reason=None if (fin and nonconst) else f"predictions finite={fin} nonconstant={nonconst}",
                    n_train_tx=len(tr) - tri["n_tx_dropped_sum_lt_200"], n_train_tx_before_filter=len(tr), n_val_tx=len(va),
                    n_test_tx=len(te), n_train_windows=tri["n_windows"], n_val_windows=vai["n_windows"], train_filter=tri, val_info=vai,
                    n_optimizer_steps=tm.n_optimizer_steps, val_metric_best=best, selected_epoch=tm.best_epoch,
                    selected_vallen=tm.best_vallen, n_validations=tm.n_validations, stopped_epoch=tm.stop_epoch, cfg=cfg,
                    seconds=round(time.time() - t0, 1))
    except Exception as e:
        meta["failure_reason"] = f"{type(e).__name__}: {e}"; meta["traceback"] = traceback.format_exc()
        log("FAILED " + meta["failure_reason"]); preds, provs = {}, {}
    d = C.write(out_root, config_id, dataset, seed, fold, preds, provs, meta)
    if meta["status"] == "OK":
        if save_ckpt: torch.save(tm.model.state_dict(), os.path.join(d, "best_model.pt"))
        json.dump(tm.history, open(os.path.join(d, "train_history.json"), "w"))
    log(f"wrote {d} status={meta['status']}")
    return 0 if meta["status"] == "OK" else 1
