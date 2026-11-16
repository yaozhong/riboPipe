"""RiboGL (Nallapareddy et al.; github.com/vam-sin/ribogl @ f3722b6) -- official code imported verbatim.

Official modules used unchanged (fetched by fetch_upstream.sh; md5-checked): rna_struct_utils.getRNASS /
mergeNTGraphToCodonGraph, graph_processing_utils.slidingWindowZeroToNan, gnn_utils.GCN_LSTM / MaskedPCCL1Loss.
Official defaults (gnn_model.ipynb cell 1 + gnn_utils.trainGCN): GCN layers [256, 128, 128, 64], dropout 0.1, AdamW
lr 1e-3 with ReduceLROnPlateau(0.1, patience 10) on val_loss, batch 1 graph, SAGE convolutions, random-walk positional
encoding length 32, <= 200 epochs, early stopping patience 10 on val_loss, loss MaskedPCCL1Loss
(1 - PCC + sqrt(mean |err|)), output softmax over the transcript; native target log(1 + c) with zero runs >= 30 set to NaN,
normalised by its nansum.  This file is a port of the paper's baseline_code/riboexp_ribogl/src/ribogl_paper.py
(md5 aa58db1e...).

Configurations
  Paper_RiboGL_heldout     native target; checkpoint selected on the frozen validation set val_ids(f) (the released
                           trainGCN passes the TEST loader as validation, gnn_utils.py:852-855 -- not used); test-time
                           output = softmax as is (scored by per-transcript correlations, which are scale invariant)
  Paper_RiboGL_L2_heldout  setting 2: prediction L_eff * softmax(z) against y = c / mean(c) (no NaN mask); the only
                           change to the official _get_loss is that rescaling
Adaptations: the official training-gene filter of the RiboGL study (coverage of non-zero, non-masked codons > 0.3, NaN
fraction <= 0.05, longest zero run <= 20; thresholds from gnn_model.ipynb) restored and applied to train_ids only; the
on-disk graph cache is sequence-only (codon ids, edges, RWPE) and test graphs carry no y; Lightning seed_everything(seed)
(official default 42) and the training order shuffled once (official FileRiboDataset); ModelCheckpoint save_top_k=1
(official 2; same best); print-only `verbose=True` dropped from ReduceLROnPlateau (removed in torch 2.7); deterministic
kernels at inference only; no length cap (an older harness skipped L > 1000).  The paper runs used a separate environment
(python 3.10.22, torch 2.6.0+cu124, torch_geometric 2.8.0, lightning 2.6.5, ViennaRNA 2.7.2, numpy 2.2.6) with
PYTHONNOUSERSITE=1 on an RTX 4090.
"""
import os, sys, json, time, itertools, hashlib, platform, random, traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

from . import common as C

GCN_LAYERS = [256, 128, 128, 64]; NUM_INP_FT = GCN_LAYERS[0]; DROPOUT = 0.1; LR = 1e-3; BS = 1
ALGO = 'SAGE'; EDGE_ATTR = 'None'; CHEB_K = 2; MODEL_TYPE = 'GCN_LSTM'; EPOCHS = 200
ANNOT_THRESH = 0.3; LONG_ZEROS_THRESH = 20; PERC_NANS_THRESH = 0.05
RW_LEN = 32
LENGTH_CAP = None
CONFIGS = {
    "Paper_RiboGL_heldout": dict(tier="T1", target="native", label="RiboGL-heldout (checkpoint selected only on frozen val)"),
    "Paper_RiboGL_L2_heldout": dict(tier="T2", target="L2", label="RiboGL-L2-heldout: L_eff*softmax(z) vs y=c/mean_full(c)"),
}
UPSTREAM_MD5 = {"models/GNN/gnn_utils.py": "50003f1cb814696bb67ec917c3ac99fb",
                "preprocessing/graph_processing_utils.py": "6488011191e45b4d6f088fa5ccb3455c",
                "preprocessing/rna_struct_utils.py": "3b79f28f753e64ae2e3bbd51d4af1e9e"}
_G = {}


def _official():
    """Import the official RiboGL modules verbatim (a stub replaces `transformers`, an unused top-level import of
    gnn_utils.py, exactly as in the paper's harness)."""
    if _G: return _G
    import types
    src = os.path.join(C.upstream_dir(), "ribogl", "src")
    for rel, m in UPSTREAM_MD5.items():
        p = os.path.join(src, rel)
        if not os.path.exists(p): raise SystemExit(f"{p} not found: run `bash baselines/fetch_upstream.sh` first")
        assert C.md5_file(p) == m, f"{p} differs from the version used for the paper"

    class _Dummy: pass

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"): raise AttributeError(name)
            return type(name, (_Dummy,), {})
    sys.modules["transformers"] = _StubModule("transformers")
    import torch
    torch.multiprocessing.set_sharing_strategy("file_system")
    sys.path.insert(0, src + "/preprocessing"); sys.path.insert(0, src + "/models/GNN")
    from rna_struct_utils import getRNASS, mergeNTGraphToCodonGraph
    from graph_processing_utils import slidingWindowZeroToNan
    import gnn_utils
    import RNA
    import lightning as L
    import torch_geometric.transforms as T
    _G.update(getRNASS=getRNASS, merge=mergeNTGraphToCodonGraph, zero_to_nan=slidingWindowZeroToNan, gnn_utils=gnn_utils,
              RNA=RNA, L=L, RW=T.AddRandomWalkPE(walk_length=RW_LEN), src=src)
    _G["classes"] = _classes(gnn_utils.GCN_LSTM)
    return _G


id_to_codon = {i: ''.join(el) for i, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v: k for k, v in id_to_codon.items()}


def build_graph_core(dna_seq):
    """graph_processing_utils.RiboDataset graph block (lines 60-83) without the target block."""
    from scipy import sparse
    from torch_geometric.utils.convert import from_scipy_sparse_matrix
    G = _official()
    L_ = len(dna_seq) // 3
    codon_ids = np.array([codon_to_id[dna_seq[i * 3:i * 3 + 3]] for i in range(L_)], dtype=np.int64)
    ss_adj = G["getRNASS"](dna_seq)
    codon_ss_graph = G["merge"](ss_adj)
    full_adj_mat = np.asarray(codon_ss_graph)
    len_Seq = full_adj_mat.shape[0]
    assert len_Seq == L_, f"codon graph {len_Seq} != L {L_}"
    adj_mat_init = np.zeros((len_Seq, len_Seq))
    for j in range(len_Seq - 1):
        adj_mat_init[j][j + 1] = 1; adj_mat_init[j + 1][j] = 1
    adj_mat_3d = full_adj_mat - adj_mat_init
    adj_mat_fin = adj_mat_3d + adj_mat_init
    ei, ew = from_scipy_sparse_matrix(sparse.csr_matrix(adj_mat_fin))
    return codon_ids, ei.numpy().astype(np.int64)


def _build_core(args):
    name, seq = args
    try:
        ci, ei = build_graph_core(seq); return name, ci, ei, None
    except Exception as e:
        return name, None, None, repr(e)


def seq_md5(s): return hashlib.md5(s.encode()).hexdigest()


def struct_cache(seqs, cache_path, nproc=8, log=print):
    """Sequence-only graph cache {name: dict(seq_md5, x, edge_index, random_walk_pe)}; resumable; no counts stored."""
    import multiprocessing as mp, torch
    from torch_geometric.data import Data
    G = _official()
    Cc = {}
    if os.path.isfile(cache_path):
        Cc = torch.load(cache_path, weights_only=False)
        assert Cc.get("_kind") == "ribogl_struct_seq_only_v1", cache_path
    entries = Cc.get("entries", {})
    need = [(k, s) for k, s in seqs.items() if k not in entries or entries[k]["seq_md5"] != seq_md5(s)]
    need.sort(key=lambda x: -len(x[1]))
    if need:
        log(f"[ribogl] folding {len(need)} transcripts (RNAfold {G['RNA'].__version__}, {nproc} procs)")
        t0 = time.time()
        with mp.get_context("fork").Pool(nproc) as pool:
            for j, (name, ci, ei, err) in enumerate(pool.imap_unordered(_build_core, need, chunksize=1)):
                if err is not None: raise RuntimeError(f"graph build failed for {name}: {err}")
                d = G["RW"](Data(edge_index=torch.from_numpy(ei), x=torch.from_numpy(ci), num_nodes=int(len(ci))))
                entries[name] = dict(seq_md5=seq_md5(seqs[name]), x=d.x, edge_index=d.edge_index, random_walk_pe=d.random_walk_pe)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + f".tmp{os.getpid()}"
        torch.save({"_kind": "ribogl_struct_seq_only_v1", "entries": entries}, tmp); os.replace(tmp, cache_path)
        log(f"[ribogl] struct cache {len(entries)} entries -> {cache_path} ({time.time()-t0:.0f}s)")
    return entries


def make_data(entry, y=None):
    import torch
    from torch_geometric.data import Data
    d = Data(edge_index=entry["edge_index"].clone(), x=entry["x"].clone(), num_nodes=int(len(entry["x"])))
    d.random_walk_pe = entry["random_walk_pe"].clone()
    if y is not None: d.y = torch.as_tensor(np.asarray(y, np.float32))
    return d


def native_target(counts):
    """graph_processing_utils.RiboDataset target block (lines 86-93): zero runs >= 30 -> NaN, log(1 + y)."""
    G = _official()
    y_i = [float(el) for el in counts]
    y_i = G["zero_to_nan"](y_i, window_size=30)
    y_i = np.log(np.asarray([1 + el for el in y_i]))
    return y_i.astype(np.float32)


def train_filter(counts):
    """Official RiboGL training-gene filter: coverage (c > 0 and not NaN-masked) > 0.3, NaN fraction <= 0.05,
    longest zero run <= 20."""
    G = _official()
    c = np.asarray(counts, float)
    y = G["zero_to_nan"](c.copy(), window_size=30)
    nan = np.isnan(y)
    cov = float(((c != 0) & ~nan).mean()); nanf = float(nan.mean())
    r = m = 0
    for v in c:
        r = r + 1 if v == 0 else 0; m = max(m, r)
    ok = (cov > ANNOT_THRESH) and (nanf <= PERC_NANS_THRESH) and (m <= LONG_ZEROS_THRESH)
    return ok, dict(cov=cov, nan_frac=nanf, max_zero_run=int(m))


def _classes(GCN_LSTM):
    import torch

    class GraphDS(torch.utils.data.Dataset):
        """FileRiboDataset.get (gnn_utils.py:50-64): edge_attr=None; y /= nansum(y) (native target only)."""
        def __init__(self, graphs, names, normalise=True):
            self.graphs = graphs; self.names = names; self.normalise = normalise
        def __len__(self): return len(self.graphs)
        def __getitem__(self, i):
            data = self.graphs[i].clone(); data.edge_attr = None
            if self.normalise and hasattr(data, "y") and data.y is not None:
                data.y = data.y / torch.nansum(data.y)
            return data, self.names[i]

    class GCN_LSTM_T1(GCN_LSTM):
        """Official GCN_LSTM; configure_optimizers (gnn_utils.py:257-265) without the print-only verbose=True."""
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'val_loss'}

    class GCN_LSTM_L2(GCN_LSTM_T1):
        """Setting 2: official _get_loss (gnn_utils.py:236-255) with y_hat = L_eff * softmax(z)."""
        def _get_loss(self, batch):
            y = batch.y
            y_pred = self.forward(batch)
            L_eff = y.shape[0]
            y_pred = L_eff * y_pred
            lengths = torch.tensor([y.shape[0]]).to(y_pred)
            mask = torch.arange(y_pred.shape[0])[None, :].to(lengths) < lengths[:, None]
            mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y)))
            mask = mask.squeeze(dim=0)
            loss, cos, mae = self.loss(y_pred, y, mask)
            return loss, 1 - cos, mae
    return dict(GraphDS=GraphDS, T1=GCN_LSTM_T1, L2=GCN_LSTM_L2)


def _cls(target): return _official()["classes"]["T1" if target == "native" else "L2"]


def load_model(ckpt, target):
    return _cls(target).load_from_checkpoint(ckpt, gcn_layers=GCN_LAYERS, dropout_val=DROPOUT, num_epochs=EPOCHS, bs=BS, lr=LR,
                                             num_inp_ft=NUM_INP_FT, model_type=MODEL_TYPE, algo=ALGO, edge_attr=EDGE_ATTR,
                                             cheb_k=CHEB_K, map_location="cuda")


def predict(model, te_graphs, te_names, target):
    import torch
    dev = torch.device("cuda"); model = model.to(dev).eval()
    prev = torch.are_deterministic_algorithms_enabled(); prev_c = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    out = {}
    with torch.no_grad():
        for g, n in zip(te_graphs, te_names):
            assert not hasattr(g, "y") or g.y is None, f"test graph {n} carries y"
            d = g.clone(); d.edge_attr = None; d = d.to(dev)
            p = model.forward(d).detach().cpu().numpy().ravel().astype(np.float64)
            if target == "L2": p = p * len(p)
            out[n] = p.astype(np.float32)
    torch.use_deterministic_algorithms(prev); torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev_c
    return out


def _epoch_from_ckpt(p):
    try: return int(os.path.basename(p).split("epoch=")[1].split("-")[0])
    except Exception: return None


def train_and_predict(tr_graphs, tr_names, va_graphs, va_names, te_graphs, te_names, save_loc, seed, target, epochs=EPOCHS):
    """Official trainGCN (gnn_utils.py:823-857) with val_dataloaders = frozen validation set (not test)."""
    import torch
    G = _official(); L = G["L"]; K = G["classes"]
    L.seed_everything(seed, workers=True)
    order = list(range(len(tr_graphs)))
    np.random.shuffle(order)
    tr_graphs = [tr_graphs[i] for i in order]; tr_names = [tr_names[i] for i in order]
    norm = (target == "native")
    train_loader = torch.utils.data.DataLoader(K["GraphDS"](tr_graphs, tr_names, norm), batch_size=1, shuffle=False, collate_fn=lambda b: b[0])
    val_loader = torch.utils.data.DataLoader(K["GraphDS"](va_graphs, va_names, norm), batch_size=1, shuffle=False, collate_fn=lambda b: b[0])
    model = _cls(target)(GCN_LAYERS, DROPOUT, epochs, BS, LR, NUM_INP_FT, MODEL_TYPE, ALGO, EDGE_ATTR, CHEB_K)
    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc, monitor='val_loss', save_top_k=1)
    trainer = L.Trainer(default_root_dir=save_loc, accelerator="auto", devices=1, accumulate_grad_batches=BS,
                        max_epochs=epochs, logger=False, enable_progress_bar=False,
                        callbacks=[ckpt_cb, L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10)])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = ckpt_cb.best_model_path
    if not (best and os.path.isfile(best)): raise RuntimeError("no best checkpoint (val_loss never finite?)")
    model = load_model(best, target)
    info = dict(best_ckpt=best, best_val_loss=float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
                selected_epoch=_epoch_from_ckpt(best), stopped_epoch=int(trainer.current_epoch), n_optimizer_steps=int(trainer.global_step))
    return model, predict(model, te_graphs, te_names, target), info


def run(config_id, dataset, fold, seed, out_root, smoke=None, max_epochs=0, overwrite=False, data=None, mfe=None, nproc=8, cache=None):
    """smoke=(ntrain, nval, ntest): first n sorted ids with L <= 300 (test L >= 45); cache default
    <out_root>/_cache_ribogl_struct/ribogl_struct_<dataset>.pt (sequence-only)."""
    import torch, platform as _pl
    G = _official()
    Cf = CONFIGS[config_id]; target = Cf["target"]; epochs = max_epochs or EPOCHS
    if C.run_exists(out_root, config_id, dataset, seed, fold) and not overwrite:
        print(f"[skip] run.json exists for {config_id} {dataset} seed{seed} fold{fold}", flush=True); return 0
    cache = cache or os.path.join(out_root, "_cache_ribogl_struct", f"ribogl_struct_{dataset}{'_smoke' if smoke else ''}.pt")
    t0 = time.time(); log = C.log_fn(f"{config_id}/{dataset}/s{seed}/f{fold}")
    import torch_geometric, scipy
    meta = dict(status="MODEL_FAILED", method="RiboGL", config_label=Cf["label"], tier=Cf["tier"], config_id_paper=config_id,
                failure_reason=None, smoke=(list(smoke) if smoke else False), length_cap=LENGTH_CAP, struct_cache=cache,
                env=dict(C.env_info(), torch_geometric=torch_geometric.__version__, lightning=G["L"].__version__,
                         viennarna=G["RNA"].__version__, scipy=scipy.__version__),
                code_md5=C.code_md5([os.path.abspath(__file__), os.path.join(C.HERE, "common.py")] +
                                    [os.path.join(G["src"], r) for r in UPSTREAM_MD5]),
                val_metric_name="val_loss = MaskedPCCL1Loss on frozen val_ids (Lightning epoch mean)",
                output_scale="softmax (sums to 1) as is" if target == "native" else "L_eff*softmax(z)")
    preds, provs = {}, {}
    try:
        ds = C.D.load_dataset(dataset, data=data, mfe=mfe, load_mfe=False)
        tr, va, te = ds.train_ids(fold), ds.val_ids(fold), ds.test_ids(fold)
        if smoke:
            short = lambda ids, lo=0: [t for t in sorted(ids) if lo <= len(ds.seq[t]) // 3 <= 300]
            tr, va, te = short(tr)[:smoke[0]], short(va)[:smoke[1]], short(te, 45)[:smoke[2]]
        keep, stats = [], dict(n_in=len(tr), fail_cov=0, fail_nan=0, fail_zero_run=0)
        for k in tr:
            ok, st = train_filter(ds.counts[k])
            if ok: keep.append(k)
            else:
                stats["fail_cov"] += st["cov"] <= ANNOT_THRESH; stats["fail_nan"] += st["nan_frac"] > PERC_NANS_THRESH
                stats["fail_zero_run"] += st["max_zero_run"] > LONG_ZEROS_THRESH
        stats["n_kept"] = len(keep); tr = keep
        log(f"train -> {len(tr)} after the official RiboGL filter {stats}; val {len(va)}; test {len(te)}")
        E = struct_cache({k: ds.seq[k] for k in tr + va + te}, cache, nproc, log)
        if target == "native":
            ytr = [native_target(ds.counts[k]) for k in tr]; yva = [native_target(ds.counts[k]) for k in va]
        else:
            ytr = [ds.truth(k).astype(np.float32) for k in tr]; yva = [ds.truth(k).astype(np.float32) for k in va]
        trg = [make_data(E[k], y) for k, y in zip(tr, ytr)]
        vag = [make_data(E[k], y) for k, y in zip(va, yva)]
        teg = [make_data(E[k]) for k in te]
        save_loc = os.path.join(out_root, config_id, dataset, f"seed{seed}", f"fold{fold}", "ckpt")
        if os.path.isdir(save_loc):
            import shutil; shutil.rmtree(save_loc)
        os.makedirs(save_loc, exist_ok=True)
        model, pred, info = train_and_predict(trg, tr, vag, va, teg, te, save_loc, seed, target, epochs)
        for k in te:
            p = pred[k]; assert len(p) == len(ds.seq[k]) // 3
            preds[k] = p; provs[k] = np.zeros(len(p), np.int8)
        fin = all(np.isfinite(v).all() for v in preds.values()); nonconst = any(np.std(v) > 0 for v in preds.values())
        meta.update(status="OK" if (fin and nonconst) else "MODEL_FAILED",
                    failure_reason=None if (fin and nonconst) else f"predictions finite={fin} nonconstant={nonconst}",
                    n_train_tx=len(tr), n_val_tx=len(va), n_test_tx=len(te), train_filter=stats,
                    n_optimizer_steps=info["n_optimizer_steps"], val_metric_best=info["best_val_loss"],
                    selected_epoch=info["selected_epoch"], stopped_epoch=info["stopped_epoch"],
                    checkpoint_id="md5:" + C.md5_file(info["best_ckpt"]), seconds=round(time.time() - t0, 1))
    except Exception as e:
        meta["failure_reason"] = f"{type(e).__name__}: {e}"; meta["traceback"] = traceback.format_exc()
        log("FAILED " + meta["failure_reason"]); preds, provs = {}, {}
    d = C.write(out_root, config_id, dataset, seed, fold, preds, provs, meta)
    log(f"wrote {d} status={meta['status']}")
    return 0 if meta["status"] == "OK" else 1
