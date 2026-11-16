"""RiboMIMO (Tian et al. 2021; github.com/tiantz17/RiboMIMO @ 094c693) -- official network, adapted training driver.

The network class ``RiboMIMO`` and ``weights_init`` come from the official model.py, fetched by fetch_upstream.sh and
patched with two compatibility lines (patches/ribomimo_model.py.diff: lengths moved to CPU for pack_padded_sequence,
torch >= 1.7; dtype=object for a ragged numpy array, numpy >= 1.24; no numerical change).  This file is a port of the
paper's baseline_code/ribomimo/ribomimo_paper.py (md5 a3966469...); data preparation, training loop and inference are
unchanged, comments translated.  The official runRiboMIMO.py driver is replaced, step by step, by the functions below
(load_model order, get_thresh, train_one_fold / train_one_epoch, evaluate('Valid')).

Configurations (others identical to the official defaults: 2-layer bidirectional GRU h=256, Linear 512->256 + ReLU,
regression head + 3-class head, loss MSE + 3 * CE, class thresholds "0,2" (mean + {0, 2} SD of log10 training density),
Adam 1e-3 / weight decay 1e-6, StepLR(5, 0.95), batch 16, <= 100 epochs, early stop after 5 epochs without improvement
of the validation R2, Xavier initialisation; inputs 64 codon + 12 nucleotide + 21 amino-acid one-hot = 97, as the
official recommended command --add_nt --add_aa):
  Paper_RiboMIMO        official target 1[c>0.5] c / mean(c[c>0.5]); first 5 / last 4 codons zeroed (ends trimmed);
                        loss / validation on density > 0; training filter mean(c > 0.5) > 0.6 on train_ids only;
                        trimmed end positions written with provenance 1 (adapter, raw network output)
  Paper_RiboMIMO_L2     setting 2: target y = c / mean(c) over the full CDS, no trimming, only padding masked,
                        training filter kept; all positions model
  Paper_RiboMIMO_impl, Paper_RiboMIMO_adapt_noMFE, Paper_RiboMIMO_adapt_MFE   task-adapted controls (target
                        c / mean(c[c>0]), no filter; adapt_MFE adds the 3 standardised MFE inputs of RiboPipe)
Adaptations vs the official driver: frozen folds and val_ids(f) as validation set (official: random 10-fold split,
first 10% of training as validation); seeds 123/124/125 passed to random/numpy/torch/cuda (official: np seed 1234);
class thresholds computed before end trimming; test counts deleted from memory before any processing and never used
(official evaluates the test fold every epoch); non-finite training loss -> MODEL_FAILED; transcripts with non-ACGT
bases raise instead of being dropped silently; no length cap.
"""
import os, sys, json, time, math, random, hashlib, platform, socket, traceback
import numpy as np

from . import common as C

RNN_HIDDEN_DIMS = 256; RNN_DEPTH = 2; THRESHOLD = "0,2"; ALPHA = 3.0; BATCH_SIZE = 16; MAX_EPOCH = 100; EARLY_STOP = 5
LEARNING_RATE = 1e-3; WEIGHT_DECAY = 1e-6; STEP_SIZE = 5; GAMMA = 0.95; ADD_NT = True; ADD_AA = True
INPUT_DIMS_BASE = 64 + 4 * 3 + 21
COVERAGE_CUT = 0.6
MFE_OFFSETS = (-17, -16, -15)
LENGTH_CAP = None

CONFIGS = {
    "Paper_RiboMIMO": dict(tier="T1", method="RiboMIMO", target_mode="official", end_trim=(5, 4), loss_mask="density>0",
                           train_filter="coverage_6", n_mfe=0, end_prov="adapter"),
    "Paper_RiboMIMO_impl": dict(tier="control", method="RiboMIMO-impl", target_mode="impl", end_trim=(0, 0), loss_mask="padding",
                                train_filter="none", n_mfe=0, end_prov="model"),
    "Paper_RiboMIMO_L2": dict(tier="T2", method="RiboMIMO-L2", target_mode="L2", end_trim=(0, 0), loss_mask="padding",
                              train_filter="coverage_6", n_mfe=0, end_prov="model"),
    "Paper_RiboMIMO_adapt_noMFE": dict(tier="T3", method="RiboMIMO-adapt", target_mode="impl", end_trim=(0, 0), loss_mask="padding",
                                       train_filter="none", n_mfe=0, end_prov="model"),
    "Paper_RiboMIMO_adapt_MFE": dict(tier="T3", method="RiboMIMO-adapt", target_mode="impl", end_trim=(0, 0), loss_mask="padding",
                                     train_filter="none", n_mfe=3, end_prov="model"),
}

_M = None


def _official():
    """Import the patched official model.py (fetch_upstream.sh)."""
    global _M
    if _M is None:
        d = os.path.join(C.upstream_dir(), "patched", "ribomimo")
        if not os.path.exists(os.path.join(d, "model.py")):
            raise SystemExit(f"{d}/model.py not found: run `bash baselines/fetch_upstream.sh` first")
        sys.path.insert(0, d)
        import model as M
        assert C.md5_file(M.__file__) == "375c7a0e8796092cc45d908be2bf4779", "patched RiboMIMO model.py differs from the paper's"
        _M = M
    return _M


def set_all_seeds(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _loader_class():
    M = _official()

    class PaperDataLoader(M.DataLoader):
        """In-memory version of the official DataLoader (model.py:81-278): same one-hot dictionaries, criteria and
        coverage list; counts=None builds sequence features only (inference on test transcripts)."""
        def __init__(self, names, seqs, counts=None, target_mode=None, extra=None, use_median=False):
            self.path = None; self.dataset = "memory"; self.filename = None
            self.aa_table = M.utils.aa_table
            self.codon2aa = M.utils.codon2aa
            self.aa2codon = {}
            for codon in self.codon2aa.keys():
                if self.codon2aa[codon] not in self.aa2codon:
                    self.aa2codon[self.codon2aa[codon]] = {codon}
                else:
                    self.aa2codon[self.codon2aa[codon]].add(codon)
            self.load_onehot()
            self.target_mode = target_mode
            self.extra = extra
            self.load_data_mem(names, seqs, counts, use_median)

        @staticmethod
        def target(count, mode, use_median=False):
            if mode == "official":        # official model.py:133,135
                if use_median:
                    avg = np.median(np.array(count)[np.array(count) > 0.5])
                else:
                    avg = np.mean(np.array(count)[np.array(count) > 0.5])
                density = (np.array(count) > 0.5) * np.array(count) / avg
            elif mode == "impl":
                avg = np.mean(np.array(count)[np.array(count) > 0])
                density = np.array(count) / avg
            elif mode == "L2":
                avg = np.mean(np.array(count))
                density = np.array(count) / avg
            else:
                raise ValueError(mode)
            return avg, density

        def load_data_mem(self, names, seqs, counts, use_median=False):
            list_name, list_seq, list_density, list_criteria, list_avg = [], [], [], [], []
            for name in names:
                s = seqs[name]
                seq = [s[3 * i:3 * i + 3] for i in range(len(s) // 3)]
                list_name.append(name); list_seq.append(seq)
                if counts is not None:
                    count = [float(e) for e in counts[name]]
                    avg, density = self.target(count, self.target_mode, use_median)
                    list_criteria.append([np.sum(count), np.mean(count), np.sum(np.array(count) > 0.5), np.mean(np.array(count) > 0.5)])
                    list_density.append(density); list_avg.append(avg)
            list_name = np.array(list_name)
            self.list_gene_all = list_name
            self.has_counts = counts is not None
            if self.has_counts:
                list_criteria = np.array(list_criteria).reshape(-1, 4)
                self.criteria = {list_name[i]: list_criteria[i] for i in range(len(list_name))}
                index = (list_criteria[:, 3] > COVERAGE_CUT)          # official model.py:150
                self.list_gene_coverage_6 = list_name[index]
            self.dict_seq, self.dict_seq_nt, self.dict_seq_aa, self.dict_seq_codon = {}, {}, {}, {}
            self.dict_density, self.dict_avg = {}, {}
            for i in range(len(list_name)):
                codons = list_seq[i]
                nts = "".join(codons)
                if set(nts) - set("ACGT"):
                    raise ValueError(f"non-ACGT base in {list_name[i]}")
                aas = [self.codon2aa[codon] for codon in codons]
                self.dict_seq[list_name[i]] = nts
                self.dict_seq_nt[list_name[i]] = np.array([self.onehot_nt[nt] for nt in nts])
                self.dict_seq_codon[list_name[i]] = np.array([self.onehot_codon[codon] for codon in codons])
                self.dict_seq_aa[list_name[i]] = np.array([self.onehot_aa[aa] for aa in aas])
                if self.has_counts:
                    self.dict_density[list_name[i]] = list_density[i]
                    self.dict_avg[list_name[i]] = list_avg[i]

        def get_data_pack(self, gene_list, add_nt=False, add_aa=False, end_trim=(5, 4)):
            """Official get_data_pack (model.py:239-259) with configurable end trimming (official fixed 5 / 4)."""
            batch_size = len(gene_list)
            x_input = [self.dict_seq_codon[gene] for gene in gene_list]
            length = [len(self.dict_seq_codon[gene]) for gene in gene_list]
            if add_nt:
                x_nt = [self.dict_seq_nt[gene] for gene in gene_list]
                x_input = [np.concatenate([x_input[i], x_nt[i].reshape((len(x_nt[i]) // 3, -1))], axis=1) for i in range(batch_size)]
            if add_aa:
                x_aa = [self.dict_seq_aa[gene] for gene in gene_list]
                x_input = [np.concatenate([x_input[i], x_aa[i]], axis=1) for i in range(batch_size)]
            if self.extra is not None:
                x_input = [np.concatenate([x_input[i], self.extra[gene_list[i]]], axis=1) for i in range(batch_size)]
            density = []
            if self.has_counts:
                t5, t3 = end_trim
                for gene in gene_list:
                    temp = self.dict_density[gene].copy()
                    if t5 > 0: temp[:t5] = 0
                    if t3 > 0: temp[-t3:] = 0
                    density.append(temp)
            self.density = density
            self.x_input = [np.asarray(x, dtype=np.float32) for x in x_input]
            self.length = np.array(length)
    return PaperDataLoader


def mfe_features(ds, tx, mu, sd):
    """The 3 MFE inputs of RiboPipe: windows starting at 3i-17/-16/-15; valid iff 0 <= p < n_valid; (raw - mu_f)/sd_f,
    invalid -> 0."""
    L = len(ds.seq[tx]) // 3
    raw = ds.mfe_raw(tx); nv = ds.mfe_valid(tx)
    out = np.zeros((L, len(MFE_OFFSETS)), dtype=np.float32); nmask = 0
    for k, rel in enumerate(MFE_OFFSETS):
        p = np.arange(L) * 3 + rel
        ok = (p >= 0) & (p < nv)
        out[ok, k] = (raw[p[ok]] - mu) / sd
        nmask += int((~ok).sum())
    return out, nmask


def get_thresh(loader, train_names, threshold=THRESHOLD):
    """Official get_thresh (runRiboMIMO.py:216-229) on the untrimmed training density."""
    list_density = []
    for gene in train_names:
        list_density.extend(loader.dict_density[gene])
    list_density = np.array(list_density)
    list_density = list_density[list_density > 0]
    list_density = np.log10(list_density)
    avg = np.mean(list_density); std = np.std(list_density)
    threshs = []
    thresholds = np.sort(np.array(threshold.split(","), dtype=float))
    for i in thresholds:
        threshs.append(avg + float(i) * std)
    return threshs


class Pack:
    """Encoded inputs of a set of transcripts (official recomputes get_data_pack each epoch; inputs are constant)."""
    def __init__(self, loader, names, end_trim, threshs=None):
        self.names = list(names)
        loader.get_data_pack(self.names, ADD_NT, ADD_AA, end_trim=end_trim)
        self.x = loader.x_input; self.length = loader.length.copy()
        self.density = [np.asarray(d, dtype=np.float32) for d in loader.density] if loader.has_counts else None
        self.level = None
        if threshs is not None:
            loader.get_data_pack_cls(self.names, threshs)
            self.level = [np.asarray(l, dtype=np.int64) for l in loader.density_level]


def get_batch(pack, index, loss_mask, dev):
    """Official get_data_batch (runRiboMIMO.py:153-178): sorted by length (descending), zero padded."""
    import torch
    index = np.asarray(index)
    order = pack.length[index].argsort()[::-1]
    index = index[order]
    lens = pack.length[index]; Lmax = int(lens.max()); Dm = pack.x[index[0]].shape[1]
    x = np.zeros((len(index), Lmax, Dm), dtype=np.float32)
    dens = np.zeros((len(index), Lmax), dtype=np.float32)
    lev = np.zeros((len(index), Lmax), dtype=np.int64)
    for k, j in enumerate(index):
        L = pack.length[j]; x[k, :L] = pack.x[j]
        if pack.density is not None: dens[k, :L] = pack.density[j]
        if pack.level is not None: lev[k, :L] = pack.level[j]
    if loss_mask == "density>0":
        mask = dens > 0
    elif loss_mask == "padding":
        mask = np.arange(Lmax)[None, :] < lens[:, None]
    else:
        raise ValueError(loss_mask)
    return (torch.tensor(x, device=dev), torch.tensor(lens, dtype=torch.int64),
            torch.tensor(dens, device=dev), torch.tensor(mask, device=dev),
            torch.tensor(lev, device=dev), index)


def train_one_epoch(model, opt, mse, ce, pack, loss_mask, dev):
    """Official train_one_epoch (runRiboMIMO.py:274-295); np.random.shuffle on the seeded global numpy RNG."""
    import torch
    n = len(pack.names)
    index_shuffle = np.array(range(n))
    num_batch = int(np.ceil(n / BATCH_SIZE))
    np.random.shuffle(index_shuffle)
    steps, losses, npos = 0, [], 0
    for batch in range(num_batch):
        index = index_shuffle[batch * BATCH_SIZE:min((batch + 1) * BATCH_SIZE, n)]
        x_input, length, density, mask, level, _ = get_batch(pack, index, loss_mask, dev)
        opt.zero_grad()
        pred_reg, pred_cls, _ = model(x_input, length)
        pred_reg = pred_reg.masked_select(mask)
        pred_cls = pred_cls.masked_select(mask.unsqueeze(2).repeat(1, 1, model.num_state)).view(-1, model.num_state)
        density = density.masked_select(mask)
        level = level.masked_select(mask)
        loss = mse(pred_reg, density) + ce(pred_cls, level) * ALPHA
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss (batch {batch}, masked positions {int(mask.sum())})")
        loss.backward(); opt.step()
        steps += 1; losses.append(float(loss)); npos += int(mask.sum())
    return steps, float(np.mean(losses)), npos


def predict_pack(model, pack, dev):
    import torch
    out = [None] * len(pack.names)
    n = len(pack.names)
    with torch.no_grad():
        for b in range(int(np.ceil(n / BATCH_SIZE))):
            index = np.array(range(b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, n)))
            x_input, length, _, _, _, idx = get_batch(pack, index, "padding", dev)
            y = model.predict(x_input, length).cpu().numpy()
            for k, j in enumerate(idx):
                out[j] = y[k, :pack.length[j]].astype(np.float64)
    return out


def evaluate_valid(model, pack, loss_mask, dev):
    """Official evaluate('Valid') (runRiboMIMO.py:318-355): pooled r2_score over the masked positions."""
    M = _official()
    preds = predict_pack(model, pack, dev)
    P, T = [], []
    for j in range(len(pack.names)):
        d = pack.density[j]; p = preds[j]
        m = (d > 0) if loss_mask == "density>0" else np.ones(len(d), bool)
        P.append(p[m]); T.append(d[m])
    P = np.concatenate(P); T = np.concatenate(T)
    return float(M.r2_score(T, P)) if len(T) > 1 and np.isfinite(P).all() else float("nan")


def infer(model, ds, names, cfg, fold, dev):
    """Inference from sequence (and MFE track) only: a loader without counts."""
    Loader = _loader_class()
    extra = None
    if cfg["n_mfe"]:
        mu, sd = ds.mfe_norm(fold)
        extra = {t: mfe_features(ds, t, mu, sd)[0] for t in names}
    lte = Loader(names, ds.seq, None, None, extra)
    yhat = predict_pack(model, Pack(lte, names, (0, 0), None), dev)
    preds, provs = {}, {}
    for j, t in enumerate(names):
        L = len(ds.seq[t]) // 3; q = np.zeros(L, np.int8); p = yhat[j]
        if tuple(cfg["end_trim"]) != (0, 0):
            code = C.ADAPTER if cfg["end_prov"] == "adapter" else C.UNDEFINED
            t5, t3 = cfg["end_trim"]; q[:t5] = code; q[max(L - t3, 0):] = code
            if code == C.UNDEFINED: p = np.where(q == C.UNDEFINED, np.nan, p)
        preds[t] = p; provs[t] = q
    return preds, provs


def run(config_id, dataset, fold, seed, out_root, device="auto", smoke=None, max_epochs=0, overwrite=False,
        data=None, mfe=None, save_ckpt=True):
    import torch, torch.nn as nn, torch.optim as optim
    M = _official(); Loader = _loader_class()
    cfg = CONFIGS[config_id]
    if C.run_exists(out_root, config_id, dataset, seed, fold) and not overwrite:
        print(f"[skip] run.json exists for {config_id} {dataset} seed{seed} fold{fold}", flush=True); return 0
    up = os.path.join(C.upstream_dir(), "patched", "ribomimo")
    meta = dict(status="MODEL_FAILED", method=cfg["method"], tier=cfg["tier"], config_id_paper=config_id, failure_reason=None,
                smoke=(list(smoke) if smoke else False), max_epochs_override=(max_epochs or None),
                code_md5=C.code_md5([os.path.abspath(__file__), os.path.join(C.HERE, "common.py"),
                                     os.path.join(up, "model.py"), os.path.join(up, "utils.py")]),
                config={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                hparams=dict(RNN_hidden_dims=RNN_HIDDEN_DIMS, RNN_depth=RNN_DEPTH, threshold=THRESHOLD, alpha=ALPHA,
                             batch_size=BATCH_SIZE, max_epoch=MAX_EPOCH, early_stop=EARLY_STOP, lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY, steplr=[STEP_SIZE, GAMMA], add_nt=ADD_NT, add_aa=ADD_AA),
                length_cap=LENGTH_CAP)
    t0 = time.time()
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    try:
        set_all_seeds(seed)
        ds = C.D.load_dataset(dataset, data=data, mfe=mfe, load_mfe=bool(cfg["n_mfe"]))
        f = fold
        te, va, tr = ds.test_ids(f), ds.val_ids(f), ds.train_ids(f)
        for t in te:                      # blind: test counts removed before any processing
            del ds.counts[t]
        ds._reps = None
        if smoke:
            tr, va, te = tr[:smoke[0]], va[:smoke[1]], te[:smoke[2]]
        max_epoch = max_epochs or MAX_EPOCH
        extra = None
        if cfg["n_mfe"]:
            mu, sd = ds.mfe_norm(f); extra, nmask = {}, 0
            for t in tr + va:
                extra[t], k = mfe_features(ds, t, mu, sd); nmask += k
            meta.update(mfe_norm=dict(mean=mu, sd=sd), mfe_offsets=list(MFE_OFFSETS), mfe_masked_entries_train_val=nmask)
        lab = Loader(tr + va, ds.seq, {t: ds.counts[t] for t in tr + va}, cfg["target_mode"], extra)
        if cfg["train_filter"] == "coverage_6":
            keep = set(lab.list_gene_coverage_6.tolist()); train_names = [t for t in tr if t in keep]
        else:
            train_names = list(tr)
        if not train_names:
            raise RuntimeError("no training transcript after the training filter")
        val_names = list(va)
        if cfg["loss_mask"] == "padding":
            bad = [t for t in train_names + val_names if not np.isfinite(lab.dict_density[t]).all()]
            if bad: raise ValueError(f"{len(bad)} training/validation transcripts with non-finite target (e.g. {bad[:3]})")
        threshs = get_thresh(lab, train_names)
        ptr = Pack(lab, train_names, cfg["end_trim"], threshs)
        pva = Pack(lab, val_names, cfg["end_trim"], None)
        meta.update(n_train_tx_before_filter=len(tr), n_train_tx=len(train_names), n_val_tx=len(val_names), n_test_tx=len(te),
                    class_thresholds_log10=[float(x) for x in threshs])
        # official load_model order: build on CPU -> Adam / StepLR -> to device -> apply(weights_init)
        input_dims = INPUT_DIMS_BASE + cfg["n_mfe"]
        model = M.RiboMIMO(input_dims, RNN_HIDDEN_DIMS, RNN_DEPTH, num_class=len(THRESHOLD.split(",")) + 1)
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        mse = nn.MSELoss(); ce = nn.CrossEntropyLoss()
        sched = optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
        model = model.to(dev)
        model.apply(M.weights_init)
        meta.update(input_dims=input_dims, n_params=int(sum(p.numel() for p in model.parameters())))
        best, best_state, best_ep, no_improve = -np.inf, None, None, 0
        steps_total, hist = 0, []
        for epoch in range(max_epoch):
            no_improve += 1
            st, ltr, npos = train_one_epoch(model, opt, mse, ce, ptr, cfg["loss_mask"], dev)
            steps_total += st
            sched.step()
            r2 = evaluate_valid(model, pva, cfg["loss_mask"], dev)
            if best < r2:
                best = r2; best_ep = epoch + 1; no_improve = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            hist.append(dict(epoch=epoch + 1, train_loss=ltr, val_r2=r2, lr=opt.param_groups[0]["lr"]))
            print(f"[{config_id}|{dataset}|f{f}|s{seed}] epoch {epoch+1} loss {ltr:.5f} val_r2 {r2:.5f} best_ep {best_ep}", flush=True)
            if no_improve >= EARLY_STOP:
                break
        if best_state is None:
            raise FloatingPointError("validation metric never finite (no checkpoint)")
        model.load_state_dict(best_state)
        preds, provs = infer(model, ds, te, cfg, f, dev)
        meta.update(status="OK", val_metric_name=("pooled R2 on density>0 positions of val_ids(f) (official evaluate)"
                                                  if cfg["loss_mask"] == "density>0" else "pooled R2 on all non-padding positions of val_ids(f)"),
                    val_metric_best=float(best), selected_epoch=best_ep, epochs_run=len(hist), n_optimizer_steps=steps_total,
                    history=hist, seconds=round(time.time() - t0, 1))
    except Exception as e:
        if type(e).__name__ == "OutOfMemoryError":
            print(f"[RETRY-OOM] {config_id} {dataset} f{fold} s{seed}: {e}", flush=True); return 75
        meta.update(status="MODEL_FAILED", failure_reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        C.write(out_root, config_id, dataset, seed, fold, {}, {}, meta)
        print(f"[FAILED] {config_id} {dataset} f{fold} s{seed}: {meta['failure_reason']}", flush=True); return 1
    d = C.write(out_root, config_id, dataset, seed, fold, preds, provs, meta)
    if save_ckpt: torch.save(best_state, os.path.join(d, "model_best.pt"))
    print(f"[OK] {config_id} {dataset} f{fold} s{seed} best_ep {best_ep} -> {d}", flush=True)
    return 0
