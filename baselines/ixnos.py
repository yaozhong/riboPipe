"""iXnos (Tunney et al. 2018, Nat Struct Mol Biol; github.com/lareaulab/iXnos) -- PyTorch re-implementation used in the paper.

The official iXnos code is Python 2 / Theano / Lasagne and is not executed here (and not redistributed).  This file is a
port of the paper's baseline_code/ixnos_ref/ixnos_ffn_paper.py (md5 43bed972...) and of the runner modules
scripts/partI/ixnos_ref/{_ixnos_driver.py, _ixnos_lr005.py, _ixnos_elig.py}; numerics, RNG consumption and order of
operations are unchanged, comments translated.  Official line references are to lareaulab/iXnos (see fetch_upstream.sh,
reference only) and Lasagne master 5d3c63c.

Official recipe kept: codon window [-5,+4] (one-hot, 640) + nucleotide window [-15,+14] (one-hot, 120) = 760 inputs;
structure model adds 3 raw RNAfold MFEs of 30-nt windows starting at nt -17/-16/-15 (763 inputs); Dense(200, tanh) ->
Dense(1, ReLU); Glorot-uniform weights, zero biases; MSE; Nesterov SGD momentum 0.9, lr_epoch = lr/(1 + epoch/decay);
batch 500, incomplete last batch dropped, optimiser (momentum buffer) re-created every epoch (Lasagne behaviour);
fixed number of epochs, last epoch kept; target c[20:L-20]/mean(c[20:L-20]) on the 20/20-trimmed interior.

Adaptations (documented in baselines/README.md):
  * frozen folds; training genes come only from train_ids(f); validation MSE on val_ids(f) logged, never used for selection;
  * official gene filters on the training fold: L > 40, interior sum(c) >= 200 and >= 100 codons with c > 0; then either
    the top 500 by interior density (the official selection, configs without "_elig") or ALL eligible genes ("_elig");
  * schedule (epochs, lr decay): human 70/32 (Iwasaki), yeast 60/16 (Weinberg), E. coli 60/16 (pre-registered);
  * base learning rate 0.005 for "_lr005" configs (value declared in official struc_series.py but not passed; the
    official code actually executes the interface default 0.01, which is used by Paper_iXnos_seq / Paper_iXnos_MFE);
  * pre-registered seeds passed to random / numpy / torch / cuda; no length cap;
  * end positions whose feature window contains zero padding (or an undefined MFE window) are written with provenance
    1 (adapter); the scored interior [20, L-20) is always provenance 0 (model).  "nocut" (setting 2): no trim, target
    c/mean(c) over the full CDS, all positions model.
"""
import os, math, random, json, time, traceback
import numpy as np

from . import common as C

COD_K = 5; NT_WIN = 15; HIDDEN = 200; MOMENTUM = 0.9; BATCH = 500
MIN_CTS = 200; MIN_COD_W_DATA = 100; TRUNC = 20
REL_STRUC = (-17, -16, -15)
DROP_LAST_BATCH = True
RESET_MOMENTUM_EACH_EPOCH = True
EXPECTED_N_PARAMS = {760: 152401, 763: 153001}
LENGTH_CAP = None
SCHEDULE = {"human": (70, 32), "yeast": (60, 16), "ecoli": (60, 16)}
SPECIES = {"DMSO_TX9P1": "human", "GSE233886_WT": "human", "GSE133393_WT": "human", "PRJNA_iPS": "human",
           "Subtelny14_ribomimo": "yeast", "Mohammad16": "ecoli", "Mohammad19-1": "ecoli", "Mohammad19-2": "ecoli"}
VARIANTS = {"seq": (TRUNC, False), "mfe": (TRUNC, True), "nocut": (0, False)}
ALL_ELIGIBLE = 10 ** 9

# config -> (variant, base lr, top_n)
CONFIGS = {
    "Paper_iXnos_seq": ("seq", 0.01, 500),
    "Paper_iXnos_MFE": ("mfe", 0.01, 500),
    "Paper_iXnos_nocut": ("nocut", 0.01, 500),
    "Paper_iXnos_seq_lr005": ("seq", 0.005, 500),
    "Paper_iXnos_MFE_lr005": ("mfe", 0.005, 500),
    "Paper_iXnos_nocut_lr005": ("nocut", 0.005, 500),
    "Paper_iXnos_seq_lr005_elig": ("seq", 0.005, ALL_ELIGIBLE),
    "Paper_iXnos_MFE_lr005_elig": ("mfe", 0.005, ALL_ELIGIBLE),
    "Paper_iXnos_nocut_lr005_elig": ("nocut", 0.005, ALL_ELIGIBLE),
}
NT2ID = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def set_seed(s):
    import torch
    s = int(s); random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ features
def windows(cod_idx, cds_nt, k=COD_K, nt_win=NT_WIN):
    """Raw 0/1 one-hot: codons [-k, k-1] + nucleotides [-nt_win, nt_win-1] relative to the first nt (3i) of codon i.
    Returns (F (L, 760) float32, complete (L,) bool): complete = window entirely inside the CDS and all ACGT."""
    L = len(cod_idx); W = 2 * k; Dm = W * 64 + 2 * nt_win * 4
    F = np.zeros((L, Dm), np.float32); complete = np.ones(L, bool)
    ii = np.arange(L); cod_idx = np.asarray(cod_idx)
    for j, off in enumerate(range(-k, k)):
        p = ii + off; inb = (p >= 0) & (p < L)
        c = np.full(L, -1, np.int64); c[inb] = cod_idx[p[inb]]
        ok = c >= 0
        F[ii[ok], j * 64 + c[ok]] = 1.0
        complete &= ok
    nt = np.array([NT2ID.get(ch, -1) for ch in cds_nt.upper()], np.int64); n = len(nt); base = W * 64
    for j, off in enumerate(range(-nt_win, nt_win)):
        p = 3 * ii + off; inb = (p >= 0) & (p < n)
        v = np.full(L, -1, np.int64); v[inb] = nt[p[inb]]
        ok = v >= 0
        F[ii[ok], base + j * 4 + v[ok]] = 1.0
        complete &= ok
    return F, complete


def struc_feats(mfe_raw, n_valid, L):
    """3 raw MFEs (kcal/mol, unstandardised).  mfe_raw[p] = MFE of seq[p:p+30]; valid iff 0 <= p < n_valid
    (the official .fold files include 13 nt of 5'UTR / 10 nt of 3'UTR; ours are CDS-only, so the first ~6 and last ~5
    codons have no defined window; the 20/20 interior is always defined)."""
    out = np.zeros((L, len(REL_STRUC)), np.float32); ok = np.ones(L, bool)
    mfe_raw = np.asarray(mfe_raw, np.float32)
    for j, r in enumerate(REL_STRUC):
        ps = np.arange(L) * 3 + r; v = (ps >= 0) & (ps < n_valid)
        out[v, j] = mfe_raw[ps[v]]; ok &= v
    return out, ok


def features(variant, cod_idx, cds_nt, mfe=None):
    F, ok = windows(cod_idx, cds_nt)
    if VARIANTS[variant][1]:
        mfe_raw, n_valid = mfe
        S, sok = struc_feats(mfe_raw, n_valid, len(cod_idx))
        F = np.concatenate([F, S], 1); ok = ok & sok
    return F, ok


def prov_for(variant, model_ok):
    if variant == "nocut":
        return np.zeros(len(model_ok), np.int8)
    return np.where(model_ok, 0, 1).astype(np.int8)


# ------------------------------------------------------------------ training-gene selection (training fold only)
def select_training(train_ids, counts, trunc, L_of, top_n):
    """Official make_codon_set_files on the current training fold: L > 2*trunc; interior sum(c) >= 200 and
    #(c > 0) >= 100; sorted by interior density (descending, ties by id); first top_n (all if fewer)."""
    led = {"n_train_ids": len(train_ids)}
    s1 = [t for t in train_ids if L_of[t] > 2 * trunc]
    led["n_len_ok"] = len(s1)
    def interior(t):
        return np.asarray(counts[t], float)[trunc:L_of[t] - trunc]
    s2 = [t for t in s1 if interior(t).sum() >= MIN_CTS]
    led["n_min_cts_ok"] = len(s2)
    s3 = [t for t in s2 if int((interior(t) > 0).sum()) >= MIN_COD_W_DATA]
    led["n_min_cod_w_data_ok"] = len(s3)
    dens = {t: float(interior(t).mean()) for t in s3}
    order = sorted(s3, key=lambda t: (-dens[t], t))
    sel = order[:top_n]
    led.update(n_eligible=len(s3), n_selected=len(sel), top_n=(top_n if top_n < ALL_ELIGIBLE else "all"),
               density_cutoff=(dens[sel[-1]] if sel else None))
    return sel, led


def target_rows(variant, c):
    trunc = VARIANTS[variant][0]; L = len(c); lo, hi = trunc, L - trunc
    if hi - lo <= 0: return None
    sub = np.asarray(c, float)[lo:hi]; m = sub.mean()
    if not (m > 0): return None
    return lo, hi, (sub / m).astype(np.float32)


def build_matrices(variant, ids, counts, seqs, mfe_of=None):
    Xs, ys, n_tx = [], [], 0
    for t in ids:
        c = counts[t]; r = target_rows(variant, c)
        if r is None: continue
        lo, hi, y = r
        L = len(c); idx = C.seq_to_idx(seqs[t], L)
        X, ok = features(variant, idx, seqs[t], mfe_of(t) if mfe_of else None)
        if variant != "nocut":
            assert ok[lo:hi].all(), (t, "interior row contains zero padding")
        Xs.append(X[lo:hi]); ys.append(y); n_tx += 1
    if not Xs:
        return np.zeros((0, 760 + 3 * VARIANTS[variant][1]), np.float32), np.zeros(0, np.float32), 0
    return np.concatenate(Xs), np.concatenate(ys), n_tx


# ------------------------------------------------------------------ model
class ModelFailure(Exception):
    pass


def _ffn(din):
    import torch.nn as nn, torch

    class FFN(nn.Module):
        """Dense(din -> 200, tanh) -> Dense(200 -> 1, rectify); Lasagne defaults GlorotUniform / zero bias."""
        def __init__(self):
            super().__init__()
            self.h = nn.Linear(din, HIDDEN); self.out = nn.Linear(HIDDEN, 1)
            nn.init.xavier_uniform_(self.h.weight); nn.init.zeros_(self.h.bias)
            nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)

        def forward(self, x):
            return torch.relu(self.out(torch.tanh(self.h(x)))).squeeze(-1)
    return FFN()


def _predict_rows(net, X, dev, chunk=65536):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            out.append(net(torch.as_tensor(X[i:i + chunk], device=dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train_once(Xtr, ytr, Xva, yva, epochs, lr_decay, seed, dev, base_lr, log=print):
    import torch, torch.nn as nn
    set_seed(seed)
    net = _ffn(Xtr.shape[1]).to(dev)
    n_par = int(sum(p.numel() for p in net.parameters()))
    assert n_par == EXPECTED_N_PARAMS[Xtr.shape[1]], (n_par, Xtr.shape)
    lossf = nn.MSELoss()
    Xt = torch.as_tensor(Xtr, device=dev); yt = torch.as_tensor(ytr, device=dev, dtype=torch.float32)
    n = len(Xt); nb = n // BATCH if DROP_LAST_BATCH else int(math.ceil(n / BATCH))
    if nb == 0:
        raise ModelFailure(f"{n} training windows < batch {BATCH}: the official iterate_minibatches yields no batch")
    hist = []; steps = 0; opt = None
    for ep in range(epochs):
        lr = base_lr / (1.0 + ep / float(lr_decay))
        if opt is None or RESET_MOMENTUM_EACH_EPOCH:
            opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=MOMENTUM, nesterov=True)
        else:
            for g in opt.param_groups: g["lr"] = lr
        net.train()
        perm = torch.randperm(n, device=dev)
        tl = 0.0
        for bi in range(nb):
            b = perm[bi * BATCH:(bi + 1) * BATCH]
            opt.zero_grad()
            l = lossf(net(Xt[b]), yt[b]); l.backward(); opt.step()
            tl += float(l.detach())
        steps += nb
        net.eval()
        if len(Xva):
            pv = _predict_rows(net, Xva, dev)
            vl = float(np.mean((pv.astype(np.float64) - yva.astype(np.float64)) ** 2))
        else:
            vl = float("nan")
        hist.append({"epoch": ep + 1, "lr": lr, "train_mse": tl / nb, "val_mse": vl})
        if ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            log(f"[ixnos] ep{ep+1:3d} lr={lr:.5f} train_mse={tl/nb:.5f} val_mse={vl:.5f}")
    net.eval()
    return {"net": net, "history": hist, "n_steps": steps, "n_windows": n, "n_batches_per_epoch": nb,
            "n_dropped_tail_windows": n - nb * BATCH, "n_params": n_par}


def predict_tx(net, variant, seq, L, dev, mfe=None):
    idx = C.seq_to_idx(seq, L)
    X, ok = features(variant, idx, seq, mfe)
    p = _predict_rows(net, X, dev).astype(np.float32)
    prov = prov_for(variant, ok)
    if L > 40:
        assert (prov[20:L - 20] == 0).all(), "non-model position inside the interior domain"
    return p, prov


# ------------------------------------------------------------------ driver
def run(config_id, dataset, out_root, folds=range(5), seeds=(123,), device="auto", smoke=None, max_epochs=0,
        overwrite=False, data=None, mfe=None, save_ckpt=True):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    variant, base_lr, top_n = CONFIGS[config_id]
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    species = SPECIES[dataset]; epochs, lr_decay = SCHEDULE[species]
    if max_epochs: epochs = max_epochs
    log = C.log_fn(f"{config_id}/{dataset}")
    use_mfe = VARIANTS[variant][1]
    ds = C.D.load_dataset(dataset, data=data, mfe=mfe, load_mfe=use_mfe)
    mfe_of = (lambda t: (ds.mfe_raw(t), ds.mfe_valid(t))) if use_mfe else None
    L_of = {t: len(ds.seq[t]) // 3 for t in ds.ids}
    log(f"variant={variant} lr={base_lr} top_n={'all' if top_n == ALL_ELIGIBLE else top_n} species={species} "
        f"epochs={epochs} decay={lr_decay} device={dev}")
    rc = 0
    for f in folds:
        todo = [s for s in seeds if overwrite or not C.run_exists(out_root, config_id, dataset, s, f)]
        if not todo:
            log(f"fold{f}: all seeds done, skipped"); continue
        tr, va, te = ds.train_ids(f), ds.val_ids(f), ds.test_ids(f)
        if smoke: tr, va, te = C.smoke_pick(tr, smoke[0]), C.smoke_pick(va, smoke[1]), C.smoke_pick(te, smoke[2])
        counts = {t: np.asarray(ds.counts[t], float) for t in list(tr) + list(va)}   # no test counts
        sel, led = select_training(tr, counts, VARIANTS[variant][0], L_of, top_n)
        Xtr, ytr, ntx = build_matrices(variant, sel, counts, ds.seq, mfe_of)
        Xva, yva, nva = build_matrices(variant, va, counts, ds.seq, mfe_of)
        log(f"fold{f}: training filter {json.dumps(led)}; train windows {len(Xtr)} ({ntx} tx) x {Xtr.shape[1]}; val windows {len(Xva)}")
        for s in todo:
            meta = dict(method=f"iXnos ({variant})", config_id_paper=config_id, variant=variant, base_lr=base_lr,
                        training_selection=("all eligible genes" if top_n == ALL_ELIGIBLE else f"top {top_n} by interior density"),
                        species=species, epochs=epochs, lr_decay=lr_decay, training_filter=led, n_val_tx=nva,
                        n_val_windows=int(len(Xva)), batch=BATCH, input_dim=int(Xtr.shape[1]), device=str(dev),
                        length_cap=LENGTH_CAP, smoke=(list(smoke) if smoke else False), max_epochs_override=(max_epochs or None),
                        code_md5=C.code_md5([os.path.abspath(__file__), os.path.join(C.HERE, "common.py")]))
            try:
                t0 = time.time()
                res = train_once(Xtr, ytr, Xva, yva, epochs, lr_decay, s, dev, base_lr, log)
                o = _predict_rows(res["net"], Xtr, dev); allzero = bool(np.all(o == 0))
                preds, provs = {}, {}
                for t in te:
                    p, q = predict_tx(res["net"], variant, ds.seq[t], L_of[t], dev, mfe_of(t) if mfe_of else None)
                    preds[t] = p; provs[t] = q
                hist = res["history"]
                meta.update(status="OK", n_train_tx=ntx, n_train_windows=res["n_windows"], n_optimizer_steps=res["n_steps"],
                            n_batches_per_epoch=res["n_batches_per_epoch"], n_dropped_tail_windows=res["n_dropped_tail_windows"],
                            n_params=res["n_params"], val_metric_name="val MSE (frozen val_ids; logged, not used for selection)",
                            val_metric_best=hist[-1]["val_mse"] if hist else None, selected_epoch=hist[-1]["epoch"] if hist else None,
                            selection_rule="fixed epochs, last epoch (official: no early stopping)",
                            failure_reason=("train_output_all_zero (fixed-seed official recipe: not replaced; scorer decides)" if allzero else None),
                            seconds=round(time.time() - t0, 1), history=hist)
                d = C.write(out_root, config_id, dataset, s, f, preds, provs, meta)
                if save_ckpt: torch.save(res["net"].state_dict(), os.path.join(d, "model.pt"))
                log(f"seed{s} fold{f}: wrote {d} (flat/non-finite {meta['n_test_tx_flat_or_nonfinite']}; prov {meta['prov_counts']})")
            except ModelFailure as e:
                meta.update(status="MODEL_FAILED", failure_reason=f"ModelFailure: {e}", n_train_tx=ntx, n_train_windows=int(len(Xtr)))
                C.write(out_root, config_id, dataset, s, f, {}, {}, meta); log(f"seed{s} fold{f}: MODEL_FAILED {e}")
            except Exception:
                traceback.print_exc(); log(f"seed{s} fold{f}: machine/environment error, no run.json written"); rc = 1
    return rc
