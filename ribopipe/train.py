"""Training and inference of RiboPipe, RiboPipe (-MFE), their variants and the BiLSTM-97 control.

This is the code path that produced the paper's results (PROTOCOL_PartI_v1.3), with file paths made configurable.
Test transcripts enter only through their sequence and MFE track; their counts are never read. All inputs are in
A-site coordinates and no register offset is applied. There is no length cap.

Recipe (shared by all configs): windows of 64 consecutive transcripts from a seeded shuffle, window start every
16 transcripts (ceil(N/16) optimiser steps per epoch); Adam (lr 1e-3, weight decay 1e-6); StepLR (step 5, gamma 0.95,
stepped after each epoch's validation); epochs 1..99; early stopping with patience 5 on the median per-transcript
Pearson correlation on the frozen validation transcripts (back-transformed scale, full CDS, L >= 20); the best
epoch's weights are kept. Loss: gamma-weighted log-MSE, mean[(1 + y)^0.5 (u - log1p(y))^2], y = c / mean(c).
"""
import os, sys, json, math, time, random, hashlib, platform, socket
import numpy as np
import torch
import torch.nn as nn

from . import arch as A
from . import data as D

LR = 1e-3
WEIGHT_DECAY = 1e-6
STEPLR_STEP = 5
STEPLR_GAMMA = 0.95
BATCH_WINDOW = 64
BATCH_STRIDE = 16
EPOCH_FIRST, EPOCH_LAST = 1, 99
PATIENCE = 5
VAL_MIN_POS = 20
PRED_BATCH = 16
NT_PAD = (15, 12)               # nucleotide kernel 30, stride 3: nt -15..+14 relative to the first A-site nucleotide
MFE_REL = (-17, -16, -15)       # codon i uses the 30-nt windows starting at 3i-17, 3i-16, 3i-15 (iXnos positions)
GAMMA_DEFAULT = 0.5
DROPOUT = 0.0

COMMON_ARCH = dict(Fc=64, Fn=32, Fa=32, ch2=64, fc=32, rnn_layers=1, first_act="exp", clamp_max=5.0,
                   cell="gru", gauge=True, span_codon=(2, 0), span_aa=(5, -1), fuse_k=3, fc_act="relu",
                   nt_pad=NT_PAD)

HUMAN, NONHUMAN = list(D.HUMAN), list(D.NONHUMAN)
CONFIGS = {
    "Paper_RiboPipe_MFE_H128": dict(
        tier="T1", method="RiboPipe", model="mfe", hidden=128, target="log", loss="gamma_logmse", gamma=GAMMA_DEFAULT,
        expected_params=228321, datasets=HUMAN + NONHUMAN,
        also_write={"Paper_RiboPipe_MFE_H128_sameCkpt": "interior"}, note="RiboPipe (evaluation setting 1)"),
    "Paper_RiboPipe_noMFE_H128": dict(
        tier="T3", method="RiboPipe (-MFE)", model="nomfe", hidden=128, target="log", loss="gamma_logmse", gamma=GAMMA_DEFAULT,
        expected_params=226017, datasets=HUMAN + NONHUMAN, note="RiboPipe without the three folding energies"),
    "Paper_RiboPipe_MFE_H110": dict(
        tier="T3", method="RiboPipe", model="mfe", hidden=110, target="log", loss="gamma_logmse", gamma=GAMMA_DEFAULT,
        expected_params=187101, datasets=HUMAN + NONHUMAN, note="hidden size 110"),
    "Paper_RiboPipe_noMFE_H110": dict(
        tier="T3", method="RiboPipe (-MFE)", model="nomfe", hidden=110, target="log", loss="gamma_logmse", gamma=GAMMA_DEFAULT,
        expected_params=185121, datasets=HUMAN + NONHUMAN, note="hidden size 110, no MFE"),
    "Paper_RiboPipe_MFE_linear": dict(
        tier="T2", method="RiboPipe", model="mfe", hidden=128, target="linear", loss="linmse", gamma=None,
        expected_params=228321, datasets=HUMAN + NONHUMAN, note="linear target y = c/mean(c), plain squared error"),
    "Paper_RiboPipe_MFE_logSE": dict(
        tier="T3", method="RiboPipe", model="mfe", hidden=128, target="log", loss="logmse", gamma=None,
        expected_params=228321, datasets=HUMAN + NONHUMAN, note="log1p(y) target, plain squared error"),
    "Paper_RiboPipe_MFEhead": dict(
        tier="PartIII", method="RiboPipe-MFEhead", model="mfehead", hidden=110, target="log", loss="gamma_logmse", gamma=GAMMA_DEFAULT,
        expected_params=185202, datasets=HUMAN, note="additive MFE head (output = f_seq + q_MFE on the log1p scale)"),
    "Paper_BiLSTM97": dict(
        tier="control", method="BiLSTM-97", model="oh97", hidden=128, target="log", loss="gamma_logmse", gamma=GAMMA_DEFAULT,
        expected_params=240705, datasets=HUMAN + NONHUMAN, note="97-dim one-hot into one BiLSTM(128), RiboPipe recipe"),
}
LOSSES = ("gamma_logmse", "logmse", "linmse")
PREREG_SEEDS = (123, 124, 125)
DEFAULT_SEED = 123

CODONS = A.CODONS
CIDX = {c: i for i, c in enumerate(CODONS)}
NT_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
PAD_IDX = 64


class TrainingFailure(RuntimeError):
    pass


def set_seed(seed, deterministic=True):
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)


def env_info():
    d = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
             cuda=torch.version.cuda, cudnn=(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
             host=socket.gethostname())
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return d


md5_file = D.md5_file


def code_md5():
    here = os.path.dirname(os.path.abspath(__file__))
    return {f: md5_file(os.path.join(here, f)) for f in ("arch.py", "train.py", "data.py")}


# ---------------------------------------------------------------- inputs (sequence and MFE track only)
def codon_idx(seq):
    """Codon index 0..63; non-ACGT triplets -> 64 (all-zero one-hot and amino acid)."""
    L = len(seq) // 3
    return np.array([CIDX.get(seq[3 * i:3 * i + 3], PAD_IDX) for i in range(L)], dtype=np.int64)


def nt3_onehot(seq, L):
    """(L, 12): one-hot of the 3 nucleotides of codon i; block j at codon i = absolute position 3i+j."""
    s = np.array([NT_MAP.get(c, -1) for c in seq], dtype=np.int64)
    out = np.zeros((L, 12), np.float32)
    for j in range(3):
        b = s[np.arange(L) * 3 + j]
        ok = b >= 0
        out[np.where(ok)[0], 4 * j + b[ok]] = 1.0
    return out


def mfe_features(raw, n_valid, L):
    """(L, 3) raw MFEs and (L, 3) valid flags; column k of codon i = track position p = 3i + MFE_REL[k],
    valid iff 0 <= p < n_valid. Invalid raw values are set to 0 (the model masks them by the flag)."""
    v = np.zeros((L, 3), np.float32); ok = np.zeros((L, 3), np.float32)
    for k, rel in enumerate(MFE_REL):
        ps = np.arange(L) * 3 + rel
        good = (ps >= 0) & (ps < n_valid)
        v[good, k] = raw[ps[good]]; ok[good, k] = 1.0
    return v, ok


def build_inputs(ds, tx, model_kind):
    """(idx (L,), bio (L, D)); D = 18 for MFE models, 12 otherwise."""
    seq = ds.seq[tx]; L = len(seq) // 3
    assert len(seq) == 3 * L
    idx = codon_idx(seq); nt = nt3_onehot(seq, L)
    if model_kind in ("mfe", "mfehead"):
        nv = ds.mfe_valid(tx); raw = ds.mfe_raw(tx)
        assert len(raw) == len(seq) and nv == len(seq) - 29, (tx, len(raw), nv)
        v, ok = mfe_features(raw, nv, L)
        return idx, np.concatenate([nt, v, ok], 1)
    return idx, nt


def targets(counts, target):
    """y = c / mean(c) over the full CDS (the scored truth); log target t = log1p(y), linear t = y."""
    c = np.asarray(counts, np.float64); m = c.mean()
    y = c / m
    return (np.log1p(y) if target == "log" else y).astype(np.float32)


def collate(items):
    """items: [(tx, idx, bio, tgt or None, L)] -> padded tensors, sorted by decreasing length."""
    items = sorted(items, key=lambda x: x[4], reverse=True)
    B = len(items); Lmax = items[0][4]; Dm = items[0][2].shape[1]
    idx = torch.full((B, Lmax), PAD_IDX, dtype=torch.long)
    bio = torch.zeros(B, Lmax, Dm); tgt = torch.zeros(B, Lmax); mask = torch.zeros(B, Lmax, dtype=torch.bool)
    for i, (_, ii, bb, tt, l) in enumerate(items):
        idx[i, :l] = torch.from_numpy(ii); bio[i, :l] = torch.from_numpy(bb); mask[i, :l] = True
        if tt is not None: tgt[i, :l] = torch.from_numpy(tt)
    return idx, bio, tgt, mask, torch.tensor([x[4] for x in items]), [x[0] for x in items]


# ---------------------------------------------------------------- model
def build_model(cfg, mfe_mu=0.0, mfe_sd=1.0):
    kind, H = cfg["model"], int(cfg["hidden"])
    arch = dict(COMMON_ARCH, rnn_hidden=H)
    if kind == "mfe":
        return A.MB1v2NoC2MFE(18, mfe_mu=mfe_mu, mfe_sd=mfe_sd, **arch)
    if kind == "nomfe":
        return A.MB1v2NoC2(12, **arch)
    if kind == "mfehead":
        return A.MB1v2EndHead(18, use_pwm=False, use_mfe=True, mfe_mu=mfe_mu, mfe_sd=mfe_sd, **arch)
    if kind == "oh97":
        a = dict(arch); a["cell"] = "lstm"
        return A.MB1v2OH97(12, fusion="lin1", **a)
    raise ValueError(kind)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def structure_gate(m, cfg):
    """Measured parameter count == expected; no dropout; kernel widths and recurrent input as defined."""
    n = count_params(m)
    if n != cfg["expected_params"]:
        raise SystemExit(f"[gate] {n} parameters != expected {cfg['expected_params']} ({cfg['model']} H{cfg['hidden']})")
    nd = sum(1 for x in m.modules() if isinstance(x, nn.Dropout))
    if nd != 0: raise SystemExit(f"[gate] {nd} dropout modules found (dropout = 0)")
    if cfg["model"] == "oh97":
        g = A.gate_x97(m)
        if not g["ok"] or not isinstance(m.rnn, nn.LSTM): raise SystemExit(f"[gate] x97 structure check failed {g}")
        ks = None
    else:
        ks = (m.cc.kernel_size[0], m.cn.kernel_size[0], m.ca.kernel_size[0])
        if ks != (3, 30, 5) or m.cn.stride[0] != 3 or tuple(m.span_nt) != NT_PAD:
            raise SystemExit(f"[gate] kernel widths / nt window {ks} stride {m.cn.stride} span_nt {m.span_nt}")
        want_in = 128 + (3 if cfg["model"] == "mfe" else 0)
        if m.c2 is not None or m.rnn.input_size != want_in or not isinstance(m.rnn, nn.GRU):
            raise SystemExit(f"[gate] fusion / GRU input c2={m.c2} rnn_in={m.rnn.input_size}, need {want_in}")
    if m.rnn.hidden_size != cfg["hidden"] or m.rnn.num_layers != 1 or not m.rnn.bidirectional:
        raise SystemExit("[gate] recurrent layer is not one bidirectional layer of the expected size")
    return dict(params=n, kernels=ks, rnn=type(m.rnn).__name__, rnn_input=m.rnn.input_size, hidden=m.rnn.hidden_size,
                dropout_modules=nd)


# ---------------------------------------------------------------- loss (dispatched by name, no fallback)
def loss_fn(name, u, t, mask, gamma):
    m = mask.bool(); u, t = u[m], t[m]
    if name == "gamma_logmse":          # mean[(1 + y)^gamma (u - log1p(y))^2], y = expm1(t) clamped at 0
        y = torch.expm1(t).clamp(min=0.0)
        return (((1.0 + y) ** gamma) * (u - t) ** 2).mean()
    if name == "logmse" or name == "linmse":
        return ((u - t) ** 2).mean()
    raise ValueError(f"unknown loss {name!r}")


def loss_gate(name, u, t, mask, gamma, target):
    """First-batch check that the loss is what the config says (gamma weighting active, target scale matches)."""
    with torch.no_grad():
        L = float(loss_fn(name, u, t, mask, gamma)); se = float(((u - t)[mask.bool()] ** 2).mean())
        info = dict(loss=name, value=L, plain_se=se, target=target)
        if not math.isfinite(L): raise TrainingFailure("first-batch loss is not finite")
        if name == "gamma_logmse":
            if target != "log": raise SystemExit("[gate] gamma_logmse needs the log target")
            w = (1.0 + torch.expm1(t[mask.bool()]).clamp(min=0)) ** gamma
            info.update(w_min=float(w.min()), w_max=float(w.max()), w_median=float(w.median()))
            if float(w.max()) <= 1.0 + 1e-6 or abs(L - se) < 1e-12:
                raise SystemExit(f"[gate] gamma weighting not active {info}")
        elif name == "logmse" and target != "log":
            raise SystemExit("[gate] logmse needs the log target")
        elif name == "linmse" and target != "linear":
            raise SystemExit("[gate] linmse needs the linear target")
    return info


# ---------------------------------------------------------------- inference and validation metric
def back_transform(u, target):
    return np.expm1(u) if target == "log" else u


@torch.no_grad()
def predict_raw(model, items, dev, batch=PRED_BATCH):
    """items: [(tx, idx, bio, _, L)] -> {tx: raw model output u (float64, length L)}."""
    model.eval(); out = {}
    order = sorted(range(len(items)), key=lambda i: items[i][4], reverse=True)
    for s in range(0, len(order), batch):
        sub = [items[i] for i in order[s:s + batch]]
        idx, bio, _, _, ll, ks = collate(sub)
        y = model(idx.to(dev), bio.to(dev), ll.to(dev)).float().cpu().numpy()
        for r, k in enumerate(ks):
            out[k] = y[r, :int(ll[r])].astype(np.float64)
    return out


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean(); d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def val_metrics(u_by_tx, y_by_tx, target):
    """native   (early-stopping criterion): median per-transcript Pearson on the back-transformed scale, full CDS,
                transcripts with L >= 20 and non-constant prediction and truth;
       interior (same-checkpoint sensitivity): the same over interior codons [20, L-20);
       dev_log  recorded only: Pearson on the log1p scale."""
    nat, inte, dlog = [], [], []
    for k, u in u_by_tx.items():
        y = y_by_tx[k]; p = back_transform(u, target); L = len(y)
        if L >= VAL_MIN_POS and np.isfinite(p).all() and p.std() > 0 and y.std() > 0:
            nat.append(_pearson(p, y))
        if L > 40:
            ii = slice(20, L - 20); pi, yi = p[ii], y[ii]
            if len(yi) >= 3 and yi.std() > 0 and np.isfinite(pi).all() and pi.std() > 0:
                inte.append(_pearson(pi, yi))
        if target == "log":
            ly = np.log1p(y)
            if L >= VAL_MIN_POS and np.isfinite(u).all() and u.std() > 0 and ly.std() > 0:
                dlog.append(_pearson(u, ly))
    med = lambda v: float(np.median(v)) if v else -1.0
    return dict(native=med(nat), interior=med(inte), dev_log=(med(dlog) if target == "log" else None),
                n_native=len(nat), n_interior=len(inte))


# ---------------------------------------------------------------- training
def make_items(ds, ids, kind, target, with_target, inputs=build_inputs):
    items = []; dropped = []
    for k in ids:
        idx, bio = inputs(ds, k, kind); L = len(idx)
        if with_target:
            c = ds.counts[k]
            if L == 0 or float(np.sum(c)) <= 0:
                dropped.append(k); continue
            items.append((k, idx, bio, targets(c, target), L))
        else:
            items.append((k, idx, bio, None, L))
    return items, dropped


_make_items = make_items


def train_one(cfg, ds, fold, seed, dev, train_ids, val_ids, max_epochs=None, deterministic=True, log=print,
              model_builder=None, gate=None, inputs=build_inputs):
    set_seed(seed, deterministic)
    needs_mfe = cfg["model"] in ("mfe", "mfehead") or cfg.get("uses_mfe", False)
    mu, sd = ds.mfe_norm(fold) if needs_mfe else (0.0, 1.0)
    model = (model_builder or build_model)(cfg, mu, sd)
    g = (gate or structure_gate)(model, cfg)
    log(f"[gate] {cfg['method']} {cfg['model']} H{cfg['hidden']}: {g['params']} parameters == expected "
        f"{cfg['expected_params']}; MFE standardisation mu={mu:.6f} sd={sd:.6f} (fold {fold} training windows)")
    model = model.to(dev)
    tr, drop_tr = make_items(ds, train_ids, cfg["model"], cfg["target"], True, inputs)
    va, drop_va = make_items(ds, val_ids, cfg["model"], cfg["target"], True, inputs)
    y_val = {k: np.asarray(ds.truth(k), np.float64) for (k, *_r) in va}
    opt = torch.optim.Adam(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEPLR_STEP, gamma=STEPLR_GAMMA)
    rng = np.random.RandomState(int(seed))
    last = EPOCH_LAST if max_epochs is None else min(EPOCH_LAST, EPOCH_FIRST + int(max_epochs) - 1)
    best = {"native": (-1.0, None, None), "interior": (-1.0, None, None)}
    bad = 0; steps = 0; hist = []; gate_info = None; stopped_early = False
    N = len(tr); steps_per_epoch = int(math.ceil(N / BATCH_STRIDE))
    for ep in range(EPOCH_FIRST, last + 1):
        model.train(); o = np.arange(N); rng.shuffle(o); t0 = time.time(); ls = []
        for i in range(0, N, BATCH_STRIDE):
            sub = [tr[j] for j in o[i:i + BATCH_WINDOW]]
            idx, bio, tgt, mask, ll, _ = collate(sub)
            idx, bio, tgt, mask, ll = [x.to(dev) for x in (idx, bio, tgt, mask, ll)]
            u = model(idx, bio, ll)
            if gate_info is None:
                gate_info = loss_gate(cfg["loss"], u, tgt, mask, cfg["gamma"], cfg["target"]); log(f"[loss-gate] {gate_info}")
            loss = loss_fn(cfg["loss"], u, tgt, mask, cfg["gamma"])
            if not torch.isfinite(loss):
                raise TrainingFailure(f"training loss not finite (epoch {ep} step {steps + 1})")
            opt.zero_grad(); loss.backward(); opt.step(); steps += 1; ls.append(float(loss))
        vm = val_metrics(predict_raw(model, va, dev), y_val, cfg["target"])
        lr_now = opt.param_groups[0]["lr"]; sched.step()
        rec = dict(epoch=ep, train_loss=float(np.mean(ls)), lr=lr_now, sec=round(time.time() - t0, 2), **vm)
        hist.append(rec)
        log(f"  [ep {ep:3d}] loss={rec['train_loss']:.5f} val native={vm['native']:.5f} interior={vm['interior']:.5f} "
            f"lr={lr_now:.6g} ({rec['sec']}s)")
        if vm["interior"] > best["interior"][0]:
            best["interior"] = (vm["interior"], ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if vm["native"] > best["native"][0]:
            best["native"] = (vm["native"], ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                stopped_early = True; break
    info = dict(n_train_tx=len(tr), n_val_tx=len(va), dropped_train=drop_tr, dropped_val=drop_va, n_optimizer_steps=steps,
                steps_per_epoch=steps_per_epoch, epochs_run=len(hist), stopped_early=stopped_early, history=hist,
                loss_gate=gate_info, mfe_norm=[mu, sd], structure=g)
    return model, best, info


def save_ckpt(path, model, cfg, config_id, dataset, seed, fold, sel_epoch, mfe_norm):
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, config_id=config_id,
                    config={k: v for k, v in cfg.items() if k != "also_write"}, dataset=dataset, seed=seed, fold=fold,
                    selected_epoch=sel_epoch, mfe_norm=mfe_norm, arch=dict(COMMON_ARCH, rnn_hidden=cfg["hidden"])), path)
    return md5_file(path)


def load_ckpt(path, map_location="cpu"):
    """Rebuild a model from a self-contained checkpoint (the paper's checkpoints load unchanged)."""
    ck = torch.load(path, map_location=map_location)
    cfg = ck["config"]; mu, sd = ck["mfe_norm"]
    m = build_model(cfg, mu, sd); m.load_state_dict(ck["state_dict"]); return m, ck


def predict_transcripts(model, ds, ids, cfg, dev="cpu"):
    """Predicted relative footprint density (back-transformed) for transcripts of a dataset, from sequence and MFE."""
    items, _ = make_items(ds, ids, cfg["model"], cfg["target"], with_target=False)
    u = predict_raw(model.to(dev), items, torch.device(dev))
    return {k: back_transform(u[k], cfg["target"]).astype(np.float32) for k in ids}
