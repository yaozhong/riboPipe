"""Evaluation setting 3: component removal and replacement (Supplementary Section S2.4 of the paper), the loss
comparison (Huber vs gamma-weighted log-MSE), training-set size, and the branch lattice.

Every variant is RiboPipe (``arch.MB1v2NoC2MFE``) with one change, trained with exactly the same recipe
(``train.train_one``). ``RiboPipeVariant`` keeps the original module creation order, so the full variant
(all branches, BiGRU, hidden 128) has 228,321 parameters and, loaded with the same weights, gives bit-identical
outputs to RiboPipe (``gate_equivalence``).

  keep      which convolutional branches are kept (codon / nt / aa); removed modules are deleted
  raw187    no convolutions: each position gives [codon one-hot 64 | one-hot of nt -15..+14 (120) | 3 MFEs] to the BiGRU
  backbone  gru (RiboPipe) / lstm / mlp (per-position network, no recurrence) / tf1, tf2, tf4 (Transformer encoder)
"Matched" variants resize the recurrent layer (or Transformer width) to the parameter count of RiboPipe.
"""
import math, os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import arch as A
from . import train as RP
from . import data as D

PATHS = ("codon", "nt", "aa")
BACKBONES = ("gru", "lstm", "mlp", "tf1", "tf2", "tf4")


def _sinusoid(L, d, device):
    pos = torch.arange(L, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(L, d, device=device); pe[:, 0::2] = torch.sin(ang); pe[:, 1::2] = torch.cos(ang)
    return pe


class RiboPipeVariant(A.MB1v2NoC2MFE):
    def __init__(s, bio_dim=18, keep=PATHS, raw187=False, backbone="gru", tf_d=128, tf_heads=4, tf_ff=256, **kw):
        super().__init__(bio_dim, **kw)
        assert backbone in BACKBONES, backbone
        keep = tuple(p for p in PATHS if p in keep); assert keep or raw187
        s.keep, s.raw187, s.backbone = (() if raw187 else keep), bool(raw187), backbone
        dims = dict(codon=s.cc.out_channels, nt=s.cn.out_channels, aa=s.ca.out_channels)
        s.nt_k = s.cn.kernel_size[0]
        for p, nm in (("codon", "cc"), ("nt", "cn"), ("aa", "ca")):
            if s.raw187 or p not in s.keep: delattr(s, nm)
        din = (64 + 4 * s.nt_k + 3) if s.raw187 else (sum(dims[p] for p in s.keep) + 3)
        s.din = din
        H, fcw = s.rnn.hidden_size, s.fc.out_features
        delattr(s, "rnn")
        if backbone in ("gru", "lstm"):
            R = nn.GRU if backbone == "gru" else nn.LSTM
            s.rnn = R(din, H, num_layers=1, batch_first=True, bidirectional=True); dout = 2 * H
        elif backbone == "mlp":
            s.mlp = nn.Sequential(nn.Linear(din, 2 * H), nn.ReLU(), nn.Linear(2 * H, 2 * H)); dout = 2 * H
        else:
            nl = int(backbone[2:]); s.tf_d = tf_d
            s.inp = nn.Linear(din, tf_d)
            layer = nn.TransformerEncoderLayer(tf_d, tf_heads, tf_ff, dropout=0.0, batch_first=True)
            s.tf = nn.TransformerEncoder(layer, nl, enable_nested_tensor=False); dout = tf_d
        if dout != s.fc.in_features: s.fc = nn.Linear(dout, fcw)
        s.dout = dout
        # Transformers run one transcript at a time at its true length (no padding, so no attention mask) with gradient
        # checkpointing, to fit the longest transcripts in GPU memory; mathematically identical to the padded batch.
        s.per_sample = backbone.startswith("tf")

    def _core(s, x, l):
        return s.tf(s.inp(x) + _sinusoid(x.size(1), s.tf_d, x.device)[None])

    def _per_sample(s, h, lengths):
        from torch.utils.checkpoint import checkpoint
        B, L, _ = h.shape; out = h.new_zeros(B, L, s.dout)
        for b in range(B):
            lb = int(lengths[b]); xb = h[b:b + 1, :lb]; lt = lengths[b:b + 1]
            ob = checkpoint(s._core, xb, lt, use_reentrant=False) if (s.training and torch.is_grad_enabled()) else s._core(xb, lt)
            out[b, :lb] = ob[0]
        return out

    def pathways(s, idx, bio, mask=None):
        s._mfe = s.mfe_input(bio, mask)
        b12 = bio[..., :12]
        cod = s.onehot(idx); aa = s.C2A[idx]
        if mask is not None:
            cod = cod * mask.unsqueeze(-1); aa = aa * mask.unsqueeze(-1); b12 = b12 * mask.unsqueeze(-1)
        L = idx.size(1); B = idx.size(0)
        lc, rc = s.span_codon; la, ra = s.span_aa; ln, rn = s.span_nt
        if s.raw187:
            win = F.pad(A.nt3_to_seq(b12), (ln, rn)).unfold(2, s.nt_k, 3)       # (B, 4, L, 30)
            assert win.size(2) == L
            return [cod, win.permute(0, 2, 1, 3).reshape(B, L, 4 * s.nt_k)]
        out = []
        if "codon" in s.keep:
            out.append(s._a1(F.conv1d(F.pad(cod.transpose(1, 2), (lc, rc)), s._w(s.cc), s.cc.bias)).transpose(1, 2))
        if "nt" in s.keep:
            out.append(s._a1(F.conv1d(F.pad(A.nt3_to_seq(b12), (ln, rn)), s._w(s.cn), s.cn.bias, stride=3)).transpose(1, 2))
        if "aa" in s.keep:
            out.append(s._a1(F.conv1d(F.pad(aa.transpose(1, 2), (la, ra)), s._w(s.ca), s.ca.bias)).transpose(1, 2))
        return out

    def forward(s, idx, bio, lengths):
        L = idx.size(1)
        m = (torch.arange(L, device=idx.device)[None, :] < lengths[:, None].to(idx.device)).float()
        parts = s.pathways(idx, bio, m)
        h = torch.cat(list(parts) + [s._mfe], -1) * m.unsqueeze(-1)
        if s.backbone in ("gru", "lstm"):
            pk = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            o, _ = s.rnn(pk); o, _ = pad_packed_sequence(o, batch_first=True, total_length=L)
        elif s.backbone == "mlp":
            o = s.mlp(h)
        elif s.per_sample:
            o = s._per_sample(h, lengths)
        else:
            x = s.inp(h) + _sinusoid(L, s.tf_d, h.device)[None]
            o = s.tf(x, src_key_padding_mask=(m == 0))
        o = o * m.unsqueeze(-1)
        return s.reg(s._m(s.fc(o))).squeeze(-1)


class RiboPipeVariantNoMFE(RiboPipeVariant):
    """Branch-lattice variant without the folding energies (BiGRU input loses the 3 MFE dimensions)."""
    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        assert s.backbone == "gru" and not s.raw187
        s.din = s.din - 3; H = s.rnn.hidden_size
        s.rnn = nn.GRU(s.din, H, num_layers=1, batch_first=True, bidirectional=True)
        s.no_mfe = True

    def pathways(s, idx, bio, mask=None):
        out = super().pathways(idx, bio, mask)
        s._mfe = s._mfe[..., :0]
        return out


_BASE = dict(model="mfe", uses_mfe=True, target="log", loss="gamma_logmse", gamma=0.5)
ALL8 = list(D.HUMAN) + list(D.NONHUMAN)


def _c(group, method, hidden, params, variant, datasets=ALL8, note="", **extra):
    return dict(_BASE, tier=group, group=group, method=method, hidden=hidden, expected_params=params,
                variant=variant, datasets=datasets, note=note, **extra)


# Configuration IDs are those of the paper's runs (Letter_* = experiments added during revision).
ABLATION_CONFIGS = {
    # component removal (Supplementary Table "Evaluation setting 3")
    "Letter_E1_A2_noNT":        _c("removal", "- nucleotide branch", 128, 199873, dict(keep=("codon", "aa"))),
    "Letter_E1_A2_noNT_pm":     _c("removal", "- nucleotide branch, matched", 140, 227209, dict(keep=("codon", "aa"))),
    "Letter_E1_A3_noAA":        _c("removal", "- amino-acid branch", 128, 200353, dict(keep=("codon", "nt"))),
    "Letter_E1_A3_noAA_pm":     _c("removal", "- amino-acid branch, matched", 140, 227689, dict(keep=("codon", "nt"))),
    "Letter_E1_A4_codonMFE":    _c("removal", "codon branch only", 128, 171905, dict(keep=("codon",))),
    "Letter_E1_A4_codonMFE_pm": _c("removal", "codon branch only, matched", 154, 228325, dict(keep=("codon",))),
    "Letter_E1_A5_noRecur":     _c("removal", "per-position network", 128, 127457, dict(backbone="mlp")),
    "Letter_E1_A5_noRecur_pm":  _c("removal", "per-position network, matched", 191, 228635, dict(backbone="mlp")),
    "Letter_E1_A6_noConv":      _c("removal", "no convolutions", 128, 251713, dict(raw187=True)),
    "Letter_E1_A6_noConv_pm":   _c("removal", "no convolutions, matched", 119, 227593, dict(raw187=True)),
    # recurrent-layer replacement
    "Letter_E2_BiLSTM_pm":      _c("replacement", "BiLSTM, matched", 106, 229137, dict(backbone="lstm")),
    "Letter_E2_BiLSTM_H128":    _c("replacement", "BiLSTM, hidden 128", 128, 295137, dict(backbone="lstm")),
    "Letter_E2_TF1_pm":         _c("replacement", "Transformer, 1 layer", 128, 231113, dict(backbone="tf1", tf_d=152, tf_ff=304)),
    "Letter_E2_TF2_pm":         _c("replacement", "Transformer, 2 layers", 128, 226393, dict(backbone="tf2", tf_d=108, tf_ff=216)),
    "Letter_E2_TF4_pm":         _c("replacement", "Transformer, 4 layers", 128, 220321, dict(backbone="tf4", tf_d=76, tf_ff=152)),
    # loss comparison: unweighted Huber (delta 1) on log1p(y), otherwise identical to RiboPipe
    "Letter_E4L_HuberLog":      _c("loss", "Huber delta=1 on log1p(y)", 128, 228321, dict(), loss="huber_log", gamma=None),
    # branch lattice with the folding energies (TX9_WT)
    "Letter_E12_aaMFE":         _c("lattice", "aa + MFE", 128, 138369, dict(keep=("aa",)), datasets=["DMSO_TX9P1"]),
    "Letter_E12_ntMFE":         _c("lattice", "nt + MFE", 128, 138849, dict(keep=("nt",)), datasets=["DMSO_TX9P1"]),
    "Letter_E12_aantMFE":       _c("lattice", "aa + nt + MFE", 128, 166817, dict(keep=("aa", "nt")), datasets=["DMSO_TX9P1"]),
}
for _cid, _keep, _p in (("Letter_E12nm_C", ("codon",), 169601), ("Letter_E12nm_A", ("aa",), 136065), ("Letter_E12nm_N", ("nt",), 136545),
                        ("Letter_E12nm_CA", ("codon", "aa"), 197569), ("Letter_E12nm_CN", ("codon", "nt"), 198049),
                        ("Letter_E12nm_AN", ("aa", "nt"), 164513), ("Letter_E12nm_CAN", ("codon", "nt", "aa"), 226017)):
    ABLATION_CONFIGS[_cid] = _c("lattice", f"{'+'.join(_keep)}, no MFE", 128, _p, dict(keep=_keep), datasets=["DMSO_TX9P1"], no_mfe=True)
# training-set size (human datasets): nested subsets of training similarity units, validation and test unchanged
E3_FRACS = (0.05, 0.1, 0.2, 0.4, 0.7)
E3_EXTRA_DRAWS = {0.05: (1, 2), 0.1: (1, 2)}
for _f in E3_FRACS:
    for _d in (0,) + E3_EXTRA_DRAWS.get(_f, ()):
        ABLATION_CONFIGS[f"Letter_E3_frac{_f:g}_d{_d}"] = _c("train_size", f"training units f={_f:g}, draw {_d}", 128, 228321, dict(),
                                                             datasets=list(D.HUMAN), train_frac=_f, sub_draw=_d)


def build_model(cfg, mfe_mu=0.0, mfe_sd=1.0):
    arch = dict(RP.COMMON_ARCH, rnn_hidden=int(cfg["hidden"]))
    cls = RiboPipeVariantNoMFE if cfg.get("no_mfe") else RiboPipeVariant
    return cls(18, mfe_mu=mfe_mu, mfe_sd=mfe_sd, **arch, **cfg["variant"])


def structure_gate(m, cfg):
    n = RP.count_params(m)
    if n != cfg["expected_params"]:
        raise SystemExit(f"[gate] {n} parameters != expected {cfg['expected_params']}")
    nd = sum(1 for x in m.modules() if isinstance(x, nn.Dropout) and x.p > 0)
    if nd: raise SystemExit(f"[gate] {nd} dropout modules with p > 0")
    v = cfg["variant"]; keep = () if v.get("raw187") else tuple(p for p in PATHS if p in v.get("keep", PATHS))
    have = tuple(p for p, nm in (("codon", "cc"), ("nt", "cn"), ("aa", "ca")) if hasattr(m, nm))
    if have != keep: raise SystemExit(f"[gate] branches {have} != {keep}")
    ks = {nm: getattr(m, nm).kernel_size[0] for nm in ("cc", "cn", "ca") if hasattr(m, nm)}
    if any(ks[k] != dict(cc=3, cn=30, ca=5)[k] for k in ks) or tuple(m.span_nt) != RP.NT_PAD:
        raise SystemExit(f"[gate] kernel widths / nt window {ks} {m.span_nt}")
    bb = v.get("backbone", "gru")
    if m.backbone != bb: raise SystemExit("[gate] backbone mismatch")
    if bb in ("gru", "lstm"):
        R = nn.GRU if bb == "gru" else nn.LSTM
        if not isinstance(m.rnn, R) or m.rnn.input_size != m.din or m.rnn.hidden_size != cfg["hidden"] or m.rnn.num_layers != 1 or not m.rnn.bidirectional:
            raise SystemExit("[gate] recurrent layer mismatch")
    return dict(params=n, kernels=ks, backbone=bb, din=m.din, hidden=cfg["hidden"], rnn=bb, rnn_input=m.din)


_RP_LOSS, _RP_GATE = RP.loss_fn, RP.loss_gate


def loss_fn(name, u, t, mask, gamma):
    if name == "huber_log":
        m = mask.bool(); return F.huber_loss(u[m], t[m], delta=1.0)
    return _RP_LOSS(name, u, t, mask, gamma)


def loss_gate(name, u, t, mask, gamma, target):
    if name != "huber_log": return _RP_GATE(name, u, t, mask, gamma, target)
    with torch.no_grad():
        m = mask.bool(); r = (u - t)[m]; L = float(loss_fn(name, u, t, mask, gamma))
        manual = float(torch.where(r.abs() <= 1.0, 0.5 * r ** 2, r.abs() - 0.5).mean())
        info = dict(loss=name, value=L, manual_huber=manual, target=target)
        if target != "log" or abs(L - manual) > 1e-6 * max(1.0, abs(L)) or not math.isfinite(L):
            raise SystemExit(f"[gate] huber_log check failed {info}")
    return info


def subsample_train(ds, tr, frac, draw):
    """Nested subset of training similarity units: RandomState(1000 + draw) permutation of the sorted units,
    first ceil(frac x n_units) units; for one draw, smaller fractions give subsets of larger ones."""
    units = sorted({ds.unit[t] for t in tr})
    order = list(np.random.RandomState(1000 + int(draw)).permutation(len(units)))
    k = int(math.ceil(frac * len(units)))
    keep_u = {units[i] for i in order[:k]}
    sub = [t for t in tr if ds.unit[t] in keep_u]
    for f2 in E3_FRACS:
        if f2 < frac:
            assert {units[i] for i in order[:int(math.ceil(f2 * len(units)))]} <= keep_u
    return sub, dict(train_frac=frac, sub_draw=int(draw), n_units_full=len(units), n_units_kept=k,
                     n_tx_full=len(tr), n_tx_kept=len(sub))


def gate_equivalence():
    """Full variant (all branches, GRU, H128) == RiboPipe: same parameter count, bit-identical outputs with the same weights."""
    arch = dict(RP.COMMON_ARCH, rnn_hidden=128)
    torch.manual_seed(0)
    ref = A.MB1v2NoC2MFE(18, mfe_mu=-5.0, mfe_sd=3.0, **arch).eval()
    new = RiboPipeVariant(18, mfe_mu=-5.0, mfe_sd=3.0, **arch).eval()
    new.load_state_dict(ref.state_dict(), strict=True)
    g = torch.Generator().manual_seed(1); B, L = 3, 57
    idx = torch.randint(0, 64, (B, L), generator=g); bio = torch.zeros(B, L, 18)
    for b in range(B):
        for t in range(L):
            c = int(idx[b, t])
            for p in range(3): bio[b, t, 4 * p + (c >> (2 * (2 - p))) % 4] = 1.0
    bio[..., 12:15] = -torch.rand(B, L, 3, generator=g) * 20; bio[..., 15:18] = 1.0; bio[:, -20:, 15:18] = 0.0
    ll = torch.tensor([57, 40, 23])
    with torch.no_grad(): a, b = ref(idx, bio, ll), new(idx, bio, ll)
    mk = torch.arange(L)[None] < ll[:, None]
    out = dict(params_ref=RP.count_params(ref), params_new=RP.count_params(new), bitwise=bool(torch.equal(a[mk], b[mk])))
    out["ok"] = out["params_ref"] == out["params_new"] and out["bitwise"]
    return out


def run(config_id, dataset, fold, seed, out_root, device="cuda:0", data=None, mfe=None, max_epochs=0, smoke_n=(0, 0, 0)):
    """One (variant, dataset, seed, fold) run with the RiboPipe recipe; writes the same run directory layout."""
    from . import cli
    cfg = ABLATION_CONFIGS[config_id]
    if dataset not in cfg["datasets"] and dataset not in ("TX9_WT",):
        print(f"[note] {config_id} was run in the paper on {cfg['datasets']}; running on {dataset} anyway")
    RP.loss_fn, RP.loss_gate = loss_fn, loss_gate          # adds 'huber_log'; every other loss goes to the original
    tt = (lambda ds, tr: subsample_train(ds, tr, cfg["train_frac"], cfg["sub_draw"])) if "train_frac" in cfg else None
    try:
        return cli._run_one(config_id, dataset, fold, seed, out_root, device, True, max_epochs, smoke_n, False, data, mfe,
                            configs=ABLATION_CONFIGS, builder=build_model, gate=structure_gate, train_transform=tt)
    finally:
        RP.loss_fn, RP.loss_gate = _RP_LOSS, _RP_GATE


def register(sp):
    p = sp.add_parser("ablation", help="evaluation setting 3: train a component-removal / replacement variant")
    p.add_argument("--config", choices=sorted(ABLATION_CONFIGS), help="variant (omit with --list)")
    p.add_argument("--list", action="store_true", help="list variants and parameter counts")
    p.add_argument("--dataset"); p.add_argument("--fold", type=int); p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-epochs", type=int, default=0); p.add_argument("--smoke-n", type=int, nargs=3, default=(0, 0, 0))
    p.add_argument("--data", default=None); p.add_argument("--mfe", default=None)
    p.set_defaults(func=_main)


def _main(a):
    if a.list or not a.config:
        g = gate_equivalence(); print(f"[gate equivalence] {g}")
        for k, c in ABLATION_CONFIGS.items(): print(f"{k:28s} {c['group']:12s} {c['expected_params']:>8,d}  {c['method']}")
        print(f"{len(ABLATION_CONFIGS)} variants")
        return 0
    folds = [a.fold] if a.fold is not None else range(5)
    rc = 0
    for f in folds:
        rc |= run(a.config, a.dataset, f, a.seed, a.out, a.device, a.data, a.mfe, a.max_epochs, tuple(a.smoke_n))
    return rc


def load_ckpt(path, map_location="cpu"):
    """Rebuild an ablation variant from a checkpoint written by ``run`` (or by the paper's runs)."""
    ck = torch.load(path, map_location=map_location)
    cfg = ABLATION_CONFIGS[ck["config_id"]]; mu, sd = ck["mfe_norm"]
    m = build_model(cfg, mu, sd); m.load_state_dict(ck["state_dict"]); return m, cfg
