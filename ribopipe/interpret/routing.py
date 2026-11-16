"""Branch routing of single edits at sampled interior codons.

The input of the BiGRU is the concatenation h = [codon branch | nt branch | aa branch | MFE] (64 + 32 + 32 + 3).
For an edited sequence the routed response of block k keeps the wild-type values of all other blocks and takes the
edited values of block k only; "total" takes all blocks from the edited sequence. Such inputs do not occur in real
sequences, so the shares measure the reliance of the fitted model, not information content.

Sites: per fold, 80 test transcripts with 81 <= L <= 1200 (RandomState(11140 + f) order, one per similarity unit),
4 codons each in [30, L-33] (RandomState(100000 (f + 1) + k)), i.e. 1,600 sites for five folds. At each site every one
of the 61 sense codons is substituted (wild type included, effect exactly 0) and u is read at j - r for r in -12..+10;
the single-base edits at the first base of codon j are read at j + 5 (nt -15). Both MFE treatments are run.

Exactness: a variant differs from the wild type only in rows [lo, hi] of h, so the forward GRU is re-run from lo with
the wild-type hidden state at lo - 1 and the backward GRU from hi with the wild-type state at hi + 1. On the stack used
for the paper (PyTorch 2.1.2, NumPy 1.26.4, CPU, one thread) this reproduces the full forward bit for bit. Other
PyTorch/BLAS builds may order float32 operations differently; independent tests on PyTorch 2.14 found differences of
at most 9.5e-7. ``gate`` therefore compares the local recomputation with the full forward, reports the largest absolute
difference and whether all comparisons were bit-identical, and passes when that difference is <= GATE_TOL (1e-5, in
units of u = log1p(relative density)). For scale: the median |effect| of a substitution is ~0.02, and perturbing every
effect by up to +-1e-5 changed the paper's route shares by < 2e-7. ``--gate-tol 0`` requires bit-identical results.

Share of block k for one effect (a set of sites and substitutions e): s_k = sum_e D_ek D_e,all / sum_e D_e,all^2.
A block is the main route when s_k > 0.5 and the lower end of its 95% unit-bootstrap interval (1,000 resamples,
seed 20261115) is > 0.5; otherwise the route is "shared". rho = 1 - sum_k s_k is the non-additive remainder.
"""
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import core as C

ROUTES = ("codon", "nt", "aa", "mfe")
GATE_TOL = 1e-5                         # allowed |local - full| in u (see module docstring)
NT_D, NT_R = -15, -5                    # nt -15 = first base of codon j, read at i = j + 5 (r = -5)
N_PER_FOLD, N_SITES = 80, 4
EFFECT_OFFSETS = (-5, -2, -1, 0, 2)     # offsets reported in the paper
AA_EFFECTS = {"Pro@P": ("P", -1), "Pro@E": ("P", -2), "Cys@A": ("C", 0)}


class WT:
    __slots__ = ("h", "o", "u", "y")


class Net:
    """Wrap one fold model of RiboPipe (MB1v2NoC2MFE): block construction and exact local recomputation."""

    def __init__(s, m):
        s.m = m
        dims = {"codon": m.cc.out_channels, "nt": m.cn.out_channels, "aa": m.ca.out_channels}
        s.slices, a = {}, 0
        for k in ("codon", "nt", "aa"):
            s.slices[k] = slice(a, a + dims[k]); a += dims[k]
        s.slices["mfe"] = slice(a, a + 3); s.D = a + 3
        assert m.rnn.input_size == s.D, (m.rnn.input_size, s.D)
        s.H = m.rnn.hidden_size
        sd = m.rnn.state_dict()
        s.gf = torch.nn.GRU(s.D, s.H, batch_first=True); s.gb = torch.nn.GRU(s.D, s.H, batch_first=True)
        s.gf.load_state_dict({k: v for k, v in sd.items() if not k.endswith("_reverse")})
        s.gb.load_state_dict({k.replace("_reverse", ""): v for k, v in sd.items() if k.endswith("_reverse")})
        s.gf.eval(); s.gb.eval()

    @torch.no_grad()
    def blocks_from_inputs(s, idx, bio):
        I = torch.from_numpy(np.asarray(idx)[None]); Bi = torch.from_numpy(np.asarray(bio)[None]); L = I.shape[1]
        mk = torch.ones(1, L)
        ac, an, aa_ = s.m.pathways(I, Bi, mk)
        return s.m._fuse(ac, an, aa_) * mk.unsqueeze(-1)

    def blocks(s, seq, track):
        idx, bio = C.inputs_from(seq, track); return s.blocks_from_inputs(idx, bio)

    @torch.no_grad()
    def rnn_full(s, h):
        L = h.shape[1]; ll = torch.tensor([L])
        pk = pack_padded_sequence(h, ll, batch_first=True, enforce_sorted=False)
        o, _ = s.m.rnn(pk); o, _ = pad_packed_sequence(o, batch_first=True, total_length=L)
        return o

    @torch.no_grad()
    def head(s, o): return s.m.reg(s.m._m(s.m.fc(o))).squeeze(-1)

    @torch.no_grad()
    def wt(s, seq, track):
        w = WT(); w.h = s.blocks(seq, track); w.o = s.rnn_full(w.h)
        w.y = s.head(w.o); w.u = w.y[0].double().numpy(); return w

    @torch.no_grad()
    def local(s, w, h, rows):
        """u (float64) of the variant blocks h at the read-out rows."""
        rows = np.asarray(rows); L = h.shape[1]
        diff = (h[0] != w.h[0]).any(-1).nonzero().flatten()
        if diff.numel() == 0: return w.u[rows].copy()
        lo, hi = int(diff[0]), int(diff[-1]); a, b = int(rows.min()), int(rows.max())
        H = s.H; o = w.o[0, a:b + 1].clone()
        if b >= lo:                                                   # forward direction: lo..b
            h0 = (w.o[0, lo - 1, :H] if lo > 0 else torch.zeros(H)).reshape(1, 1, H)
            of, _ = s.gf(h[:, lo:b + 1], h0)
            st = max(lo, a); o[st - a:, :H] = of[0, st - lo:]
        if hi >= a:                                                   # backward direction: hi..a
            h0 = (w.o[0, hi + 1, H:] if hi + 1 < L else torch.zeros(H)).reshape(1, 1, H)
            ob, _ = s.gb(torch.flip(h[:, a:hi + 1], [1]), h0); ob = torch.flip(ob, [1])
            en = min(hi, b); o[:en - a + 1, H:] = ob[0, :en - a + 1]
        O = w.o.clone(); O[0, a:b + 1] = o
        return s.head(O)[0].double().numpy()[rows]

    def routed(s, w, h, rows, routes=ROUTES):
        """{k: u(rows)} with only block k taken from the variant, plus 'total'."""
        out = {"total": s.local(w, h, rows)}
        for k in routes:
            hr = w.h.clone(); sl = s.slices[k]; hr[..., sl] = h[..., sl]
            out[k] = s.local(w, hr, rows)
        return out


# ---------------------------------------------------------------------------------------------- sites
def sample_sites(ds, folds=None, n_per_fold=N_PER_FOLD, n_sites=N_SITES):
    out = []
    for f in (ds.folds if folds is None else folds):
        pool = sorted(t for t in ds.test_ids(f) if 81 <= len(ds.counts[t]) <= 1200)
        order = [pool[i] for i in np.random.RandomState(11140 + f).permutation(len(pool))]
        used, k = set(), 0
        for t in order:
            u = ds.unit[t]
            if u in used: continue
            used.add(u)
            L = len(ds.seq[t]) // 3
            js = sorted(np.random.RandomState(100000 * (f + 1) + k).choice(np.arange(30, L - 33 + 1), n_sites, replace=False).tolist())
            out.append(dict(tx=t, fold=f, k=k, unit=u, L=L, sites=js)); k += 1
            if k == n_per_fold: break
        if k < n_per_fold:
            raise ValueError(f"{ds.tag} fold {f}: only {k} eligible transcripts for {n_per_fold} requested")
    return out


def _nt_states(seq, q):
    ci, k = q // 3, q % 3; cod = seq[3 * ci:3 * ci + 3]; st = []
    for b in C.NT:
        c2 = cod[:k] + b + cod[k + 1:]
        if c2 in C.STOPS: continue
        st.append(b)
    return st


# ---------------------------------------------------------------------------------------------- workers
def _site_task(e):
    o = C.G["oof"]; nets = C.G["nets"]; tx, f = e["tx"], e["fold"]; o.assert_oof(tx, f)
    seq = o.ds.seq[tx]; tr = o.track(tx); t0 = time.time()
    net = nets[f]; W = net.wt(seq, tr)
    dev = float(np.abs(W.u - np.log1p(o.pred[tx])).max()) if tx in o.pred else None
    nS, nO = len(e["sites"]), len(C.OFFSETS)
    D = np.full((nS, 2, nO, 61), np.nan); DR = np.full((nS, 2, nO, 61, 4), np.nan)
    DN = np.full((nS, 2, nO, 4), np.nan); DNR = np.full((nS, 2, nO, 4, 4), np.nan)
    wt, drop = [], np.zeros(nS, bool)
    for s, j in enumerate(e["sites"]):
        w = seq[3 * j:3 * j + 3]; wt.append(w)
        if w not in C.SENSE: drop[s] = True; continue
        rows = np.array([j - r for r in C.OFFSETS])
        for bi, b in enumerate(C.SENSE):
            sv = C.sub_codon(seq, j, b); tv, _ = C.mfe_update(sv, tr, 3 * j, 3 * j + 3)
            for mi, trk in enumerate((tv, tr)):
                idx, bio = C.inputs_from(sv, trk)
                rr = net.routed(W, net.blocks_from_inputs(idx, bio), rows)
                D[s, mi, :, bi] = rr["total"] - W.u[rows]
                for ki, k in enumerate(ROUTES): DR[s, mi, :, bi, ki] = rr[k] - W.u[rows]
        q = 3 * j
        assert q - 3 * (j - NT_R) == NT_D
        for b in _nt_states(seq, q):
            bi = C.NT.index(b)
            sv = seq[:q] + b + seq[q + 1:]; tv, _ = C.mfe_update(sv, tr, q, q + 1)
            for mi, trk in enumerate((tv, tr)):
                idx, bio = C.inputs_from(sv, trk)
                rr = net.routed(W, net.blocks_from_inputs(idx, bio), rows)
                DN[s, mi, :, bi] = rr["total"] - W.u[rows]
                for ki, k in enumerate(ROUTES): DNR[s, mi, :, bi, ki] = rr[k] - W.u[rows]
    self_ok = True
    for s, w in enumerate(wt):
        if drop[s]: continue
        self_ok &= bool(np.all(D[s, :, :, C.SENSE.index(w)] == 0.0)) and bool(np.all(DR[s, :, :, C.SENSE.index(w)] == 0.0))
    return dict(tx=tx, D=D, DR=DR, DN=DN, DNR=DNR, wt=wt, drop=drop, self_exact=self_ok, wt_dev_vs_pred=dev,
                sec=round(time.time() - t0, 2))


def _gate_task(e):
    """Local recomputation == plain batch-1 forward (total) and == full GRU (each route), route-none == WT, at the
    first site of a transcript, all 61 codons x both MFE treatments."""
    o = C.G["oof"]; net = C.G["nets"][e["fold"]]; tx = e["tx"]; seq = o.ds.seq[tx]; tr = o.track(tx)
    w = net.wt(seq, tr)
    u1 = C.forward_one(net.m, seq, tr)
    g = dict(wt_engine_eq_plain=bool(np.array_equal(w.u, u1)), wt_maxdiff=float(np.abs(w.u - u1).max()),
             total_eq_plain=True, total_maxdiff=0.0, route_local_eq_full=True, route_maxdiff=0.0,
             route_none_eq_wt=True, none_maxdiff=0.0, n=0)
    j = e["sites"][0]; rows = np.array([j - r for r in C.OFFSETS])
    if seq[3 * j:3 * j + 3] not in C.SENSE: return g
    for b in C.SENSE:
        sv = C.sub_codon(seq, j, b); tv, _ = C.mfe_update(sv, tr, 3 * j, 3 * j + 3)
        for trk in (tv, tr):
            idx, bio = C.inputs_from(sv, trk); h = net.blocks_from_inputs(idx, bio)
            up = C.forward_same_len(net.m, [idx], [bio])[0]; ul = net.local(w, h, rows)
            g["total_eq_plain"] &= bool(np.array_equal(ul, up[rows])); g["total_maxdiff"] = max(g["total_maxdiff"], float(np.abs(ul - up[rows]).max()))
            rr = net.routed(w, h, rows)
            for k in ROUTES:
                hr = w.h.clone(); hr[..., net.slices[k]] = h[..., net.slices[k]]
                uf = net.head(net.rnn_full(hr))[0].double().numpy()[rows]
                g["route_local_eq_full"] &= bool(np.array_equal(rr[k], uf)); g["route_maxdiff"] = max(g["route_maxdiff"], float(np.abs(rr[k] - uf).max()))
            un = net.head(net.rnn_full(w.h.clone()))[0].double().numpy()
            g["route_none_eq_wt"] &= bool(np.array_equal(un, w.u)); g["none_maxdiff"] = max(g["none_maxdiff"], float(np.abs(un - w.u).max()))
            g["n"] += 1
    return g


# ---------------------------------------------------------------------------------------------- drivers
def run_routing(oof, sites=None, workers=1, gate=True, gate_tol=GATE_TOL, log=print):
    """Routed substitutions at the sampled sites -> dict of arrays (+ 'gate' results).

    The gate passes when every |local - full| <= gate_tol; 'bit_exact' records whether all were identical."""
    sites = sample_sites(oof.ds, oof.folds) if sites is None else sites
    C.G["oof"] = oof; C.G["nets"] = {f: Net(oof.models[f]) for f in oof.folds}
    t0 = time.time(); out = {}
    if gate:
        gs = C.pmap(_gate_task, sites, workers)
        agg = {k: (all(g[k] for g in gs) if isinstance(gs[0][k], bool) else (max(g[k] for g in gs) if "max" in k else sum(g[k] for g in gs)))
               for k in gs[0]}
        agg["bit_exact"] = all(agg[k] for k in ("wt_engine_eq_plain", "total_eq_plain", "route_local_eq_full", "route_none_eq_wt"))
        agg["max_abs_diff"] = max(agg[k] for k in ("wt_maxdiff", "total_maxdiff", "route_maxdiff", "none_maxdiff"))
        agg["tol"] = float(gate_tol); agg["passed"] = bool(agg["max_abs_diff"] <= gate_tol)
        out["gate"] = agg; log(f"[routing gate] {oof.dataset}: {agg}")
        if not agg["passed"]: raise RuntimeError(f"routing gate failed: {agg}")
    res = C.pmap(_site_task, sites, workers)
    log(f"[routing] {oof.dataset}: {len(res)} transcripts x {len(sites[0]['sites'])} sites, {time.time() - t0:.1f}s")
    sv = dict(tx=np.array([e["tx"] for e in sites]), fold=np.array([e["fold"] for e in sites]), unit=np.array([e["unit"] for e in sites]),
              L=np.array([e["L"] for e in sites]), sites=np.array([e["sites"] for e in sites]), wt=np.array([r["wt"] for r in res]),
              drop=np.stack([r["drop"] for r in res]), D=np.stack([r["D"] for r in res]), DR=np.stack([r["DR"] for r in res]),
              DN=np.stack([r["DN"] for r in res]), DNR=np.stack([r["DNR"] for r in res]))
    devs = [r["wt_dev_vs_pred"] for r in res if r["wt_dev_vs_pred"] is not None]
    out.update(arrays=sv, self_substitution_exact_all=bool(all(r["self_exact"] for r in res)),
               wt_dev_vs_pred_max=(max(devs) if devs else None), cpu_sec=round(sum(r["sec"] for r in res), 1))
    return out


# ---------------------------------------------------------------------------------------------- summaries
def shares(DR, Dt, units):
    """s_k = sum D_k D_tot / sum D_tot^2 with a unit-bootstrap interval. DR (n, ..., 4), Dt (n, ...)."""
    n = Dt.shape[0]
    num = np.stack([np.nansum((DR[..., k] * Dt).reshape(n, -1), 1) for k in range(4)], 1)
    den = np.nansum((Dt * Dt).reshape(n, -1), 1)
    uq, inv = np.unique(units, return_inverse=True); W = C.unit_weights(len(uq))[:, inv]
    s = num.sum(0) / den.sum(); bs = (W @ num) / (W @ den)[:, None]
    out = {k: dict(s=float(s[i]), ci=C.ci95(bs[:, i])) for i, k in enumerate(ROUTES)}
    out["rho"] = dict(s=float(1 - s.sum()), ci=C.ci95(1 - bs.sum(1)))
    mains = [k for k in ROUTES if out[k]["s"] > 0.5 and out[k]["ci"][0] > 0.5]
    out["reading"] = ("main route: " + mains[0]) if mains else "shared"
    return out


def summarize_routing(z):
    """Routing shares for every offset, for G at nt -15, and for the amino-acid effects Pro@P, Pro@E, Cys@A."""
    RIDX = {r: k for k, r in enumerate(C.OFFSETS)}
    ok = ~z["drop"] & np.isfinite(z["D"]).all((2, 3, 4)) & np.isfinite(z["DR"]).all((2, 3, 4, 5))
    units = np.repeat(z["unit"], z["D"].shape[1]).reshape(len(z["tx"]), z["D"].shape[1])[ok]
    DR = z["DR"][ok]; Dt = z["D"][ok]
    out = dict(n_sites_valid=int(ok.sum()), shares={})
    for mi, mode in enumerate(("recompute", "fixed")):
        sh = {}
        for r in C.OFFSETS:
            sh[str(r)] = shares(DR[:, mi, RIDX[r]], Dt[:, mi, RIDX[r]], units)
        DN = z["DN"][ok][:, mi, RIDX[NT_R]]; DNR = z["DNR"][ok][:, mi, RIDX[NT_R]]
        cN = DN - np.nanmean(DN, 1, keepdims=True); cR = DNR - np.nanmean(DNR, 1, keepdims=True)
        gi = C.NT.index("G"); okg = np.isfinite(cN[:, gi])
        sh["nt15_G"] = shares(cR[okg, gi][:, None, :], cN[okg, gi][:, None], units[okg])
        sh["nt15_G"]["n_sites"] = int(okg.sum())
        aa_eff = {}
        for eff, (aa, off) in AA_EFFECTS.items():
            cols = [C.SENSE.index(c) for c in C.SENSE if C.CODON_TABLE[c] == aa]
            tot = Dt[:, mi, RIDX[off]][:, cols].mean(1); rk = DR[:, mi, RIDX[off]][:, cols, :].mean(1)
            uq, inv = np.unique(units, return_inverse=True); W = C.unit_weights(len(uq))[:, inv]
            e = dict(total=dict(v=float(tot.mean()), ci=C.ci95((W @ tot) / W.sum(1))))
            for k, kk in enumerate(ROUTES):
                e[kk] = dict(v=float(rk[:, k].mean()), ci=C.ci95((W @ rk[:, k]) / W.sum(1)), share_of_total=float(rk[:, k].mean() / tot.mean()))
            e["rho"] = dict(v=float(1 - rk.mean(0).sum() / tot.mean()))
            aa_eff[eff] = e
        sh["aa_effects"] = aa_eff
        out["shares"][mode] = sh
    out["limitation"] = ("A routed input is internally inconsistent (one block from the variant, the others from the wild type); "
                         "shares measure reliance of the fitted model, not information content.")
    return out


def positional_profile(z, mode="recompute"):
    """Mean |Delta u| per offset over the 61 substitutions and the valid sites (the sampled-site positional profile)."""
    mi = ("recompute", "fixed").index(mode)
    ok = ~z["drop"] & np.isfinite(z["D"]).all((2, 3, 4))
    return np.abs(z["D"][ok][:, mi]).mean((0, 2))
