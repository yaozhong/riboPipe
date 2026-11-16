"""Window transplantation.

Donors (selection half X): per fold, the 6 transcripts of half X whose representative peak p has the highest observed
density (one per similarity unit). Acceptors (evaluation half Y != X): per fold, 10 transcripts of half Y with
81 <= L <= 1200 in RandomState(3) order (one per unit), each with one random neutral site q:
q - 6 >= 20, q + 4 < L - 20, the out-of-fold prediction at q between the 40th and 60th interior percentiles, and no
observed or predicted interior top-5% codon within [q - 10, q + 10]. Donors and acceptors come from the same test fold
and are scored by that fold's model (strictly out of fold). Both directions (A->B, B->A) are run.

Conditions grafted into [q - 6, q + 4]: T = the donor window [p - 6, p + 4]; S = the same 11 codons in random order
(RandomState(donor index)); R = an 11-codon window from a neutral site of the donor transcript. Folding energies of
every 30-nt window overlapping the graft are recomputed on the chimeric sequence ("recompute") or the acceptor's
values are kept ("fixed"). Read-out Delta = u_new(q) - u_orig(q). Donor elevation e = u_d(p) - median interior u_d.
Transfer ratio of a donor = mean over its acceptors of Delta / e.

Sub-window grafts (same donors and acceptors; codon offsets relative to p and q): W_full -6..+4, W_EPA -2..0,
W_up -6..-3, W_down +1..+4, W_core -3..+1, W_noEPA = W_full without -2..0, W_wide -12..+10, and W_shuffled (the S
window). They use batch-1 forwards. Ratios are summarised as medians over donors with unit-bootstrap 95% intervals.
"""
import time
import numpy as np

from . import core as C

WL, WR = C.FF_FIELD
N_DONOR, N_ACC = 6, 10
SUBWINDOWS = {"W_full": list(range(-6, 5)), "W_EPA": [-2, -1, 0], "W_up": [-6, -5, -4, -3], "W_down": [1, 2, 3, 4],
              "W_core": [-3, -2, -1, 0, 1], "W_noEPA": [k for k in range(-6, 5) if k not in (-2, -1, 0)],
              "W_wide": list(range(-12, 11))}
MODES = ("recompute", "fixed")


# ---------------------------------------------------------------------------------------------- selection
def neutral_sites(y, yhat, L):
    lo, hi = C.INTERIOR_MARGIN, L - C.INTERIOR_MARGIN
    n = hi - lo; k = int(np.ceil(0.05 * n))
    tops = {lo + i for i in C.topk_positions(y[lo:hi], k)} | {lo + i for i in C.topk_positions(yhat[lo:hi], k)}
    q40, q60 = np.quantile(yhat[lo:hi], [0.4, 0.6])
    out = []
    for q in range(lo - WL, hi - WR):
        if not (q40 <= yhat[q] <= q60): continue
        if any(t in tops for t in range(q - 10, q + 11)): continue
        out.append(q)
    return out


def select(oof, halves, fold, X, n_donor=N_DONOR, n_acc=N_ACC):
    ds = oof.ds; Y = "B" if X == "A" else "A"
    cands = []
    for t in sorted(ds.test_ids(fold)):
        if halves[ds.unit[t]] != X: continue
        y = ds.truth(t); c = C.candidate_peaks(y, oof.pred[t])
        if not c: continue
        p = max(c, key=lambda i: (y[i], -i))
        if p + WL < 0 or p + WR >= len(y): continue
        cands.append((float(y[p]), t, p))
    cands.sort(key=lambda x: (-x[0], x[1]))
    donors, used = [], set()
    for yv, t, p in cands:
        if ds.unit[t] in used: continue
        used.add(ds.unit[t]); donors.append((t, p, yv))
        if len(donors) == n_donor: break
    pool = sorted(t for t in ds.test_ids(fold) if halves[ds.unit[t]] == Y and 81 <= len(ds.counts[t]) <= 1200)
    rng = np.random.RandomState(3); order = [pool[i] for i in rng.permutation(len(pool))]
    acc, used = [], set()
    for t in order:
        if ds.unit[t] in used: continue
        ns = neutral_sites(ds.truth(t), oof.pred[t], len(ds.counts[t]))
        if not ns: continue
        used.add(ds.unit[t]); acc.append((t, int(ns[rng.randint(len(ns))])))
        if len(acc) == n_acc: break
    return donors, acc


def donor_tasks(oof, folds=None, halves=None):
    """[(donor index, donor tx, p, [(acceptor tx, q)])] and per-donor info, in the paper's order (X = A then B, folds)."""
    folds = list(oof.folds) if folds is None else folds
    halves = C.halves_for(oof.ds, halves); tasks, info = [], []
    for X in ("A", "B"):
        for f in folds:
            donors, acc = select(oof, halves, f, X)
            for t, p, yv in donors:
                tasks.append((len(tasks), t, p, acc))
                info.append(dict(direction=f"{X}->{'B' if X == 'A' else 'A'}", fold=f, unit=oof.ds.unit[t], y_peak=yv, n_acc=len(acc)))
    return tasks, info


# ---------------------------------------------------------------------------------------------- whole-window transplant
def _donor_task(arg):
    di, t_d, p, acc = arg
    o = C.G["oof"]; ds = o.ds; t0 = time.time()
    sd = ds.seq[t_d]; m_d = o.model_for(t_d)
    u_d = C.forward_variants(m_d, [sd], [o.track(t_d)])[0]
    Ld = len(sd) // 3
    e = float(u_d[p] - np.median(u_d[C.INTERIOR_MARGIN:Ld - C.INTERIOR_MARGIN]))
    win_T = sd[3 * (p + WL):3 * (p + WR + 1)]
    rng = np.random.RandomState(di)
    cods = [win_T[3 * k:3 * k + 3] for k in range(len(win_T) // 3)]
    win_S = "".join(cods[k] for k in rng.permutation(len(cods)))
    ns = neutral_sites(ds.truth(t_d), o.pred[t_d], Ld)
    qR = int(ns[rng.randint(len(ns))]) if ns else None
    win_R = sd[3 * (qR + WL):3 * (qR + WR + 1)] if qR is not None else None
    res = dict(di=di, tx=t_d, p=p, e=e, qR=qR, win_T=win_T, win_S=win_S, win_R=win_R, acc=[], dT={}, n_rec=[],
               u_orig=np.full(len(acc), np.nan))
    for cond in "TSR":
        for mode in MODES: res["dT"][(cond, mode)] = np.full(len(acc), np.nan)
    for ai, (t_a, q) in enumerate(acc):
        sa = ds.seq[t_a]; tra = o.track(t_a); m_a = o.model_for(t_a)
        assert o.fold_of[t_a] == o.fold_of[t_d]                      # same test fold (strictly out of fold)
        lo, hi = 3 * (q + WL), 3 * (q + WR + 1)
        seqs, trs, tags = [sa], [tra], [None]
        for cond, win in (("T", win_T), ("S", win_S), ("R", win_R)):
            if win is None: continue
            assert len(win) == hi - lo
            sn = sa[:lo] + win + sa[hi:]
            trn, st = C.mfe_update(sn, tra, lo, hi); res["n_rec"].append(len(st))
            seqs += [sn, sn]; trs += [trn, tra]; tags += [(cond, "recompute"), (cond, "fixed")]
        U = C.forward_variants(m_a, seqs, trs)[:, q]
        res["u_orig"][ai] = U[0]
        for tg, v in zip(tags[1:], U[1:]):
            res["dT"][tg][ai] = v - U[0]
        res["acc"].append((t_a, q))
    res["sec"] = round(time.time() - t0, 2)
    return res


def run_transplant(oof, folds=None, halves=None, workers=1, log=print):
    """Whole-window transplant (T, S, R x recompute, fixed) -> dict of arrays (layout of the paper's transplant.npz,
    plus u_orig)."""
    tasks, info = donor_tasks(oof, folds, halves)
    C.G["oof"] = oof; t0 = time.time()
    res = C.pmap(_donor_task, tasks, workers)
    log(f"[transplant] {oof.dataset}: {len(res)} donors, {time.time() - t0:.1f}s")
    nmax = max(len(r["acc"]) for r in res)
    sv = dict(direction=np.array([x["direction"] for x in info]), fold=np.array([x["fold"] for x in info]),
              unit=np.array([x["unit"] for x in info]), y_peak=np.array([x["y_peak"] for x in info]),
              tx=np.array([r["tx"] for r in res]), p=np.array([r["p"] for r in res]), e=np.array([r["e"] for r in res]),
              n_acc=np.array([len(r["acc"]) for r in res]),
              acc_tx=np.array([[a for a, _ in r["acc"]] + [""] * (nmax - len(r["acc"])) for r in res]),
              acc_q=np.array([[q for _, q in r["acc"]] + [-1] * (nmax - len(r["acc"])) for r in res]),
              win_T=np.array([r["win_T"] for r in res]), win_S=np.array([r["win_S"] for r in res]),
              win_R=np.array([r["win_R"] or "" for r in res]))
    uo = np.full((len(res), nmax), np.nan)
    for i, r in enumerate(res): uo[i, :len(r["acc"])] = r["u_orig"][:len(r["acc"])]
    sv["u_orig"] = uo
    for cond in "TSR":
        for mode in MODES:
            a = np.full((len(res), nmax), np.nan)
            for i, r in enumerate(res): a[i, :len(r["acc"])] = r["dT"][(cond, mode)][:len(r["acc"])]
            sv[f"d_{cond}_{mode}"] = a
    nrec = np.concatenate([r["n_rec"] for r in res])
    sv["mfe_windows_recomputed"] = np.array([nrec.min(), nrec.max(), nrec.mean()])
    sv["cpu_sec"] = np.array([r["sec"] for r in res])
    return sv


def summarize_transplant(z):
    """Per direction and MFE treatment: mean Delta u of T, S, R (cluster-bootstrap over donor units), paired T - S and
    T - R, fraction of donors with Delta_T > 0 and Delta_T > Delta_S, and the median transfer ratio Delta_T / e."""
    out = {}
    for direc in ("A->B", "B->A"):
        m = z["direction"] == direc; res = {}
        for mode in MODES:
            per = {c: np.nanmean(z[f"d_{c}_{mode}"][m], 1) for c in "TSR"}
            u = z["unit"][m]
            res[mode] = dict({c: C.cluster_boot_ci(per[c], u, seed=ord(c)) for c in "TSR"},
                             T_minus_S=C.cluster_boot_ci(per["T"] - per["S"], u, seed=1),
                             T_minus_R=C.cluster_boot_ci(per["T"] - per["R"], u, seed=2),
                             frac_donor_T_pos=float(np.mean(per["T"] > 0)),
                             frac_donor_T_gt_S=float(np.mean(per["T"] > per["S"])),
                             transfer_ratio_median=float(np.nanmedian(per["T"] / z["e"][m])))
        res["n_donors"] = int(m.sum()); res["n_acceptors_per_donor"] = int(np.median(z["n_acc"][m]))
        res["n_donors_without_R"] = int((z["win_R"][m] == "").sum())
        res["donor_in_situ_elevation_median"] = float(np.median(z["e"][m]))
        out[direc] = res
    return out


# ---------------------------------------------------------------------------------------------- sub-window transplant
def graft(sa, sd, q, p, offs):
    s = list(sa)
    for k in offs:
        a, b = 3 * (q + k), 3 * (p + k)
        s[a:a + 3] = sd[b:b + 3]
    return "".join(s)


def _sub_task(arg):
    di, t_d, p, acc, win_S, windows = arg
    o = C.G["oof"]; ds = o.ds; t0 = time.time()
    sd = ds.seq[t_d]; Ld = len(sd) // 3
    ud = C.forward_one(o.model_for(t_d), sd, o.track(t_d))
    e = float(ud[p] - np.median(ud[C.INTERIOR_MARGIN:Ld - C.INTERIOR_MARGIN]))
    names = list(windows) + (["W_shuffled"] if win_S else [])
    d = {(w, m): np.full(len(acc), np.nan) for w in names for m in MODES}
    for ai, (t_a, q) in enumerate(acc):
        assert o.fold_of[t_a] == o.fold_of[t_d], (t_a, t_d)
        sa = ds.seq[t_a]; tra = o.track(t_a); m = o.model_for(t_a)
        u0 = C.forward_one(m, sa, tra)[q]
        for w in names:
            if w == "W_shuffled":
                lo, hi = 3 * (q + WL), 3 * (q + WR + 1); sn = sa[:lo] + win_S + sa[hi:]
            else:
                offs = windows[w]; sn = graft(sa, sd, q, p, offs); lo, hi = 3 * (q + min(offs)), 3 * (q + max(offs) + 1)
            assert sn[:lo] == sa[:lo] and sn[hi:] == sa[hi:] and len(sn) == len(sa)
            trn = C.mfe_update(sn, tra, lo, hi)[0]
            for mode in MODES:
                d[(w, mode)][ai] = C.forward_one(m, sn, trn if mode == "recompute" else tra)[q] - u0
    return dict(di=di, e=e, d=d, sec=round(time.time() - t0, 2))


def run_subwindows(oof, tp=None, folds=None, halves=None, windows=None, shuffled=True, workers=1, log=print):
    """Sub-window grafts on the donors/acceptors of the whole-window transplant ``tp`` (arrays from run_transplant or
    the paper's transplant.npz; selected afresh when None). Returns per-donor arrays d|<window>|<mode> and e."""
    windows = dict(SUBWINDOWS if windows is None else windows)
    if tp is None:
        tasks, info = donor_tasks(oof, folds, halves)
        C.G["oof"] = oof
        winS = [r["win_S"] for r in C.pmap(_shuffled_only, tasks, 1)]
        meta = dict(direction=[x["direction"] for x in info], fold=[x["fold"] for x in info], unit=[x["unit"] for x in info],
                    tx=[t[1] for t in tasks], p=[t[2] for t in tasks])
    else:
        n = len(tp["tx"])
        tasks = [(i, str(tp["tx"][i]), int(tp["p"][i]),
                  [(str(a), int(q)) for a, q in zip(tp["acc_tx"][i], tp["acc_q"][i]) if str(a) != "" and int(q) >= 0][:int(tp["n_acc"][i])])
                 for i in range(n)]
        winS = [str(w) for w in tp["win_S"]]
        meta = dict(direction=list(tp["direction"]), fold=[int(x) for x in tp["fold"]], unit=list(tp["unit"]),
                    tx=[t[1] for t in tasks], p=[t[2] for t in tasks])
    C.G["oof"] = oof; t0 = time.time()
    res = C.pmap(_sub_task, [(t[0], t[1], t[2], t[3], (winS[k] if shuffled else None), windows) for k, t in enumerate(tasks)], workers)
    log(f"[subwindow] {oof.dataset}: {len(res)} donors, {time.time() - t0:.1f}s")
    nmax = max(len(t[3]) for t in tasks)
    sv = {k: np.array(v) for k, v in meta.items()}
    sv["e"] = np.array([r["e"] for r in res])
    for (w, mode) in res[0]["d"]:
        a = np.full((len(res), nmax), np.nan)
        for i, r in enumerate(res): a[i, :len(r["d"][(w, mode)])] = r["d"][(w, mode)]
        sv[f"d|{w}|{mode}"] = a
    sv["cpu_sec"] = np.array([r["sec"] for r in res])
    return sv


def _shuffled_only(arg):
    di, t_d, p, acc = arg
    sd = C.G["oof"].ds.seq[t_d]; win_T = sd[3 * (p + WL):3 * (p + WR + 1)]
    cods = [win_T[3 * k:3 * k + 3] for k in range(len(win_T) // 3)]
    return dict(win_S="".join(cods[k] for k in np.random.RandomState(di).permutation(len(cods))))


def _ratio(z, w, mode):
    d = np.asarray(z[f"d|{w}|{mode}"], float); e = np.asarray(z["e"], float)
    return np.nanmean(d, 1) / e, np.nanmean(d, 1)


def summarize_subwindows(z):
    """Transfer ratios per window (median over donors, unit-bootstrap 95% CI; pooled and per direction), mean Delta u,
    additivity (ratio(EPA) + ratio(noEPA) - ratio(full), per donor), wide minus full, and the shuffled-window control."""
    units = np.asarray(z["unit"]).astype(str); dirs = np.asarray(z["direction"]).astype(str)
    wins = sorted({k.split("|")[1] for k in z if str(k).startswith("d|")})
    out = dict(n_donors=int(len(units)), windows={}, paired={}, additivity={}, e_median=float(np.median(z["e"])),
               frac_e_le_0=float(np.mean(np.asarray(z["e"]) <= 0)))
    for w in wins:
        for mode in MODES:
            rat, dm = _ratio(z, w, mode)
            out["windows"][f"{w}|{mode}"] = dict(pooled=C.median_ci(rat, units),
                                                 **{d_: C.median_ci(rat, units, dirs == d_) for d_ in ("A->B", "B->A")},
                                                 mean_delta_u=C.mean_ci(dm, units), frac_donor_delta_pos=float(np.nanmean(dm > 0)))
    for mode in MODES:
        if "W_wide" in wins and "W_full" in wins:
            a, _ = _ratio(z, "W_wide", mode); b, _ = _ratio(z, "W_full", mode)
            out["paired"][f"W_wide_minus_W_full|{mode}"] = C.median_ci(a - b, units)
        if {"W_EPA", "W_noEPA", "W_full"} <= set(wins):
            e_, _ = _ratio(z, "W_EPA", mode); n_, _ = _ratio(z, "W_noEPA", mode); f_, _ = _ratio(z, "W_full", mode)
            out["additivity"][mode] = dict(median_EPA_plus_median_noEPA=float(np.nanmedian(e_) + np.nanmedian(n_)),
                                           median_full=float(np.nanmedian(f_)), per_donor_excess=C.median_ci(e_ + n_ - f_, units))
        if "W_shuffled" in wins and "W_full" in wins:
            rs, _ = _ratio(z, "W_shuffled", mode); rt, _ = _ratio(z, "W_full", mode)
            out.setdefault("shuffle_control", {})[mode] = dict(S=C.median_ci(rs, units), T=C.median_ci(rt, units),
                                                              T_minus_S=C.median_ci(rt - rs, units),
                                                              frac_donor_T_gt_S=float(np.nanmean(rt > rs)))
    return out
