"""Single-codon and single-nucleotide substitutions (in silico mutagenesis, ISM).

(1) Peak ISM. At representative peaks (observed and predicted top 5%), the codon at every offset r of ``offsets`` is
    replaced by each of the other 60 sense codons and the change of the peak prediction is recorded,
    Delta(r, b) = u_variant(p) - u_WT(p), with folding energies recomputed (dR) and held fixed (dF).
    Sampling: per fold and similarity-unit half, ``n_per_half`` transcripts with 81 <= L <= 1200 in the order of
    RandomState(2).permutation, one per unit, each with a candidate peak.
(2) Centred sensitivity maps. At 4 random interior codons per transcript (j in [30, L-33]) every one of the 61 sense
    codons is substituted (the wild type included) and u is read at j - r for every r; values are centred over the 61
    states at each site and offset. A nucleotide version substitutes single bases at 12 random positions per
    transcript (bases that create a stop codon are excluded) and centres over the allowed states.
    Sampling: per fold and half, ``n_per_half`` transcripts (RandomState(1) order); site seeds 1000 f + 500 [half B] + k.

``summarize_ism`` / ``summarize_map`` reproduce the summary statistics of the paper (cluster-bootstrap means over
peaks / transcripts, selection in one half and evaluation in the other, response range and far-offset controls).
"""
import os, time
import numpy as np

from . import core as C

D_RANGE = list(range(-39, 33))       # nucleotide offsets d of the nucleotide map


# ---------------------------------------------------------------------------------------------- sampling
def sample_transcripts(oof, halves, fold, half, n, seed, need_peak):
    ds = oof.ds
    pool = sorted(t for t in ds.test_ids(fold) if halves[ds.unit[t]] == half and 81 <= len(ds.counts[t]) <= 1200)
    rng = np.random.RandomState(seed); order = [pool[i] for i in rng.permutation(len(pool))]
    out, used = [], set()
    for t in order:
        u = ds.unit[t]
        if u in used: continue
        pk = None
        if need_peak:
            c = C.candidate_peaks(ds.truth(t), oof.pred[t])
            if not c: continue
            y = ds.truth(t); pk = max(c, key=lambda i: (y[i], -i))
        used.add(u); out.append((t, pk))
        if len(out) == n: break
    return out


def peak_set(oof, halves, folds, n_per_half=10, seed=2):
    """[(tx, peak, fold, half, unit)] in the order of the paper's peak ISM."""
    out = []
    for f in folds:
        for h in ("A", "B"):
            for t, pk in sample_transcripts(oof, halves, f, h, n_per_half, seed, True):
                out.append((t, pk, f, h, oof.ds.unit[t]))
    return out


# ---------------------------------------------------------------------------------------------- workers
def _ism_task(arg):
    tx, p, offsets = arg
    o = C.G["oof"]; m = o.model_for(tx); seq = o.ds.seq[tx]; tr = o.track(tx); L = len(seq) // 3
    t0 = time.time()
    u0 = C.forward_variants(m, [seq], [tr])[0]
    dF = np.full((len(offsets), 64), np.nan); dR = np.full((len(offsets), 64), np.nan)
    wt = []; seqs, trs_r, keys, nrec = [], [], [], []
    for a, r in enumerate(offsets):
        j = p + r; assert 0 <= j < L, (tx, p, r)
        w = seq[3 * j:3 * j + 3]; wt.append(w)
        if w not in C.SENSE: continue
        for b in C.SENSE:
            if b == w: continue
            sv = C.sub_codon(seq, j, b); tv, st = C.mfe_update(sv, tr, 3 * j, 3 * j + 3)
            seqs.append(sv); trs_r.append(tv); keys.append((a, C.CIDX[b])); nrec.append(len(st))
    uf = C.forward_variants(m, seqs, [tr] * len(seqs))[:, p]
    ur = C.forward_variants(m, seqs, trs_r)[:, p]
    for (a, b), x, y in zip(keys, uf, ur):
        dF[a, b] = x - u0[p]; dR[a, b] = y - u0[p]
    pred = o.pred.get(tx)
    return dict(tx=tx, p=p, L=L, u0p=float(u0[p]), u0_stored=(float(np.log1p(pred[p])) if pred is not None else np.nan),
                u_dev_max=(float(np.abs(u0 - np.log1p(pred)).max()) if pred is not None else np.nan),
                wt=wt, dF=dF, dR=dR, n_var=len(seqs), mfe_windows_recomputed_mean=float(np.mean(nrec)) if nrec else 0.0,
                sec=round(time.time() - t0, 2))


def _map_task(arg):
    tx, seed, offsets = arg
    o = C.G["oof"]; m = o.model_for(tx); seq = o.ds.seq[tx]; tr = o.track(tx); L = len(seq) // 3
    rng = np.random.RandomState(seed)
    t0 = time.time()
    js = sorted(rng.choice(np.arange(30, L - 33 + 1), 4, replace=False).tolist())
    qlo, qhi = 92, 3 * (L - 21) - 39
    qs = sorted(rng.choice(np.arange(qlo, qhi + 1), 12, replace=False).tolist())
    nO = len(offsets)
    out = dict(tx=tx, L=L, sites=js, nt_sites=qs)
    for mode in ("fixed", "recompute"):
        cod_c = np.full((len(js), nO, 61), np.nan)
        for s, j in enumerate(js):
            w = seq[3 * j:3 * j + 3]
            if w not in C.SENSE: continue
            seqs = [C.sub_codon(seq, j, b) for b in C.SENSE]
            trs = [tr] * 61 if mode == "fixed" else [C.mfe_update(sv, tr, 3 * j, 3 * j + 3)[0] for sv in seqs]
            U = C.forward_variants(m, seqs, trs)
            for a, r in enumerate(offsets):
                v = U[:, j - r]; cod_c[s, a] = v - v.mean()
        nt_sum = np.zeros((len(D_RANGE), 4)); nt_cnt = np.zeros((len(D_RANGE), 4))
        for q in qs:
            ci = q // 3; k = q % 3; cod = seq[3 * ci:3 * ci + 3]
            states = []
            for b in C.NT:
                c2 = cod[:k] + b + cod[k + 1:]
                if c2 in C.STOPS or any(ch not in C.NT for ch in c2): continue
                states.append(b)
            if len(states) < 2: continue
            seqs = [seq[:q] + b + seq[q + 1:] for b in states]
            trs = [tr] * len(seqs) if mode == "fixed" else [C.mfe_update(sv, tr, q, q + 1)[0] for sv in seqs]
            U = C.forward_variants(m, seqs, trs)
            for a, d in enumerate(D_RANGE):
                if (q - d) % 3: continue
                i = (q - d) // 3
                v = U[:, i]; cv = v - v.mean()
                for b, x in zip(states, cv):
                    nt_sum[a, C.NT.index(b)] += x; nt_cnt[a, C.NT.index(b)] += 1
        out[mode] = dict(cod_c=cod_c, nt_sum=nt_sum, nt_cnt=nt_cnt)
    out["sec"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------------------------- drivers
def run_peak_ism(oof, folds=None, offsets=None, peaks=None, n_per_half=10, seed=2, halves=None, workers=1, log=print):
    """Peak ISM -> dict of arrays (same layout as the paper's ism.npz / ism_ext.npz).
    ``peaks`` (list of (tx, p, fold, half, unit)) re-uses a fixed peak set, e.g. for the far-offset controls."""
    folds = list(oof.folds) if folds is None else folds
    offsets = list(C.OFFSETS if offsets is None else offsets)
    if peaks is None:
        peaks = peak_set(oof, C.halves_for(oof.ds, halves), folds, n_per_half, seed)
    C.G["oof"] = oof
    t0 = time.time()
    res = C.pmap(_ism_task, [(t, p, offsets) for t, p, *_ in peaks], workers)
    log(f"[ism] {oof.dataset}: {len(res)} peaks x {len(offsets)} offsets, {time.time() - t0:.1f}s")
    return dict(tx=np.array([r["tx"] for r in res]), p=np.array([r["p"] for r in res]), L=np.array([r["L"] for r in res]),
                fold=np.array([x[2] for x in peaks]), half=np.array([x[3] for x in peaks]), unit=np.array([x[4] for x in peaks]),
                offsets=np.array(offsets), wt=np.array([r["wt"] for r in res]), dF=np.stack([r["dF"] for r in res]),
                dR=np.stack([r["dR"] for r in res]), u0p=np.array([r["u0p"] for r in res]),
                u0_stored=np.array([r["u0_stored"] for r in res]), u_dev_max=np.array([r["u_dev_max"] for r in res]),
                mfe_windows_recomputed_mean=np.array([r["mfe_windows_recomputed_mean"] for r in res]),
                n_var=np.array([r["n_var"] for r in res]), cpu_sec=np.array([r["sec"] for r in res]))


def run_maps(oof, folds=None, n_per_half=15, seed=1, offsets=None, halves=None, workers=1, log=print):
    """Centred codon and nucleotide sensitivity maps -> dict of arrays (layout of the paper's map.npz)."""
    folds = list(oof.folds) if folds is None else folds
    offsets = list(C.OFFSETS if offsets is None else offsets)
    halves = C.halves_for(oof.ds, halves)
    tasks, info = [], []
    for f in folds:
        for h in ("A", "B"):
            for k, (t, _) in enumerate(sample_transcripts(oof, halves, f, h, n_per_half, seed, False)):
                tasks.append((t, 1000 * f + (0 if h == "A" else 500) + k, offsets)); info.append((f, h, oof.ds.unit[t]))
    C.G["oof"] = oof; t0 = time.time()
    res = C.pmap(_map_task, tasks, workers)
    log(f"[map] {oof.dataset}: {len(res)} transcripts, {time.time() - t0:.1f}s")
    sv = dict(tx=np.array([r["tx"] for r in res]), fold=np.array([x[0] for x in info]), half=np.array([x[1] for x in info]),
              unit=np.array([x[2] for x in info]), sites=np.array([r["sites"] for r in res]),
              nt_sites=np.array([r["nt_sites"] for r in res]), offsets=np.array(offsets), d_range=np.array(D_RANGE))
    for mode in ("fixed", "recompute"):
        sv[f"cod_c_{mode}"] = np.stack([r[mode]["cod_c"] for r in res])          # (n_tx, 4, nO, 61)
        sv[f"nt_sum_{mode}"] = np.stack([r[mode]["nt_sum"] for r in res])
        sv[f"nt_cnt_{mode}"] = np.stack([r[mode]["nt_cnt"] for r in res])
    return sv


# ---------------------------------------------------------------------------------------------- summaries
def summarize_ism(z, offsets=None):
    """Per-offset mean effect and mean |effect| (cluster-bootstrap 95% intervals, one unit per peak) for both MFE
    treatments, the recompute - fixed difference (a perturbation-based MFE-route estimate, not an exact split), and the
    selection/evaluation check across the two unit halves."""
    offsets = list(z["offsets"]) if offsets is None and "offsets" in z else list(C.OFFSETS if offsets is None else offsets)
    unit = np.asarray(z["unit"]); half = np.asarray(z["half"]); out = {}
    for key, nm in (("dR", "recompute"), ("dF", "fixed")):
        d = z[key]
        mean_eff = np.nanmean(d, 2); mean_abs = np.nanmean(np.abs(d), 2)
        out[nm] = dict(mean_effect=[C.cluster_boot_ci(mean_eff[:, a], unit, seed=a) for a in range(d.shape[1])],
                       mean_abs=[C.cluster_boot_ci(mean_abs[:, a], unit, seed=100 + a) for a in range(d.shape[1])])
    dd = z["dR"] - z["dF"]
    out["recompute_minus_fixed"] = dict(
        label="perturbation-based MFE route estimate (RiboPipe is not additive; not an exact decomposition)",
        mean_effect=[C.cluster_boot_ci(np.nanmean(dd, 2)[:, a], unit, seed=200 + a) for a in range(dd.shape[1])],
        mean_abs=[C.cluster_boot_ci(np.nanmean(np.abs(dd), 2)[:, a], unit, seed=300 + a) for a in range(dd.shape[1])])
    sel = {}
    for key, nm in (("dR", "recompute"), ("dF", "fixed")):
        d = z[key]; mean_abs = np.nanmean(np.abs(d), 2); support = -np.nanmean(d, 2)
        for X, Y in (("A", "B"), ("B", "A")):
            mX, mY = half == X, half == Y
            prof = np.nanmean(mean_abs[mX], 0); a = int(np.nanargmax(prof)); r = offsets[a]
            wt_aa = np.array([C.CODON_TABLE.get(str(w[a]), "?") for w in z["wt"]])
            aa_rank = []
            for aa in sorted(set(wt_aa[mX])):
                v = support[mX & (wt_aa == aa), a]
                aa_rank.append((float(np.nanmean(v)), aa, int(np.isfinite(v).sum())))
            aa_rank.sort(key=lambda x: (-x[0], x[1])); top = aa_rank[:3]
            ev = []
            for _, aa, nX in top:
                mm = mY & (wt_aa == aa)
                ci = C.cluster_boot_ci(support[mm, a], unit[mm], seed=7) if mm.any() else (float("nan"),) * 3
                ev.append(dict(aa=aa, n_select=nX, select_mean=float(np.nanmean(support[mX & (wt_aa == aa), a])),
                               n_eval=int(mm.sum()), eval_mean_ci=ci))
            tgt = np.nanmean(d[mX][:, a, :], 0); order = [int(i) for i in np.argsort(-np.nan_to_num(tgt, nan=-1e9))]
            up = [b for b in order if np.isfinite(tgt[b])][:3]
            dn = [b for b in order[::-1] if np.isfinite(tgt[b])][:3]
            tg = []
            for b in up + dn:
                tg.append(dict(codon=C.CODONS[b], aa=C.CODON_TABLE[C.CODONS[b]], select_mean=float(tgt[b]),
                               eval_mean_ci=C.cluster_boot_ci(d[mY][:, a, b], unit[mY], seed=9)))
            rY = np.nanmean(mean_abs[mY], 0)
            sel[f"{nm}:{X}->{Y}"] = dict(r_star=r, r_star_label=C.lab(r), select_profile_max=float(prof[a]),
                                         eval_value_at_r_star=C.cluster_boot_ci(mean_abs[mY, a], unit[mY], seed=3),
                                         eval_argmax_r=offsets[int(np.nanargmax(rY))],
                                         profile_spearman_select_vs_eval=C.spearman(prof, rY),
                                         wt_aa_support_top3=ev, target_codon_supplementary_not_preregistered=tg)
    out["select_eval"] = sel
    out["n_peaks"] = int(len(z["tx"])); out["n_peaks_by_half"] = {h: int((half == h).sum()) for h in ("A", "B")}
    out["n_variants_per_mode"] = int(np.isfinite(z["dF"]).sum())
    out["u_cpu_vs_stored_max_abs"] = float(np.nanmax(z["u_dev_max"]))
    out["mfe_windows_recomputed_per_variant_mean"] = float(np.mean(z["mfe_windows_recomputed_mean"]))
    return out


def summarize_response_range(z_core, z_ext, threshold=0.1):
    """Response range of the peak ISM: the longest run of consecutive core offsets containing the maximum mean |Delta u|
    with mean |Delta u| >= threshold * maximum, and the far-offset control level (same peaks, offsets +-15, +-20)."""
    core = list(z_core["offsets"]); ext = list(z_ext["offsets"])
    assert (np.asarray(z_core["tx"]) == np.asarray(z_ext["tx"])).all() and (np.asarray(z_core["p"]) == np.asarray(z_ext["p"])).all()
    unit = np.asarray(z_core["unit"]); offs = core + ext; out = dict(edited_offsets=sorted(offs))
    out["n_peaks_not_editable"] = {str(r): int(np.isnan(z_ext["dF"][:, k]).all(1).sum()) for k, r in enumerate(ext)}
    out["n_peaks_not_editable"].update({str(r): int(np.isnan(z_core["dF"][:, k]).all(1).sum()) for k, r in enumerate(core)})
    for key, nm in (("dR", "recompute"), ("dF", "fixed")):
        d_all = np.concatenate([z_core[key], z_ext[key]], 1)
        ma = np.nanmean(np.abs(d_all), 2)
        prof = {r: C.cluster_boot_ci(ma[:, k], unit, seed=k) for k, r in enumerate(offs)}
        M = max(v[0] for v in prof.values()); rmax = max(prof, key=lambda r: prof[r][0])
        lo = hi = rmax
        while lo - 1 in core and prof[lo - 1][0] >= threshold * M: lo -= 1
        while hi + 1 in core and prof[hi + 1][0] >= threshold * M: hi += 1
        far = np.nanmean(ma[:, len(core):], 1)
        out[nm] = dict(profile_mean_abs={str(r): prof[r] for r in sorted(offs)}, max_mean_abs=M, r_at_max=rmax,
                       effective_response_range=[lo, hi], threshold=threshold * M,
                       edge_values={str(r): prof[r][0] for r in (core[0], core[-1])},
                       far_control_mean_abs=C.cluster_boot_ci(far, unit, seed=99), far_over_max=float(np.mean(far) / M),
                       ext_mean_effect={str(r): C.cluster_boot_ci(np.nanmean(z_ext[key][:, k], 1), unit, seed=50 + k)
                                        for k, r in enumerate(ext)})
    return out


def summarize_map(z):
    """Codon map (mean centred value per offset and codon), positional importance (mean over sites of the SD over the
    61 states, cluster-bootstrap interval), half-to-half reproducibility and the nucleotide map."""
    offsets = list(z["offsets"]) if "offsets" in z else list(C.OFFSETS)
    out = {}; unit = np.asarray(z["unit"]); half = np.asarray(z["half"])
    for nm in ("recompute", "fixed"):
        c = z[f"cod_c_{nm}"]
        M = np.nanmean(c.reshape(-1, c.shape[2], 61), 0)
        imp_site = np.sqrt(np.nanmean(c ** 2, 3))
        imp_tx = np.nanmean(imp_site, 1)
        MA = np.nanmean(c[half == "A"].reshape(-1, c.shape[2], 61), 0); MB = np.nanmean(c[half == "B"].reshape(-1, c.shape[2], 61), 0)
        ns = z[f"nt_sum_{nm}"].sum(0); nc = z[f"nt_cnt_{nm}"].sum(0); NTm = ns / np.maximum(nc, 1)
        nt_imp = np.sqrt((NTm ** 2).mean(1))
        top = []
        for a, r in enumerate(offsets):
            k = int(np.argmax(M[a])); k2 = int(np.argmin(M[a]))
            top.append(dict(r=r, label=C.lab(r), max_codon=C.SENSE[k], max_aa=C.CODON_TABLE[C.SENSE[k]], max_val=float(M[a, k]),
                            min_codon=C.SENSE[k2], min_aa=C.CODON_TABLE[C.SENSE[k2]], min_val=float(M[a, k2])))
        out[nm] = dict(map=M.tolist(), importance=[C.cluster_boot_ci(imp_tx[:, a], unit, seed=a) for a in range(c.shape[2])],
                       half_map_pearson=float(np.corrcoef(MA.ravel(), MB.ravel())[0, 1]),
                       half_importance_spearman=C.spearman(np.nanmean(imp_tx[half == "A"], 0), np.nanmean(imp_tx[half == "B"], 0)),
                       per_offset_extremes=top, nt_map=NTm.tolist(), nt_importance=nt_imp.tolist(),
                       centering_check_max_abs_rowsum=float(np.nanmax(np.abs(np.nansum(c, 3)))))
    out["n_tx"] = int(len(z["tx"])); out["n_sites"] = int(np.isfinite(z["cod_c_fixed"][:, :, 0, 0]).sum())
    return out

