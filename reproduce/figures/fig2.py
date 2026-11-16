#!/usr/bin/env python3
"""Figure 2 (Part II): predicted versus sparse observed profiles against an independent replicate.

Ported from the published generator make_fig_part2_1115_v8.py (md5 d7255aab527f589e61ca89071941f648); the plotting code is
unchanged. Reads only paper_results/:
  partII/partII_eval.json                 crossover analysis (main analysis = v2_primary, thinning draw 0) for RiboPipe, ridge,
                                          adapted RiboMIMO, noise predictor; operational criterion at D*
  partII/ixnos_addon_results.json         iXnos (MFE / sequence only) crossover results and ten-draw summaries
  partII/more_addon_results.json          RUST crossover results (reference curve of panel a)
  partII/thinning_seeds.json              outcome counts over the ten thinning draws (labels where no crossover was located)
  partII/eval_v2_ixnos/eval_v2_<ds>.json  iXnos (sequence only) per-bin curves
  partII/fig2_depth_cdf.json              panel d curve: depth of every low-coverage record, sorted, without identifiers
                                          (made by paper_results/derive_fig2.py from the internal record tables)
Panels: (a) paired difference vs depth, RiboPipe + RUST + noise predictor, D* with 95% CI; (b) D* by predictor;
(c) D* by design (main analysis, controlled thinning, pooled-replicate split); (d) depth distribution of the low-coverage
population and the share of scorable transcripts below D*.
Usage: python3 reproduce/figures/fig2.py OUTDIR [--results paper_results]
"""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
A = C.parse_args(__doc__); IN = C.Inputs(A.results); OUT = A.outdir
FNAME = "fig2"
evx = IN.json("partII/ixnos_addon_results.json"); assert evx["gates_all_pass"] is True
evm = IN.json("partII/more_addon_results.json"); assert evm["gates"]["G2_all_pass"] is True and evm["gates"]["G1_pass_excluding_BiLSTM97"] is True
evs = IN.json("partII/thinning_seeds.json"); assert evs["primary_seed"] == 0   # counts over the ten thinning draws (labels in a and b)
SRCXE = {t: f"partII/eval_v2_ixnos/eval_v2_{t}.json" for t in ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]}
ev = IN.json("partII/partII_eval.json")
CDF = IN.json("partII/fig2_depth_cdf.json")
DS = ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]
DN = {"DMSO_TX9P1": "TX9_WT", "GSE233886_WT": "GSE233886_WT", "GSE133393_WT": "GSE133393_WT", "PRJNA_iPS": "PRJNA_iPS"}
CL = {"DMSO_TX9P1": "HEK293T", "GSE233886_WT": "HEK293, Flp-In T-REx", "GSE133393_WT": "HEK293", "PRJNA_iPS": "iPSC 409B2"}
RP, RPn, RMm, RMn, RG, DU, IXm, IXs = "Paper_RiboPipe_MFE_H128", "Paper_RiboPipe_noMFE_H128", "Paper_RiboMIMO_adapt_MFE", "Paper_RiboMIMO_adapt_noMFE", "Paper_Ridge", "Dummy_noise", "Paper_iXnos_MFE_lr005_elig", "Paper_iXnos_seq_lr005_elig"
RU = "Paper_RUST"
COL = {RP: "#d55e00", RPn: "#f4a582", RMm: "#009e73", RMn: "#8fd1b8", RG: "#0072b2", DU: "#8a8985", IXm: "#882255", IXs: "#882255", RU: "#1a1a1a"}
COL[RPn] = COL[RP]; COL[RMn] = COL[RMm]   # one colour per family; filled = +MFE, open = -MFE
FILLED = {RP: True, RPn: False, RMm: True, RMn: False, IXm: True, IXs: False, RG: True}
MK = {RP: "o", RPn: "o", RMm: "s", RMn: "s", RG: "D", DU: "x", IXm: "^", IXs: "^", RU: "v"}
LAB = {RP: "RiboPipe", RPn: "RiboPipe (−MFE)", RMm: "adapted RiboMIMO + MFE", RMn: "adapted RiboMIMO", RG: "ridge", DU: "noise predictor", IXm: "iXnos (MFE)", IXs: "iXnos (sequence only)", RU: "RUST"}
VARS = [("v2_primary", "main analysis"), ("v2_sens_gridthin", "same transcripts,\ncontrolled thinning"), ("v2_sens_binom", "pooled-replicate split\nas reference")]
STX = {"NO_SIGN_CHANGE": "no sign change", "NOT_LOCATABLE": "not located", None: "not analysed"}   # the two pre-registered phrases
def draw_label(k0, k1, k2, n):
    """Outcome over the ten thinning draws for a predictor without a located crossover in the pre-specified draw (two pre-registered phrases)."""
    assert k0 + k1 + k2 == n
    if k1 == n: return f"no sign change ({n}/{n} draws)"
    if k2 == n: return f"not located ({n}/{n} draws)"
    seg = []
    if k1: seg.append(f"no sign change in {k1}/{n} draws")
    if k2: seg.append(f"not located in {k2}/{n}" + ("" if seg else " draws"))
    if k0: seg.append(f"located in {k0}/{n}")
    return ", ".join(seg)
def nocross(t, p, d):
    """Plain label for a predictor with no located crossover in the main analysis (draw 0), with the ten-draw counts read from thinning_seeds.json."""
    if p in (IXm, IXs):
        s = evx["thinning_seed_summary"].get(f"{p}/{t}"); s0 = evx["seeds"][f"{p}/{t}"][0]; assert str(s0["seed"]) == "0"; st0 = s0["status"]; srcn = "ixnos_addon/thinning_seed_summary"
    else:
        s = evs["results"].get(t, {}).get(p); st0 = None if s is None else s["primary_seed0"]["status"]; srcn = "rerun76/thinning_seeds/thinning_seeds.json"
    used_seeds[f"{t}|{p}"] = None if s is None else dict({k: s[k] for k in ("n_seeds", "n_locatable", "n_no_sign_change", "n_not_locatable")}, source=srcn)
    if s is None: return STX.get(d["status"], d["status"]) + " (pre-specified draw)"
    assert st0 == d["status"], (t, p)
    return draw_label(s["n_locatable"], s["n_no_sign_change"], s["n_not_locatable"], s["n_seeds"])
def wrap_a(lab):
    """Line breaks for the narrow a panels: 'no sign change' on its own line."""
    head = "no sign change" if lab.startswith("no sign change") else "not located"
    rest = lab[len(head):].strip()
    return head + "\n" + rest.replace(", ", ",\n")
used_seeds = {}; labels = {}
FS = 9   # title hierarchy as in Figure 1 (panel title FS+0.5, bold header FS+1.5, letter FS+3)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "axes.titlesize": FS + 0.5, "pdf.fonttype": 42})
# layout arithmetic: row 2 (b, c, d) 0.9 x the height of the earlier version; row 1, row gap and top/bottom margins kept in inches
H0 = 10.2; _c0 = (0.885 - 0.22) * H0 / (2 + 0.30); _r1, _r2, _gap = 2 * _c0 * 0.72 / 1.87, 2 * _c0 * 1.15 / 1.87, 0.30 * _c0
HN = (1 - 0.885) * H0 + _r1 + _gap + 0.9 * _r2 + 0.22 * H0; _HSN = _gap / ((_r1 + 0.9 * _r2) / 2); FYN = lambda y: 1 - (1 - y) * H0 / HN
# width = 1.2 x the earlier version; left/right margins and panel gaps kept in inches (wspace recomputed), panels widen
W7 = 7.2 * 1.05; W8 = W7 * 1.2; _PR7 = (0.98 - 0.15) * W7; _PR8 = _PR7 + (W8 - W7); FXN = lambda x: x * W7 / W8
_WSN = lambda n, ws: (ws * _PR7 / (n + ws * (n - 1))) / ((_PR8 - (n - 1) * ws * _PR7 / (n + ws * (n - 1))) / n)
fig = plt.figure(figsize=(W8, HN))
outer = fig.add_gridspec(2, 1, height_ratios=[0.72, 1.15 * 0.9], hspace=_HSN, left=0.15 * W7 / W8, right=1 - 0.02 * W7 / W8, top=FYN(0.885), bottom=0.22 * H0 / HN)
rowA = outer[0].subgridspec(1, 4, wspace=_WSN(4, 0.18)); rowB = outer[1].subgridspec(1, 3, width_ratios=[1.3, 0.62, 0.95], wspace=_WSN(3, 0.55))
used = {}; YMIN, YMAX = -0.2, 0.2
PA = [RP, RU, DU]   # reference curve = RUST; RiboPipe drawn on top
TICK_Y = {RP: -0.02, RU: -0.05}
def getd_a(t, p, V):
    if p == IXs:
        e = IN.json(SRCXE[t]); d = dict(e["variants"]["v2_primary"]["dstar"][p]); assert e["thin_seed"] == 0
        x = evx["primary"][f"{p}/{t}"]; assert (d["status"], d["D_star"], d["ci95"]) == (x["status"], x["D_star"], x["ci95"]); d["_src"] = f"eval_v2_ixnos/{t}"; return d
    if p == RU:
        d = dict(evm["primary"][f"{p}/{t}"]); assert str(d["seed"]) == "0"; d["_src"] = "more_predictors_addon/more_addon_results.json"; return d
    return V["dstar"][p]
for c, t in enumerate(DS):
    ax = fig.add_subplot(rowA[0, c]); V = ev["datasets"][t]["variants"]["v2_primary"]
    dp = ev["dummy_positive_bins"][t]; band_hi = max(z[1] for z in dp); band_lo = [b for b in V["dstar"][RP]["bins"] if b["valid"]][0]["lo"]
    ax.axvspan(band_lo, band_hi, color="#efeeea", lw=0, zorder=0); ax.axhline(0, color="#52514e", lw=0.6)
    for k, p in enumerate(PA):
        d = getd_a(t, p, V); xs, ys = [], []
        for j, b in enumerate(d["bins"]):
            if b["valid"]: xs.append(10 ** b["x_log10"]); ys.append(b["y"])
        main = p in (RP, DU)
        ax.plot(xs, ys, color=COL[p], lw=(0.8 if p == DU else (1.3 if p == RP else 0.75)), ls=("--" if p == DU else "-"), marker=MK[p],
                ms=(2.5 if p == DU else (3 if p == RP else 2.2)), mfc=("white" if main else COL[p]), mec=COL[p], mew=(0.8 if main else 0.4),
                zorder=(4 if p == RP else 3), alpha=(1 if main else 0.9), clip_on=True)
        if d["status"] == "OK" and p != DU:
            yv = TICK_Y[p]
            halo = [pe.withStroke(linewidth=3.2, foreground="white")]   # D* tick drawn above the curves with a white halo
            ax.plot(d["ci95"], [yv, yv], color=COL[p], lw=(2 if p == RP else 1.4), alpha=0.6, solid_capstyle="butt", zorder=6, path_effects=halo)
            ax.plot([d["D_star"]], [yv], marker="|", ms=(7 if p == RP else 5), mew=(1.6 if p == RP else 1.2), color=COL[p], zorder=7, path_effects=halo)
        used[f"a|{t}|{p}"] = {"status": d["status"], "D_star": d.get("D_star"), "ci95": d.get("ci95"), "x": xs, "y": ys, "source": d.get("_src", "rerun76/partII_eval.json")}
    s = V["dstar"][RP]
    if s["status"] == "OK": ax.text(0.04, 0.97, f"$D^*$ = {s['D_star']:.3f}", transform=ax.transAxes, va="top", fontsize=FS - 0.5, color=COL[RP])
    else:
        lab_ = nocross(t, RP, s); labels[f"a|{t}|{RP}"] = lab_
        ax.text(0.04, 0.97, wrap_a(lab_), transform=ax.transAxes, va="top", fontsize=FS - 1.5, color=COL[RP], linespacing=1.1, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.set_xscale("log"); ax.set_xlim(5e-5, 2); ax.set_ylim(YMIN, YMAX); ax.set_yticks([-0.2, -0.1, 0, 0.1])
    ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    ax.set_title(f"a-{c+1}  {DN[t]}\n{CL[t]}", loc="left", fontsize=FS + 0.5)
    ax.set_xlabel("depth (reads/codon)", fontsize=FS - 0.5)
    if c == 0: ax.set_ylabel("Δ median r with replicate\n(estimate − sparse reads)")
    else: ax.set_yticklabels([])
# (b) D* by predictor
axb = fig.add_subplot(rowB[0, 0]); PB = [RP, RPn, RMm, RMn, IXm, IXs, RG]
def getd(t, p):
    if p in (IXm, IXs):
        d = evx["primary"][f"{p}/{t}"]; assert str(d["seed"]) == "0"; return d
    return ev["datasets"][t]["variants"]["v2_primary"]["dstar"][p]
yy = 0; ylab, ypos = [], []
for t in DS:
    for k, p in enumerate(PB):
        d = getd(t, p); y = -(yy + k * 0.2 + (0.06 if k >= 2 else 0) + (0.06 if k >= 4 else 0) + (0.06 if k >= 6 else 0))   # row pitch 0.2: neighbouring labels do not overlap
        if d["status"] == "OK":
            axb.plot(d["ci95"], [y, y], color=COL[p], lw=1.2); axb.plot([d["D_star"]], [y], marker=MK[p], ms=3.6, color=COL[p], mfc=(COL[p] if FILLED[p] else "white"), mec=COL[p], mew=0.8)
        else:
            lb_ = ("" if p == RG else ("+MFE: " if FILLED[p] else "−MFE: ")) + nocross(t, p, d)
            axb.text(0.02, y, lb_, fontsize=FS - 2.6, color=COL[p], va="center", ha="left", clip_on=False); labels[f"b|{t}|{p}"] = lb_
        used[f"b|{t}|{p}"] = {"status": d["status"], "D_star": d.get("D_star"), "ci95": d.get("ci95")}
    ypos.append(-(yy + 0.69)); ylab.append(DN[t] + "\n" + CL[t].replace(", ", ",\n")); yy += 1.6
axb.set_yticks(ypos); axb.set_yticklabels(ylab, fontsize=FS - 1); axb.set_xlim(0, 1.25); axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axb.set_xlabel("$D^*$ (reads/codon)"); axb.set_title("b  $D^*$ by predictor", loc="left", fontsize=FS + 1.5, fontweight="bold")
axb.grid(axis="x", color="#e6e6e6", lw=0.5); axb.set_axisbelow(True); axb.tick_params(axis="y", length=0)
axb.legend(handles=[Line2D([], [], color=COL[p], marker=MK[p], ms=3.5, lw=1.2, mfc=(COL[p] if FILLED[p] else "white"), mec=COL[p], label=LAB[p]) for p in PB], fontsize=FS - 1.5, frameon=False, title="filled = with folding energies (MFE), open = without MFE\ntext = outcome over the ten thinning draws\nwhere no crossover was located", title_fontsize=FS - 1.5, ncol=2, columnspacing=0.8,
           loc="upper center", bbox_to_anchor=(0.5, -0.13), handlelength=1.4)
# (c) D* by reference / axis
axc = fig.add_subplot(rowB[0, 1]); VC = ["#d55e00", "#e69f00", "#cc79a7", "#56b4e9"]; MV = ["o", "^", "v", "s"]
yy = 0; ypos = []
for t in DS:
    for k, (var, nm) in enumerate(VARS):
        d = ev["datasets"][t]["variants"].get(var, {}).get("dstar", {}).get(RP); y = -(yy + 0.2 + k * 0.4)
        stt = d["status"] if d else None
        if stt == "OK":
            axc.plot(d["ci95"], [y, y], color=VC[k], lw=1.2); axc.plot([d["D_star"]], [y], marker=MV[k], ms=3.6, color=VC[k], mec="white", mew=0.4)
        else:
            lc_ = nocross(t, RP, d) if var == "v2_primary" else (STX.get(stt, stt) + " (pre-specified draw)" if stt else "not analysed")
            lc_ = lc_.replace(" (", "\n(", 1).replace(", ", ",\n", 1).replace(" in ", "\nin ", 1) if len(lc_) > 22 else lc_
            n_l = lc_.count("\n") + 1   # three-line labels move up so they do not reach the next row
            axc.text(0.03, y + (0.12 if n_l >= 3 else 0), lc_, fontsize=FS - 2.6, color=VC[k], va="center", ha="left", linespacing=0.95); labels[f"c|{t}|{var}"] = lc_
        used[f"c|{t}|{var}"] = {"status": stt, "D_star": d.get("D_star") if d else None, "ci95": d.get("ci95") if d else None}
    ypos.append(-(yy + 0.6)); yy += 1.6
axc.set_yticks(ypos); axc.set_yticklabels([]);   # rows aligned with b; dataset labels not repeated
axc.set_xlim(0, 1.25); axc.set_xticks([0, 0.5, 1.0]); axc.set_ylim(-6.45, 0.25); axb.set_ylim(-6.45, 0.25)
axc.set_xlabel("RiboPipe $D^*$ (reads/codon)"); axc.set_title("c  $D^*$ by design", loc="left", fontsize=FS + 1.5, fontweight="bold")
axc.grid(axis="x", color="#e6e6e6", lw=0.5); axc.set_axisbelow(True); axc.tick_params(axis="y", length=0)
axc.legend(handles=[Line2D([], [], color=VC[k], marker=MV[k], ms=3.5, lw=1.2, label=nm) for k, (v, nm) in enumerate(VARS)], fontsize=FS - 1.5,
           frameon=False, loc="upper left", bbox_to_anchor=(-0.05, -0.13), ncol=1, handlelength=1.4)
# (d) cumulative depth of T_low
axd = fig.add_subplot(rowB[0, 2]); DC = {"DMSO_TX9P1": "#1a1a1a", "GSE233886_WT": "#6b6b6b", "GSE133393_WT": "#1a1a1a", "PRJNA_iPS": "#6b6b6b"}
DLS = {"DMSO_TX9P1": "-", "GSE233886_WT": "-", "GSE133393_WT": "--", "PRJNA_iPS": "--"}   # datasets in grey + line style, not confused with predictor colours
txt = []
for t in DS:
    V = ev["datasets"][t]["variants"]["v2_primary"]; cd = CDF["datasets"][t]; assert CDF["meta"]["records"][t]["md5"] == V["records_md5"]
    n = cd["n"]; nz = cd["n_zero"]; pos = np.asarray(cd["depth_pos"], float); assert len(pos) + nz == n and np.all(np.diff(pos) >= 0) and pos[0] > 0
    axd.step(pos, (np.arange(1, len(pos) + 1) + nz) / n, where="post", color=DC[t], ls=DLS[t], lw=1.1, label=DN[t])
    o = ev["datasets"][t]["opcrit"][RP]
    if o["status"] == "OK":
        x0 = o["at_D_star"]["threshold"]; f0 = o["at_D_star"]["frac_scorable_below"]
        axd.plot([x0], [o["at_D_star"]["frac_eligible_below"]], marker="o", ms=4.5, color=COL[RP], mec="white", mew=0.5, zorder=5)
        txt.append(f"{DN[t]}: {f0:.0%}")
        used[f"d|{t}"] = {"D_star": x0, "frac_scorable_below": f0, "frac_eligible_below": o["at_D_star"]["frac_eligible_below"]}
axd.set_xscale("log"); axd.set_xlim(5e-5, 2); axd.set_ylim(0, 1); axd.set_xlabel("depth (reads/codon)"); axd.set_ylabel("cumulative fraction of\nlow-coverage transcripts")
from matplotlib.ticker import FixedLocator, NullLocator; axd.xaxis.set_major_locator(FixedLocator([1e-3, 1e-1])); axd.xaxis.set_minor_locator(NullLocator())   # fixed ticks (on the wider axis the automatic log locator would add 1e-4, 1e-2, 1)
axd.grid(color="#e6e6e6", lw=0.5); axd.set_axisbelow(True); axd.set_title("d  Population", loc="left", fontsize=FS + 1.5, fontweight="bold")
axd.text(0.04, 0.97, "below $D^*$, share of\nscorable transcripts:\n" + "\n".join(txt), transform=axd.transAxes, va="top", fontsize=FS - 1.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
axd.legend(fontsize=FS - 1.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, handlelength=1.4)
fig.text(FXN(0.008), FYN(0.944), "a", fontsize=FS + 3, fontweight="bold"); fig.text(FXN(0.035), FYN(0.946), "Predicted vs sparse observed profiles against an independent replicate", fontsize=FS + 1.5, fontweight="bold", color="#333333")
fig.legend(handles=[Line2D([], [], color=COL[RP], marker="o", mfc="white", lw=1.3, label="RiboPipe")] +
                   [Line2D([], [], color=COL[RU], marker=MK[RU], ms=3, mfc=COL[RU], mec=COL[RU], lw=0.75, label="RUST (reference)")] +
                   [Line2D([], [], color=COL[DU], marker="x", ls="--", lw=0.8, label="noise predictor"), Patch(color="#efeeea", label="noise predictor also above 0"),
                    Line2D([], [], color="#333333", marker="|", ls="none", ms=7, mew=1.6, label="crossover depth $D^*$ (bar = 95% CI)")],
           loc="upper center", ncol=3, fontsize=FS - 1, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handlelength=1.5)
fig.savefig(os.path.join(OUT, FNAME + ".pdf")); fig.savefig(os.path.join(OUT, FNAME + ".png"), dpi=300)
# same keys as the published provenance: numbers_drawn, labels_drawn, seed_counts_used
p = C.write_numbers(OUT, FNAME, used, IN, extra={"published_generator": "make_fig_part2_1115_v8.py", "published_pdf_md5": "0789aef1390f9565e1d7e76b0588bdcc",
                                                  "labels_drawn": labels, "seed_counts_used": used_seeds})
print("wrote", FNAME, p)
