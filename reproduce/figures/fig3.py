#!/usr/bin/env python3
"""Figure 3 (fig:part2use): using the prediction on low-coverage transcripts.

Ported from the published generator make_fig_part2_use_1115_v4.py (md5 0aa8e41036f36117d9bcd27eb00c0897); the plotting code
is unchanged. Reads only paper_results/ (all files copied unchanged):
  partII/posterior_partII_results.json   posterior (Dirichlet-multinomial posterior mean of the reads with the RiboPipe prediction
                                         as prior) on the main-analysis low-coverage bins, draw 0
  partII/partII_eval.json                main analysis (bins, noise band), used for the shared x axis and consistency checks
  partII/PARTII_CROSSOVER_v2.json        protocol (bin validity rule, asserted)
  application/a1_<ds>.json               controlled thinning of held-out high-coverage transcripts, top-5% precision
  application/n2_tercile_ci.json         E-P Pro-Pro sites in naturally low-coverage transcripts by predicted third (RiboPipe)
  application/pp_tercile_cmp.json, gate_g1.json   the same for adapted RiboMIMO + MFE and ridge (gates asserted)
Panels: (a) posterior - reads, posterior - RiboPipe and RiboPipe - reads (reference) per depth bin with 95% similarity-unit
bootstrap intervals; (b) controlled thinning, precision for the reference top-5% codons; (c) observed/expected occupancy of
E-P Pro-Pro sites grouped by each model's predicted third.
Usage: python3 reproduce/figures/fig3.py OUTDIR [--results paper_results]
"""
import os, sys, json, math
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

A = C.parse_args(__doc__); IN = C.Inputs(A.results); OUT = A.outdir; STEM = "fig3"
POST = IN.json("partII/posterior_partII_results.json")
EV = IN.json("partII/partII_eval.json")
assert POST["gates_all_pass"] is True and POST["meta"]["draw_primary"] == 0
DS = ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]
A1 = {t: IN.json(f"application/a1_{t}.json") for t in DS}; TC = IN.json("application/n2_tercile_ci.json")
CMP = IN.json("application/pp_tercile_cmp.json"); G1 = IN.json("application/gate_g1.json")
assert CMP["gate_G2"]["passed"] is True and CMP["crosschecks"]["passed"] is True and G1["passed"] is True
DN = {"DMSO_TX9P1": "TX9_WT", "GSE233886_WT": "GSE233886_WT", "GSE133393_WT": "GSE133393_WT", "PRJNA_iPS": "PRJNA_iPS"}
CL = {"DMSO_TX9P1": "HEK293T", "GSE233886_WT": "HEK293, Flp-In T-REx", "GSE133393_WT": "HEK293", "PRJNA_iPS": "iPSC 409B2"}   # as in Fig. 2
RPC = "#d55e00"                       # RiboPipe orange, as in Fig. 2 and the application figure
C_PR, C_PP = "#1baf7a", "#6a4fc2"     # posterior - reads (green, = posterior colour of panel b), posterior - RiboPipe (violet)
# palette {#d55e00, #1baf7a, #6a4fc2} checked with the dataviz validate_palette.js (--pairs all, light): all checks PASS,
# worst CVD pair dE 10.4 (deutan, green vs orange), normal-vision floor 26.7; contrast WARN for the green -> markers + legend carry identity.
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e6e6", "#efeeea"
FS = 9   # title hierarchy as in Figure 1
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "pdf.fonttype": 42, "axes.edgecolor": INK2})
NMIN = 30   # protocol bin validity (>= 30 scored transcripts per role); asserted against the protocol below
proto = IN.json("partII/PARTII_CROSSOVER_v2.json")
assert f"N_scored >= {NMIN}" in proto["per_bin_value"]

W0, W = 7.2, 7.2 * 1.05; XS = W0 / W   # canvas 1.05 x wider than the earlier version; absolute left/right margins and header x positions kept (relative coords x XS)
fig = plt.figure(figsize=(W, 9.6))
gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 0.9, 0.9], hspace=1.3, wspace=0.2, left=0.095 * XS, right=1 - (1 - 0.985) * XS, top=0.86, bottom=0.065)
used = {"a": {}, "b": {}, "c": {}}
SER = [("RiboPipe-reads", "RiboPipe − reads (reference)", RPC, "o", 0.0), ("posterior-reads", "posterior − reads", C_PR, "D", -0.035),
       ("posterior-RiboPipe", "posterior − RiboPipe", C_PP, "^", 0.035)]
YL = (-0.14, 0.185)
for c, t in enumerate(DS):
    ax = fig.add_subplot(gs[0, c]); d = POST["datasets"][t]; bins = d["bins"]
    vmain = EV["datasets"][t]["variants"]["v2_primary"]["dstar"]["Paper_RiboPipe_MFE_H128"]["bins"]
    vb = [b for b in vmain if b["valid"]]
    assert [b["bin"] for b in bins] == [b["bin"] for b in vb], t          # same valid bins as the main analysis
    for b, m in zip(bins, vb):
        assert min(b["N_scored_role"]) >= NMIN and b["N_scored_role"] == m["N_scored_role"], (t, b["bin"])
        assert abs(b["RiboPipe-reads"]["y"] - m["y"]) < 1e-12, (t, b["bin"])   # reference curve equals the main analysis
    dp = EV["dummy_positive_bins"][t]; band_hi = max(z[1] for z in dp); band_lo = vb[0]["lo"]
    ax.axvspan(band_lo, band_hi, color=BAND, lw=0, zorder=0); ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.axhline(POST["meta"]["margin"], color=INK2, lw=0.6, ls=":", zorder=1)
    for key, lab, col, mk, dx in SER:
        xs = np.array([math.sqrt(b["lo"] * b["hi"]) for b in bins]); xd = xs * 10 ** dx
        for b, m in zip(bins, vb): assert abs(math.log10(math.sqrt(b["lo"] * b["hi"])) - m["x_log10"]) < 1e-9
        ys = np.array([b[key]["y"] for b in bins]); lo = np.array([b[key]["ci95"][0] for b in bins]); hi = np.array([b[key]["ci95"][1] for b in bins])
        ax.vlines(xd, lo, hi, color=col, lw=0.7, alpha=0.8, zorder=2)
        ax.plot(xd, ys, color=col, lw=(0.9 if key == "RiboPipe-reads" else 1.2), ls=("--" if key == "RiboPipe-reads" else "-"), marker=mk,
                ms=2.8, mfc=("white" if key == "RiboPipe-reads" else col), mec=col, mew=0.7, zorder=3)
        assert lo.min() >= YL[0] and hi.max() <= YL[1], (t, key, lo.min(), hi.max())   # nothing clipped
        used["a"][f"{t}|{key}"] = {"x_geometric_centre": xs.tolist(), "y": ys.tolist(), "ci95_lo": lo.tolist(), "ci95_hi": hi.tolist(),
                                    "N_scored_role": [b["N_scored_role"] for b in bins]}
    used["a"][f"{t}|noise_band"] = [band_lo, band_hi]
    ax.set_xscale("log"); ax.set_xlim(5e-5, 2); ax.set_ylim(*YL); ax.set_yticks([-0.1, 0, 0.1])
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(f"a-{c+1}  {DN[t]}\n{CL[t]}", loc="left", fontsize=FS + 0.5)
    ax.set_xlabel("depth (reads/codon)", fontsize=FS - 0.5)
    if c == 0: ax.set_ylabel("Δ median r with replicate")
    else: ax.set_yticklabels([])
# (b) controlled thinning, top-5% precision (as make_fig_part2_application.py)
EST = [("Raw", "sparse reads (thinned input replicate)", INK2, "o"), ("RiboPipe", "RiboPipe prediction", RPC, "s"),
       ("Post_RiboPipe", "reads + RiboPipe posterior", C_PR, "D")]
for k, t in enumerate(DS):
    ax = fig.add_subplot(gs[1, k]); a = A1[t]; ds = a["depths"]
    for key, lab, col, mk in EST:
        m = [np.mean([a["cells"][f"main_Bge1|role{r}|d{d}"]["est"][key]["prec"]["mean"] for r in (0, 1)]) for d in ds]
        lo = [min(a["cells"][f"main_Bge1|role{r}|d{d}"]["est"][key]["prec"]["lo"] for r in (0, 1)) for d in ds]
        hi = [max(a["cells"][f"main_Bge1|role{r}|d{d}"]["est"][key]["prec"]["hi"] for r in (0, 1)) for d in ds]
        ax.fill_between(ds, lo, hi, color=col, alpha=0.15, lw=0); ax.plot(ds, m, color=col, marker=mk, ms=2.8, lw=1.2)
        used["b"][f"{t}|{key}"] = {"depths": ds, "mean_of_roles": [float(x) for x in m], "band_lo": lo, "band_hi": hi}
    rnd = float(np.mean([a["cells"][f"main_Bge1|role{r}|d{d}"]["rand_prec"] for r in (0, 1) for d in ds]))
    ax.axhline(rnd, color=INK2, lw=0.8, ls=":"); used["b"][f"{t}|random"] = rnd
    if k == 0: ax.text(ds[0], rnd + 0.015, "random", fontsize=FS - 1.5, color=INK2)
    ax.set_xscale("log"); ax.set_xticks([0.02, 0.1, 0.5]); ax.set_xticklabels(["0.02", "0.1", "0.5"]); ax.set_ylim(0, 0.72)
    ax.set_title(f"b-{k+1}  {DN[t]}\n{CL[t]}", fontsize=FS + 0.5, loc="left"); ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_xlabel("input depth (reads/codon)", fontsize=FS - 0.5)
    if k == 0: ax.set_ylabel("precision for reference\ntop-5% codons")
    else: ax.set_yticklabels([])
# (c) E-P Pro-Pro thirds in naturally low-coverage transcripts: RiboPipe + comparators
lab = ["low", "middle", "high"]
CMPM = [("RiboPipe", "RiboPipe", RPC, "o", -0.13), ("RiboMIMO-adapt (MFE)", "adapted RiboMIMO + MFE", "#009e73", "s", 0.0),
        ("Ridge", "ridge", "#0072b2", "^", 0.13)]   # three representative models; iXnos and others are in the supplementary table
YLC = (0.1, 12)
for k, t in enumerate(DS):
    ax = fig.add_subplot(gs[2, k]); r = TC[t]; dm = CMP["datasets"][t]
    assert dm["n_occ"] == r["n_occ"] and dm["n_tx"] == r["n_tx"]
    for g in range(3):   # RiboPipe in the comparison JSON equals n2_tercile_ci.json exactly
        for kk in ("est", "lo", "hi", "n_occ", "n_tx"): assert dm["models"]["RiboPipe"][f"t{g}"][kk] == r[f"t{g}"][kk], (t, g, kk)
    used["c"][t] = {"n_occ": r["n_occ"], "n_tx": r["n_tx"]}
    for key, lb, col, mk, dx in CMPM:
        v = dm["models"][key]; xs = np.arange(3) + dx
        est = [v[f"t{g}"]["est"] for g in range(3)]; lo = [v[f"t{g}"]["lo"] for g in range(3)]; hi = [v[f"t{g}"]["hi"] for g in range(3)]
        assert min(lo) > YLC[0] and max(hi) < YLC[1], (t, key, min(lo), max(hi))   # nothing clipped
        ax.vlines(xs, lo, hi, color=col, lw=1.0)
        ax.plot(xs, est, color=col, lw=0.7, alpha=0.6, marker=mk, ms=3.2, mfc=col, mec=col)
        used["c"][t][key] = {f"t{g}": {"est": est[g], "lo": lo[g], "hi": hi[g]} for g in range(3)}
    ax.axhline(1, color=INK2, lw=0.7, ls=":")
    ax.axhline(dm["pooled_all_sites"], color=INK2, lw=0.9, ls="--"); used["c"][t]["pooled_all_sites"] = dm["pooled_all_sites"]
    ax.set_yscale("log"); ax.set_ylim(*YLC); ax.set_yticks([0.1, 0.3, 1, 3, 10])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10"] if k == 0 else [])
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=FS - 0.5); ax.set_xlim(-0.45, 2.45); ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title(f"c-{k+1}  {DN[t]}\n{CL[t]}", fontsize=FS + 0.5, loc="left"); ax.set_xlabel("predicted group (thirds)", fontsize=FS - 0.5)
    ax.text(0.03, 0.97, f"{r['n_occ']:,} sites\n{r['n_tx']:,} transcripts", transform=ax.transAxes, fontsize=FS - 1.5, va="top", color=INK2)
    if k == 0: ax.set_ylabel("observed / expected\n(pooled over sites)")
# row headers and legends
def head(row, letter, text, handles, ncol):
    bb = fig.add_subplot(gs[row, 0]).get_position(); fig.axes[-1].remove()
    nrow = -(-len(handles) // ncol) if handles else 0          # legend rows
    y = bb.y1 + 0.055 + 0.026 * nrow
    fig.text(0.008 * XS, y, letter, fontsize=FS + 3, fontweight="bold", va="bottom")
    fig.text(0.035 * XS, y + 0.002, text, fontsize=FS + 1.5, fontweight="bold", color="#333333", va="bottom")
    fig.legend(handles=handles, loc="upper left", ncol=ncol, fontsize=FS - 0.5, frameon=False, bbox_to_anchor=(0.03 * XS, y - 0.004), handlelength=2.0,
               columnspacing=1.2)
head(0, "a", "Posterior versus reads and prediction on low-coverage transcripts",
     [Line2D([], [], color=C_PR, marker="D", ms=3, lw=1.2, label="posterior − reads"),
      Line2D([], [], color=C_PP, marker="^", ms=3, lw=1.2, label="posterior − RiboPipe"),
      Line2D([], [], color=RPC, marker="o", mfc="white", ms=3, lw=0.9, ls="--", label="RiboPipe − reads (Fig. 2a)"),
      Patch(color=BAND, label="noise predictor > 0"), Line2D([], [], color=INK2, lw=0.6, ls=":", label="−0.01")], 3)
head(1, "b", "Controlled thinning of held-out high-coverage transcripts",
     [Line2D([], [], color=c, marker=m, lw=1.2, ms=3, label=l) for _, l, c, m in EST], 3)
head(2, "c", "E–P Pro–Pro sites in low-coverage transcripts, grouped by each model's prediction",
     [Line2D([], [], color=c, marker=m, ms=3, lw=0.9, label=l) for _, l, c, m, _ in CMPM]
     + [Line2D([], [], color=INK2, lw=0.9, ls="--", label="all E–P Pro–Pro sites"), Line2D([], [], color=INK2, lw=0.7, ls=":", label="transcript mean")], 3)
pdf, png = os.path.join(OUT, STEM + ".pdf"), os.path.join(OUT, STEM + ".png")
fig.savefig(pdf); fig.savefig(png, dpi=300)
p = C.write_numbers(OUT, STEM, used, IN, extra={"published_generator": "make_fig_part2_use_1115_v4.py", "published_pdf_md5": "3f4be56493a185049b813fc55f41112f",
                                                 "colours": {"RiboPipe-reads": RPC, "posterior-reads": C_PR, "posterior-RiboPipe": C_PP, "panel_c": {k: c for k, _, c, _, _ in CMPM}}})
print("wrote", pdf, p)
