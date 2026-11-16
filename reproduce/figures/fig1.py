#!/usr/bin/env python3
"""Figure 1 (fig:benchmark): Part I benchmark under three evaluation settings.

Ported from the published generator make_fig_partI_main_1115_v5.py (md5 9ce1e6da027ebe973d80f6b4185666c2); the plotting
code is unchanged. Reads only paper_results/:
  partI/fig1_derived.json        per-fold medians (a-1), parameter counts (legend), TX9_WT pooled-site summary + hexbin cells (a-2)
                                 (made by paper_results/derive_fig1.py from the internal prediction runs)
  partI/partI_eval_elig.json     Part I scores; the mean of the five fold medians must equal S (asserted)
  partI/T2_matched_eval_elig.json  Setting 2, RiboPipe minus each method retrained to the common target (paired difference, 95% CI)
  partI/e1_results.json          Setting 3, ablations of RiboPipe (E1)
Panels:
  a-1  per-transcript Pearson r (interior [20, L-20)), median per fold, bar = mean of 5 folds, error bar = SD across folds; 8 datasets, 10 methods
  a-2  TX9_WT, RiboPipe out-of-fold prediction vs observed density, pooled codons (hexbin; log1p axes for display only)
  b    Setting 2: RiboPipe minus RiboMIMO / iXnos / RiboGL retrained to the common target (95% bootstrap CI)
  c    Setting 3: ablations of RiboPipe (parameter-matched variants; -MFE without matching), * = contributes
Seed 123; iXnos = iXnos (MFE), lr 0.005, all eligible genes.
Usage: python3 reproduce/figures/fig1.py OUTDIR [--results paper_results]
"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

A = C.parse_args(__doc__); IN = C.Inputs(A.results)
STEM = "fig1"
DER = IN.json("partI/fig1_derived.json")
HUMAN = ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]
NONHUMAN = ["Subtelny14_ribomimo", "Mohammad16", "Mohammad19-1", "Mohammad19-2"]
# panel a keeps all ten methods
METH = [("Paper_CodonMean", "Codon-mean", "#d9d9d9", None), ("Paper_TriCodon", "Tri-codon", "#bdbdbd", "////"),
        ("Paper_Ridge", "Ridge", "#8c8c8c", "...."), ("Paper_RUST", "RUST", "#636363", "xxxx"),
        ("Paper_iXnos_MFE_lr005_elig", "iXnos (MFE)", "#56b4e9", None), ("Paper_RiboMIMO", "RiboMIMO", "#009e73", None),
        ("Paper_RiboGL_heldout", "RiboGL", "#e69f00", None), ("Paper_RiboExp", "RiboExp", "#cc79a7", "\\\\\\\\"),   # hatch as second code: deutan dE 7.6 vs RiboMIMO
        ("Paper_BiLSTM97", "BiLSTM-97", "#0072b2", None), ("Paper_RiboPipe_MFE_H128", "RiboPipe", "#d55e00", None)]
DS = HUMAN + NONHUMAN
OURS = "Paper_RiboPipe_MFE_H128"
# legend: trainable parameter count of the neural methods (from each method's run.json; iXnos = input_dim*200+200+200+1; RiboGL = predictive path)
NEURAL = ("Paper_iXnos_MFE_lr005_elig", "Paper_RiboMIMO", "Paper_RiboGL_heldout", "Paper_RiboExp", "Paper_BiLSTM97", OURS)
NPAR = {c: DER["params_legend"][c] for c in NEURAL}
assert NPAR[OURS] == 228321
_fmt = lambda n: f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.0f}K"
LEG = {c: (f"{nm} ({_fmt(NPAR[c])})" if c in NPAR else nm) for c, nm, *_ in METH}
res = {"datasets": {}, "params_legend": NPAR}
for d in DS:
    R = {}
    for c, *_ in METH:
        out = {}
        for metric in ("pearson", "recall5"):
            fm = DER["panel_a1"][d][c][metric]["folds"]
            out[metric] = {"mean": float(np.mean(fm)), "sd": float(np.std(fm, ddof=1)), "folds": fm, "n": DER["panel_a1"][d][c][metric]["n"]}
        R[c] = out
    res["datasets"][d] = R
# consistency: the 5-fold mean of the fold medians must equal S in partI_eval_elig
EV = IN.json("partI/partI_eval_elig.json")
for d in DS:
    for c, *_ in METH:
        S = EV["datasets"][d]["absolute"][c]["pearson_interior"]["S"]
        assert abs(S - res["datasets"][d][c]["pearson"]["mean"]) < 1e-9, (d, c, S, res["datasets"][d][c]["pearson"]["mean"])
        R5 = EV["datasets"][d]["absolute"][c]["recall5_interior"]["S"]
        assert abs(R5 - res["datasets"][d][c]["recall5"]["mean"]) < 1e-9, (d, c, "recall5")
# (a-2) TX9_WT pooled sites: summary values and the non-empty hexbin cells (centre, number of codons)
d = "DMSO_TX9P1"; P2 = DER["panel_a2"]
pooled, med, ntx, ncod = P2["pooled_r"], P2["median_tx_r"], P2["n_tx"], P2["n_codons"]
HXC = np.asarray(P2["hexbin"]["centres"], float); HXN = np.asarray(P2["hexbin"]["counts"], float)
assert P2["hexbin"]["gridsize"] == 32 and HXN.sum() <= ncod
res["panel_b"] = {"pooled_r": pooled, "median_tx_r": med, "n_tx": ntx, "n_codons": ncod, "partI_S": EV["datasets"][d]["absolute"][OURS]["pearson_interior"]["S"]}
T2 = IN.json("partI/T2_matched_eval_elig.json")
L2 = [("Paper_RiboMIMO_L2", "RiboMIMO", "#009e73", "o"), ("Paper_iXnos_nocut_lr005_elig", "iXnos", "#56b4e9", "s"), ("Paper_RiboGL_L2_heldout", "RiboGL", "#e69f00", "^")]
res["panel_d"] = {d: {c: {k: T2["datasets"][d]["ours_vs_matched"][c].get(k) for k in ("delta_hat", "ci95", "class")} for c, *_ in L2} for d in HUMAN + NONHUMAN}
# ---- Setting 3: ablations of RiboPipe (E1; parameter-matched variants _pm; A1 = -MFE without matching, 1.0% fewer parameters)
E1 = IN.json("partI/e1_results.json"); assert E1["ref"]["config"] == OURS
ALL8 = HUMAN + NONHUMAN
ABL = [("CNN-Motif\nFront End", [("A2_pm", "−nt\nbranch"), ("A3_pm", "−aa\nbranch"),
                                                      ("A4_pm", "codon\nonly"), ("A6_pm", "no\nconv.")]),
       ("MFE", [("A1", "−MFE")]),
       ("BiGRU", [("A5_pm", "per-\nposition")])]
res["panel_e"] = {}
for g, rows in ABL:
    for k, nm in rows:
        res["panel_e"][k] = {"label": nm, "config": E1["groups"][k]["config"], "params": E1["groups"][k]["params"], "datasets": {}}
        for d in ALL8:
            p = E1["datasets"][d]["pairs"][k]["pearson"]; assert p["status"] == "OK", (k, d)
            # final_class "有贡献" (= "contributes") is the class label recorded in e1_results.json; contributing cells are bold with *
            res["panel_e"][k]["datasets"][d] = {"delta_hat": p["delta_hat"], "ci95": p["ci95"], "final_class": p["final_class"], "contributes": p["final_class"] == "有贡献"}
# ---- drawing: designed at print size (width 7.2 in), smallest font 6 pt. Rows: a = Setting 1 (a-1, a-2), b = Setting 2, c = Setting 3
FS = 7
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "axes.titlesize": FS + 0.5, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "hatch.linewidth": 0.6})
H5 = 8.3 - 0.1 * 0.78 / 2.36 * (0.865 - 0.06) * 8.3   # row a (outer[0]) 10% shorter; other rows and margins keep their absolute size
fig = plt.figure(figsize=(7.2, H5))
outer = fig.add_gridspec(3, 1, height_ratios=[0.78 * 0.9, 0.58, 1.0], hspace=0.0, left=0.075, right=0.925, top=1 - (1 - 0.865) * 8.3 / H5, bottom=0.06 * 8.3 / H5)
rowA = outer[0].subgridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.32)
row2 = outer[2].subgridspec(1, 2, width_ratios=[0.95, 1.35], wspace=0.42)
labe = {"DMSO_TX9P1": "TX9_WT\nHEK293T", "GSE233886_WT": "GSE233886_WT\nHEK293 Flp-In", "GSE133393_WT": "GSE133393_WT\nHEK293", "PRJNA_iPS": "PRJNA_iPS\niPSC 409B2",
        "Subtelny14_ribomimo": "Subtelny14\nyeast", "Mohammad16": "Mohammad16\nE. coli", "Mohammad19-1": "Mohammad19-1\nE. coli", "Mohammad19-2": "Mohammad19-2\nE. coli"}
labx = {"DMSO_TX9P1": "TX9_WT", "GSE233886_WT": "GSE233886\n_WT", "GSE133393_WT": "GSE133393\n_WT", "PRJNA_iPS": "PRJNA_iPS"}
def bars(ax, metric, ylab, title):
    w = 0.84 / len(METH); xs = np.arange(len(DS))
    for i, (c, nm, col, hat) in enumerate(METH):
        m = [res["datasets"][d][c][metric]["mean"] for d in DS]; s_ = [res["datasets"][d][c][metric]["sd"] for d in DS]
        ax.bar(xs - 0.42 + w * (i + 0.5), m, w * 0.92, yerr=s_, color=col, hatch=hat, edgecolor="#333333", linewidth=0.25,
               error_kw=dict(lw=0.5, capsize=0, ecolor="#333333"), label=LEG[c])
    ax.set_xticks(xs); ax.set_xticklabels([labe[d] for d in DS], fontsize=FS - 1, linespacing=0.95, rotation=30, ha="right", rotation_mode="anchor", multialignment="right")   # dataset + cell line / species, as in b and c
    ax.axvline(len(HUMAN) - 0.5, color="#999999", lw=0.6, ls="--")
    ax.text((len(HUMAN) - 1) / 2, 0.985, "Human", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS - 0.5, color="#555555")
    ax.text(len(HUMAN) + (len(DS) - len(HUMAN) - 1) / 2, 0.985, "Yeast and E. coli", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS - 0.5, color="#555555")
    ax.set_ylabel(ylab); ax.set_title(title, loc="left", fontsize=FS + 0.5); ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
axa3 = None
axa1 = fig.add_subplot(rowA[0, 0]); bars(axa1, "pearson", "Per-transcript Pearson r", "a-1  Per-transcript accuracy"); axa1.set_ylim(0.0, 0.86)
axa2 = fig.add_subplot(rowA[0, 1])
# same hexbin call as the published generator, fed with one point per non-empty cell at its centre weighted by its count
hb = axa2.hexbin(HXC[:, 0], HXC[:, 1], C=HXN, reduce_C_function=np.sum, gridsize=32, bins="log", cmap="magma", mincnt=1,
                 extent=(0, np.log1p(60), 0, np.log1p(60)), linewidths=0)
assert np.array_equal(np.asarray(hb.get_array(), float), HXN) and np.array_equal(np.asarray(hb.get_offsets(), float), HXC)   # cells redrawn exactly
axa2.plot([0, np.log1p(60)], [0, np.log1p(60)], ls="--", lw=0.6, color="#888888")
tk = [0, 1, 3, 10, 30]; axa2.set_xticks(np.log1p(tk)); axa2.set_xticklabels(tk); axa2.set_yticks(np.log1p(tk)); axa2.set_yticklabels(tk)
axa2.set_xlabel("Observed density (codon)"); axa2.set_ylabel("Predicted density (codon)")
axa2.set_title(f"a-2  Pooled sites, TX9_WT\nRiboPipe, site-level r = {pooled:.3f}", loc="left", fontsize=FS + 0.5)
cb = fig.colorbar(hb, ax=axa2, fraction=0.05, pad=0.03); cb.set_label("Codons", fontsize=FS - 0.5, labelpad=1); cb.ax.tick_params(labelsize=FS - 1, width=0.5, length=2)
# (b) Setting 2: 8 datasets
NDS = len(ALL8); YPOS = {d: NDS - 1 - k for k, d in enumerate(ALL8)}
axb = fig.add_subplot(row2[0, 0])
axb.axvspan(-0.01, 0.01, color="#eeeeee", zorder=0); axb.axvline(0, color="#555", lw=0.6)
for d in ALL8:
    if YPOS[d] % 2 == 1: axb.axhspan(YPOS[d] - 0.5, YPOS[d] + 0.5, color="#f3f3f3", zorder=-1, lw=0)
    for i, (c, nm, col, mk) in enumerate(L2):
        p = res["panel_d"][d][c]; yy = YPOS[d] + (1 - i) * 0.26
        axb.plot(p["ci95"], [yy, yy], color=col, lw=1.1); axb.plot(p["delta_hat"], yy, mk, color=col, ms=4, mew=0, label=nm if d == ALL8[0] else None)
axb.set_yticks([YPOS[d] for d in ALL8]); axb.set_yticklabels([]); axb.tick_params(axis="y", length=0); axb.spines["left"].set_visible(False)
axb.set_ylim(-0.5, NDS - 0.5); axb.axhline(NDS - len(HUMAN) - 0.5, color="#999999", lw=0.6, ls="--")
axb.set_xlabel("Δ Pearson r (RiboPipe − method)")
axb.legend(fontsize=FS - 0.5, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#dddddd", handlelength=1.0, borderpad=0.4)
axb.grid(axis="x", color="#e6e6e6", lw=0.5); axb.set_axisbelow(True); axb.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
# (c) Setting 3: heat map, rows = datasets, shares the y axis with b
axc = fig.add_subplot(row2[0, 1], sharey=axb)
cols = [(g, k, nm) for g, rr in ABL for k, nm in rr]
M = np.array([[res["panel_e"][k]["datasets"][d]["delta_hat"] for _, k, _n in cols] for d in ALL8])
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import blended_transform_factory
VMIN = -0.11; assert M.min() >= VMIN and M.max() <= 0.03
im = axc.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=0.03), aspect="auto",
                extent=(-0.5, len(cols) - 0.5, -0.5, NDS - 0.5), origin="upper", interpolation="nearest")
for i, d in enumerate(ALL8):
    for j, (_, k, _n) in enumerate(cols):
        v = res["panel_e"][k]["datasets"][d]
        t = ("0.000" if abs(v['delta_hat']) < 0.0005 else f"{v['delta_hat']:+.3f}".replace("-", "−")) + ("*" if v["contributes"] else "")   # a signed zero is shown as 0.000
        axc.text(j, YPOS[d], t, ha="center", va="center", fontsize=FS - 0.5, color="white" if v["delta_hat"] < -0.06 else "#222222",
                 fontweight="bold" if v["contributes"] else "normal")
axc.set_xticks(range(len(cols))); axc.set_xticklabels([n for _, _, n in cols], fontsize=FS - 0.5, linespacing=0.95); axc.xaxis.tick_top()
plt.setp(axc.get_yticklabels(), visible=False)
axc.axhline(NDS - len(HUMAN) - 0.5, color="white", lw=2)
b0 = 0
for g, rr in ABL:
    if b0: axc.axvline(b0 - 0.5, color="white", lw=2)
    axc.annotate("", xy=(b0 - 0.42, NDS - 0.5 + 1.0), xytext=(b0 + len(rr) - 0.58, NDS - 0.5 + 1.0), xycoords="data",
                 arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8), annotation_clip=False)
    axc.text((2 * b0 + len(rr) - 1) / 2, NDS - 0.5 + 1.08, g.replace("\n", " "), ha="center", va="bottom", fontsize=FS, fontweight="bold", color="#333333", clip_on=False)
    b0 += len(rr)
for sp in axc.spines.values(): sp.set_visible(False)
axc.tick_params(length=0)
cb2 = fig.colorbar(im, ax=axc, fraction=0.045, pad=0.02); cb2.set_label("Δ Pearson r (variant − RiboPipe)", fontsize=FS - 0.5, labelpad=2)
cb2.ax.tick_params(labelsize=FS - 1, width=0.5, length=2); cb2.set_ticks([-0.10, -0.05, 0.0, 0.02])
h, l = axa1.get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, fontsize=FS - 0.5, frameon=False, bbox_to_anchor=(0.5, 0.997),
                                                     handlelength=1.3, columnspacing=0.9, handletextpad=0.4)
# dataset labels shared by b and c
fig.canvas.draw(); rr_ = fig.canvas.get_renderer()
_xm = (axb.get_position().x1 + axc.get_position().x0) / 2
_trm = blended_transform_factory(fig.transFigure, axb.transData)
for d in ALL8:
    fig.text(_xm, YPOS[d], labe[d], transform=_trm, ha="center", va="center", fontsize=FS - 0.5, color="#222222", linespacing=0.95)
XL = min(a_.get_tightbbox(rr_).transformed(fig.transFigure.inverted()).x0 for a_ in (axa1, axb)) - 0.004
def head(x0, y, letter, text):
    fig.text(x0, y, letter, fontsize=FS + 3, fontweight="bold", ha="left", va="bottom")
    fig.text(x0 + 0.026, y + 0.001, text, fontsize=FS + 1.5, fontweight="bold", ha="left", va="bottom", color="#333333")
ya = max(a_.get_tightbbox(rr_).transformed(fig.transFigure.inverted()).y1 for a_ in (axa1, axa2)) + 0.008
head(XL, ya, "a", "Evaluation Setting 1: Original method-centric configurations*")   # asterisk = necessary changes, see the caption
yb = axc.get_tightbbox(rr_).transformed(fig.transFigure.inverted()).y1 + 0.008
head(XL, yb, "b", "Evaluation Setting 2: Common Training Target")
head(axc.get_tightbbox(rr_).transformed(fig.transFigure.inverted()).x0 - 0.01, yb, "c", "Evaluation Setting 3: Ablation of RiboPipe")
OUT = os.path.join(A.outdir, STEM)
fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=300)
# numbers_drawn has the same keys as the published provenance (datasets, params_legend, panel_b, panel_d, panel_e)
p = C.write_numbers(A.outdir, STEM, res, IN, extra={"published_generator": "make_fig_partI_main_1115_v5.py", "published_pdf_md5": "0464e228de2aa385af8e5fca1d5dba59"})
print("wrote", OUT + ".pdf", p)
