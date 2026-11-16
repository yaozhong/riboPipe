#!/usr/bin/env python3
"""Figure 4 (Part III main figure): what the model relies on, and where.

Ported from the published generator make_fig_part3_merge_1115_v5.py (md5 9dec4b57d4ed3ad7ab37965daeae799b); the layout and
plotting code are unchanged, and draw_logo is copied from make_fig_part3_motif.py. Reads only paper_results/:
  partIII/fig4_logos.json            the three activation logos of panel a (from the E12 logo archive; paper_results/derive_fig4.py)
  partIII/xr_e12_fwd_logorep.json    cross-seed reproducibility of the top filters (selects the logos, panel a titles)
  partIII/results_X1ext.json         single-branch models vs the full model (panel b)
  partIII/figdata_partIII_levels.json  seed floor (panel b threshold)
  partIII/results_X2.json            branch reliance (c) and routing shares (e), seed 123
  partIII/results_X7.json            seeds 124/125 for c, e, f
  partIII/partIII_addendum_v2.json   mutagenesis profiles (d)
  partIII/results_X3.json            transplant summaries (consistency checks for f, g)
  partIII/results_X5.json            shuffled-window control (f) and sub-window transplants (g)
Panels: a filter logos (TX9_WT); b single-branch agreement; c branch reliance; d edit profiles; e routing shares;
f whole-window transplant (ordered vs shuffled); g sub-window transplants, RiboPipe vs adapted RiboMIMO.
Usage: python3 reproduce/figures/fig4.py OUTDIR [--results paper_results]
"""
import os, sys, json, itertools
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import matplotlib.transforms as mtr

A = C.parse_args(__doc__); IN = C.Inputs(A.results); OUTDIR = A.outdir; STEM = "fig4"
md5 = C.md5
J = lambda name: IN.json(f"partIII/{name}")
X1E = J("results_X1ext.json"); X2 = J("results_X2.json"); X3 = J("results_X3.json")
X5 = J("results_X5.json"); X7 = J("results_X7.json"); FD = J("figdata_partIII_levels.json")
AD = J("partIII_addendum_v2.json")
LR = J("xr_e12_fwd_logorep.json")
_LG = J("fig4_logos.json")["logos"]
Z = {k: np.array(v["values"], dtype=np.float32).reshape(v["shape"]) for k, v in _LG.items()}   # exact float32 values of the E12 logo archive

# ---------------------------------------------------------------- logo letters and draw_logo (from make_fig_part3_motif.py)
INK2 = "#52514e"
CODONS = ["".join(p) for p in itertools.product("ACGT", repeat=3)]
AAL = sorted(set("ACDEFGHIKLMNPQRSTVWY*"))
LET = {"codon": CODONS, "aa": AAL, "nt": list("ACGT")}
POS = {"codon": ["E", "P", "A"], "aa": [str(x) for x in range(-5, 0)], "nt": [str(x) for x in range(-15, 15)]}

STOP = {"*", "TAA", "TAG", "TGA"}; STOPLOG = []   # stop letters only occur in windows reaching the CDS end; not drawn in the logo


def draw_logo(ax, M, letters, pos_labels, k_top=None, thr=0.0):
    """Enrichment logo: only positive log2 enrichments (letters over-represented in the top-1% windows relative to background) are
    drawn, stacked upwards; depleted letters (negative values) are not drawn (stated in the caption)."""
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    pal = {"A": "#1baf7a", "C": "#2a78d6", "G": "#e0a020", "T": "#d6452a"}
    npos = M.shape[0]; ymax = ymin = 0
    for q in range(npos):
        v = M[q]; idx = np.argsort(-np.abs(v)); idx = [i for i in idx if v[i] > thr][: (k_top or len(idx))]
        up = sorted([i for i in idx if v[i] > 0 and letters[i] not in STOP], key=lambda i: v[i]); dn = []
        STOPLOG.append(max([v[i] for i in range(len(v)) if letters[i] in STOP] + [-np.inf]))
        for group, sign in ((up, 1), (dn, -1)):
            base = 0.0
            for i in group:
                h = abs(v[i]); lab = letters[i]; tp = TextPath((0, 0), lab, size=1, prop=fp); bb = tp.get_extents()
                sx = 0.9 / max(bb.width, 1e-6); sy = h / max(bb.height, 1e-6)
                y0 = base if sign > 0 else -base - h
                tr = mtr.Affine2D().translate(-bb.x0, -bb.y0).scale(sx, sy).translate(q - 0.45, y0) + ax.transData
                col = pal.get(lab, "#2a78d6" if len(lab) == 1 else INK2)
                if len(lab) == 3: col = {"A": "#1baf7a", "C": "#2a78d6", "G": "#e0a020", "T": "#d6452a"}[lab[0]]
                ax.add_patch(PathPatch(tp, transform=tr, color=col if len(lab) != 1 or lab in pal else ("#c0392b" if lab == "P" else ("#7a7a7a" if lab == "*" else "#2a78d6")), lw=0))
                base += h
            if sign > 0: ymax = max(ymax, base)
            else: ymin = min(ymin, -base)
    ax.set_xlim(-0.6, npos - 0.4); ax.set_ylim(0, max(ymax * 1.08, 0.3)); ax.axhline(0, color=INK2, lw=0.5)
    step = 1 if npos <= 6 else 3
    ax.set_xticks(range(0, npos, step)); ax.set_xticklabels(pos_labels[::step], fontsize=5.8)



# ---------------------------------------------------------------- style
FS = 8; FT = FS + 0.5; FH = FS + 1.5; FL = FS + 3
WIDTH_MM = 185.0; W = WIDTH_MM / 25.4          # full journal text width, inserted at scale 1.0
JOURNAL_MM = 185.0
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e6e6", "#e9e8e4"
RPC = "#d55e00"; C_FIX = "#0072b2"; C_GREY = "#7a7a7a"; C_RM = "#6a3d9a"
DSC = {"DMSO_TX9P1": "#332288", "GSE233886_WT": "#cc6677", "GSE133393_WT": "#117733", "PRJNA_iPS": "#ddaa33"}
DSM = {"DMSO_TX9P1": "o", "GSE233886_WT": "s", "GSE133393_WT": "^", "PRJNA_iPS": "D"}
BRC = {"codon": "#56b4e9", "nt": "#0072b2", "aa": "#009e73", "mfe": "#d55e00"}
BRL = {"codon": "codon", "nt": "nt", "aa": "aa", "mfe": "MFE"}
HATCH_AA = "////"; HATCH_INK = "#333333"; HATCH_LW = 0.5
BRH = {"codon": None, "nt": None, "aa": HATCH_AA, "mfe": None}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "axes.titlesize": FT, "axes.labelsize": FS,
                     "pdf.fonttype": 42, "hatch.linewidth": HATCH_LW, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
TAGS = ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]; assert TAGS == ["DMSO_TX9P1", "GSE233886_WT", "GSE133393_WT", "PRJNA_iPS"]
DN = {"DMSO_TX9P1": "TX9_WT", "GSE233886_WT": "GSE233886_WT", "GSE133393_WT": "GSE133393_WT", "PRJNA_iPS": "PRJNA_iPS"}
USED = {"L1": {}, "L2": {}, "L3": {}}       # same key structure as the published provenance
NEW = {}                                    # numbers_drawn_new of the published provenance (shuffled window, MFE fixed)
OUTS = {}; FONTCHECK = {}; LAYOUT = {}; OVERLAP = {}


def newfig(H):
    fig = plt.figure(figsize=(W, H))
    def box(x0, ytop, w, h):                # x from the left, y of the TOP edge measured from the top of the figure (inches)
        return fig.add_axes([x0 / W, (H - ytop - h) / H, w / W, h / H])
    return fig, box


def plabel(fig, H, x, ytop, letter, title):
    fig.text(x / W, (H - ytop) / H, letter, fontsize=FL, fontweight="bold", va="baseline")
    fig.text((x + 0.2) / W, (H - ytop) / H, title, fontsize=FT, va="baseline")


def save(fig, stem):
    w_in, h_in = fig.get_size_inches()
    assert abs(w_in - W) < 1e-9, (w_in, W)
    fig.canvas.draw(); sizes = []
    # tick-label Text objects of ticks outside the view limits exist but are not drawn; exclude them from all text checks
    drawn, alltl = set(), set()
    for ax_ in fig.axes:
        for axis in (ax_.xaxis, ax_.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks(): alltl.update((id(tk.label1), id(tk.label2)))
            for tk in axis._update_ticks(): drawn.update((id(tk.label1), id(tk.label2)))
    shown = lambda t: t.get_visible() and t.get_text().strip() and (id(t) not in alltl or id(t) in drawn)
    for t in fig.findobj(Text):
        if shown(t): sizes.append(t.get_fontsize())
    mn = min(sizes); assert mn >= FS, (stem, mn)
    # every visible text inside the canvas
    r = fig.canvas.get_renderer(); bb_fig = fig.bbox; out = []
    for t in fig.findobj(Text):
        if shown(t):
            e = t.get_window_extent(r)
            if e.x0 < bb_fig.x0 - 0.5 or e.x1 > bb_fig.x1 + 0.5 or e.y0 < bb_fig.y0 - 0.5 or e.y1 > bb_fig.y1 + 0.5: out.append(t.get_text())
    assert not out, (stem, out)
    # pairwise text-overlap self-check (window extents in display pixels). Rotated labels have axis-aligned boxes larger than the glyphs,
    # so pairs involving a rotated text are reported separately and checked visually.
    tx = [(t, t.get_window_extent(r)) for t in fig.findobj(Text) if shown(t)]
    hard, rot = [], []
    for i in range(len(tx)):
        for j in range(i + 1, len(tx)):
            (ta, ea), (tb, eb) = tx[i], tx[j]
            w_ = min(ea.x1, eb.x1) - max(ea.x0, eb.x0); h_ = min(ea.y1, eb.y1) - max(ea.y0, eb.y0)
            if w_ > 0.5 and h_ > 0.5:
                rec = {"a": ta.get_text(), "b": tb.get_text(), "overlap_px": [round(w_, 2), round(h_, 2)]}
                (rot if (ta.get_rotation() % 360) not in (0, 90) or (tb.get_rotation() % 360) not in (0, 90) else hard).append(rec)
    # every figure-level legend box (handles + labels) against every text that is not part of that legend
    leg = []
    for lg in fig.legends:
        el = lg.get_window_extent(r); own = {id(t) for t in lg.get_texts()}
        for t, e in tx:
            if id(t) in own: continue
            w_ = min(el.x1, e.x1) - max(el.x0, e.x0); h_ = min(el.y1, e.y1) - max(el.y0, e.y0)
            if w_ > 0.5 and h_ > 0.5: leg.append({"legend": [x.get_text() for x in lg.get_texts()], "text": t.get_text(), "overlap_px": [round(w_, 2), round(h_, 2)]})
    OVERLAP[stem] = {"n_text": len(tx), "text_pairs_overlapping": hard, "rotated_pairs_bbox_overlapping": rot, "legend_box_vs_text_overlapping": leg,
                     "n_legends": len(fig.legends)}
    assert not hard, (stem, hard)
    assert not leg, (stem, leg)
    FONTCHECK[stem] = {"width_in": w_in, "height_in": h_in, "width_mm": w_in * 25.4, "height_mm": h_in * 25.4, "min_font_pt_scale1": mn,
                       "effective_min_pt_journal_185mm": mn * JOURNAL_MM / WIDTH_MM, "n_text": len(sizes)}
    pdf, png = os.path.join(OUTDIR, stem + ".pdf"), os.path.join(OUTDIR, stem + ".png")
    fig.savefig(pdf); fig.savefig(png, dpi=300); plt.close(fig); OUTS[stem + ".pdf"] = md5(pdf); OUTS[stem + ".png"] = md5(png)


def ds_handles():
    return [Line2D([], [], marker=DSM[t], color=DSC[t], ls="none", ms=5, label=DN[t]) for t in TAGS]


def ci_err(m, ci): return [[m - ci[0]], [ci[1] - m]]


# ======================================================================================== MAIN FIGURE
DY = 0.2                                      # room for the b/c dataset legend
H = 8.0 + DY; fig, box = newfig(H); LAYOUT["main"] = {"height_in": H}
XR0 = 3.5                                    # left edge of the right column of layer 1

# ---------------- (a) = old L1d, activation logos (left column)
PATHS = ("codon", "aa", "nt")
TOP5 = {p: LR["logo_reproducibility"][p]["top5"] for p in PATHS}
rep = lambda r: r["best_r_seed124"] >= 0.75 and r["best_r_seed125"] >= 0.75
REC = {p: [r["filter"] for r in TOP5[p] if rep(r)] for p in PATHS}
assert len(REC["nt"]) == 3 and len(REC["aa"]) == 2 and len(REC["codon"]) == 0, REC
MAIN_LOGOS = [("nt", REC["nt"][0]), ("aa", REC["aa"][0]), ("codon", TOP5["codon"][0]["filter"])]
assert MAIN_LOGOS == [("nt", 21), ("aa", 19), ("codon", 7)], MAIN_LOGOS
plabel(fig, H, 0.02, 0.2, "a", "Filters show local preferences, TX9_WT")
u = USED["L1"].setdefault("d", {}); LOGO_AX = {}
for (p, j), (x0, yt, w), lab in zip(MAIN_LOGOS, ((0.55, 0.47, 2.8), (0.55, 1.5, 1.35), (2.65, 1.5, 0.7)), ("nt offset", "residue offset", "site")):
    ax = box(x0, yt, w, 0.44); M = Z[f"logo|CAN|0|{p}|{j}"]
    draw_logo(ax, M, LET[p], POS[p], k_top=(3 if p == "codon" else (4 if p == "aa" else None)), thr=0.05)
    ax.tick_params(axis="x", labelsize=FS); ax.tick_params(axis="y", labelsize=FS); ax.yaxis.set_major_locator(MaxNLocator(2))
    r = {x["filter"]: x for x in TOP5[p]}[j]
    u[f"{p}|{j}"] = {"best_r_seed124": r["best_r_seed124"], "best_r_seed125": r["best_r_seed125"], "recurs": rep(r)}
    ttl = {"nt": f"nt, recurring (r {r['best_r_seed124']:.2f}, {r['best_r_seed125']:.2f})", "aa": "aa, recurring", "codon": "codon,\nnot recurring"}[p]
    ax.set_title(ttl, loc="left", fontsize=FS, pad=2)
    ax.set_xlabel(lab, labelpad=1); LOGO_AX[p] = ax
fig.text(0.07 / W, (H - 1.25) / H, "log2 enrichment", rotation=90, va="center", ha="left", fontsize=FS)

# ---------------- (b) = old L1c, single-branch sufficiency (right column, top)
plabel(fig, H, XR0, 0.2, "b", "Single-branch models: agreement with the full model")
u = USED["L1"].setdefault("c", {}); floor = FD["rows"]["L1"]["lattice_sufficiency_TX9"]["seed_floor"]
assert abs(floor["E"] - 0.9714285714285714) < 1e-12
BX0, BW, BG = 4.02, 1.02, 0.06
for k, (site, key) in enumerate((("E site", "aa_E"), ("P site", "aa_P"), ("A site", "aa_A"))):
    ax = box(BX0 + k * (BW + BG), 0.34, BW, 0.82)
    thr = floor[site[0]] - 0.05; ax.axhline(thr, color=INK2, lw=0.8, ls="--"); vals = []
    for i, a in enumerate(("C", "A", "N")):
        for j, t in enumerate(TAGS):
            x = i + (j - 1.5) * 0.18
            if t == "DMSO_TX9P1":
                rd = X1E["X9"]["readings"]["recompute"]; v = rd[a][key]["spearman_vs_CAN"]
                sv = [rd[f"{a}_s{s}"][key]["spearman_vs_CAN"] for s in ("124", "125")]
                ax.plot([x, x], [min([v] + sv), max([v] + sv)], color=DSC[t], lw=1.0)
                src = f"results_X1ext.json X9.readings.recompute.{a}[_s124|_s125].{key}.spearman_vs_CAN"
            else:
                v = X1E["datasets"][t]["readings"]["recompute"][a][key]["spearman_vs_CAN"]; sv = []
                src = f"results_X1ext.json datasets.{t}.readings.recompute.{a}.{key}.spearman_vs_CAN"
            ax.plot(x, v, marker=DSM[t], color=DSC[t], ms=4, ls="none", zorder=3); vals += [v] + sv
            u[f"{site}|{t}|{a}"] = {"rho": v, "seed_values_124_125": sv, "threshold_floor_minus_0.05": thr, "src": src}
    ax.set_ylim(0.7, 1.005); assert min(vals) > 0.7, min(vals)
    ax.set_xticks(range(3)); ax.set_xticklabels(["codon", "aa", "nt"]); ax.set_xlim(-0.5, 2.5)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.text(0.04, 0.05, site, transform=ax.transAxes, fontsize=FS, color=INK, va="bottom", ha="left")
    if k == 0: ax.set_ylabel("Spearman ρ\nvs full model", labelpad=2)
    else: ax.set_yticklabels([])

# ---------------- (c) = old L1b, branch reliance (right column, bottom)
plabel(fig, H, XR0, 1.62 + DY, "c", "Reliance is largest on nt, or MFE in PRJNA_iPS")
u = USED["L1"].setdefault("b", {}); BR = ("codon", "nt", "aa", "mfe"); top = 0
ax = box(BX0, 1.73 + DY, 3 * BW + 2 * BG, 0.6)
for i, b in enumerate(BR):
    for j, t in enumerate(TAGS):
        x = i + (j - 1.5) * 0.2
        v123 = X2["L1_reliance_mean_ablation"]["datasets"][t]["test_mean_E12"]["mean_dP"][b]
        ma = X7["datasets"][t]["mean_ablation"]
        assert abs(ma["123"]["mean_over_folds"][b] - v123) < 1e-6
        ax.bar(x, v123, width=0.18, color=DSC[t], lw=0)
        vs = [ma[s]["mean_over_folds"][b] for s in ("124", "125")]
        ax.scatter([x, x], vs, s=7, color=INK, zorder=3, lw=0)
        top = max(top, v123, *vs)
        u[f"{t}|{b}"] = {"seed123": v123, "seed124": vs[0], "seed125": vs[1],
                         "src": f"results_X2.json L1_reliance_mean_ablation.datasets.{t}.test_mean_E12.mean_dP.{b}; results_X7.json datasets.{t}.mean_ablation.<seed>.mean_over_folds.{b}"}
ax.set_xticks(range(4)); ax.set_xticklabels(["codon", "nt", "aa", "MFE"]); ax.set_ylim(0, 0.35); assert top < 0.35, top
ax.set_ylabel("accuracy\nloss", labelpad=2); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.yaxis.set_major_locator(MaxNLocator(4))
ax.set_xlim(-0.5, 3.5)

# ---------------- dataset legend for b and c (one row, right column, between the b tick labels and the c title)
fig.legend(handles=[Line2D([], [], marker=DSM[t], color=DSC[t], ls="none", ms=5, label=DN[t]) for t in TAGS],
           loc="center left", ncol=4, frameon=False, bbox_to_anchor=((XR0 + 0.05) / W, (H - 1.6) / H), columnspacing=0.8, handletextpad=0.2,
           handlelength=1.0, borderaxespad=0, borderpad=0, fontsize=FS)

# ---------------- column grid for d-g (four dataset columns, same order as the colours in b and c)
X0, PITCH, PW = 1.02, 1.58, 1.4
COLX = [X0 + k * PITCH for k in range(4)]
fig.add_artist(Line2D([0.02 / W, (W - 0.05) / W], [(H - 2.66 - DY) / H] * 2, color=GRID, lw=0.6, transform=fig.transFigure))

# ---------------- (d) = old L2a, mutagenesis profiles
YD = 3.17 + DY                                     # d axes top = YD + 0.12 (unchanged)
YDT = 2.9 + DY                                     # d title + legend line (moved up into the old dataset-row slot)
plabel(fig, H, 0.02, YDT, "d", "Edits near the peak change it most")
# dataset column titles directly above the d panels: the only dataset legend of the figure (marker + colour for b, c) and the column
# header for d-g (e, f, g columns share the same column left edges COLX, d panels sit at COLX + 0.33 because of their own y ticks)
YHEAD = 3.15 + DY; _r = fig.canvas.get_renderer(); HEAD = {}
for k, t in enumerate(TAGS):
    cx = COLX[k] + 0.33 + (PW - 0.33) / 2 + 0.07
    tx = fig.text(cx / W, (H - YHEAD) / H, DN[t], fontsize=FS, va="baseline", ha="center", color=INK)
    x0_in = tx.get_window_extent(_r).x0 / fig.dpi
    fig.add_artist(Line2D([(x0_in - 0.09) / W], [(H - YHEAD + 0.042) / H], marker=DSM[t], color=DSC[t], ms=5.5, ls="none", transform=fig.transFigure))
    HEAD[DN[t]] = {"center_in": cx, "baseline_from_top_in": YHEAD}
fig.legend(handles=[Line2D([], [], color=RPC, lw=1.2, marker="o", ms=2.5, label="MFE recomputed"),
                    Line2D([], [], color=C_FIX, lw=1.2, ls="--", marker="o", ms=2.5, label="MFE fixed"),
                    Patch(color=BAND, label="Local window")],
           loc="lower right", ncol=3, frameon=False, bbox_to_anchor=((W - 0.05) / W, (H - YDT - 0.06) / H), columnspacing=1.0, handletextpad=0.4,
           borderaxespad=0, borderpad=0)
u = USED["L2"].setdefault("a", {})
for k, t in enumerate(TAGS):
    ax = box(COLX[k] + 0.33, YD + 0.12, PW - 0.33, 0.62); a1 = AD["datasets"][t]["A1"]; uu = u.setdefault(t, {})
    ax.axvspan(-6.5, 4.5, color=BAND, lw=0, zorder=0); ymax = 0
    for mode, col, ls in (("recompute", RPC, "-"), ("fixed", C_FIX, "--")):
        pm = a1[mode]["profile_mean_abs"]; rs = sorted(int(x) for x in pm)
        inner = [r for r in rs if -12 <= r <= 10]; outer = [r for r in rs if r < -12 or r > 10]; assert sorted(set(outer)) == [-20, -15, 15, 20], outer
        m = np.array([pm[str(r)][0] for r in inner]); lo = np.array([pm[str(r)][1] for r in inner]); hi = np.array([pm[str(r)][2] for r in inner])
        ax.fill_between(inner, lo, hi, color=col, alpha=0.18, lw=0); ax.plot(inner, m, color=col, ls=ls, lw=1.0, marker="o", ms=1.6)
        ax.plot(outer, [pm[str(r)][0] for r in outer], ls="none", marker="x", ms=3.2, color=col)
        ymax = max(ymax, hi.max(), max(pm[str(r)][2] for r in outer))
        uu[mode] = {"r": inner + outer, "mean_abs_du": [pm[str(r)][0] for r in inner + outer], "src": f"partIII_addendum_v2.json datasets.{t}.A1.{mode}.profile_mean_abs"}
    tp = {0: 0.65, 1: 1.3, 2: 0.4, 3: 0.35}[k]; ax.set_ylim(0, tp); assert ymax < tp, (t, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlim(-21.5, 21.5); ax.set_xticks([-20, 0, 20]); ax.set_xticklabels(["−20", "A", "+20"])
    ax.set_xticks([-2, -1], minor=True); ax.tick_params(axis="x", which="minor", length=4, width=0.9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    if k == 0: ax.set_ylabel("mean |Δu|", labelpad=2)
fig.text((COLX[0] - 0.1) / W, (H - (YD + 0.12 + 0.62 + 0.07)) / H, "edited codon", fontsize=FS, ha="right", va="top", color=INK)

# ---------------- (e) = old L2c, routing shares (aa hatched, as v4d)
YE = 4.37 + DY
plabel(fig, H, 0.02, YE, "e", "Branch routing depends on position")
fig.legend(handles=[(Patch(facecolor=BRC[b], hatch=BRH[b], edgecolor=HATCH_INK, lw=0, label=BRL[b]) if BRH[b] else Patch(color=BRC[b], label=BRL[b]))
                    for b in ("codon", "nt", "aa", "mfe")] + [Line2D([], [], color=INK, lw=0.7, ls=":", label="0.5")],
           loc="lower right", ncol=5, frameon=False, bbox_to_anchor=((W - 0.05) / W, (H - YE - 0.06) / H), columnspacing=1.0, handletextpad=0.4,
           borderaxespad=0, borderpad=0)
EFF = [("-5", "−5"), ("-2", "E"), ("-1", "P"), ("0", "A"), ("2", "+2"), ("nt15_G", "−15(G)")]
XE = [1.8, 2.8, 3.8, 4.8, 5.8, 0.0]; XSEP = (XE[5] + XE[0]) / 2   # x per EFF entry; −15(G) leftmost, extra gap ~1 bar width (0.8 units) before −5
u = USED["L2"].setdefault("c", {})
for k, t in enumerate(TAGS):
    ax = box(COLX[k], YE + 0.12, PW, 0.6)
    for i, (ek, el) in enumerate(EFF):
        sh = X2["datasets"][t]["shares"]["recompute"][ek]
        k7 = "nt_m15_G" if ek == "nt15_G" else ek
        per = {s: X7["datasets"][t]["per_seed"][s]["recompute"]["shares"][k7] for s in ("123", "124", "125")}
        base = 0.0; bounds = {s: 0.0 for s in per}
        for b in ("codon", "nt", "aa", "mfe"):
            v = sh[b]["s"]; assert abs(per["123"][b]["s"] - v) < 1e-6 and v > -0.03, (t, ek, b, v)
            ax.bar(XE[i], v, bottom=base, width=0.72, color=BRC[b], lw=0, **({"hatch": BRH[b], "edgecolor": HATCH_INK} if BRH[b] else {}))
            base += v
            for s in per: bounds[s] += per[s][b]["s"]
            if b != "mfe":
                bv = [bounds[s] for s in per]; ax.plot([XE[i], XE[i]], [min(bv), max(bv)], color=INK, lw=1.0, solid_capstyle="butt")
            u[f"{t}|{ek}|{b}"] = {"seed123": v, "ci95": sh[b]["ci"], "seed124": per["124"][b]["s"], "seed125": per["125"][b]["s"],
                                  "src": f"results_X2.json datasets.{t}.shares.recompute.{ek}.{b}; results_X7.json datasets.{t}.per_seed.<s>.recompute.shares.{k7}.{b}"}
    ax.axhline(0.5, color=INK, lw=0.7, ls=":")
    ax.axvline(XSEP, color=C_GREY, lw=0.5, ls="--", zorder=0)        # separates the five codon edits from the single-base edit
    _o = sorted(range(len(EFF)), key=lambda q: XE[q])
    ax.set_xticks([XE[q] for q in _o]); ax.set_xticklabels([EFF[q][1] for q in _o]); ax.set_xlim(-0.6, max(XE) + 0.6); ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    if k == 0: ax.set_ylabel("share", labelpad=2)
    else: ax.set_yticklabels([])

# ---------------- (f) = old L3a, whole-window transplant, plus shuffled with MFE held fixed (seed 123)
YF = 5.55 + DY
plabel(fig, H, 0.02, YF, "f", "The ordered window transfers far more than the shuffled one")
fig.legend(handles=[Line2D([], [], marker="o", color=RPC, ls="none", ms=4.5, label="MFE recomputed"),
                    Line2D([], [], marker="o", color=C_FIX, ls="none", ms=4.5, label="MFE fixed"),
                    Line2D([], [], marker="^", color=INK2, mfc="white", ls="none", ms=4.5, label="Shuffled (seed 123)")],
           loc="lower right", ncol=3, frameon=False, bbox_to_anchor=((W - 0.05) / W, (H - YF - 0.06) / H), columnspacing=1.0, handletextpad=0.3,
           handlelength=1.0,                                                # marker-only handles, frees room for the longer f title
           labelspacing=0.15, borderaxespad=0, borderpad=0)
u = USED["L3"].setdefault("a", {}); nf = NEW.setdefault("main_f_shuffled_fixed_seed123", {})
XPOS = {("recompute", "123"): 0.0, ("recompute", "124"): 0.25, ("recompute", "125"): 0.5,
        ("fixed", "123"): 0.9, ("fixed", "124"): 1.15, ("fixed", "125"): 1.4, ("S", "recompute"): 2.0, ("S", "fixed"): 2.35}
for k, t in enumerate(TAGS):
    ax = box(COLX[k], YF + 0.12, PW, 0.6); vals = []
    tw = X7["datasets"][t]["transplant_W_full_per_seed"]; sc = X5["datasets"][t]["shuffle_control"]
    for j, (mode, col) in enumerate((("recompute", RPC), ("fixed", C_FIX))):
        for si, s in enumerate(("123", "124", "125")):
            v = tw[f"CAN_s{s}|{mode}"]; x = XPOS[(mode, s)]
            ax.errorbar(x, v["median"], yerr=ci_err(v["median"], v["ci95"]), fmt="o", color=col, ms=3.4, lw=0.8, capsize=0)
            vals += v["ci95"]; u[f"{t}|ordered|{mode}|seed{s}"] = {"median": v["median"], "ci95": v["ci95"], "src": f"results_X7.json datasets.{t}.transplant_W_full_per_seed.CAN_s{s}|{mode}"}
    sv = sc["CAN_s123|recompute"]["S"]; x = XPOS[("S", "recompute")]
    ax.errorbar(x, sv["median"], yerr=ci_err(sv["median"], sv["ci95"]), fmt="^", color=RPC, mfc="white", ms=4, lw=0.8, capsize=0)
    vals += sv["ci95"]; u[f"{t}|shuffled|recompute|seed123"] = {"median": sv["median"], "ci95": sv["ci95"], "src": f"results_X5.json datasets.{t}.shuffle_control.CAN_s123|recompute.S"}
    sf = sc["CAN_s123|fixed"]["S"]; x = XPOS[("S", "fixed")]
    ax.errorbar(x, sf["median"], yerr=ci_err(sf["median"], sf["ci95"]), fmt="^", color=C_FIX, mfc="white", ms=4, lw=0.8, capsize=0)
    vals += sf["ci95"]; nf[f"{t}|shuffled|fixed|seed123"] = {"median": sf["median"], "ci95": sf["ci95"], "n": sf["n"],
                                                             "src": f"results_X5.json datasets.{t}.shuffle_control.CAN_s123|fixed.S"}
    assert abs(tw["CAN_s123|recompute"]["median"] - X3["datasets"][t]["summary"]["W_full|recompute"]["pooled"][0]) < 1e-9
    ax.axhline(0, color=INK2, lw=0.6); ax.axhline(1, color=INK2, lw=0.6, ls=":")
    ax.set_xticks([0.7, 2.175]); ax.set_xticklabels(["ordered", "shuffled"]); ax.set_xlim(-0.3, 2.65)
    ax.set_ylim(-0.1, 1.1); assert min(vals) > -0.1 and max(vals) < 1.1, (t, min(vals), max(vals))
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_yticks([0, 0.5, 1])
    if k == 0: ax.set_ylabel("transfer\nratio", labelpad=2)
    else: ax.set_yticklabels([])

# ---------------- (g) = old L3b, sub-windows RiboPipe vs adapted RiboMIMO
YG = 6.74 + DY
plabel(fig, H, 0.02, YG, "g", "Among the parts, the core transfers most")
fig.legend(handles=[Line2D([], [], marker="s", color=RPC, ls="none", ms=4.5, label="RiboPipe"),
                    Line2D([], [], marker="D", color=C_RM, mfc="white", ls="none", ms=4.5, label="Adapted RiboMIMO")],
           loc="lower right", ncol=2, frameon=False, bbox_to_anchor=((W - 0.05) / W, (H - YG - 0.06) / H), columnspacing=1.0, handletextpad=0.3,
           borderaxespad=0, borderpad=0)
WIN = [("W_full", "−6..+4"), ("W_wide", "−12..+10"), ("W_core", "−3..+1"), ("W_EPA", "E,P,A"), ("W_noEPA", "outside E,P,A"), ("W_up", "−6..−3"), ("W_down", "+1..+4")]
u = USED["L3"].setdefault("b", {})
for k, t in enumerate(TAGS):
    ax = box(COLX[k], YG + 0.1, PW, 0.88); ww = X5["datasets"][t]["windows"]
    for i, (wk, wl) in enumerate(WIN):
        for j, (mk, col, mfc, mrk) in enumerate((("CAN_s123", RPC, RPC, "s"), ("RM_s123", C_RM, "white", "D"))):
            v = ww[f"{mk}|{wk}|recompute"]["pooled"]; y = i + (j - 0.5) * 0.32
            ax.errorbar(v["median"], y, xerr=ci_err(v["median"], v["ci95"]), fmt=mrk, color=col, mfc=mfc, ms=3.4, lw=0.8, capsize=0)
            u[f"{t}|{wk}|{mk}"] = {"median": v["median"], "ci95": v["ci95"], "src": f"results_X5.json datasets.{t}.windows.{mk}|{wk}|recompute.pooled"}
            if mk == "CAN_s123" and wk != "W_wide":
                x3 = X3["datasets"][t]["summary"][f"{wk}|recompute"]["pooled"][0]; assert abs(x3 - v["median"]) < 1e-9, (t, wk)
            assert -0.1 < v["ci95"][0] and v["ci95"][1] < 1.1
    ax.axvline(0, color=INK2, lw=0.6); ax.axvline(1, color=INK2, lw=0.6, ls=":")
    ax.set_yticks(range(len(WIN))); ax.set_yticklabels([w[1] for w in WIN] if k == 0 else []); ax.set_ylim(len(WIN) - 0.5, -0.5)
    ax.set_xlim(-0.1, 1.1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.text((COLX[0] - 0.1) / W, (H - (YG + 0.1 + 0.88 + 0.05)) / H, "transfer ratio", fontsize=FS, ha="right", va="top", color=INK)
save(fig, STEM)

p = C.write_numbers(OUTDIR, STEM, USED, IN, extra={"published_generator": "make_fig_part3_merge_1115_v5.py", "published_pdf_md5": "757a83d0d4abb04caba45e6b52bdc193",
                                                   "numbers_drawn_new": NEW, "main_logos": MAIN_LOGOS, "font_check": FONTCHECK, "overlap_check": OVERLAP,
                                                   "main_column_titles": HEAD})
print("wrote", os.path.join(OUTDIR, STEM + ".pdf"), p)
