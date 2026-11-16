#!/usr/bin/env python3
"""Verify the figure-reproduction scripts against the published figures.

For each of the four main figures:
  1. numbers: every value in the new <fig>_numbers_drawn.json must equal the published provenance exactly (same keys, same
     types, float equality without tolerance; NaN == NaN).
  2. pixels: the new PDF and the published PDF are rendered to PNG with pdftoppm at the same resolution and compared pixel by
     pixel (max absolute difference over RGB channels, number of differing pixels). The 300-dpi PNGs written by matplotlib are
     compared the same way.
Writes tests/verification/figures_verification.json and figures_verification.md.

Usage: python3 tests/verification/figures_verify.py NEW_DIR PUBLISHED_DIR SCRATCH_DIR [--dpi 200] [--alt OTHER_ENV_DIR]
  NEW_DIR        output directory of reproduce/figures/make_all.sh (fig1.pdf, fig1.png, fig1_numbers_drawn.json, ...)
  PUBLISHED_DIR  directory holding the published files under their published names (see PUB below)
  SCRATCH_DIR    where the rendered PNGs are written
"""
import os, sys, json, math, hashlib, subprocess, argparse
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser(); ap.add_argument("new"); ap.add_argument("pub"); ap.add_argument("scratch"); ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--alt", default=None, help="optional second output directory made with another Python/matplotlib (reported, not part of the pass criterion)")
a = ap.parse_args(); os.makedirs(a.scratch, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
PUB = {"fig1": ("fig_partI_main_1115_v5.pdf", "fig_partI_main_1115_v5.png", "fig_partI_main_1115_v5_provenance.json", ["datasets", "params_legend", "panel_b", "panel_d", "panel_e"]),
       "fig2": ("fig_part2_1115_v8.pdf", "fig_part2_1115_v8.png", "fig_part2_1115_v8_provenance.json", ["numbers_drawn", "labels_drawn", "seed_counts_used"]),
       "fig3": ("fig_part2_use_1115_v4.pdf", "fig_part2_use_1115_v4.png", "fig_part2_use_1115_v4_provenance.json", ["numbers_drawn"]),
       "fig4": ("fig_part3_main_1115_v5.pdf", "fig_part3_main_1115_v5.png", "fig_part3_merge_1115_v5_provenance.json", ["numbers_drawn", "numbers_drawn_new", "main_logos"])}


def leaves(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, f"{pre}[{i}]")
    else: yield pre, x


def compare(new, pub, pre, bad):
    """Exact structural comparison; appends mismatches to bad; returns number of leaves compared."""
    if isinstance(pub, dict):
        if not isinstance(new, dict) or list(new.keys()) != list(pub.keys()):
            if not isinstance(new, dict) or set(new) != set(pub): bad.append({"path": pre, "why": "keys differ"}); return 0
        return sum(compare(new[k], pub[k], f"{pre}.{k}", bad) for k in pub)
    if isinstance(pub, list):
        if not isinstance(new, list) or len(new) != len(pub): bad.append({"path": pre, "why": "list length differs"}); return 0
        return sum(compare(n, p, f"{pre}[{i}]", bad) for i, (n, p) in enumerate(zip(new, pub)))
    same = type(new) is type(pub) and (new == pub or (isinstance(pub, float) and math.isnan(pub) and math.isnan(new)))
    if not same: bad.append({"path": pre, "new": new, "published": pub})
    return 1


def render(pdf, stem):
    out = os.path.join(a.scratch, stem)
    subprocess.run(["pdftoppm", "-r", str(a.dpi), "-png", "-singlefile", pdf, out], check=True)
    return out + ".png"


def pixdiff(p1, p2):
    A = np.asarray(Image.open(p1).convert("RGB"), dtype=np.int16); B = np.asarray(Image.open(p2).convert("RGB"), dtype=np.int16)
    if A.shape != B.shape: return {"same_size": False, "shape_new": list(A.shape), "shape_published": list(B.shape)}
    D = np.abs(A - B).max(axis=2)
    return {"same_size": True, "shape": list(A.shape), "max_abs_diff": int(D.max()), "n_pixels_differing": int((D > 0).sum()), "n_pixels": int(D.size)}


REP = {"dpi_pdf_render": a.dpi, "renderer": subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True).stderr.strip().splitlines()[0], "figures": {}}


def check(newdir, tag):
    out = {}; ok_all = True
    for fig, (pdf, png, prov, keys) in PUB.items():
        N = json.load(open(os.path.join(newdir, f"{fig}_numbers_drawn.json"))); P = json.load(open(os.path.join(a.pub, prov)))
        if fig == "fig1": newv = {k: N["numbers_drawn"][k] for k in keys}
        elif fig == "fig4": newv = {"numbers_drawn": N["numbers_drawn"], "numbers_drawn_new": N["numbers_drawn_new"], "main_logos": N["main_logos"]}
        elif fig == "fig2": newv = {"numbers_drawn": N["numbers_drawn"], "labels_drawn": N["labels_drawn"], "seed_counts_used": N["seed_counts_used"]}
        else: newv = {"numbers_drawn": N["numbers_drawn"]}
        newv = json.loads(json.dumps(newv, default=float))           # tuples -> lists, as in the published JSON
        bad = []; n = 0; nnum = 0
        for k in keys:
            n += compare(newv[k], P[k], k, bad)
            nnum += sum(1 for _, v in leaves(P[k]) if isinstance(v, (int, float)) and not isinstance(v, bool))
        r_pdf = pixdiff(render(os.path.join(newdir, f"{fig}.pdf"), f"{fig}_{tag}"), render(os.path.join(a.pub, pdf), f"{fig}_published"))
        r_png = pixdiff(os.path.join(newdir, f"{fig}.png"), os.path.join(a.pub, png))
        ok = not bad and r_pdf.get("max_abs_diff") == 0 and r_png.get("max_abs_diff") == 0
        ok_all &= ok
        out[fig] = {"published_pdf": pdf, "published_pdf_md5": md5(os.path.join(a.pub, pdf)), "new_pdf_md5": md5(os.path.join(newdir, f"{fig}.pdf")),
                    "published_png_md5": md5(os.path.join(a.pub, png)), "new_png_md5": md5(os.path.join(newdir, f"{fig}.png")),
                    "numbers": {"keys_compared": keys, "n_leaves_compared": n, "n_numeric_leaves_published": nnum, "n_mismatches": len(bad), "mismatches": bad[:20]},
                    "pdf_render_pixel_diff": r_pdf, "png_300dpi_pixel_diff": r_png, "pass": ok,
                    "python": N.get("python"), "matplotlib": N.get("matplotlib")}
        print(tag, fig, "numbers", n, "leaves,", len(bad), "mismatches | pdf render", r_pdf, "| png", r_png, flush=True)
    return out, ok_all


REP["figures"], ok_all = check(a.new, "new")
# negative controls: the checks must detect a one-ulp change of one number and a one-pixel change
_N = json.loads(json.dumps(json.load(open(os.path.join(a.new, "fig3_numbers_drawn.json")))["numbers_drawn"]))
_P = json.load(open(os.path.join(a.pub, PUB["fig3"][2])))["numbers_drawn"]
_k = "DMSO_TX9P1|random"; _N["b"][_k] = float(np.nextafter(_N["b"][_k], 1.0)); _bad = []; compare(_N, _P, "numbers_drawn", _bad)
_im = np.asarray(Image.open(os.path.join(a.new, "fig3.png")).convert("RGB")).copy(); _im[100, 100, 0] = 255 - _im[100, 100, 0]
_t = os.path.join(a.scratch, "negctl_fig3.png"); Image.fromarray(_im).save(_t); _pd = pixdiff(_t, os.path.join(a.pub, PUB["fig3"][1]))
REP["negative_controls"] = {"one_ulp_change_detected": len(_bad) == 1, "one_pixel_change_detected": _pd.get("n_pixels_differing") == 1}
ok_all &= all(REP["negative_controls"].values())
print("negative controls", REP["negative_controls"])
REP["all_pass"] = ok_all
if a.alt:
    _alt, _ok = check(a.alt, "alt")
    REP["other_environment"] = {"all_pass": _ok, "figures": _alt}
json.dump(REP, open(os.path.join(HERE, "figures_verification.json"), "w"), indent=1)
L = ["# Figure reproduction: verification", "",
     f"Each script in `reproduce/figures/` was run from `paper_results/` only. Numbers: every value in `<fig>_numbers_drawn.json` compared with the "
     f"published provenance (exact equality, same types). Pixels: new and published PDFs rendered with `{REP['renderer']}` at {a.dpi} dpi, plus the 300-dpi "
     f"PNGs written by matplotlib, compared pixel by pixel (max absolute RGB difference).", "",
     "| figure | published file | numeric leaves compared | mismatches | PDF render max diff (differing px) | 300-dpi PNG max diff (differing px) | pass |", "|---|---|---|---|---|---|---|"]
for f, r in REP["figures"].items():
    q = r["pdf_render_pixel_diff"]; g = r["png_300dpi_pixel_diff"]
    L.append(f"| {f} | `{r['published_pdf']}` | {r['numbers']['n_leaves_compared']} | {r['numbers']['n_mismatches']} | "
             f"{q.get('max_abs_diff', 'size differs')} ({q.get('n_pixels_differing', '-')}) | {g.get('max_abs_diff', 'size differs')} ({g.get('n_pixels_differing', '-')}) | {'yes' if r['pass'] else 'NO'} |")
nc = REP["negative_controls"]
L += ["", f"Negative controls (the checks detect a one-ulp change of one drawn number and a one-pixel change of one PNG): "
      f"{'detected' if nc['one_ulp_change_detected'] else 'NOT detected'} / {'detected' if nc['one_pixel_change_detected'] else 'NOT detected'}.", "",
      f"All pass: **{'yes' if ok_all else 'no'}**. The 300-dpi PNGs are byte-identical to the published PNGs (same md5) for "
      f"{sum(r['new_png_md5'] == r['published_png_md5'] for r in REP['figures'].values())}/4 figures and the PDFs for "
      f"{sum(r['new_pdf_md5'] == r['published_pdf_md5'] for r in REP['figures'].values())}/4; matplotlib writes the creation date into the PDF, "
      "so the rendered pages are compared instead. Environment of the run: Python " + str(REP['figures']['fig1']['python']) + ", matplotlib "
      + str(REP['figures']['fig1']['matplotlib']) + " (the versions that made the published figures). Details: `figures_verification.json`."]
if a.alt:
    o = REP["other_environment"]["figures"]; f1 = o["fig1"]
    L += ["", f"Second environment (not part of the pass criterion): Python {f1['python']}, matplotlib {f1['matplotlib']}.", "",
          "| figure | number mismatches | PDF render max diff (differing px) | 300-dpi PNG max diff (differing px) |", "|---|---|---|---|"]
    for f, r in o.items():
        q = r["pdf_render_pixel_diff"]; g = r["png_300dpi_pixel_diff"]
        L.append(f"| {f} | {r['numbers']['n_mismatches']} | {q.get('max_abs_diff', 'size differs')} ({q.get('n_pixels_differing', '-')}) | "
                 f"{g.get('max_abs_diff', 'size differs')} ({g.get('n_pixels_differing', '-')}) |")
open(os.path.join(HERE, "figures_verification.md"), "w").write("\n".join(L) + "\n")
print("all_pass", ok_all)
