# Figure reproduction: verification

Each script in `reproduce/figures/` was run from `paper_results/` only. Numbers: every value in `<fig>_numbers_drawn.json` compared with the published provenance (exact equality, same types). Pixels: new and published PDFs rendered with `pdftoppm version 22.02.0` at 200 dpi, plus the 300-dpi PNGs written by matplotlib, compared pixel by pixel (max absolute RGB difference).

| figure | published file | numeric leaves compared | mismatches | PDF render max diff (differing px) | 300-dpi PNG max diff (differing px) | pass |
|---|---|---|---|---|---|---|
| fig1 | `fig_partI_main_1115_v5.pdf` | 1645 | 0 | 0 (0) | 0 (0) | yes |
| fig2 | `fig_part2_1115_v8.pdf` | 694 | 0 | 0 (0) | 0 (0) | yes |
| fig3 | `fig_part2_use_1115_v4.pdf` | 1716 | 0 | 0 (0) | 0 (0) | yes |
| fig4 | `fig_part3_main_1115_v5.pdf` | 1577 | 0 | 0 (0) | 0 (0) | yes |

Negative controls (the checks detect a one-ulp change of one drawn number and a one-pixel change of one PNG): detected / detected.

All pass: **yes**. The 300-dpi PNGs are byte-identical to the published PNGs (same md5) for 4/4 figures and the PDFs for 0/4; matplotlib writes the creation date into the PDF, so the rendered pages are compared instead. Environment of the run: Python 3.10.18, matplotlib 3.10.9 (the versions that made the published figures). Details: `figures_verification.json`.

Second environment (not part of the pass criterion): Python 3.11.15, matplotlib 3.10.7.

| figure | number mismatches | PDF render max diff (differing px) | 300-dpi PNG max diff (differing px) |
|---|---|---|---|
| fig1 | 0 | 0 (0) | 0 (0) |
| fig2 | 0 | 0 (0) | 0 (0) |
| fig3 | 0 | 0 (0) | 0 (0) |
| fig4 | 0 | 0 (0) | 1 (2) |
