# Verification of ribopipe.interpret against the paper's Part III results

Generated 2026-11-16 15:06:31 JST. Machine 76 (CPU), enformer-pilot Python (torch 2.1.2, numpy 1.26, ViennaRNA 2.7.0, PYTHONNOUSERSITE=1). Paper checkpoints (Paper_RiboPipe_MFE_H128, seed 123, five out-of-fold models) loaded with `ribopipe.train.load_ckpt`; nothing retrained. Same sites, peaks, donors/acceptors, shuffles and bootstrap seeds as the paper (they are re-derived by the port, not read from the paper's files). PASS = absolute difference <= 1e-9 with identical NaN pattern.

**Total: 8108 PASS / 0 FAIL of 8112 compared quantities** (4 excluded: the descriptive label string of the recompute-minus-fixed block, which the port gives in English).

| Dataset | Analysis | Compared | PASS | FAIL | Raw arrays bit-identical | Wall (s) | Reference |
|---|---|---|---|---|---|---|---|
| TX9_WT (DMSO_TX9P1) | ism | 339 | 338 | 0 | 14/14 | 569.0 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].ism + raw/<ds>/ism.npz |
| TX9_WT (DMSO_TX9P1) | ext | 111 | 111 | 0 | 5/5 | 111.7 | Paper_2026-11-11/report/partIII_results/partIII_addendum_v2.json datasets[ds].A1 + raw/<ds>/ism_ext.npz |
| TX9_WT (DMSO_TX9P1) | map | 626 | 626 | 0 | 12/12 | 242.4 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].map + raw/<ds>/map.npz |
| TX9_WT (DMSO_TX9P1) | transplant | 60 | 60 | 0 | 20/20 | 16.3 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].transplant + raw/<ds>/transplant.npz (+ tp2x2.npz u_orig) |
| TX9_WT (DMSO_TX9P1) | subwin | 259 | 259 | 0 | 17/17 | 83.8 | 2026-11-15_overnight/experiments/results_X5.json (CAN_s123 rows), results_X3.json + addA/raw/<ds>/x5_transplant.npz, x5s_shuffle.npz |
| TX9_WT (DMSO_TX9P1) | routing | 633 | 633 | 0 | 11/11 | 1774.7 | 2026-11-15_overnight/experiments/results_X2.json shares + raw/<ds>/s1_X12.npz (D/DR/DN/DNR_CAN) |
| GSE233886_WT | ism | 339 | 338 | 0 | 14/14 | 579.4 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].ism + raw/<ds>/ism.npz |
| GSE233886_WT | ext | 111 | 111 | 0 | 5/5 | 113.7 | Paper_2026-11-11/report/partIII_results/partIII_addendum_v2.json datasets[ds].A1 + raw/<ds>/ism_ext.npz |
| GSE233886_WT | map | 626 | 626 | 0 | 12/12 | 244.0 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].map + raw/<ds>/map.npz |
| GSE233886_WT | transplant | 60 | 60 | 0 | 20/20 | 17.2 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].transplant + raw/<ds>/transplant.npz (+ tp2x2.npz u_orig) |
| GSE233886_WT | subwin | 259 | 259 | 0 | 17/17 | 87.3 | 2026-11-15_overnight/experiments/results_X5.json (CAN_s123 rows), results_X3.json + addA/raw/<ds>/x5_transplant.npz, x5s_shuffle.npz |
| GSE233886_WT | routing | 633 | 633 | 0 | 11/11 | 1748.0 | 2026-11-15_overnight/experiments/results_X2.json shares + raw/<ds>/s1_X12.npz (D/DR/DN/DNR_CAN) |
| GSE133393_WT | ism | 339 | 338 | 0 | 14/14 | 588.4 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].ism + raw/<ds>/ism.npz |
| GSE133393_WT | ext | 111 | 111 | 0 | 5/5 | 118.6 | Paper_2026-11-11/report/partIII_results/partIII_addendum_v2.json datasets[ds].A1 + raw/<ds>/ism_ext.npz |
| GSE133393_WT | map | 626 | 626 | 0 | 12/12 | 230.3 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].map + raw/<ds>/map.npz |
| GSE133393_WT | transplant | 60 | 60 | 0 | 20/20 | 18.6 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].transplant + raw/<ds>/transplant.npz (+ tp2x2.npz u_orig) |
| GSE133393_WT | subwin | 259 | 259 | 0 | 17/17 | 83.1 | 2026-11-15_overnight/experiments/results_X5.json (CAN_s123 rows), results_X3.json + addA/raw/<ds>/x5_transplant.npz, x5s_shuffle.npz |
| GSE133393_WT | routing | 633 | 633 | 0 | 11/11 | 1694.3 | 2026-11-15_overnight/experiments/results_X2.json shares + raw/<ds>/s1_X12.npz (D/DR/DN/DNR_CAN) |
| PRJNA_iPS | ism | 339 | 338 | 0 | 14/14 | 498.9 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].ism + raw/<ds>/ism.npz |
| PRJNA_iPS | ext | 111 | 111 | 0 | 5/5 | 76.7 | Paper_2026-11-11/report/partIII_results/partIII_addendum_v2.json datasets[ds].A1 + raw/<ds>/ism_ext.npz |
| PRJNA_iPS | map | 626 | 626 | 0 | 12/12 | 190.1 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].map + raw/<ds>/map.npz |
| PRJNA_iPS | transplant | 60 | 60 | 0 | 20/20 | 20.0 | Paper_2026-11-11/report/partIII_results/partIII_eval.json results[ds].transplant + raw/<ds>/transplant.npz (+ tp2x2.npz u_orig) |
| PRJNA_iPS | subwin | 259 | 259 | 0 | 17/17 | 98.5 | 2026-11-15_overnight/experiments/results_X5.json (CAN_s123 rows), results_X3.json + addA/raw/<ds>/x5_transplant.npz, x5s_shuffle.npz |
| PRJNA_iPS | routing | 633 | 633 | 0 | 11/11 | 1625.5 | 2026-11-15_overnight/experiments/results_X2.json shares + raw/<ds>/s1_X12.npz (D/DR/DN/DNR_CAN) |

Analyses: **ism** = peak ISM, offsets -12..+10: mean effect / mean |effect| per offset (recompute, fixed, recompute-fixed), selection/evaluation across halves; **ext** = peak ISM control offsets -20,-15,+15,+20 and effective response range; **map** = centred codon / nucleotide sensitivity maps, positional importance, half reproducibility; **transplant** = whole-window transplant T/S/R x recompute/fixed: mean delta u, T-S, T-R, transfer ratio median; **subwin** = sub-window, wide-window and shuffled-window transfer ratios (median, unit-bootstrap CI), additivity, wide-full; **routing** = branch routing shares (codon/nt/aa/MFE, rho, main-route call) for every offset, G at nt -15 and Pro@P/Pro@E/Cys@A.

## Selected quantities (reference = paper artefact, reproduced = port)

| Dataset | Quantity | Reference | Reproduced | Status |
|---|---|---|---|---|
| TX9_WT (DMSO_TX9P1) | ISM mean abs, P site (r=-1), MFE recomputed | [0.5276, 0.4685, 0.5888] | [0.5276, 0.4685, 0.5888] | PASS |
| TX9_WT (DMSO_TX9P1) | ISM mean abs, P site (r=-1), MFE fixed | [0.4572, 0.4039, 0.5116] | [0.4572, 0.4039, 0.5116] | PASS |
| TX9_WT (DMSO_TX9P1) | ISM effective response range (recompute) | [-7, 6] | [-7.0000, 6.0000] | PASS |
| TX9_WT (DMSO_TX9P1) | far-offset controls / maximum (recompute) | 0.030703 | 0.030703 | PASS |
| TX9_WT (DMSO_TX9P1) | transfer ratio A->B (recompute) | 0.982482 | 0.982482 | PASS |
| TX9_WT (DMSO_TX9P1) | T - S mean delta u, B->A (fixed) | [0.9782, 0.7369, 1.2020] | [0.9782, 0.7369, 1.2020] | PASS |
| TX9_WT (DMSO_TX9P1) | whole-window transfer ratio, pooled (recompute) | 0.958072 | 0.958072 | PASS |
| TX9_WT (DMSO_TX9P1) | shuffled-window ratio (recompute) | 0.165383 | 0.165383 | PASS |
| TX9_WT (DMSO_TX9P1) | ordered - shuffled ratio CI (fixed) | [0.5775, 0.7563] | [0.5775, 0.7563] | PASS |
| TX9_WT (DMSO_TX9P1) | core (-3..+1) ratio (recompute) | 0.475167 | 0.475167 | PASS |
| TX9_WT (DMSO_TX9P1) | wide - whole ratio (recompute) | 0.030387 | 0.030387 | PASS |
| TX9_WT (DMSO_TX9P1) | routing share nt, codon -5 (recompute) | 0.736393 | 0.736393 | PASS |
| TX9_WT (DMSO_TX9P1) | routing share nt CI, codon -5 | [0.7220, 0.7500] | [0.7220, 0.7500] | PASS |
| TX9_WT (DMSO_TX9P1) | routing share nt, G at nt -15 | 0.733349 | 0.733349 | PASS |
| TX9_WT (DMSO_TX9P1) | routing share MFE, codon +2 | 0.655969 | 0.655969 | PASS |
| TX9_WT (DMSO_TX9P1) | main-route call, E site | shared | shared | PASS |
| GSE233886_WT | ISM mean abs, P site (r=-1), MFE recomputed | [0.4095, 0.3534, 0.4728] | [0.4095, 0.3534, 0.4728] | PASS |
| GSE233886_WT | ISM mean abs, P site (r=-1), MFE fixed | [0.3755, 0.3248, 0.4348] | [0.3755, 0.3248, 0.4348] | PASS |
| GSE233886_WT | ISM effective response range (recompute) | [-6, 4] | [-6.0000, 4.0000] | PASS |
| GSE233886_WT | far-offset controls / maximum (recompute) | 0.014268 | 0.014268 | PASS |
| GSE233886_WT | transfer ratio A->B (recompute) | 0.966055 | 0.966055 | PASS |
| GSE233886_WT | T - S mean delta u, B->A (fixed) | [1.5338, 1.2973, 1.7704] | [1.5338, 1.2973, 1.7704] | PASS |
| GSE233886_WT | whole-window transfer ratio, pooled (recompute) | 0.946367 | 0.946367 | PASS |
| GSE233886_WT | shuffled-window ratio (recompute) | 0.016509 | 0.016509 | PASS |
| GSE233886_WT | ordered - shuffled ratio CI (fixed) | [0.7567, 0.8673] | [0.7567, 0.8673] | PASS |
| GSE233886_WT | core (-3..+1) ratio (recompute) | 0.687953 | 0.687953 | PASS |
| GSE233886_WT | wide - whole ratio (recompute) | 0.012597 | 0.012597 | PASS |
| GSE233886_WT | routing share nt, codon -5 (recompute) | 0.668387 | 0.668387 | PASS |
| GSE233886_WT | routing share nt CI, codon -5 | [0.6530, 0.6828] | [0.6530, 0.6828] | PASS |
| GSE233886_WT | routing share nt, G at nt -15 | 0.708182 | 0.708182 | PASS |
| GSE233886_WT | routing share MFE, codon +2 | 0.351811 | 0.351811 | PASS |
| GSE233886_WT | main-route call, E site | shared | shared | PASS |
| GSE133393_WT | ISM mean abs, P site (r=-1), MFE recomputed | [0.3170, 0.2798, 0.3592] | [0.3170, 0.2798, 0.3592] | PASS |
| GSE133393_WT | ISM mean abs, P site (r=-1), MFE fixed | [0.2706, 0.2380, 0.3076] | [0.2706, 0.2380, 0.3076] | PASS |
| GSE133393_WT | ISM effective response range (recompute) | [-7, 5] | [-7.0000, 5.0000] | PASS |
| GSE133393_WT | far-offset controls / maximum (recompute) | 0.026915 | 0.026915 | PASS |
| GSE133393_WT | transfer ratio A->B (recompute) | 0.960361 | 0.960361 | PASS |
| GSE133393_WT | T - S mean delta u, B->A (fixed) | [0.6619, 0.4918, 0.8380] | [0.6619, 0.4918, 0.8380] | PASS |
| GSE133393_WT | whole-window transfer ratio, pooled (recompute) | 0.939031 | 0.939031 | PASS |
| GSE133393_WT | shuffled-window ratio (recompute) | 0.050701 | 0.050701 | PASS |
| GSE133393_WT | ordered - shuffled ratio CI (fixed) | [0.6023, 0.7238] | [0.6023, 0.7238] | PASS |
| GSE133393_WT | core (-3..+1) ratio (recompute) | 0.493569 | 0.493569 | PASS |
| GSE133393_WT | wide - whole ratio (recompute) | 0.027305 | 0.027305 | PASS |
| GSE133393_WT | routing share nt, codon -5 (recompute) | 0.731265 | 0.731265 | PASS |
| GSE133393_WT | routing share nt CI, codon -5 | [0.7194, 0.7424] | [0.7194, 0.7424] | PASS |
| GSE133393_WT | routing share nt, G at nt -15 | 0.766832 | 0.766832 | PASS |
| GSE133393_WT | routing share MFE, codon +2 | 0.577981 | 0.577981 | PASS |
| GSE133393_WT | main-route call, E site | shared | shared | PASS |
| PRJNA_iPS | ISM mean abs, P site (r=-1), MFE recomputed | [0.2963, 0.2652, 0.3308] | [0.2963, 0.2652, 0.3308] | PASS |
| PRJNA_iPS | ISM mean abs, P site (r=-1), MFE fixed | [0.2125, 0.1926, 0.2353] | [0.2125, 0.1926, 0.2353] | PASS |
| PRJNA_iPS | ISM effective response range (recompute) | [-7, 5] | [-7.0000, 5.0000] | PASS |
| PRJNA_iPS | far-offset controls / maximum (recompute) | 0.027104 | 0.027104 | PASS |
| PRJNA_iPS | transfer ratio A->B (recompute) | 0.919699 | 0.919699 | PASS |
| PRJNA_iPS | T - S mean delta u, B->A (fixed) | [0.4087, 0.3029, 0.5055] | [0.4087, 0.3029, 0.5055] | PASS |
| PRJNA_iPS | whole-window transfer ratio, pooled (recompute) | 0.958927 | 0.958927 | PASS |
| PRJNA_iPS | shuffled-window ratio (recompute) | 0.161500 | 0.161500 | PASS |
| PRJNA_iPS | ordered - shuffled ratio CI (fixed) | [0.3600, 0.4893] | [0.3600, 0.4893] | PASS |
| PRJNA_iPS | core (-3..+1) ratio (recompute) | 0.353337 | 0.353337 | PASS |
| PRJNA_iPS | wide - whole ratio (recompute) | 0.034639 | 0.034639 | PASS |
| PRJNA_iPS | routing share nt, codon -5 (recompute) | 0.667302 | 0.667302 | PASS |
| PRJNA_iPS | routing share nt CI, codon -5 | [0.6541, 0.6799] | [0.6541, 0.6799] | PASS |
| PRJNA_iPS | routing share nt, G at nt -15 | 0.805399 | 0.805399 | PASS |
| PRJNA_iPS | routing share MFE, codon +2 | 0.776223 | 0.776223 | PASS |
| PRJNA_iPS | main-route call, E site | main route: mfe | main route: mfe | PASS |

## Activation logos (Fig. 4a), cuda0

TX9_WT, RiboPipe seed 123, 5 fold(s), 128 filters per fold (64 codon, 32 nt, 32 aa): 640/640 logos identical to the paper's E12 logos (e12_logos.npz, CAN arm, computed on GPU); max |diff| 0. codon: 320/320 identical, max |diff| 0; nt: 160/160 identical, max |diff| 0; aa: 160/160 identical, max |diff| 0

## Activation logos (Fig. 4a), cpu

TX9_WT, RiboPipe seed 123, 1 fold(s), 128 filters per fold (64 codon, 32 nt, 32 aa): 70/128 logos identical to the paper's E12 logos (e12_logos.npz, CAN arm, computed on GPU); max |diff| 1.58. codon: 22/64 identical, max |diff| 1.58; nt: 16/32 identical, max |diff| 0.0219; aa: 32/32 identical, max |diff| 0

## Routing with another training seed (GSE233886_WT|seed124)

Paper checkpoints of seed 124; raw per-site arrays compared with the paper's addA/raw/GSE233886_WT/s1_X7.npz: 7/7 PASS (raw:routing_s124.tx max |diff| None; raw:routing_s124.sites max |diff| 0.0; raw:routing_s124.D max |diff| 0.0; raw:routing_s124.DR max |diff| 0.0; raw:routing_s124.DN max |diff| 0.0; raw:routing_s124.DNR max |diff| 0.0; routing gate max |diff| None). Wall 321.0 s.
Routing shares (recompute, rounded to 2 decimals as in Supplementary Table S2.6.11): {"-5": {"codon": 0.04, "nt": 0.66, "aa": 0.26, "mfe": 0.04}, "-2": {"codon": 0.25, "nt": 0.31, "aa": 0.34, "mfe": 0.12}, "-1": {"codon": 0.32, "nt": 0.25, "aa": 0.31, "mfe": 0.13}, "0": {"codon": 0.44, "nt": 0.54, "aa": 0.07, "mfe": 0.0}, "2": {"codon": 0.08, "nt": 0.57, "aa": 0.02, "mfe": 0.34}, "nt15_G": {"codon": 0.01, "nt": 0.67, "aa": 0.33, "mfe": 0.02}}

The paper computed the logos on the GPU. On the CPU, float-level differences in the activations reorder near-tied windows at the top-1% cut, so some codon and nucleotide logos differ; run `ribopipe logos --device cuda:0` to reproduce Fig. 4a exactly.

## Not ported / not verified here

- Adapted RiboMIMO runs (transplant and sub-window comparisons, enrichment comparison): the RiboMIMO model is not part of this package.
- Single-branch and two-branch (lattice) models, branch mean-replacement reliance, filter importance and cross-seed filter matching.
- Pro-Pro double-substitution interaction (S2 sites), A2 (r = -5 composition) and A3 (2x2 delta-yhat / e >= 0.1 ratio) summaries of the addendum; the arrays they need (ISM, maps, u_orig of the transplant) are produced.
- MFEhead exact decomposition and the MFE perturbation (training-mean) analysis.
- Seeds 124/125: supported through --ckpt-dir; verified here only for routing on GSE233886_WT seed 124.

## Failures

None.
