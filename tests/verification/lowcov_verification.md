# Low-coverage (Part II) port: verification against the paper's stored results

Written 2026-11-16 15:08:02 JST. 589 PASS, 0 FAIL. Floating-point values compared exactly (full stored precision). Reproduction from the paper's stored predictions (no retraining) on machine 76. Runtime: 54 min wall (2026-11-16 14:13:03-15:07:10 JST) on 76 (64 cores, shared with other jobs); crossover all 4 variants x 6 predictors x 4 datasets, posterior draws 0-9, kappa, ranking.

| Group | PASS | FAIL |
|---|---|---|
| crossover | 12 | 0 |
| headline | 61 | 0 |
| headline_bins | 72 | 0 |
| crossover_full | 109 | 0 |
| self_check | 8 | 0 |
| posterior_kappa | 4 | 0 |
| posterior | 13 | 0 |
| posterior_bins | 216 | 0 |
| posterior_full | 16 | 0 |
| posterior_draws | 18 | 0 |
| ranking | 56 | 0 |
| ranking_full | 4 | 0 |

## Headline quantities

| Quantity | Reference | Reproduced | Status |
|---|---|---|---|
| TX9_WT: RiboPipe D* (reads/codon) | 0.34449871473422966 | 0.34449871473422966 | PASS |
| TX9_WT: RiboPipe D* status | OK | OK | PASS |
| TX9_WT: RiboPipe D* 95% bootstrap CI | [0.3059811917330045, 0.3777309736264181] | [0.3059811917330045, 0.3777309736264181] | PASS |
| TX9_WT: RiboPipe bootstrap failure fraction | 0.001 | 0.001 | PASS |
| TX9_WT: opcrit status | OK | OK | PASS |
| TX9_WT: frac_eligible_below at_D_star | 0.6696802276912774 | 0.6696802276912774 | PASS |
| TX9_WT: frac_with_reference_below at_D_star | 0.6404228175687989 | 0.6404228175687989 | PASS |
| TX9_WT: frac_scorable_below at_D_star | 0.6325009317927693 | 0.6325009317927693 | PASS |
| TX9_WT: frac_eligible_below at_ci_lo | 0.6410513979574753 | 0.6410513979574753 | PASS |
| TX9_WT: frac_with_reference_below at_ci_lo | 0.6092582467650811 | 0.6092582467650811 | PASS |
| TX9_WT: frac_scorable_below at_ci_lo | 0.600633619083116 | 0.600633619083116 | PASS |
| TX9_WT: frac_eligible_below at_ci_hi | 0.6919470952620124 | 0.6919470952620124 | PASS |
| TX9_WT: frac_with_reference_below at_ci_hi | 0.6646619281939129 | 0.6646619281939129 | PASS |
| TX9_WT: frac_scorable_below at_ci_hi | 0.657286619455833 | 0.657286619455833 | PASS |
| TX9_WT: opcrit N_eligible | 11946 | 11946 | PASS |
| TX9_WT: opcrit N_with_reference | 10974 | 10974 | PASS |
| TX9_WT: opcrit N_scorable | 10732 | 10732 | PASS |
| TX9_WT: noise-predictor positive bins (lo, hi, y) | [[0.0008165, 0.0012396, 0.001286446238604215], [0.0012396, 0.0018821, 0.005787682207834418], [0.0018821, 0.0028574, 0.006108768543692332], [0.0028574, 0.0043383, 0.007541643467685527], [0.0043383, 0.0065866, 0.004223313397554274], [0.0065866, 0.01, 0.010621520326067321], [0.01, 0.0151824, 0.009494603484217736]] | [[0.0008165, 0.0012396, 0.001286446238604215], [0.0012396, 0.0018821, 0.005787682207834418], [0.0018821, 0.0028574, 0.006108768543692332], [0.0028574, 0.0043383, 0.007541643467685527], [0.0043383, 0.0065866, 0.004223313397554274], [0.0065866, 0.01, 0.010621520326067321], [0.01, 0.0151824, 0.009494603484217736]] | PASS |
| GSE233886_WT: RiboPipe D* (reads/codon) | 0.7320191546820658 | 0.7320191546820658 | PASS |
| GSE233886_WT: RiboPipe D* status | OK | OK | PASS |
| GSE233886_WT: RiboPipe D* 95% bootstrap CI | [0.6842741302039704, 0.7864254691320928] | [0.6842741302039704, 0.7864254691320928] | PASS |
| GSE233886_WT: RiboPipe bootstrap failure fraction | 0.08 | 0.08 | PASS |
| GSE233886_WT: opcrit status | OK | OK | PASS |
| GSE233886_WT: frac_eligible_below at_D_star | 0.918627684018699 | 0.918627684018699 | PASS |
| GSE233886_WT: frac_with_reference_below at_D_star | 0.9114578843003707 | 0.9114578843003707 | PASS |
| GSE233886_WT: frac_scorable_below at_D_star | 0.9093241330502477 | 0.9093241330502477 | PASS |
| GSE233886_WT: frac_eligible_below at_ci_lo | 0.9067427303700182 | 0.9067427303700182 | PASS |
| GSE233886_WT: frac_with_reference_below at_ci_lo | 0.8985257349771532 | 0.8985257349771532 | PASS |
| GSE233886_WT: frac_scorable_below at_ci_lo | 0.8960544939844303 | 0.8960544939844303 | PASS |
| GSE233886_WT: frac_eligible_below at_ci_hi | 0.9313049679106251 | 0.9313049679106251 | PASS |
| GSE233886_WT: frac_with_reference_below at_ci_hi | 0.9252521769118027 | 0.9252521769118027 | PASS |
| GSE233886_WT: frac_scorable_below at_ci_hi | 0.9233899504600142 | 0.9233899504600142 | PASS |
| GSE233886_WT: opcrit N_eligible | 12621 | 12621 | PASS |
| GSE233886_WT: opcrit N_with_reference | 11599 | 11599 | PASS |
| GSE233886_WT: opcrit N_scorable | 11304 | 11304 | PASS |
| GSE233886_WT: noise-predictor positive bins (lo, hi, y) | [[0.0008165, 0.0012396, 0.007373509983105368], [0.0018821, 0.0028574, 0.0005216870582804454], [0.0028574, 0.0043383, 0.005648817721070675], [0.0043383, 0.0065866, 0.004482153158787474], [0.0065866, 0.01, 0.005372470247342219], [0.01, 0.0151824, 0.011119368522189466]] | [[0.0008165, 0.0012396, 0.007373509983105368], [0.0018821, 0.0028574, 0.0005216870582804454], [0.0028574, 0.0043383, 0.005648817721070675], [0.0043383, 0.0065866, 0.004482153158787474], [0.0065866, 0.01, 0.005372470247342219], [0.01, 0.0151824, 0.011119368522189466]] | PASS |
| GSE133393_WT: RiboPipe D* (reads/codon) | null | null | PASS |
| GSE133393_WT: RiboPipe D* status | NO_SIGN_CHANGE | NO_SIGN_CHANGE | PASS |
| GSE133393_WT: RiboPipe D* 95% bootstrap CI | null | null | PASS |
| GSE133393_WT: RiboPipe bootstrap failure fraction | 0.9995 | 0.9995 | PASS |
| GSE133393_WT: opcrit status | NO_SIGN_CHANGE | NO_SIGN_CHANGE | PASS |
| GSE133393_WT: opcrit N_eligible | 13604 | 13604 | PASS |
| GSE133393_WT: opcrit N_with_reference | 12533 | 12533 | PASS |
| GSE133393_WT: opcrit N_scorable | 12268 | 12268 | PASS |
| GSE133393_WT: noise-predictor positive bins (lo, hi, y) | [[0.0008165, 0.0012396, 0.004890037033115251], [0.0012396, 0.0018821, 0.006666482754006532], [0.0018821, 0.0028574, 0.002573549758660805], [0.0028574, 0.0043383, 0.0027364618781432247], [0.0043383, 0.0065866, 0.0053897685507031945], [0.0065866, 0.01, 0.0039488091862210604], [0.01, 0.0151824, 0.00823799597282161], [0.0151824, 0.0230506, 0.012437914907616077]] | [[0.0008165, 0.0012396, 0.004890037033115251], [0.0012396, 0.0018821, 0.006666482754006532], [0.0018821, 0.0028574, 0.002573549758660805], [0.0028574, 0.0043383, 0.0027364618781432247], [0.0043383, 0.0065866, 0.0053897685507031945], [0.0065866, 0.01, 0.0039488091862210604], [0.01, 0.0151824, 0.00823799597282161], [0.0151824, 0.0230506, 0.012437914907616077]] | PASS |
| PRJNA_iPS: RiboPipe D* (reads/codon) | null | null | PASS |
| PRJNA_iPS: RiboPipe D* status | NO_SIGN_CHANGE | NO_SIGN_CHANGE | PASS |
| PRJNA_iPS: RiboPipe D* 95% bootstrap CI | null | null | PASS |
| PRJNA_iPS: RiboPipe bootstrap failure fraction | 0.733 | 0.733 | PASS |
| PRJNA_iPS: opcrit status | NO_SIGN_CHANGE | NO_SIGN_CHANGE | PASS |
| PRJNA_iPS: opcrit N_eligible | 14151 | 14151 | PASS |
| PRJNA_iPS: opcrit N_with_reference | 12451 | 12451 | PASS |
| PRJNA_iPS: opcrit N_scorable | 11898 | 11898 | PASS |
| PRJNA_iPS: noise-predictor positive bins (lo, hi, y) | <structure, 407 chars> | <structure, 407 chars> | PASS |
| pooled bins with 95% interval above zero, posterior - reads | [71, 72] | [71, 72] | PASS |
| pooled bins with 95% interval below zero, posterior - reads | [0, 72] | [0, 72] | PASS |
| pooled bins with point estimate > 0, posterior - reads | 72 | 72 | PASS |
| pooled bins with 95% interval above zero, posterior - RiboPipe | [49, 72] | [49, 72] | PASS |
| pooled bins with 95% interval below zero, posterior - RiboPipe | [6, 72] | [6, 72] | PASS |
| pooled bins with point estimate > 0, posterior - RiboPipe | 66 | 66 | PASS |
| pooled bins with both differences > 0 (draw 0) | [66, 72] | [66, 72] | PASS |

## Failures

none

## Scope and notes

- Crossover: `ribopipe crossover` on the four human datasets with the paper's stored Part II predictions (runs/partII/preds/<config>/<dataset>, md5 identical to those recorded in the reference), all five model predictors plus the noise predictor, all four variants (main analysis, input-depth axis, per-bin grid thinning, binomial split), B = 2000, matching seed 0. Reference: report/partII_results/rerun76/partII_eval.json.
- Posterior: `ribopipe posterior-kappa` (prior strength chosen on Part I transcripts of the other folds from the stored Part I out-of-fold predictions) and `ribopipe posterior` for matching draws 0-9 (B = 2000 each, as in the paper's final run). Reference: report/partII_results/posterior_partII/posterior_partII_results.json; kappa reference: Application_2026-11-13/results/a1_<dataset>.json.
- Ranking: `ribopipe peak-ranking` with the stored Part I out-of-fold predictions of RiboPipe, CodonMean and Ridge. Reference: Application_2026-11-13/results/a1_<dataset>.json (all cells, chosen prior strengths, descriptive crossover).
- Not reproduced here: the crossover matching-seed sensitivity (seeds 1-9; supported via `ribopipe crossover --seed k`), the descriptive p_seq-versus-raw-prediction table of the posterior analysis, and Part II inference itself (`ribopipe lowcov-predict` was only smoke-tested: 40 transcripts per fold of PRJNA_iPS, max |difference| to the stored predictions 3.8e-6, 140/200 bitwise equal; the remaining differences are float32 rounding from a different batch composition).

All 589 comparisons (including per-bin values, full dstar objects, per-role tables, posterior draws 1-9 and the ranking cells) are listed in lowcov_verification.json.
