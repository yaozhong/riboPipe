# Verification of `ribopipe.evaluate` (Part I evaluation)

Generated 2026-11-16 14:42:23 JST on dl-box3. Runs: `<storage>/20261102_human_reAnalysis/2026-11-04_newSplit/Paper_2026-11-11/runs/partI`. Data: `<storage>/software/github_0901/verify_v2/data_full`. Supplement: `<storage>/software/20260901_riboPipe_BIB_interp/2026-11-12_paperData/2026-11-16/03_supp/supplementary_2_1116_v7.tex` (md5 a726250f164bd3ab5cde7dbf6f2a0e9f).

**Overall: PASS**

## 1. Field-by-field comparison with the paper's evaluation JSONs

| Quantity | Reference file | Leaves compared | Equal | Max abs numeric diff | Status |
|---|---|---|---|---|---|
| setting1_seed123 | `partI_eval_elig.json` (md5 a102fe4b1c) | 14788 | 14788 | 0 | PASS |
| setting1_3seed | `partI_eval_elig_3seed.json` (md5 00ae555dc6) | 7613 | 7613 | 0 | PASS |
| setting2_seed123 | `T2_matched_eval_elig.json` (md5 6f3c84daed) | 773 | 773 | 0 | PASS |
| ribopipe_seeds | `ribopipe_seed_noise_1114.json` (md5 7521bb0ddc) | 166 | 166 | 0 | PASS |

Numbers must agree within 1e-12; strings, classes, Holm decisions and set sizes exactly. Run-status bookkeeping is not compared leaf by leaf, but every configuration that was COMPLETE in the reference is required to be COMPLETE again.

Not compared (1 cells): the comparator's runs were not complete when the reference file was written (reference status 'NA (method not complete or absent)' / 'PENDING'); the reproduction scores the runs present now.

- setting1_3seed `datasets.PRJNA_iPS.extra_pairs.Paper_RiboPipe_MFE_H128|Paper_RiboExp.status`: reference 'NA (method not complete or absent)', now 'OK'

## 2. Published Supplementary tables of Part I rebuilt from the reproduced numbers

460/460 cells identical to the text of the supplement.

| Table | Cells | PASS | FAIL |
|---|---|---|---|
| tab:s2l1pearson | 104 | 104 | 0 |
| tab:s2l1spearman | 104 | 104 | 0 |
| tab:s2l1recall | 104 | 104 | 0 |
| tab:s2l1pairs | 56 | 56 | 0 |
| tab:s2l2 | 24 | 24 | 0 |
| tab:s2threeseedabs | 32 | 32 | 0 |
| tab:s2threeseed | 24 | 24 | 0 |
| tab:s2seeds | 8 | 8 | 0 |
| holm | 4 | 4 | 0 |

Selected cells (Table of setting-1 Pearson, RiboPipe and RiboMIMO rows; setting-2 RiboGL):

| Table | Quantity | Supplement | Reproduced | Status |
|---|---|---|---|---|
| tab:s2l1pearson | pearson_interior DMSO_TX9P1 Paper_RiboMIMO | 0.578 | 0.578 | PASS |
| tab:s2l1pearson | pearson_interior GSE233886_WT Paper_RiboMIMO | 0.756 | 0.756 | PASS |
| tab:s2l1pearson | pearson_interior GSE133393_WT Paper_RiboMIMO | 0.518 | 0.518 | PASS |
| tab:s2l1pearson | pearson_interior PRJNA_iPS Paper_RiboMIMO | 0.502 | 0.502 | PASS |
| tab:s2l1pearson | pearson_interior Subtelny14_ribomimo Paper_RiboMIMO | 0.643 | 0.643 | PASS |
| tab:s2l1pearson | pearson_interior Mohammad16 Paper_RiboMIMO | 0.732 | 0.732 | PASS |
| tab:s2l1pearson | pearson_interior Mohammad19-1 Paper_RiboMIMO | 0.719 | 0.719 | PASS |
| tab:s2l1pearson | pearson_interior Mohammad19-2 Paper_RiboMIMO | 0.745 | 0.745 | PASS |
| tab:s2l1pearson | pearson_interior DMSO_TX9P1 Paper_RiboPipe_MFE_H128 | 0.641 | 0.641 | PASS |
| tab:s2l1pearson | pearson_interior GSE233886_WT Paper_RiboPipe_MFE_H128 | 0.781 | 0.781 | PASS |
| tab:s2l1pearson | pearson_interior GSE133393_WT Paper_RiboPipe_MFE_H128 | 0.571 | 0.571 | PASS |
| tab:s2l1pearson | pearson_interior PRJNA_iPS Paper_RiboPipe_MFE_H128 | 0.631 | 0.631 | PASS |
| tab:s2l1pearson | pearson_interior Subtelny14_ribomimo Paper_RiboPipe_MFE_H128 | 0.663 | 0.663 | PASS |
| tab:s2l1pearson | pearson_interior Mohammad16 Paper_RiboPipe_MFE_H128 | 0.750 | 0.750 | PASS |
| tab:s2l1pearson | pearson_interior Mohammad19-1 Paper_RiboPipe_MFE_H128 | 0.715 | 0.715 | PASS |
| tab:s2l1pearson | pearson_interior Mohammad19-2 Paper_RiboPipe_MFE_H128 | 0.740 | 0.740 | PASS |
| tab:s2l2 | setting 2 DMSO_TX9P1 RiboPipe-Paper_RiboGL_L2_heldout | $+$0.083 & [$+$0.078, $+$0.088] & $+$0.027 | $+$0.083 & [$+$0.078, $+$0.088] & $+$0.027 | PASS |
| tab:s2l2 | setting 2 GSE233886_WT RiboPipe-Paper_RiboGL_L2_heldout | $+$0.035 & [$+$0.031, $+$0.040] & $+$0.066 | $+$0.035 & [$+$0.031, $+$0.040] & $+$0.066 | PASS |
| tab:s2l2 | setting 2 GSE133393_WT RiboPipe-Paper_RiboGL_L2_heldout | $+$0.092 & [$+$0.083, $+$0.099] & $+$0.013 | $+$0.092 & [$+$0.083, $+$0.099] & $+$0.013 | PASS |
| tab:s2l2 | setting 2 PRJNA_iPS RiboPipe-Paper_RiboGL_L2_heldout | $+$0.172 & [$+$0.157, $+$0.183] & $-$0.013 | $+$0.172 & [$+$0.157, $+$0.183] & $-$0.013 | PASS |
| tab:s2l2 | setting 2 Subtelny14_ribomimo RiboPipe-Paper_RiboGL_L2_heldout | $+$0.037 & [$+$0.034, $+$0.041] & $+$0.048 | $+$0.037 & [$+$0.034, $+$0.041] & $+$0.048 | PASS |
| tab:s2l2 | setting 2 Mohammad16 RiboPipe-Paper_RiboGL_L2_heldout | $+$0.025 & [$+$0.018, $+$0.031] & $+$0.081 | $+$0.025 & [$+$0.018, $+$0.031] & $+$0.081 | PASS |
| tab:s2l2 | setting 2 Mohammad19-1 RiboPipe-Paper_RiboGL_L2_heldout | $+$0.029 & [$+$0.023, $+$0.034] & $+$0.040 | $+$0.029 & [$+$0.023, $+$0.034] & $+$0.040 | PASS |
| tab:s2l2 | setting 2 Mohammad19-2 RiboPipe-Paper_RiboGL_L2_heldout | $+$0.018 & [$+$0.012, $+$0.021] & $+$0.031 | $+$0.018 & [$+$0.012, $+$0.021] & $+$0.031 | PASS |
| holm | superiority family seed 123: rejected / m | 22/24 | 22/24 | PASS |
| holm | superiority family 3 seeds: rejected / m | 22/24 | 22/24 | PASS |
| holm | equivalence family seed 123: rejected / m | 0/24 | 0/24 | PASS |
| holm | equivalence family 3 seeds: rejected / m | 2/24 | 2/24 | PASS |
