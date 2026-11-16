# Verification of RiboPipe v2.0.0 against the paper

The authors ran these checks on 2026-11-16, on the machine that produced the paper's results
(RTX 4090, Python 3.11.15, PyTorch 2.1.2+cu121, NumPy 1.26.4, ViennaRNA 2.7.0). Every check compares this
release with the original code, data and stored outputs of the paper. TX9_WT, which is withheld from `data/`, was
included through a local copy of the full release data.

| Area | What was compared | Result | Report |
|---|---|---|---|
| Data | every array of every dataset and population (ids and order, sequences, counts, replicate counts, MFE tracks, valid windows, fold-wise MFE normalisation, units, folds) against the paper's loader | 10/10 populations identical | `core_equiv_data_ckpt.json` |
| Checkpoints, GPU | paper checkpoints (RiboPipe, RiboPipe (−MFE), BiLSTM-97) loaded with `ribopipe.train.load_ckpt`; test predictions | 6/6 bit-identical | `core_equiv_data_ckpt.json` |
| Checkpoints, CPU | same on CPU | per-transcript Pearson differs by ≤ 6.4e-4, fold medians by ≤ 7.3e-5 | `core_equiv_data_ckpt.json` |
| Retraining | `ribopipe train` with seed 123 (Mohammad16 fold 0, GSE133393_WT fold 0) | bit-identical predictions; same selected epoch, validation value and optimiser steps | `core_equiv_retrain.json` |
| Setting 3 | parameter count and structure of 35 variants; full variant vs RiboPipe; 7 paper variant checkpoints | 35/35; bit-identical; 7/7 bit-identical | `core_ablation_equiv.json` |
| `prepare` rules | fold rule on the paper's 17,977 human units; non-human folds rebuilt from sequence (31-mer units); MFE tracks recomputed with ViennaRNA; fold-wise MFE normalisation | all identical (normalisation within 1e-14) | `core_prepare_equiv.json` |
| Part I evaluation | settings 1 and 2, three seeds, Holm families, against the paper's evaluation JSONs; Supplementary Tables of Part I rebuilt | 23,340 leaves identical; 460/460 table cells identical | `evaluate_verification.md` |
| Baselines | fetch and patch of the official code (commits and md5); smoke runs of every wrapper; in-house baselines on 8 datasets x 5 folds | 215 PASS, 0 FAIL; in-house predictions bit-identical | `baselines_verification.md` |
| Part II | D*, intervals, per-bin values, noise-control bins, fractions below D*, posterior and its decision rule, prior strength, controlled-thinning ranking, from the paper's stored predictions | 589/589 identical | `lowcov_verification.md` |
| Part III | single-codon and single-nucleotide edits, controls, sensitivity maps, branch routing, window transplants, sub-windows, from the paper's checkpoints (all four human datasets) | 8,108/8,108 identical (4 descriptive labels excluded) | `interpret_verification.md` |
| Figures | Fig. 1–4 rebuilt from `paper_results/` | all drawn values identical; renders pixel-identical | `figures_verification.md` |

The scripts are `tests/paper_equivalence.py`, `tests/ablation_equivalence.py`, `tests/prepare_equivalence.py` and
`tests/verification/*_verify.py`. They need the paper's original analysis directory and are kept to document the
checks. The unit tests (`pytest`) need no paper data.

Known differences:
- Predictions on CPU differ from the GPU predictions at the floating-point level (see the table).
- The activation logos of Fig. 4a must be computed on the GPU, as in the paper. On CPU, near-tied windows can be
  reordered at the top-1% cut.
- Retraining is bit-identical only on the same hardware and software stack. Elsewhere, expect seed-level variation
  (the paper reports three seeds).
