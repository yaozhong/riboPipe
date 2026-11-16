# Reproducing the results of the manuscript

[← back to the README](../README.md)

This page lists the commands that reproduce every analysis of the accompanying manuscript from this repository,
and how the release was checked against the original analysis. Runs use the layout
`<runs>/<config>/<dataset>/seed<k>/fold<f>/` (`pred.npz`, `model.pt`, `train_log.json`, `run.json`).

## Setup

```bash
pip install ".[mfe]"              # ViennaRNA is needed by the model analyses (folding energies are recomputed after edits)
python tools/download_mfe.py      # MFE tracks of the released datasets
bash baselines/fetch_upstream.sh  # official code of the published methods, pinned and patched (see baselines/README.md)
```

The analyses were computed with Python 3.11, PyTorch 2.1.2 (CUDA 12.1), NumPy 1.26 and ViennaRNA 2.7.0; the exact
versions are in [`requirements/paper-stack.txt`](../requirements/paper-stack.txt) (RiboExp and RiboGL:
[`requirements/gnn-baselines.txt`](../requirements/gnn-baselines.txt)). The baselines also need
`pip install ".[baselines]"` (scikit-learn, for RiboMIMO). The default training seed is 123. The prespecified seeds
are 123, 124 and 125.

## Commands

| Analysis | What | Command |
|---|---|---|
| Benchmark, setting 1 | RiboPipe and its variants, all datasets, five folds | `ribopipe cv5 --config Paper_RiboPipe_MFE_H128 --dataset D --out runs/` (`--seed 124`, `125` for the three-seed check) |
| | published methods (iXnos, RiboMIMO, RiboExp, RiboGL) and in-house references (codon-mean, tri-codon, ridge, RUST) | `python -m baselines.run --config Paper_RiboMIMO --dataset D --out runs/` (`python -m baselines.run --list`) |
| Benchmark, setting 2 | comparators retrained on the scored relative density | `python -m baselines.run --config Paper_RiboMIMO_L2 ...` (also `Paper_RiboGL_L2_heldout`, `Paper_iXnos_nocut_lr005_elig`) |
| Benchmark, setting 3 | component removal and replacement, loss comparison, training-set size, branch lattice | `ribopipe ablation --list`; `ribopipe ablation --config Letter_E2_BiLSTM_pm --dataset D --out runs/` |
| Benchmark, evaluation | per-transcript Pearson, Spearman, top-5% recall, paired similarity-unit bootstrap, 24-comparison Holm family, setting 2, three seeds | `ribopipe evaluate-partI --runs runs/ --out eval/ --what all`; single comparisons: `ribopipe score`, `ribopipe compare` |
| Low coverage | out-of-fold predictions for all low-coverage transcripts | `ribopipe lowcov-predict --runs runs/ --dataset D --out preds/` |
| | agreement with the other replicate across depth, crossover depth D\*, noise-predictor control | `ribopipe crossover --dataset D --pred RiboPipe=preds/... --out cross/`; `ribopipe crossover-summary` |
| | posterior with the prediction as prior | `ribopipe posterior-kappa`, `ribopipe posterior`, `ribopipe posterior-summary` |
| | ranking high-density codons under controlled thinning | `ribopipe peak-ranking --dataset D --oof runs/Paper_RiboPipe_MFE_H128/D/seed123 --out a1.json` |
| Model analyses | single-codon and single-nucleotide edits, sensitivity maps | `ribopipe ism`, `ribopipe ism-map` (`--ckpt-dir runs/Paper_RiboPipe_MFE_H128/D/seed123`) |
| | branch routing | `ribopipe routing` |
| | window transplantation and sub-windows | `ribopipe transplant [--subwindows]` |
| | activation logos (an inspection tool; compute on GPU) | `ribopipe logos --device cuda:0` |
| Main figures 1–4 | redrawn from the result files in `paper_results/` | `bash reproduce/figures/make_all.sh out/` |

`ribopipe <command> --help` lists the options. Analyses of TX9_WT need its count matrices, which are released on
publication.

Approximate times on one RTX 4090 and 64 CPU cores:
- one RiboPipe fold: about 2 minutes;
- the low-coverage crossover of one dataset (all variants): about 1 hour on CPU;
- the model analyses of one human dataset: about 45 minutes on CPU.

## Verification

This release was checked against the code, data and stored outputs of the original analysis, on the same machine:
- The data loader returns the original arrays.
- The original checkpoints reproduce their GPU predictions bit for bit.
- Retraining with seed 123 reproduces the original runs bit for bit.
- The benchmark evaluation reproduces all 460 cells of the supplementary benchmark tables.
- The low-coverage analysis (589 quantities) and the model analyses (8,108 quantities) reproduce exactly.
- Figures 1–4 are pixel-identical.

Details: [`tests/verification/SUMMARY.md`](../tests/verification/SUMMARY.md). The equivalence scripts in `tests/`
need the original analysis directory and document the checks. The unit tests run anywhere:
`pip install ".[test]" && pytest`.

## Tolerances

Retraining is bit-identical only on the same hardware and software stack. On other GPUs or PyTorch/NumPy versions,
the results differ at the level of training noise. Use these criteria:

| Quantity | Criterion | Basis |
|---|---|---|
| validation and test Pearson | within about 0.01 of the reference | across seeds 123–125, the five-fold score of RiboPipe ranged by 0.002–0.007 per dataset (`tests/verification/seed_range_RiboPipe.json`) |
| selected epoch | within a few epochs | early stopping with patience 5 |
| D\* (`ribopipe crossover`) | same status; for `OK`, the point estimate lies inside the reference's 95% bootstrap interval | see below |
| routing gate (`ribopipe routing`) | largest \|local − full\| ≤ 1e-5 (reported as `max_abs_diff`) | see below |

**D\*.** D\* is a crossing point of two curves, so it moves with the predictions and with the random thinning. With
the predictions fixed, ten thinning seeds gave, for RiboPipe on GSE233886_WT, a crossover in nine (D\* 0.701–0.732
reads/codon) and `NOT_LOCATABLE` in one; on GSE133393_WT all ten gave `NO_SIGN_CHANGE`
(`paper_results/partII/thinning_seeds.json`). `ribopipe crossover` uses thinning seed 0, as in the manuscript. The
reference values are:

| Dataset | Status | D\* (reads/codon) | 95% bootstrap interval |
|---|---|---|---|
| GSE233886_WT | `OK` | 0.732 | 0.684–0.786 |
| GSE133393_WT | `NO_SIGN_CHANGE` | – | – |

Other datasets and predictors: `paper_results/partII/partII_eval.json`. The criterion checks that a retrained model
reaches the same conclusion; it does not test whether two D\* values are equal.

**Routing gate.** `ribopipe routing` recomputes only the part of the BiGRU that an edit changes. On the stack above
this equals the full forward pass bit for bit. Other PyTorch builds can order float32 operations differently; an
independent test with PyTorch 2.14 found differences up to 9.5e-7. The gate therefore passes when the largest
difference is ≤ 1e-5 (in log1p relative density), reports it as `max_abs_diff`, and reports `bit_exact`. For scale,
the median effect of a codon substitution is about 0.02, and perturbing every effect by up to 1e-5 changed the route
shares by less than 2e-7. `--gate-tol 0` requires bit-identical results.
