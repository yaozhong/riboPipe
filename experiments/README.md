# Paper experiments → scripts index

This maps each part of the paper to how it is reproduced from this package.

| Paper part | What it shows | Reproduce with |
|---|---|---|
| **Part 1** — benchmark (gene-level 5-fold CV) | RiboPipe = best per-transcript accuracy at ~0.35 M params | `reproduce/run_cv5.sh` (train) or `reproduce/predict_with_checkpoint.py` (from a released checkpoint) |
| **Part 2** — low-coverage reliability | crossover depth D\*≈0.22–0.57 reads/codon; depth-weighted hybrid | independent-read-split protocol (below) |
| **Part 3** — interpretability | readable E/P/A motif filters; motif-vs-ISM disagreement flags context-driven peaks | `experiments/motif_filters.py` (filters); ISM = model forward under A-site codon substitution |

## Part 2 — crossover / reliability / hybrid

For a sample, score three estimators — the sequence prediction, the down-sampled raw
reads, and a depth-weighted blend — against an **independent read partition** of the same
transcript (a binomial read split, or a biological replicate). The depth at which the
prediction overtakes the reads is the crossover D\*; below it the model is the more
reliable estimate. Inputs are counts + sequence only. The analysis operates on the
per-transcript predictions produced in Part 1.

## Part 3 — interpretability

Two readouts of the *same* trained model:

- **Motif filters** (local): the first-layer k=7 convolutions read out directly as
  amino-acid sequence logos over the E/P/A register. `experiments/motif_filters.py`
  extracts them from any headline checkpoint (no data needed).
- **In-silico mutagenesis / CAS** (global): substitute each of the 61 sense codons at the
  A-site *in situ* and re-predict; the context-averaged change is the codon attribution.
  Adjudicated against the read-derived empirical A-site dwell, ISM (not the filters) is the
  quantitative account of elongation biology; where the two disagree flags occupancy the
  local codon window cannot explain (candidate context-driven pauses).

The exact figure-generation scripts used for the manuscript depend on the internal
Ribo-seq data layout and are available from the authors on request.
