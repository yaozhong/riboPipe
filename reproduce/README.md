# Reproducing the paper

This folder reproduces the paper's **leak-free, gene-level 5-fold cross-validation**
headline — the honest evaluation of the RiboPipe headline model
(**motif-CNN k=7 + BiGRU-128**, `ribopipe.model.RiboPipeCNN`, ~0.35 M parameters,
covered-mean-norm `log(1+µ)` target, unweighted Huber loss).

## What is here

| File | Purpose |
|------|---------|
| `run_cv5.sh` | End-to-end: gene-level 5-fold CV of the headline model + baselines on one dataset (trains from scratch). |
| `predict_with_checkpoint.py` | **No training**: score a released headline checkpoint (`../checkpoints/*.pt`) on a held-out transcript list. |
| `enst2ensg_grch38.json.gz` | ENST→ENSG map (GENCODE/GRCh38) that keeps every isoform of a gene inside one fold. |

## Datasets (paper Data Availability)

All are re-processed identically to per-codon P-site counts (human against GENCODE v47):

| Label in paper | Source |
|---|---|
| GSE133393_WT (HEK293)   | NCBI GEO **GSE133393** |
| GSE233886_WT (HEK293F)  | NCBI GEO **GSE233886** (DMSO/WT control) |
| PRJNA_iPS (iPSC 409B2)   | NCBI BioProject **PRJNA976655** / GEO **GSE233555** |
| TX9_WT (in-house HEK293) | deposited on publication (reviewer access on request) |

## Expected headline numbers (gene-level 5-fold, median per transcript)

| Dataset | Pearson *r* | Top-5 % peak recall |
|---|---:|---:|
| TX9_WT         | 0.598 | 0.405 |
| GSE233886_WT   | 0.690 | 0.500 |
| GSE133393_WT   | 0.529 | 0.345 |
| PRJNA_iPS      | 0.518 | 0.329 |

> **On the ~0.98 transcript-level number.** A plain transcript-level split leaks
> (isoforms of one gene are near-identical, so a held-out isoform's twin is in
> training). That regime measures *deployment recall*, is a different quantity, and the
> paper never leads with it. Gene-level folding is the honest headline above.

## A. Reproduce from a released checkpoint (fast, no GPU training)

The four headline checkpoints in [`../checkpoints/`](../checkpoints) are the exact
weights behind the table. Preprocess a dataset to the package NPZ format, then score:

```bash
pip install "git+https://github.com/yaozhong/riboPipe.git"     # or: pip install -e ".[struct]"

# one-time per dataset (from raw codon-count CSV + CDS FASTA):
ribopipe preprocess --csv counts.csv --fasta cds.fa --out-dir ./npz
ribopipe struct    --npz ./npz/DATASET.npz          # ViennaRNA MFE cache

python reproduce/predict_with_checkpoint.py \
    --checkpoint checkpoints/ribopipe_headline_TX9_WT.pt \
    --npz ./npz/DATASET.npz \
    --struct-npz ./npz/struct_cache/DATASET_struct.npz \
    --ids test_ids.txt
```

## B. Retrain the headline benchmark (gene-level 5-fold CV)

```bash
ribopipe struct --npz /path/to/DATASET.npz          # one-time MFE cache

NPZ=/path/to/DATASET.npz \
STRUCT=/path/to/struct_cache/DATASET_struct.npz \
bash reproduce/run_cv5.sh
```

Per-method mean ± SD across folds (Pearson / Spearman / peak recall@5 % / Jaccard) is
printed and written to `reproduce/cv5_result.json`. The headline uses `--backbone cnn
--loss huber`; the paper's ablation rows are recovered with the feature toggles
(`--no-nt`, `--no-struct`) and `--backbone bilstm`.

## Low-coverage crossover, hybrid and interpretability

The crossover / reliability protocol (Part 2) and the motif-vs-ISM interpretability
analyses (Part 3) build on these predictions; their analysis scripts and figure code
are under [`../experiments/`](../experiments) with a per-figure index.
