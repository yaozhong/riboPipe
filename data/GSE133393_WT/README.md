# GSE133393_WT benchmark dataset (public: GSE133393)

Processed per-codon ribosome-profiling data for **GSE133393_WT** (HEK293, WT), re-processed by the
authors' pipeline and packaged so peers can reproduce the GSE133393_WT column of the headline
gene-level 5-fold benchmark (Table 1 / Figure 1) from this repository alone. The raw data
are public (GSE133393; Han et al. 2020); this release redistributes the derived per-codon count matrices
only.

## Scope
Benchmark universe (gene-longest, high-coverage `T_high`): **1088 transcripts** referenced by
the frozen folds; **1056** are scoreable and enter the reported metrics (matches the paper).
Full-transcriptome and per-replicate data are obtainable from the original public accession.

## Files
| file | description |
|---|---|
| `gse133393_wt_benchmark.npz` | per-transcript data; `numpy.load(path, allow_pickle=True)`, `z[tid].item()` -> record. |
| `gse133393_wt_benchmark_index.csv` | index: transcript_id, gene_id, n_codons, cds_start/end, sum_avg_count, sum_avg_count_norm, cv5_test_fold. |

Record schema and target definition are identical to `data/TX9_WT/README.md`
(ship raw `avg_count`; the covered-mean `log(1+mu)` target `covmean0_log` is applied by the code).

## Reproduce
```bash
ribopipe cv5 \
  --npz       data/GSE133393_WT/gse133393_wt_benchmark.npz \
  --enst2ensg reproduce/enst2ensg_grch38.json.gz \
  --folds     reproduce/folds/cv5_folds_GSE133393_WT.json \
  --methods   codon_mean,ribopipe --out-json gse133393_wt_cv5.json
```
Verified: `codon_mean` gives per-transcript Pearson **P = 0.1553** with the paper's
`n_per_fold` (sum 1056), identical to the canonical harness.
