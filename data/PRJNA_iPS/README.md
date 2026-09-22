# PRJNA_iPS benchmark dataset (public: PRJNA976655 / GSE233555)

Processed per-codon ribosome-profiling data for **PRJNA_iPS** (iPSC (409B2)), re-processed by the
authors' pipeline and packaged so peers can reproduce the PRJNA_iPS column of the headline
gene-level 5-fold benchmark (Table 1 / Figure 1) from this repository alone. The raw data
are public (PRJNA976655 / GSE233555; Tomuro et al. (calibrated)); this release redistributes the derived per-codon count matrices
only.

## Scope
Benchmark universe (gene-longest, high-coverage `T_high`): **1656 transcripts** referenced by
the frozen folds; **1618** are scoreable and enter the reported metrics (matches the paper).
Full-transcriptome and per-replicate data are obtainable from the original public accession.

## Files
| file | description |
|---|---|
| `prjna_ips_benchmark.npz` | per-transcript data; `numpy.load(path, allow_pickle=True)`, `z[tid].item()` -> record. |
| `prjna_ips_benchmark_index.csv` | index: transcript_id, gene_id, n_codons, cds_start/end, sum_avg_count, sum_avg_count_norm, cv5_test_fold. |

Record schema and target definition are identical to `data/TX9_WT/README.md`
(ship raw `avg_count`; the covered-mean `log(1+mu)` target `covmean0_log` is applied by the code).

## Reproduce
```bash
ribopipe cv5 \
  --npz       data/PRJNA_iPS/prjna_ips_benchmark.npz \
  --enst2ensg reproduce/enst2ensg_grch38.json.gz \
  --folds     reproduce/folds/cv5_folds_PRJNA_iPS.json \
  --methods   codon_mean,ribopipe --out-json prjna_ips_cv5.json
```
Verified: `codon_mean` gives per-transcript Pearson **P = 0.1277** with the paper's
`n_per_fold` (sum 1618), identical to the canonical harness.
