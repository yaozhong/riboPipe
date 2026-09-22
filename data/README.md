# Benchmark datasets

Processed per-codon ribosome-profiling data for the four human datasets used in the paper,
packaged so the headline gene-level 5-fold benchmark (Table 1 / Figure 1) reproduces from this
repository alone. Each folder ships a compact per-transcript `.npz`, a human-readable index
`.csv`, and a `README.md` with the record schema and a one-line reproduction command.

| dataset | cell type | source | transcripts | scored | codon_mean P |
|---|---|---|---|---|---|
| `TX9_WT` | HEK293 WT/DMSO | in-house (raw to be deposited in GEO on publication) | 3,257 | 3,101 | 0.1629 |
| `GSE233886_WT` | HEK293F WT/DMSO | public GSE233886 | 1,704 | 1,643 | 0.1391 |
| `GSE133393_WT` | HEK293 WT | public GSE133393 | 1,088 | 1,056 | 0.1553 |
| `PRJNA_iPS` | iPSC 409B2 | public PRJNA976655 / GSE233555 | 1,656 | 1,618 | 0.1277 |

"transcripts" = frozen-fold universe (gene-longest, high-coverage); "scored" = transcripts with a
defined per-transcript metric that enter the reported numbers. Each shipped `codon_mean` value was
verified to reproduce the canonical harness bit-for-bit.

Reproduce any dataset:

```bash
ribopipe cv5 \
  --npz       data/<TAG>/<tag>_benchmark.npz \
  --enst2ensg reproduce/enst2ensg_grch38.json.gz \
  --folds     reproduce/folds/cv5_folds_<TAG>.json \
  --methods   codon_mean,ribopipe
```

The shipped data are raw per-codon P-site counts plus CDS (and 5'UTR) sequence; the covered-mean
`log(1+mu)` training target (`covmean0_log`) is applied inside `ribopipe`, so no transform is hidden
in preprocessing. Per-replicate raw data (for the reliability-crossover analysis) and the full
transcriptome (for the coverage filter) come from the original accessions / the GEO deposit on
publication.
