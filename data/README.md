# Released data

The analysis input of the paper (PaperData_v1.0, manifest MD5 `63080cc9a3144c83834f938560ba6b18`) reduced to the
transcripts of the paper's populations. `tools/build_release_data.py` made these files from PaperData_v1.0 and
`data/RELEASE_MANIFEST.json` records the MD5 of every file and of every source file.
`tests/paper_equivalence.py` checks that `ribopipe.data` returns the same arrays, in the same order, as the paper's
loader.

| Dataset | Organism, cells | Source | Transcripts (Part I / Part II) | Counts |
|---|---|---|---|---|
| TX9_WT (`DMSO_TX9P1`) | human, HEK293T | in-house, GEO GSE350848 | 3,144 / 15,090 | released on publication |
| GSE233886_WT | human, HEK293 Flp-In T-REx | Schneider-Poetsch et al. 2025 | 1,825 / 14,446 | two replicates |
| GSE133393_WT | human, HEK293 | Han et al. 2020 | 1,289 / 14,893 | two replicates |
| PRJNA_iPS | human, iPSC 409B2 | Tomuro et al. 2024 (PRJNA976655) | 753 / 14,904 | two replicates |
| Subtelny14 (`Subtelny14_ribomimo`) | *S. cerevisiae* | GSE53268, matrix released with RiboMIMO | 3,921 | released average |
| Mohammad16 | *E. coli* | GSE72899, matrix released with RiboMIMO | 1,179 | mean of 3 |
| Mohammad19-1, Mohammad19-2 | *E. coli* | GSE119104, matrices released with RiboMIMO | 1,426, 1,598 | released average |

Per dataset directory:

- `<tag>.npz`: transcript ids, A-site-aligned CDS sequences and per-codon A-site counts, concatenated with offsets.
  Human datasets store the two replicates (`reps`, int32); the per-codon count used for training and as the truth
  is their mean. Non-human datasets store the released average (`avg_count`).
- `universe_partI.json` and `universe_partII.json` (human), or `universe.json` (non-human): the transcripts of
  each population. Part I contains the high-coverage transcripts (coverage > 0.5 reads per nucleotide). Part II
  contains all transcripts with at least one read, split into `strata.high` and `strata.low`.
- `folds_partI.json` and `folds_partII.json` (or `folds.json`): test and validation ids of each of the five
  folds. Training is the universe minus test minus validation.
- `units.json`: the sequence-similarity unit of each transcript. A unit stays on one side of every split.
- `mfe_valid.json`: number of valid 30-nt MFE windows (len(seq) - 29).

Global files:

- `MFE_NORM_FOLDWISE.json`: mean and SD of the MFE over the valid windows of each fold's training transcripts.
- `REFERENCE_PROTOCOL.json`: rules of the replicate reference.

The MFE tracks themselves are release assets (`python tools/download_mfe.py`, see `mfe/README.md`).

**TX9_WT.** The in-house HEK293T data (GEO GSE350848) are private until publication. Their count matrices will be
added to this directory on publication, and analyses of TX9_WT need them. The two global JSON files above already
contain TX9_WT summary statistics (MFE normalisation constants and reference-depth summaries) but no per-codon
counts.
