# Changelog

All notable changes to RiboPipe are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-09

Turns the headline model into an end-to-end, submission-quality toolkit: raw reads to
prediction, biological interpretation, and low-coverage reliability/imputation.

### Added
- **Raw processing** (`ribopipe raw2csv`, `ribopipe.rawcount`): CDS-aligned BAM(s) ->
  P-site assignment (fixed or `--auto-offset` start-codon metagene) -> the codon-count CSV
  that `preprocess` consumes. Optional `[raw]` extra (`pysam`). A standard riboWaltz-style
  step, not a byte-for-byte reproduction of the paper's riboWaltz run.
- **Interpretability** (`ribopipe motifs` / `ribopipe ism`, `ribopipe.interpret`):
  read the exp-motif CNN first-layer filters as E/P/A amino-acid logos, and an end-to-end
  in-silico-mutagenesis A-site codon/AA attribution with a local-vs-ISM agreement score.
- **Low-coverage reliability** (`ribopipe crossover` / `ribopipe impute`,
  `ribopipe.reliability`): the independent-read-split crossover depth D* (reads/codon), the
  model-favoured fraction of expressed genes, the sweep-free `r^-1(m)` inversion, and the
  depth-weighted hybrid estimator (logistic gate in log-depth, slope 0.5, centred at D*).
- Tests for all three (`tests/test_rawcount.py`, `test_interpret.py`, `test_reliability.py`).

### Removed
- **Hand-crafted biological features (bioFeat)** entirely: the `biofeat` preprocessing
  step, `--bio-npz` / `--with-bio` flags, and the `use_bio` code paths. The headline model
  never used them (`use_bio=False`); inputs are codon + NT(+/-15) + structure only. Released
  checkpoints are unaffected.

## [1.1.0] - 2026-09

Headline-model release matching the published paper (RiboPipe). The installable
headline is now the **interpretable motif-CNN + BiGRU-128**, not the BiLSTM, and the
training objective is the paper's unweighted Huber loss.

### Changed
- **Headline architecture** (`ribopipe.model.RiboPipeCNN`, `--backbone cnn`, default):
  a k=7 **exp-motif CNN** (readable first-layer filters) -> k=3 taper conv -> a single
  **bidirectional GRU (h=128)** -> FC(32) -> per-codon head. **~0.35 M parameters**
  (a fifth of RiboMIMO, an eighth of RiboGL). Input per codon = 64-d codon one-hot +
  120-d nucleotide (+/-15 nt) + 3-d local-structure MFE = **187 first-conv channels**;
  hand-crafted biological features are **off by default** (`use_bio=False`), matching
  the paper's `use_bio=False` headline.
- **Default loss is now `huber`** (unweighted, delta=1), replacing `peakmse`.
- `predict` auto-detects the backbone from the checkpoint (CNN `c1.*` vs BiLSTM `l1.*`)
  and infers `bio_dim` from the first-conv shape; the paper's released checkpoints load
  directly via `ribopipe.model.load_cnn_from_paper_checkpoint`.
- README headline numbers updated to the honest **gene-level 5-fold** per-transcript
  Pearson on the four human datasets (TX9_WT 0.598, GSE233886_WT 0.690,
  GSE133393_WT 0.529, PRJNA_iPS 0.518).

### Retained
- The legacy BiLSTM (`--backbone bilstm`) and all feature/loss toggles remain available
  for the paper's ablation rows.

## [0.3.0] - 2026-07

Faithful headline-model release: the installable package now reproduces the paper's
headline configuration (motif-CNN + BiGRU-128, ~0.35M params, unweighted Huber loss)
rather than a codon+bio-only variant.

### Added
- **Nucleotide context features** (`use_nt`): +/-15 nt one-hot window around each A-site
  (120 dims), matching the paper.
- **Local mRNA-structure features** (`use_struct`): 3 ViennaRNA MFE dims per codon, with a
  `ribopipe struct` subcommand to (re)generate the per-nucleotide MFE cache from FASTA.
- **Parameter-free peak-gated loss** (`peakmse`, `ribopipe.losses.huber_peak_mse`): now the
  training default; squared error on peaks (target > tau), robust Huber on the background.
- **Early stopping** on a gene-level validation hold-out (median per-transcript Pearson;
  patience 20, up to 200 epochs).
- **Gene-level 5-fold cross-validation harness** (`ribopipe cv5` / `ribopipe.cv5.run_cv5`),
  the leak-free protocol behind the paper's headline benchmark table.
- **Self-contained baselines** (`ribopipe.baselines`): codon-mean, tri-codon, ridge.
- `LICENSE` (MIT), `tests/` smoke test, GitHub Actions CI, and this changelog.

### Changed
- `RiboDataset` now builds the extended (codon + bio + NT + struct) feature block via
  feature toggles; the same flags reproduce every feature-ablation row in the paper.
- Checkpoints are saved as config dicts (feature flags + dims), so `predict` restores the
  exact training configuration automatically.
- README rewritten around the leak-free **gene-level** evaluation (the paper's honest
  headline); transcript-level ("deployment recall") numbers are labelled as such.

## [0.2.0]
- Initial packaged BiLSTM (codon + biological features, masked Huber loss) and
  preprocessing pipeline (CSV -> NPZ, coverage matrix, biological features).
