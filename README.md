# RiboPipe

*An interpretable sequence-to-occupancy model for reliability-aware imputation of
low-depth ribosome profiling.*

**Within-sample Ribo-seq imputation**: learn a sequence-to-occupancy mapping from a
sample's own high-coverage transcripts and use it to recover codon-resolution ribosome
pause profiles for the same sample's sparse transcripts.

## Installation

```bash
# from GitHub (pin the release tag)
pip install "git+https://github.com/yaozhong/riboPipe.git@v1.2.1"

# or from a clone (editable), with the optional extras you need
git clone https://github.com/yaozhong/riboPipe
cd riboPipe
pip install -e ".[struct,raw]"
```

This installs the **`ribopipe`** command-line tool and the `ribopipe` Python package
(headline `RiboPipeCNN`, training / 5-fold CV / prediction, and the baselines).
Requirements: Python ≥ 3.8, PyTorch ≥ 1.12, NumPy, Pandas, SciPy, scikit-learn,
Biopython, matplotlib. Optional extras: **`[struct]`** (`ViennaRNA>=2.5`) to (re)generate
the local-structure MFE cache; **`[raw]`** (`pysam`) for `raw2csv` (BAM → codon-count CSV);
**`[test]`** (`pytest`). Training/prediction on an existing CSV + cache import neither.

The four pre-trained headline checkpoints are in `checkpoints/` in the repository (not
shipped in the pip wheel); load one with
`ribopipe.model.load_cnn_from_paper_checkpoint(path)`.

## Data processing

RiboPipe consumes **per-codon Ribo-seq counts**, not raw reads: the entry point is a
codon-level counts CSV plus a CDS FASTA. The preprocessing that turns these into
model-ready tensors is shared across all datasets in the paper and ships with the package
under `ribopipe/preprocess/`.

**Input CSV schema** — one row per codon, validated by `ribopipe.preprocess.schema`:

| Column | Meaning |
|---|---|
| `transcript` | transcript ID (matches the FASTA header) |
| `start`, `end` | codon coordinates |
| `from_cds_start`, `from_cds_stop` | codon offset from the CDS start / stop |
| `region` | `5UTR` / `CDS` / `3UTR` |
| one column **per sample** | per-codon read count for that sample |

Sample columns follow `Celltype|genotype|treatment|replicate|type|remarks`
(e.g. `HEK293T|WT|DMSO|rep1|mono|x`); the legacy `<condition>_repN` form is also accepted.

**From raw reads.** If you start from aligned reads rather than a counts CSV,
`ribopipe raw2csv` turns **CDS-aligned BAM(s)** (one per sample) into the codon-count CSV
above by standard P-site assignment (fixed offset, or `--auto-offset` from the start-codon
metagene). It needs the `[raw]` extra (`pip install -e ".[raw]"`, pulls in `pysam`).

```bash
ribopipe raw2csv \
  --bam sampleA.bam sampleB.bam \
  --sample "HEK293T|WT|DMSO|rep1|mono|x" "HEK293T|WT|DMSO|rep2|mono|x" \
  --fasta transcripts_cds.fa --out-csv codon_counts.csv --auto-offset
```

> The paper's human data were P-site-assigned upstream with **riboWaltz** (R). `raw2csv`
> re-implements that standard step so the toolkit is runnable end-to-end from BAM; it is
> **not** a byte-for-byte reproduction of the paper's riboWaltz run. For the published
> numbers, use the released count CSVs / checkpoints.

**Preprocessing chain** (the three steps of the Quick start below):

1. `ribopipe preprocess` — CSV (+ CDS FASTA) → per-transcript NPZ (`cds.sequence`, `cds.avg_count`).
2. `ribopipe matrix` — the NPZ directory → a transcript × sample coverage matrix, used for the high-/low-coverage split.
3. `ribopipe struct` — the ViennaRNA local-structure (MFE) feature cache.

## Quick start (one sample, end to end)

```bash
# 1. Preprocess raw CSV to per-transcript NPZ
ribopipe preprocess --csv codon_counts.csv --fasta transcripts_cds.fa --out-dir ./npz

# 2. Build transcript × sample coverage matrix
ribopipe matrix --npz-dir ./npz --out-csv coverage_matrix.csv

# 3. Precompute the ViennaRNA local-structure (MFE) cache  (one-time per dataset)
ribopipe struct --npz ./npz/my_sample.npz
#   -> ./npz/struct_cache/my_sample_struct.npz

# 4. Train the headline model
ribopipe train \
  --npz ./npz/my_sample.npz \
  --struct-npz ./npz/struct_cache/my_sample_struct.npz \
  --coverage-csv coverage_matrix.csv \
  --sample my_sample \
  --enst2ensg reproduce/enst2ensg_grch38.json.gz \
  --out-dir ./model

# 5. Predict pause profiles for all transcripts
ribopipe predict \
  --checkpoint ./model/ribopipe_model.pt \
  --npz ./npz/my_sample.npz \
  --struct-npz ./npz/struct_cache/my_sample_struct.npz \
  --out-csv predictions.csv
```

The checkpoint stores its own feature configuration, so `predict` restores the exact
training setup automatically. Pass `--no-struct` at step 4 (and drop `--struct-npz`) for
the codon+NT ablation that needs no ViennaRNA.

See [`examples/run_ribopipe.sh`](examples/run_ribopipe.sh) for a complete annotated script.

## Reproduce the headline benchmark

```bash
NPZ=./npz/my_sample.npz \
STRUCT=./npz/struct_cache/my_sample_struct.npz \
bash reproduce/run_cv5.sh
```

Gene-level 5-fold CV of the headline model against the baselines (codon-mean, tri-codon,
ridge, BiLSTM-base). See [`reproduce/README.md`](reproduce/README.md).

## Interpretability

RiboPipe is interpretable in two complementary ways.

**Local — motif filters.** The k=7 exp-motif CNN's first-layer filters read out directly as
E/P/A amino-acid sequence logos (e.g. P-site proline, acidic/aromatic A-site residues).
No data needed — weights only:

```bash
ribopipe motifs --checkpoint checkpoints/ribopipe_headline_TX9_WT.pt \
  --top 3 --out-csv motif_weights.csv --out-png motif_logos.png
```

**Non-local — in-silico mutagenesis (ISM).** Substituting the A-site codon with all 64
codons and re-running the full network gives a per-codon / per-amino-acid A-site
attribution — the model-derived counterpart of an empirical A-site dwell profile:

```bash
ribopipe ism --checkpoint checkpoints/ribopipe_headline_TX9_WT.pt \
  --npz ./npz/my_sample.npz --struct-npz ./npz/struct_cache/my_sample_struct.npz \
  --out-csv asite_attribution.csv
```

The Python API exposes `ribopipe.interpret.{filter_aa_matrices, motif_report,
asite_codon_attribution, motif_vs_ism}`; `motif_vs_ism` scores where the local and ISM
readouts agree, and disagreement flags occupancy the local codon window cannot explain.

## Low-coverage reliability (when to trust the model)

RiboPipe says not only *what* the profile is but *when* to trust it over the reads.

**Crossover D\*.** An independent read-split (disjoint halves of a transcript's reads) scores
the model and the down-sampled reads against a held-out reference; the depth where the reads
overtake the model is the crossover **D\* ≈ 0.22–0.57 reads/codon**. Below it the model is the
more reliable estimator. `crossover` also reports the model-favoured fraction of expressed
genes and locates D\* by the sweep-free `r⁻¹(m)` inversion:

```bash
ribopipe crossover --checkpoint checkpoints/ribopipe_headline_TX9_WT.pt \
  --npz ./npz/my_sample.npz --struct-npz ./npz/struct_cache/my_sample_struct.npz \
  --out-json crossover.json
```

**Depth-weighted imputation.** The drop-in low-coverage estimator blends prediction and reads
with a logistic gate in log-depth centred at D* (`w = sigmoid((ln rc − ln D*)/0.5)`): mostly
model where coverage is sparse, mostly reads where it is deep. It matches or exceeds either
source alone at every depth.

```bash
ribopipe impute --checkpoint checkpoints/ribopipe_headline_TX9_WT.pt \
  --npz ./npz/my_sample.npz --struct-npz ./npz/struct_cache/my_sample_struct.npz \
  --out-csv imputed_profiles.csv          # --dstar is estimated if not given
```

Python API: `ribopipe.reliability.{estimate_dstar, fraction_below, invert_dstar,
depth_weight, hybrid_profile, impute}`. Everything is covered-mean-normalised Pearson; the
read split is binomial by default (repsplit where genuine replicates exist).

## Python API

```python
import ribopipe

# Train the headline model on a set of high-coverage transcript IDs
model = ribopipe.train_on_ids(
    "sample.npz", train_ids, val_ids=val_ids,
    struct_npz_path="struct_cache/sample_struct.npz",
    use_nt=True, use_struct=True,   # headline: codon + NT(+/-15) + struct MFE
    loss_name="huber",              # unweighted Huber (delta=1)
)   # backbone="cnn" (default) = motif-CNN k=7 + BiGRU-128 (h=128), ~0.35M params

# Predict
preds = ribopipe.predict(
    model, "sample.npz", test_ids,
    struct_npz_path="struct_cache/sample_struct.npz",
    use_nt=True, use_struct=True,
)  # dict: transcript_id -> np.ndarray (pause scores, len = CDS codons)

# Gene-level 5-fold CV
summary = ribopipe.run_cv5(
    "sample.npz", all_ids,
    enst2ensg_path="reproduce/enst2ensg_grch38.json.gz",
    struct_npz_path="struct_cache/sample_struct.npz",
    methods=["ribopipe", "bilstm_base", "tricodon"],
)
```

## Design

RiboPipe frames Ribo-seq imputation as **within-sample learning**:

1. **T_high** (top-coverage transcripts) — used for training and evaluation.
2. **T_low** (the sparse remainder of the transcriptome) — prediction targets.

All model-selection and benchmarking splits are **gene-level** to avoid isoform leakage.

An **independent-read-split** analysis identifies a crossover depth
**D\* ≈ 0.22–0.57 reads/codon**, below which the model's prediction is more reliable than
a transcript's own sparse reads — a regime that covers **52–82 % of expressed genes**. This
is the *reliability-aware* part of the title: RiboPipe reports not only what the pause
profile is, but when to trust it over the raw reads. A depth-weighted hybrid of prediction
and reads then matches or exceeds either source alone at every depth tested.

## Data availability

Public datasets used in the paper: GEO **GSE133393** and **GSE233886**, and BioProject
**PRJNA976655**. The in-house HEK293 dataset (TX9_WT) will be deposited in GEO upon
publication, with reviewer access available on request.

## License

MIT — see [LICENSE](LICENSE).
