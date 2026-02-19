# RiboPipe

**RiboPipe** is a lightweight end-to-end pipeline for processing
Ribo-seq P-site count data and training codon-level translation
prediction models.

------------------------------------------------------------------------

## Installation

``` bash
wget ribopipe_mvp.zip
unzip ribopipe_mvp.zip

cd ribopipe_mvp
pip install -e .
```

------------------------------------------------------------------------

## Workflow Overview

    Raw P-site CSV
          │
          ▼
    ribopipe preprocess
          │
          ▼
    Per-sample transcript NPZ files
          │
          ├── ribopipe matrix  → coverage matrix (transcript × sample)
          │
          ├── ribopipe biofeat → biological feature file (.npz)
          │
          ▼
    ribopipe train_pipeline → codon-level prediction model

------------------------------------------------------------------------

## Usage

### Step 1: Preprocess Ribo-seq CSV

``` bash
ribopipe preprocess   --csv /mnt/ws1/ribo_seq_work/inada_lab/Psite_data_human/Psite_human/GSE233886_HEK293F_Psite_rawcount.csv   --fasta /mnt/ws1/ribo_seq_work/inada_lab/software/ribo-impute/data/gencode.v47.transcripts.fa   --out-dir GSE233886_out   --fasta-cache GSE233886_out/fasta_cache.pkl
```

------------------------------------------------------------------------

### Step 2: Generate Coverage Matrix

``` bash
ribopipe matrix   --npz-dir GSE233886_out   --out-csv GSE233886_out/coverage_matrix_transcript_x_sample.csv
```

------------------------------------------------------------------------

### Step 3: Generate Biological Features

``` bash
ribopipe biofeat   --cds-npz GSE233886_out/GSE233886_HEK293F_Psite_rawcount__HEK293F_WT_DMSO.npz   --trna-json /mnt/ws1/ribo_seq_work/inada_lab/software/ribo-impute/data/trna_copy_numbers.json   --out-npz GSE233886_out/bio_features.npz
```

------------------------------------------------------------------------

### Step 4: Train Codon-Level Prediction Model

``` bash
ribopipe train_pipeline   --coverage-csv GSE233886_out/coverage_matrix_transcript_x_sample.csv   --npz-dir GSE233886_out   --bio-feat GSE233886_out/bio_features.npz   --threshold P75   --epochs 100   --train-split 1.0   --max-codons 1000
```

------------------------------------------------------------------------

## Key Parameters

  Parameter       Description
  --------------- ------------------------------------------------
  --threshold     Select high-expression transcripts (P75 / P95)
  --train-split   Fraction of transcripts used for training
  --epochs        Number of training epochs
  --max-codons    Maximum CDS length (truncated/padded)

------------------------------------------------------------------------

## Output Structure

    GSE233886_out/
    ├── *.npz
    ├── coverage_matrix_transcript_x_sample.csv
    ├── bio_features.npz
    ├── model_checkpoint.pt
    └── predictions/

------------------------------------------------------------------------

## License

MIT License
