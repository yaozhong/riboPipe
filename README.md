# RiboPipe


**RiboPipe** is a lightweight, modular pipeline for processing Ribo-seq
P-site count data and training codon-level translation prediction
models.

The pipeline supports:

-   Transcript-level preprocessing from raw P-site count matrices
-   Codon-resolution coverage extraction
-   Biological feature integration (e.g., tRNA copy number)
-   Length-aware training for codon-level prediction
-   Export of coverage matrices for downstream analysis

![Figure](./supp_case_grid_GSE233886.jpg) 

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

    Raw P-site CSV (generated using ribowaltz)
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

Prepare:

-   A P-site count CSV file
-   A transcript FASTA file

Example:

``` bash
ribopipe preprocess   --csv data/psite_counts.csv   --fasta data/transcripts.fa   --out-dir output_dir   --fasta-cache output_dir/fasta_cache.pkl
```

------------------------------------------------------------------------

### Step 2: Generate Coverage Matrix

``` bash
ribopipe matrix   --npz-dir output_dir   --out-csv output_dir/coverage_matrix_transcript_x_sample.csv
```

------------------------------------------------------------------------

### Step 3: Generate Biological Features

``` bash
ribopipe biofeat   --cds-npz output_dir/sample_name.npz   --trna-json trna_copy_numbers.json   --out-npz output_dir/bio_features.npz
```

trna_copy_numbers.json are generated based on GtRNAdb for Homo sapiens (https://gtrnadb.ucsc.edu/genomes/eukaryota/Hsapi38/Hsapi38-summary-all.html). For preparing bioFeat for other spieces, please based on the GtRNAdb information for generating the JSON file of trna_copy_numbers.json.  

------------------------------------------------------------------------

### Step 4: Train Codon-Level Prediction Model

``` bash
ribopipe train_pipeline   --coverage-csv output_dir/coverage_matrix_transcript_x_sample.csv   --npz-dir output_dir   --bio-feat output_dir/bio_features.npz   --threshold P75   --epochs 200   --train-split 0.8   --max-codons 5000 --output-dir ./predictions
```

------------------------------------------------------------------------

## Key Parameters

  |Parameter      | Description |
  |--------------- |-------------------------------------------------------|
  |--threshold     |Select high-expression transcripts (e.g., P75 or P95)|
  |--train-split   |Fraction of transcripts used for training|
  |--epochs        |Number of training epochs|
  |--max-codons    |Maximum CDS length (truncated/padded)|
  |--output-dir    |File Fold for saving prediction results| 

------------------------------------------------------------------------

## Output Structure

    output_dir/
    ├── *.npz
    ├── coverage_matrix_transcript_x_sample.csv
    ├── bio_features.npz
    ├── model_checkpoint.pt
    
    predictions/ (determined by --output-dir)

------------------------------------------------------------------------



## License

MIT License
