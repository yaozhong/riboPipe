#!/usr/bin/env bash
# End-to-end RiboPipe example (single Ribo-seq sample), headline model.
# Assumes ribopipe is installed (pip install -e ".[struct]") and a GPU is available.
#
# Replace the paths below with your actual data files.
set -euo pipefail

CSV="path/to/codon_counts.csv"          # codon-level counts table
FASTA="path/to/transcripts_cds.fa"      # CDS FASTA sequences
E2G="reproduce/enst2ensg_grch38.json.gz" # ENST→ENSG map (gene-level validation split)
OUT="./ribopipe_out"                     # output directory

SAMPLE="my_sample"                       # column name in the coverage matrix

mkdir -p "$OUT"

echo "=== Step 1: Preprocess CSV → NPZ ==="
ribopipe preprocess \
  --csv "$CSV" \
  --fasta "$FASTA" \
  --out-dir "$OUT/npz"

echo "=== Step 2: Build coverage matrix ==="
ribopipe matrix \
  --npz-dir "$OUT/npz" \
  --out-csv "$OUT/coverage_matrix.csv"

echo "=== Step 3: Precompute ViennaRNA local-structure (MFE) cache ==="
# one-time per dataset; writes $OUT/npz/struct_cache/${SAMPLE}_struct.npz
ribopipe struct --npz "$OUT/npz/${SAMPLE}.npz"
STRUCT="$OUT/npz/struct_cache/${SAMPLE}_struct.npz"

echo "=== Step 4: Train the headline model (codon+NT+struct, Huber loss, motif-CNN + BiGRU-128) ==="
ribopipe train \
  --npz "$OUT/npz/${SAMPLE}.npz" \
  --struct-npz "$STRUCT" \
  --coverage-csv "$OUT/coverage_matrix.csv" \
  --sample "$SAMPLE" \
  --enst2ensg "$E2G" \
  --out-dir "$OUT/model"

echo "=== Step 5: Predict pause profiles ==="
ribopipe predict \
  --checkpoint "$OUT/model/ribopipe_model.pt" \
  --npz "$OUT/npz/${SAMPLE}.npz" \
  --struct-npz "$STRUCT" \
  --out-csv "$OUT/predictions.csv"

echo "Done. Results in $OUT/"
