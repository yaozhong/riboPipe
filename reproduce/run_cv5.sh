#!/usr/bin/env bash
# Reproduce the paper's headline benchmark: gene-level 5-fold cross-validation of the
# headline model (motif-CNN k=7 + BiGRU-128, RiboPipeCNN) against the baselines, on one dataset.
#
# This is the LEAK-FREE protocol behind the paper's Table 2: every isoform of a gene
# is kept in a single fold, so no gene is ever both trained on and tested. Expect median per-transcript Pearson ~0.52-0.69 (dataset-dependent):
# TX9_WT 0.598, GSE233886_WT 0.690, GSE133393_WT 0.529, PRJNA_iPS 0.518.
#
# Prerequisites (one per dataset), produced by the preprocessing + struct steps:
#   $NPZ         per-transcript NPZ         (ribopipe preprocess)
#   $STRUCT      ViennaRNA MFE cache        (ribopipe struct)
#
# The ENST->ENSG map used for the gene-level folds ships gzipped alongside this script.
set -euo pipefail
cd "$(dirname "$0")/.."

NPZ=${NPZ:?set NPZ=/path/to/<dataset>.npz}
STRUCT=${STRUCT:?set STRUCT=/path/to/struct_cache/<dataset>_struct.npz}
E2G=reproduce/enst2ensg_grch38.json.gz
OUT=${OUT:-reproduce/cv5_result.json}

# If you have not built the struct cache yet, do it once:
#   ribopipe struct --npz "$NPZ"

ribopipe cv5 \
  --npz "$NPZ" \
  --struct-npz "$STRUCT" \
  --enst2ensg "$E2G" \
  --backbone cnn --loss huber \
  --methods "ribopipe,bilstm_base,tricodon,ridge,codon_mean" \
  --n-folds 5 --epochs 200 --patience 20 \
  --out-json "$OUT"

echo "Wrote $OUT"
