"""RiboPipe: within-sample Ribo-seq imputation.

Learn a sequence-to-occupancy mapping from a sample's high-coverage transcripts and
impute codon-resolution pause profiles for its sparsely covered transcripts.

Headline model: ``ribopipe`` (``RiboPipeCNN``) -- a k=7 exp-motif CNN + BiGRU-128 over codon + nucleotide
+ nucleotide (+/-15 nt) + local mRNA-structure (MFE) features, trained on the
covered-mean-normalised log pause score with an unweighted Huber loss and
selected by early stopping on a gene-level validation hold-out.
"""
__version__ = "1.1.0"

from .model import BiLSTM, seq_to_idx, PAD_IDX
from .dataset import RiboDataset, collate_fn, build_split
from .losses import huber_peak_mse, huber_mask, huber_peak_weighted, wmse_mask
from .metrics import per_tx_metrics, per_tx_medians, true_pause
from .folds import gene_folds, split_val, load_enst2ensg
from .train import train, train_on_ids, save_checkpoint
from .predict import predict, predict_dataset, predict_from_checkpoint, load_items
from .cv5 import run_cv5
from .struct import compute_struct_cache, mfe_track, struct_cache_path

__all__ = [
    "__version__",
    "BiLSTM", "seq_to_idx", "PAD_IDX",
    "RiboDataset", "collate_fn", "build_split",
    "huber_peak_mse", "huber_mask", "huber_peak_weighted", "wmse_mask",
    "per_tx_metrics", "per_tx_medians", "true_pause",
    "gene_folds", "split_val", "load_enst2ensg",
    "train", "train_on_ids", "save_checkpoint",
    "predict", "predict_dataset", "predict_from_checkpoint", "load_items",
    "run_cv5",
    "compute_struct_cache", "mfe_track", "struct_cache_path",
]
