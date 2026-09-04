"""RiboPipe BiLSTM model (paper §Methods: Architecture)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CODON_IDX: dict[str, int] = {c: i for i, c in enumerate(CODONS)}
PAD_IDX = 64   # row 64 in embedding = all-zeros pad vector


def seq_to_idx(seq: str, L: int) -> np.ndarray:
    """Convert a CDS nucleotide string to codon-index array of length L."""
    idx = np.full(L, PAD_IDX, dtype=np.int64)
    for i in range(L):
        codon = seq[i * 3: i * 3 + 3]
        idx[i] = CODON_IDX.get(codon, PAD_IDX)
    return idx


class BiLSTM(nn.Module):
    """Two-layer bidirectional LSTM for codon-level pause-score prediction.

    Input per codon:
        - 64-dim fixed one-hot codon encoding (rows 0-63 = identity; row 64 = pad)
        - bio_dim non-codon feature channels (headline = 123: 120 nt one-hot + 3 struct MFE)

    Output:
        - scalar mean-normalised pause score (count / transcript mean) per codon

    Parameters
    ----------
    bio_dim : int
        Number of non-codon per-codon feature channels concatenated after the
        64-d codon one-hot (headline = 123: 120 nt one-hot + 3 struct MFE).
    hidden : int
        Hidden units per direction per LSTM layer.  h=256 is the primary model
        in the paper; h=128 is the efficient variant (trains in ~32 min on one GPU).
    fc : int
        Feed-forward bottleneck dimension before the output regression head.
    """

    def __init__(self, bio_dim: int = 6, hidden: int = 256, fc: int = 64):
        super().__init__()

        # Fixed one-hot encoding: 64 codons (rows 0-63) + 1 pad (row 64 = zeros)
        self.onehot = nn.Embedding(65, 64)
        with torch.no_grad():
            w = torch.zeros(65, 64)
            for i in range(64):
                w[i, i] = 1.0
            self.onehot.weight.copy_(w)
        self.onehot.weight.requires_grad_(False)

        self.l1 = nn.LSTM(64 + bio_dim, hidden, batch_first=True, bidirectional=True)
        self.l2 = nn.LSTM(hidden * 2, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, fc)
        self.reg = nn.Linear(fc, 1)

    def forward(
        self,
        idx: torch.Tensor,       # (B, L_max)  codon indices (int64)
        bio: torch.Tensor,       # (B, L_max, bio_dim)  biological features
        lengths: torch.Tensor,   # (B,) actual CDS lengths, sorted descending
    ) -> torch.Tensor:           # (B, L_max)  predicted pause scores
        oh = self.onehot(idx)
        x = torch.cat([oh, bio], dim=-1)
        p = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        h, _ = self.l1(p)
        h, _ = self.l2(h)
        h, _ = pad_packed_sequence(h, batch_first=True)
        return self.reg(torch.relu(self.fc(h))).squeeze(-1)


def load_model(
    pt_path: str,
    bio_dim: int = 6,
    hidden: int = 256,
    device: Optional[str] = None,
) -> BiLSTM:
    """Load a trained BiLSTM checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = BiLSTM(bio_dim=bio_dim, hidden=hidden).to(dev)
    model.load_state_dict(torch.load(pt_path, map_location=dev))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Headline model (paper v1.1): exp-motif CNN (k=7) -> k=3 taper -> BiGRU-128.
# Faithful port of the research backbone `LocalCNNv2RNN` used for every
# headline number in the paper (decoupled-model backbone with use_named=False,
# so the effective model is the backbone alone).  Input per codon:
#   64-dim fixed codon one-hot (built inside)  +  bio_dim contextual features
#   (headline: 120 nt one-hot + 3 struct MFE = 123).
# Total first-conv in-channels = 64 + 123 = 187, matching the released ckpts.
# ---------------------------------------------------------------------------
class RiboPipeCNN(nn.Module):
    """K=7 exp-motif CNN front-end (readable first-layer filters) + a k=3 taper
    conv + a single bidirectional GRU (h=128) for mid-range (queuing) context.

    Parameters
    ----------
    bio_dim : int
        Per-codon contextual feature dims concatenated after the 64-d codon
        one-hot (headline = 123: 120 nt + 3 struct).
    ch1, ch2 : int
        First (motif) and taper conv widths (128, 64).
    k1 : int
        Motif kernel in codons (7).
    rnn_hidden : int
        GRU hidden units per direction (128).
    fc : int
        Feed-forward bottleneck before the regression head (32).
    first_act : str
        First-layer activation: ``'exp'`` (readable multiplicative motifs,
        the headline) or ``'relu'``.
    cell : str
        ``'gru'`` (headline) or ``'lstm'``.
    """

    def __init__(self, bio_dim: int, ch1: int = 128, ch2: int = 64, k1: int = 7,
                 rnn_hidden: int = 128, fc: int = 32, first_act: str = "exp",
                 cell: str = "gru"):
        super().__init__()
        self.onehot = nn.Embedding(65, 64)
        with torch.no_grad():
            w = torch.zeros(65, 64)
            for i in range(64):
                w[i, i] = 1.0
            self.onehot.weight.copy_(w)
        self.onehot.weight.requires_grad_(False)
        cin = 64 + bio_dim
        self.c1 = nn.Conv1d(cin, ch1, k1, padding=k1 // 2)   # k=7 exp motif (readable)
        self.c2 = nn.Conv1d(ch1, ch2, 3, padding=1)          # taper
        RNN = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = RNN(ch2, rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * rnn_hidden, fc)
        self.reg = nn.Linear(fc, 1)
        self.first_act = first_act
        self.config = dict(model="cnn", bio_dim=bio_dim, ch1=ch1, ch2=ch2, k1=k1,
                           rnn_hidden=rnn_hidden, fc=fc, first_act=first_act, cell=cell)

    def _a1(self, z):
        return torch.exp(torch.clamp(z, max=5.0)) if self.first_act == "exp" else torch.relu(z)

    def act1(self, idx, bio):
        """First-layer motif activations (B, L, ch1) -- for interpretability/ISM."""
        x = torch.cat([self.onehot(idx), bio], dim=-1).transpose(1, 2)
        return self._a1(self.c1(x)).transpose(1, 2)

    def forward(self, idx, bio, lengths):
        x = torch.cat([self.onehot(idx), bio], dim=-1)
        L = x.size(1)
        m = (torch.arange(L, device=x.device)[None, :] < lengths[:, None].to(x.device)).float()
        h = (x * m.unsqueeze(-1)).transpose(1, 2)
        a = self._a1(self.c1(h))
        a = torch.relu(self.c2(a))
        a = (a * m.unsqueeze(1)).transpose(1, 2)             # (B, L, ch2)
        pk = pack_padded_sequence(a, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.rnn(pk)
        o, _ = pad_packed_sequence(o, batch_first=True, total_length=L)
        return self.reg(torch.relu(self.fc(o))).squeeze(-1)


def load_cnn_from_paper_checkpoint(pt_path: str, bio_dim: int = 123, device=None) -> "RiboPipeCNN":
    """Load the paper's released decoupled checkpoint into :class:`RiboPipeCNN`.

    The research checkpoints store the backbone under a ``bb.`` prefix alongside
    inert named heads (bA/bP/bE/U/V/Wt, zeroed because ``use_named=False``).  We
    keep only the backbone weights, which are exactly this model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    sd = torch.load(pt_path, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and "c1.weight" not in sd and "bb.c1.weight" not in sd:
        sd = sd["state_dict"]
    # Accept both the research checkpoint (backbone under `bb.` + inert named heads) and a
    # clean backbone-only state dict (`c1.*`). Drop the named heads; strip any `bb.` prefix.
    drop = ("bA.", "bP.", "bE.", "U.", "V.", "Wt.", "wchg", "chg_", "log_")
    bb = {(k[3:] if k.startswith("bb.") else k): v
          for k, v in sd.items() if not any(k.startswith(p) for p in drop)}
    if "c1.weight" in bb:
        bio_dim = int(bb["c1.weight"].shape[1]) - 64
    model = RiboPipeCNN(bio_dim=bio_dim).to(dev)
    model.load_state_dict(bb, strict=True)
    model.eval()
    return model
