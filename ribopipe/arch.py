"""RiboPipe network definitions.

The class hierarchy (MB1v2 -> MB1v2Fuse -> MB1v2NoC2 -> MB1v2NoC2MFE) is kept exactly as in the code that produced
the paper's results: each constructor creates and then replaces sub-modules in the same order, so that a given seed
produces the same initial weights, and parameter names match the paper's checkpoints.

RiboPipe (``MB1v2NoC2MFE``, 228,321 trainable parameters):
  codon branch   Conv1d(64 -> 64, k=3) over the E, P and A codons, exponential activation
  nucleotide     Conv1d(4 -> 32, k=30, stride 3) over nucleotides -15..+14 relative to the first A-site nucleotide
  amino acid     Conv1d(21 -> 32, k=5) over the five residues preceding the A site
  folding        three MFEs of 30-nt windows starting 17, 16 and 15 nt upstream of the first A-site nucleotide,
                 standardised with fold-wise training statistics, invalid windows set exactly to 0
  recurrent      the 128 branch channels and the 3 MFEs enter one bidirectional GRU (hidden 128)
  output         Linear(256 -> 32) + ReLU + Linear(32 -> 1), on the log1p(relative density) scale
First-layer filters are centred across input channels at each position (``gauge``), which removes an offset that
is not identifiable from the data.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
NT_ORDER = "ACGT"
_CODON2AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}
AAS = sorted(set(_CODON2AA.values()))            # 20 amino acids + '*' = 21
AAIDX = {a: i for i, a in enumerate(AAS)}


def codon2aa_matrix():
    """(65, 21) lookup: row k = one-hot amino acid of codon k; row 64 (padding / unknown) is all zero."""
    M = np.zeros((65, 21), dtype=np.float32)
    for k, c in enumerate(CODONS):
        M[k, AAIDX[_CODON2AA[c]]] = 1.0
    return M


def nt3_to_seq(bio):
    """(B, L, 12) per-codon nucleotide one-hot -> (B, 4, 3L) nucleotide sequence one-hot.
    Block j at codon i is absolute position 3i+j, so out[b, c, 3i+j] = bio[b, i, 4j+c]."""
    B, L, D = bio.shape
    assert D == 12, "the nucleotide branch needs the 12-dim per-codon nucleotide input"
    return bio.view(B, L, 3, 4).permute(0, 3, 1, 2).reshape(B, 4, 3 * L)


def _zero_sum(w):
    return w - w.mean(dim=1, keepdim=True)


class MB1v2(nn.Module):
    """Three convolutional branches (codon, nucleotide, amino acid) followed by a fusion convolution and a BiGRU."""
    def __init__(s, bio_dim, Fc=64, Fn=32, Fa=32, ch2=64, rnn_hidden=128, rnn_layers=1,
                 fc=32, first_act='exp', clamp_max=5.0, cell='gru', gauge=True,
                 span_codon=(2, 0), nt_win=3, span_aa=(5, -1), fuse_k=3, fc_act='relu',
                 nt_pad=None):
        super().__init__()
        assert bio_dim == 12, f"bio_dim must be 12 (nt3), got {bio_dim}"
        s.first_act, s.clamp_max, s.gauge, s.fc_act = first_act, clamp_max, bool(gauge), fc_act
        s.span_codon = tuple(span_codon); s.span_nt = (int(nt_win), int(nt_win)); s.span_aa = tuple(span_aa)
        if nt_pad is not None: s.span_nt = (int(nt_pad[0]), int(nt_pad[1]))   # paper: (15, 12) -> kernel 30, nt -15..+14
        s.onehot = nn.Embedding(65, 64)
        with torch.no_grad():
            w = torch.zeros(65, 64)
            for i in range(64): w[i, i] = 1.0
            s.onehot.weight.copy_(w)
        s.onehot.weight.requires_grad_(False)
        s.register_buffer("C2A", torch.from_numpy(codon2aa_matrix()))
        s.cc = nn.Conv1d(64, Fc, s.span_codon[0] + s.span_codon[1] + 1, padding=0)
        s.cn = nn.Conv1d(4,  Fn, s.span_nt[0] + s.span_nt[1] + 3, stride=3, padding=0)
        s.ca = nn.Conv1d(21, Fa, s.span_aa[0] + s.span_aa[1] + 1, padding=0)
        s.c2 = nn.Conv1d(Fc + Fn + Fa, ch2, fuse_k, padding=fuse_k // 2)
        R = nn.GRU if cell == 'gru' else nn.LSTM
        s.rnn = R(ch2, rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        s.fc = nn.Linear(2 * rnn_hidden, fc)
        s.reg = nn.Linear(fc, 1)

    def _a1(s, z):
        if s.first_act == 'exp':
            return torch.exp(z if s.clamp_max is None else torch.clamp(z, max=s.clamp_max))
        return torch.relu(z)

    def _m(s, z):
        if s.fc_act == 'relu': return torch.relu(z)
        if s.fc_act == 'gelu': return F.gelu(z)
        return z

    def _w(s, c): return _zero_sum(c.weight) if s.gauge else c.weight

    def pathways(s, idx, bio, mask=None):
        """Branch activations (B, L, Fc), (B, L, Fn), (B, L, Fa)."""
        cod = s.onehot(idx); aa = s.C2A[idx]
        if mask is not None:
            cod = cod * mask.unsqueeze(-1); aa = aa * mask.unsqueeze(-1); bio = bio * mask.unsqueeze(-1)
        L = idx.size(1)
        lc, rc = s.span_codon; la, ra = s.span_aa; ln, rn = s.span_nt
        ac = s._a1(F.conv1d(F.pad(cod.transpose(1, 2), (lc, rc)), s._w(s.cc), s.cc.bias)).transpose(1, 2)
        aa_ = s._a1(F.conv1d(F.pad(aa.transpose(1, 2), (la, ra)), s._w(s.ca), s.ca.bias)).transpose(1, 2)
        an = s._a1(F.conv1d(F.pad(nt3_to_seq(bio), (ln, rn)), s._w(s.cn), s.cn.bias, stride=3)).transpose(1, 2)
        assert ac.size(1) == an.size(1) == aa_.size(1) == L
        return ac, an, aa_

    def act1(s, idx, bio): return s.pathways(idx, bio, None)

    def forward(s, idx, bio, lengths):
        L = idx.size(1)
        m = (torch.arange(L, device=idx.device)[None, :] < lengths[:, None].to(idx.device)).float()
        ac, an, aa_ = s.pathways(idx, bio, m)
        h = s.c2(torch.cat([ac, an, aa_], -1).transpose(1, 2)).transpose(1, 2)
        h = h * m.unsqueeze(-1)
        pk = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = s.rnn(pk); o, _ = pad_packed_sequence(o, batch_first=True, total_length=L)
        return s.reg(s._m(s.fc(o))).squeeze(-1)


class MB1v2Fuse(MB1v2):
    """Fusion after the branches: 'conv3' (Conv1d k=3) or 'lin1' (Conv1d k=1, i.e. additive across branches)."""
    FUSIONS = ('conv3', 'lin1')

    def __init__(s, bio_dim, fusion='conv3', **kw):
        super().__init__(bio_dim, **kw)
        assert fusion in s.FUSIONS, f"fusion must be one of {s.FUSIONS}, got {fusion}"
        s.fusion = fusion
        Fsum, ch2 = s.cc.out_channels + s.cn.out_channels + s.ca.out_channels, s.c2.out_channels
        if fusion == 'lin1':
            s.c2 = nn.Conv1d(Fsum, ch2, 1)

    def _fuse(s, ac, an, aa_):
        return s.c2(torch.cat([ac, an, aa_], -1).transpose(1, 2)).transpose(1, 2)

    def forward(s, idx, bio, lengths):
        L = idx.size(1)
        m = (torch.arange(L, device=idx.device)[None, :] < lengths[:, None].to(idx.device)).float()
        ac, an, aa_ = s.pathways(idx, bio, m)
        h = s._fuse(ac, an, aa_) * m.unsqueeze(-1)
        pk = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = s.rnn(pk); o, _ = pad_packed_sequence(o, batch_first=True, total_length=L)
        return s.reg(s._m(s.fc(o))).squeeze(-1)


class MB1v2NoC2(MB1v2Fuse):
    """No fusion layer: the three branches are concatenated and enter the BiGRU directly. = RiboPipe (-MFE)."""
    def __init__(s, bio_dim, fusion='lin1', **kw):
        super().__init__(bio_dim, fusion='lin1', **kw)
        Fsum = s.cc.out_channels + s.cn.out_channels + s.ca.out_channels
        R, H, nl = type(s.rnn), s.rnn.hidden_size, s.rnn.num_layers
        s.c2 = None
        s.rnn = R(Fsum, H, num_layers=nl, batch_first=True, bidirectional=True)
        s.fusion = 'identity'
        s.fsum = Fsum

    def _fuse(s, ac, an, aa_):
        return torch.cat([ac, an, aa_], -1)


class MB1v2NoC2MFE(MB1v2NoC2):
    """RiboPipe: MB1v2NoC2 plus three standardised footprint-scale MFEs concatenated to the BiGRU input.
    Input ``bio`` is 18-dim = [per-codon nucleotides 12 | MFE raw 3 | MFE valid flag 3]. A window is valid when its
    start p satisfies 0 <= p < n_valid (n_valid = len(seq) - 29); invalid windows enter as exactly 0."""
    def __init__(s, bio_dim, fusion='lin1', mfe_mu=0.0, mfe_sd=1.0, **kw):
        assert bio_dim == 18, f"bio_dim must be 18 (nt3 12 + MFE 3 + valid flags 3), got {bio_dim}"
        super().__init__(12, fusion='lin1', **kw)
        R, H, nl = type(s.rnn), s.rnn.hidden_size, s.rnn.num_layers
        s.rnn = R(s.fsum + 3, H, num_layers=nl, batch_first=True, bidirectional=True)
        s.register_buffer("mfe_mu", torch.tensor(float(mfe_mu)))
        s.register_buffer("mfe_sd", torch.tensor(float(mfe_sd)))
        s.fusion = 'identity+mfe'; s._mfe = None

    def mfe_input(s, bio, mask=None):
        """Standardised and masked MFE input (B, L, 3) = ((raw - mu) / sd) * valid [* padding mask]."""
        assert bio.size(-1) == 18, f"bio must be 18-dim (with MFE valid flags), got {bio.size(-1)}"
        valid = bio[..., 15:18]
        mfe = ((bio[..., 12:15] - s.mfe_mu) / s.mfe_sd) * valid
        mfe = torch.where(valid > 0, mfe, torch.zeros_like(mfe))
        if mask is not None: mfe = mfe * mask.unsqueeze(-1)
        return mfe

    def pathways(s, idx, bio, mask=None):
        s._mfe = s.mfe_input(bio, mask)
        return MB1v2.pathways(s, idx, bio[..., :12], mask)

    def _fuse(s, ac, an, aa_):
        m = s._mfe if (s._mfe is not None and s._mfe.shape[:2] == ac.shape[:2]) else ac.new_zeros(ac.shape[:2] + (3,))
        return torch.cat([ac, an, aa_, m], -1)


RiboPipe = MB1v2NoC2MFE
RiboPipeNoMFE = MB1v2NoC2


class MB1v2OH97(MB1v2Fuse):
    """BiLSTM-97 control: [codon one-hot 64 | per-codon nucleotides 12 | amino acid 21] at each position enter one
    bidirectional LSTM directly (the convolutional modules are removed)."""
    def __init__(s, bio_dim, **kw):
        super().__init__(bio_dim, **kw)
        hid = s.rnn.hidden_size; nl = s.rnn.num_layers
        cell = nn.GRU if isinstance(s.rnn, nn.GRU) else nn.LSTM
        for nm in ("cc", "cn", "ca", "c2"):
            if hasattr(s, nm): delattr(s, nm)
        s.rnn = cell(97, hid, num_layers=nl, batch_first=True, bidirectional=True)
        s.in_dim = 97

    def x97(s, idx, bio):
        return torch.cat([s.onehot(idx), bio, s.C2A[idx]], dim=-1)

    def forward(s, idx, bio, lengths):
        L = idx.size(1)
        m = (torch.arange(L, device=idx.device)[None, :] < lengths[:, None].to(idx.device)).float()
        x = s.x97(idx, bio) * m.unsqueeze(-1)
        pk = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = s.rnn(pk); o, _ = pad_packed_sequence(o, batch_first=True, total_length=L)
        return s.reg(s._m(s.fc(o))).squeeze(-1)


def gate_x97(model):
    """Structure check of the BiLSTM-97 input: codon rows sum to 1, nucleotide rows to 3, amino-acid rows to 1,
    exactly 5 non-zeros per position, and the convolutional modules removed."""
    g = torch.Generator().manual_seed(0); L = 23
    idx = torch.randint(0, 64, (2, L), generator=g); bio = torch.zeros(2, L, 12)
    for b in range(2):
        for t in range(L):
            c = int(idx[b, t])
            for p in range(3): bio[b, t, 4 * p + (c >> (2 * (2 - p))) % 4] = 1.0
    with torch.no_grad(): x = model.x97(idx, bio)
    out = dict(codon_rowsum=sorted(set(x[:, :, :64].sum(-1).flatten().tolist())),
               nt_rowsum=sorted(set(x[:, :, 64:76].sum(-1).flatten().tolist())),
               aa_rowsum=sorted(set(x[:, :, 76:].sum(-1).flatten().tolist())),
               nnz_per_pos=sorted(set((x != 0).sum(-1).flatten().tolist())),
               removed=[nm for nm in ("cc", "cn", "ca", "c2") if not hasattr(model, nm)],
               rnn_input_size=int(model.rnn.input_size))
    out["ok"] = bool(out["codon_rowsum"] == [1.0] and out["nt_rowsum"] == [3.0] and out["aa_rowsum"] == [1.0]
                     and out["nnz_per_pos"] == [5] and len(out["removed"]) == 4 and out["rnn_input_size"] == 97)
    return out


END_OFFS = list(range(-18, -11)) + list(range(10, 17))


class MB1v2EndHead(MB1v2NoC2):
    """RiboPipe with additive heads, used in the model analyses (Part III): output = f_seq + q_MFE (+ p_PWM), where
    q_MFE = MLP(standardised MFEs). ``predict_parts`` returns the exact decomposition."""
    def __init__(s, bio_dim, fusion='lin1', use_pwm=True, use_mfe=False, mfe_mu=0.0, mfe_sd=1.0, **kw):
        assert bio_dim in (12, 18), bio_dim
        super().__init__(12, fusion='lin1', **kw)
        s.use_pwm, s.use_mfe = bool(use_pwm), bool(use_mfe)
        assert (bio_dim == 18) == s.use_mfe, f"use_mfe={use_mfe} requires bio_dim={'18' if use_mfe else '12'}, got {bio_dim}"
        if s.use_pwm: s.pwm_head = nn.Linear(4 * len(END_OFFS), 1)
        if s.use_mfe:
            s.mfe_head = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
            s.register_buffer("mfe_mu", torch.tensor(float(mfe_mu))); s.register_buffer("mfe_sd", torch.tensor(float(mfe_sd)))
        s.fusion = 'identity+heads'

    @staticmethod
    def end_onehot(nt12):
        B, L, _ = nt12.shape; out = []
        for o in END_OFFS:
            dj, k = o // 3, o % 3
            x = nt12[..., 4 * k:4 * k + 4]
            if dj > 0: x = torch.cat([x[:, dj:], x.new_zeros(B, dj, 4)], 1)
            elif dj < 0: x = torch.cat([x.new_zeros(B, -dj, 4), x[:, :dj]], 1)
            out.append(x)
        return torch.cat(out, -1)

    def pathways(s, idx, bio, mask=None):
        return MB1v2.pathways(s, idx, bio[..., :12], mask)

    def predict_parts(s, idx, bio, lengths):
        L = idx.size(1)
        m = (torch.arange(L, device=idx.device)[None, :] < lengths[:, None].to(idx.device)).float()
        ac, an, aa_ = s.pathways(idx, bio, m)
        h = s._fuse(ac, an, aa_) * m.unsqueeze(-1)
        pk = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = s.rnn(pk); o, _ = pad_packed_sequence(o, batch_first=True, total_length=L)
        f = s.reg(s._m(s.fc(o))).squeeze(-1)
        zero = torch.zeros_like(f)
        p = s.pwm_head(s.end_onehot(bio[..., :12] * m.unsqueeze(-1))).squeeze(-1) * m if s.use_pwm else zero
        if s.use_mfe:
            assert bio.size(-1) == 18, f"use_mfe needs 18-dim bio (with valid flags), got {bio.size(-1)}"
            valid = bio[..., 15:18]
            z = ((bio[..., 12:15] - s.mfe_mu) / s.mfe_sd) * valid
            z = torch.where(valid > 0, z, torch.zeros_like(z))
            q = s.mfe_head(z * m.unsqueeze(-1)).squeeze(-1) * m
        else:
            q = zero
        return f, p, q

    def forward(s, idx, bio, lengths):
        f, p, q = s.predict_parts(idx, bio, lengths)
        return f + p + q
