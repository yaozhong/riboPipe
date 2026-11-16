"""Activation logos of the first-layer filters (inspection only; Fig. 4a of the paper).

For every filter of the codon, amino-acid and nucleotide branches, the windows of the test transcripts of one fold are
ranked by the filter's activation; the logo is the log2 enrichment of each letter at each window position among the
top 1% windows (at least 100) relative to all windows, with a pseudo-count of 0.5:
    log2((n_top + 0.5) / (N_top + 0.5 K)) - log2((n_all + 0.5) / (N_all + 0.5 K)),  K = number of letters.
Windows (relative to the A site): codon branch codons -2..0 (64 letters), amino-acid branch residues -5..-1 (21 letters,
'*' included), nucleotide branch nt -15..+14 (4 letters). Positions outside the transcript are not counted.
Logos show which inputs a filter prefers, not whether the filter changes the prediction.
"""
import numpy as np
import torch

from .. import arch as A
from .. import train as T

WIN = {"codon": (-2, 0), "aa": (-5, -1)}
NTWIN = (-15, 14)
NLET = {"codon": 64, "aa": 21, "nt": 4}
BRANCHES = ("codon", "nt", "aa")                   # order of the outputs of model.pathways
_C2A = A.codon2aa_matrix(); AA_OF = {i: int(np.argmax(_C2A[i])) for i in range(64)}


def window_letters(branch, idx, nt, L, pos):
    if branch == "codon":
        o = np.arange(WIN["codon"][0], WIN["codon"][1] + 1); j = pos[:, None] + o[None]
        ok = (j >= 0) & (j < L); v = np.where(ok, idx[np.clip(j, 0, L - 1)], -1); v[v >= 64] = -1; return v
    if branch == "aa":
        o = np.arange(WIN["aa"][0], WIN["aa"][1] + 1); j = pos[:, None] + o[None]
        ok = (j >= 0) & (j < L); c = np.where(ok, idx[np.clip(j, 0, L - 1)], 64)
        return np.where(c < 64, np.vectorize(lambda k: AA_OF.get(int(k), -1))(c), -1)
    o = np.arange(NTWIN[0], NTWIN[1] + 1); j = pos[:, None] * 3 + o[None]
    ok = (j >= 0) & (j < len(nt)); return np.where(ok, nt[np.clip(j, 0, len(nt) - 1)], -1)


@torch.no_grad()
def branch_activations(model, ds, ids, device="cpu", batch=T.PRED_BATCH):
    """{branch: [(tx, (L, F) activations)]} of the first-layer branches, computed in the prediction batches."""
    model = model.to(device).eval()
    items, _ = T.make_items(ds, ids, "mfe", "log", with_target=False)
    store = {b: [] for b in BRANCHES}
    order = sorted(range(len(items)), key=lambda i: items[i][4], reverse=True)
    for s in range(0, len(order), batch):
        sub = [items[i] for i in order[s:s + batch]]
        idx, bio, _, _, ll, ks = T.collate(sub)
        L = idx.size(1); dv = torch.device(device)
        m = (torch.arange(L, device=dv)[None, :] < ll[:, None].to(dv)).float()
        out = model.pathways(idx.to(dv), bio.to(dv), m)
        for k, b in enumerate(BRANCHES):
            a = out[k].float().cpu().numpy()
            for r, t in enumerate(ks): store[b].append((t, a[r, :int(ll[r])]))
    return store


def activation_logos(model, ds, ids, device="cpu", top_frac=0.01, min_top=100):
    """{branch: (F, n_positions, n_letters) float32 logos} over the windows of transcripts ``ids``."""
    acts = branch_activations(model, ds, ids, device)
    logos = {}
    for b in BRANCHES:
        allv, alll = [], []
        for t, ac in acts[b]:
            seq = ds.seq[t]; L = len(seq) // 3; idx = T.codon_idx(seq)
            nt = np.array([T.NT_MAP.get(c, -1) for c in seq])
            allv.append(ac); alll.append(window_letters(b, idx, nt, L, np.arange(L)))
        A_ = np.concatenate(allv); Lt = np.concatenate(alll); nlet = NLET[b]; npos = Lt.shape[1]
        bg = np.stack([np.bincount(Lt[:, q][Lt[:, q] >= 0], minlength=nlet) for q in range(npos)], 0).astype(float)
        ntop = max(min_top, int(top_frac * len(A_)))
        out = np.zeros((A_.shape[1], npos, nlet), np.float32)
        for j in range(A_.shape[1]):
            top = np.argsort(-A_[:, j])[:ntop]; Tt = Lt[top]
            fg = np.stack([np.bincount(Tt[:, q][Tt[:, q] >= 0], minlength=nlet) for q in range(npos)], 0).astype(float)
            out[j] = (np.log2((fg + 0.5) / (fg.sum(1, keepdims=True) + 0.5 * nlet))
                      - np.log2((bg + 0.5) / (bg.sum(1, keepdims=True) + 0.5 * nlet)))
        logos[b] = out
    return logos


def letter_labels(branch):
    if branch == "codon": return list(A.CODONS)
    if branch == "aa": return list(A.AAS)
    return list(A.NT_ORDER)
