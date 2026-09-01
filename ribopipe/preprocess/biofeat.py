from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from tqdm import tqdm

# ---- Tables copied from original bioFeat_gen.py (semantic-preserving) ----
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

AA_PROPERTIES = {
    'A': [1, 0, 0], 'V': [1, 0, 0], 'L': [1, 0, 0], 'I': [1, 0, 0], 'M': [1, 0, 0],
    'F': [1, 0, 0], 'W': [1, 0, 0], 'Y': [0, 1, 0], 'S': [0, 1, 0], 'T': [0, 1, 0],
    'N': [0, 1, 0], 'Q': [0, 1, 0], 'C': [0, 1, 0], 'G': [0, 1, 0], 'P': [0, 1, 0],
    'D': [0, 1, 1], 'E': [0, 1, 1], 'K': [0, 1, 1], 'R': [0, 1, 1], 'H': [0, 1, 1],
    '*': [0, 0, 0]
}

# NOTE:
# The upstream codebase uses DNA alphabet (A/C/G/T) for codons.
# The original script mixed DNA codons (T) with RNA wobble tables (U),
# which makes anticodon matching fail (tAI becomes all zeros).
# Here we keep CODON_TABLE in DNA alphabet and adapt the wobble-pair table to DNA (T).
WOBBLE_PAIRS = {
    # anticodon base -> allowed codon base(s)
    'G': ['C', 'T'],
    'C': ['G'],
    'A': ['T'],
    'T': ['A', 'G'],
    'I': ['A', 'T', 'C'],
}


def _precompute_anticodon_matches() -> Dict[str, list]:
    """Brute force anticodon matches once, not per transcript (semantic-preserving)."""
    matches: Dict[str, list] = {}
    # DNA alphabet for codons; keep I for inosine
    bases = 'ATCGI'
    for codon in CODON_TABLE:
        anti_list = []
        for b1 in bases:
            for b2 in bases:
                for b3 in bases:
                    anti = b1 + b2 + b3
                    ok = True
                    # original code: codon[i] in wobble_pairs[anti[2-i]]
                    for i in range(3):
                        if codon[i] not in WOBBLE_PAIRS.get(anti[2 - i], []):
                            ok = False
                            break
                    if ok:
                        anti_list.append(anti)
        matches[codon] = anti_list
    return matches


_ANTICODON_MATCHES = _precompute_anticodon_matches()


def compute_tai_weights(trna_copy_numbers: Dict[str, float]) -> Dict[str, float]:
    weights = {}
    for codon, anti_list in _ANTICODON_MATCHES.items():
        weights[codon] = sum(trna_copy_numbers.get(ac, 0) for ac in anti_list)
    max_w = max(weights.values()) if weights else 1.0
    if max_w <= 0:
        max_w = 1.0
    return {c: (weights[c] / max_w) for c in weights}


def _precompute_is_wobble_codon() -> Dict[str, int]:
    """Semantic-preserving precompute of is_wobble_codon for 64 codons."""
    bases = 'ATCGI'
    out: Dict[str, int] = {}
    for codon in CODON_TABLE:
        wobble_only = True
        matched = False
        for b1 in bases:
            for b2 in bases:
                for b3 in bases:
                    anti = b1 + b2 + b3
                    match = True
                    is_wobble = False
                    for i in range(3):
                        c_base = codon[i]
                        a_base = anti[2 - i]
                        if c_base not in WOBBLE_PAIRS.get(a_base, []):
                            match = False
                            break
                        if c_base != a_base:
                            is_wobble = True
                    if match:
                        matched = True
                        if not is_wobble:
                            wobble_only = False
        if not matched:
            out[codon] = 0
        else:
            out[codon] = 1 if wobble_only else 0
    return out


_IS_WOBBLE = _precompute_is_wobble_codon()


def compute_codon_frequency(cds_seq: str) -> Dict[str, float]:
    total_codons = len(cds_seq) // 3
    if total_codons <= 0:
        return {c: 0.0 for c in CODON_TABLE}

    counts = {c: 0 for c in CODON_TABLE}
    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i:i+3]
        if codon in counts:
            counts[codon] += 1
    return {c: counts[c] / total_codons for c in CODON_TABLE}


def build_bio_features(
    cds_npz_path: str,
    trna_copy_numbers_json: str,
    out_npz_path: str,
    show_progress: bool = True,
) -> str:
    """Generate biological per-codon features.

    Output format matches original bioFeat_gen.py:
      bio_features[transcript_id] = (num_codons, 6) float32
      columns = [codon_freq, tAI, wobble_only, hydrophobic, polar, charged]
    """
    cds_data = np.load(cds_npz_path, allow_pickle=True)
    with open(trna_copy_numbers_json, "r") as f:
        trna_copy = json.load(f)

    tai_weights = compute_tai_weights(trna_copy)

    bio_features: Dict[str, np.ndarray] = {}
    keys = list(cds_data.files)
    it = tqdm(keys, desc="Building bio features") if show_progress else keys

    # Diagnostics (helps catch upstream issues instead of silently writing an empty NPZ)
    n_total = 0
    n_used = 0
    skip_no_seq = 0
    skip_too_short = 0

    for key in it:
        n_total += 1
        entry = cds_data[key].item()

        # Accept both old and new structures
        if isinstance(entry, dict) and "cds" in entry and isinstance(entry.get("cds"), dict):
            cds_seq = entry["cds"].get("sequence", "")
            cov = entry["cds"].get("avg_count_norm", None)
        elif isinstance(entry, dict):
            cds_seq = entry.get("sequence", "")
            cov = entry.get("avg_count_norm", None)
        else:
            cds_seq = ""
            cov = None

        if not isinstance(cds_seq, str):
            try:
                cds_seq = str(cds_seq)
            except Exception:
                cds_seq = ""

        cds_seq = cds_seq.strip().upper().replace("U", "T")
        if not cds_seq:
            skip_no_seq += 1
            continue

        # If upstream slicing produced non-multiple-of-3 sequences, truncate to the largest multiple.
        # This is conservative and avoids dropping everything.
        if len(cds_seq) >= 3:
            cds_seq = cds_seq[: (len(cds_seq) // 3) * 3]

        if len(cds_seq) < 3:
            skip_too_short += 1
            continue

        codon_freq = compute_codon_frequency(cds_seq)
        feats = []
        for i in range(0, len(cds_seq) - 2, 3):
            codon = cds_seq[i:i+3]
            freq = codon_freq.get(codon, 0.0)
            tai = tai_weights.get(codon, 0.0)
            wobble = float(_IS_WOBBLE.get(codon, 0))
            aa = CODON_TABLE.get(codon, "*")
            aa_feat = AA_PROPERTIES.get(aa, [0, 0, 0])
            feats.append([freq, tai, wobble] + aa_feat)
        if feats:
            bio_features[key] = np.asarray(feats, dtype=np.float32)
            n_used += 1

    out_path = Path(out_npz_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if n_used == 0:
        raise RuntimeError(
            "No bio features were produced. This usually means the input NPZ does not contain valid "
            "CDS sequences under entry['cds']['sequence'], or sequences are not codon-aligned. "
            f"Diagnostics: total_keys={n_total}, no_seq={skip_no_seq}, too_short={skip_too_short}. "
            "Please verify that preprocess wrote per-transcript cds sequences (length multiple of 3)."
        )

    np.savez_compressed(out_path, **bio_features)
    return str(out_path)
