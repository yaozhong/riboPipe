"""Raw reads -> P-site -> per-codon count CSV (the input to ``ribopipe preprocess``).

A self-contained, standard P-site assignment for **CDS-aligned** ribosome-profiling
BAM files.  It turns one or more aligned BAMs (one per sample/replicate) into the
codon-level counts CSV that :mod:`ribopipe.preprocess` consumes.

Scope / honesty note
--------------------
The paper's human Ribo-seq was P-site-assigned with **riboWaltz** (an R package) as a
standard upstream step; the repository always started from the resulting
``*_P-site_rawcount.csv`` files.  This module is a lightweight, dependency-light
re-implementation of that same standard step so the released toolkit is end-to-end
runnable from BAM.  It is **not** a byte-for-byte reproduction of the paper's
riboWaltz run — for the published numbers use the released count CSVs / checkpoints.

Assumptions
-----------
* The BAM is aligned to the **CDS FASTA** later passed to ``ribopipe preprocess``
  (i.e. reference = CDS sequences, forward strand, one contig per transcript).  Then
  a read's P-site codon is ``(reference_start + offset[len]) // 3`` and every codon is
  in the CDS region, so the emitted CSV has ``region == "cds"`` throughout.
* P-site offset is the distance (nt) from a read's 5' end to its P-site first nt.
  A fixed default (12) is used unless a per-length table is supplied or
  ``--auto-offset`` derives one from the start-codon metagene (riboWaltz-style).

Output CSV schema (one row per codon) matches ``ribopipe.preprocess.schema``:
``transcript, region, start, end, from_cds_start, from_cds_stop`` + one integer count
column per sample (header = the sample column name you pass, e.g.
``HEK293T|WT|DMSO|rep1|mono|x`` or the legacy ``<condition>_repN``).
"""
from __future__ import annotations

import collections
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_OFFSET = 12          # canonical P-site offset (nt) for ~28-30 nt footprints
DEFAULT_MIN_LEN = 25
DEFAULT_MAX_LEN = 35


def _fasta_cds_lengths(fasta_path: str) -> Dict[str, int]:
    """transcript_id -> CDS length in nt (from the CDS FASTA)."""
    lengths: Dict[str, int] = {}
    tid: Optional[str] = None
    n = 0
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if tid is not None:
                    lengths[tid] = n
                tid = line[1:].strip().split()[0]
                n = 0
            else:
                n += len(line.strip())
    if tid is not None:
        lengths[tid] = n
    return lengths


def detect_offsets(
    bam_path: str,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
    flank: int = 30,
) -> Dict[int, int]:
    """Per-read-length P-site offset from the start-codon metagene (riboWaltz-style).

    For each read length, histogram the 5'-end position of reads whose 5' end lands in
    ``[0, flank)`` of any transcript (near the CDS start).  The dominant 5'-end distance
    ``d`` to the start codon implies P-site offset ``= d`` chosen so the P-site sits on
    codon 0.  Falls back to :data:`DEFAULT_OFFSET` for lengths with too few reads.
    """
    import pysam

    by_len: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped or r.is_reverse:
                continue
            L = r.query_length or (r.reference_length or 0)
            if L < min_len or L > max_len:
                continue
            start = r.reference_start
            if 0 <= start < flank:
                by_len[L][start] += 1

    offsets: Dict[int, int] = {}
    for L in range(min_len, max_len + 1):
        c = by_len.get(L)
        if not c or sum(c.values()) < 10:
            offsets[L] = DEFAULT_OFFSET
            continue
        # Most common 5'-end distance from the start codon; snap the P-site onto a codon.
        d = c.most_common(1)[0][0]
        off = int(round(d / 3.0) * 3) + 0  # keep the read frame; P-site = 5'end + off
        # keep within a sane biological window
        offsets[L] = off if 8 <= off <= 18 else DEFAULT_OFFSET
    return offsets


def psite_codon_counts(
    bam_path: str,
    cds_lengths: Dict[str, int],
    offsets: Optional[Dict[int, int]] = None,
    default_offset: int = DEFAULT_OFFSET,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> Dict[str, np.ndarray]:
    """Assign P-sites and tally reads per codon for one BAM.

    Returns ``{transcript_id: np.ndarray(int, length = n_codons)}``.  Only transcripts
    present in ``cds_lengths`` (i.e. in the CDS FASTA) are counted.
    """
    import pysam

    counts: Dict[str, np.ndarray] = {
        tid: np.zeros(L // 3, dtype=np.int64) for tid, L in cds_lengths.items() if L >= 3
    }
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped or r.is_reverse:
                continue
            tid = r.reference_name
            arr = counts.get(tid)
            if arr is None:
                continue
            L = r.query_length or (r.reference_length or 0)
            if L < min_len or L > max_len:
                continue
            off = (offsets.get(L, default_offset) if offsets else default_offset)
            psite = r.reference_start + off
            codon = psite // 3
            if 0 <= codon < arr.shape[0]:
                arr[codon] += 1
    return counts


def bams_to_codon_csv(
    bam_paths: List[str],
    sample_cols: List[str],
    fasta_path: str,
    out_csv: str,
    offsets: Optional[Dict[int, int]] = None,
    auto_offset: bool = False,
    default_offset: int = DEFAULT_OFFSET,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
    keep_uncovered: bool = False,
) -> str:
    """Convert per-sample CDS-aligned BAMs into one ``ribopipe preprocess`` CSV.

    Parameters
    ----------
    bam_paths, sample_cols
        Parallel lists: one BAM and one output count-column name per sample/replicate.
    fasta_path
        The CDS FASTA the BAMs are aligned to (also passed to ``ribopipe preprocess``).
    auto_offset
        If True, derive per-length P-site offsets from the first BAM's start-codon
        metagene; otherwise use ``offsets`` (if given) or a fixed ``default_offset``.
    keep_uncovered
        If False (default) transcripts with zero reads across all samples are omitted.
    """
    if len(bam_paths) != len(sample_cols):
        raise ValueError("bam_paths and sample_cols must have the same length")

    cds_lengths = _fasta_cds_lengths(fasta_path)
    if not cds_lengths:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    if auto_offset and offsets is None:
        offsets = detect_offsets(bam_paths[0], min_len=min_len, max_len=max_len)
        print(f"[rawcount] auto P-site offsets (len->off): {dict(sorted(offsets.items()))}")

    per_sample: List[Dict[str, np.ndarray]] = []
    for bam, col in zip(bam_paths, sample_cols):
        print(f"[rawcount] {col}: assigning P-sites from {bam}")
        per_sample.append(
            psite_codon_counts(bam, cds_lengths, offsets=offsets,
                               default_offset=default_offset, min_len=min_len, max_len=max_len)
        )

    rows: List[dict] = []
    for tid, L_nt in cds_lengths.items():
        n_cod = L_nt // 3
        if n_cod == 0:
            continue
        sample_arrs = [ps.get(tid) for ps in per_sample]
        if not keep_uncovered and all(a is None or a.sum() == 0 for a in sample_arrs):
            continue
        for i in range(n_cod):
            row = {
                "transcript": tid,
                "region": "cds",
                "start": 3 * i,
                "end": 3 * i + 3,
                "from_cds_start": i,
                "from_cds_stop": i - n_cod,
            }
            for col, arr in zip(sample_cols, sample_arrs):
                row[col] = int(arr[i]) if arr is not None else 0
            rows.append(row)

    df = pd.DataFrame(rows, columns=["transcript", "region", "start", "end",
                                     "from_cds_start", "from_cds_stop"] + list(sample_cols))
    df.to_csv(out_csv, index=False)
    print(f"[rawcount] wrote {out_csv}: {df['transcript'].nunique():,} transcripts, "
          f"{len(df):,} codon rows, {len(sample_cols)} sample column(s)")
    return out_csv
