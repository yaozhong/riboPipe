from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import time
import pandas as pd

from .schema import detect_format, parse_new_sample_col, LEGACY_REP_RE, validate_csv_columns
from .utils import load_fasta_dict


def _safe_slice(seq: str, start: Optional[int], end: Optional[int]) -> str:
    if start is None or end is None:
        return ""
    if pd.isna(start) or pd.isna(end):
        return ""
    try:
        s = int(start)
        e = int(end)
        if s < 0:
            s = 0
        if e <= s:
            return ""
        return seq[s:e]
    except Exception:
        return ""


def _bounds(x: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    if x.empty:
        return (None, None)
    return (x["start"].min(), x["end"].max())


def _is_5utr(region: str) -> bool:
    r = str(region).lower()
    return r in {"5utr", "utr5", "5'utr", "5_utr"}


def _is_3utr(region: str) -> bool:
    r = str(region).lower()
    return r in {"3utr", "utr3", "3'utr", "3_utr"}


def build_npz_for_one_signal(
    df: pd.DataFrame,
    fasta_dict: Dict[str, str],
    signal_cols: List[str],
) -> Dict[str, dict]:
    """Build the NPZ dictionary for one signal definition.

    This matches the *semantic outputs* of the original `extract_read_coverage`:
      transcripts_data[tid] = {'5utr':..., 'cds':{'avg_count','avg_count_norm'...}, '3utr':...}

    - avg_count: mean across signal_cols (replicates)
    - avg_count_norm: avg_count / cds_length

    Note: We keep coordinates as they appear in CSV (treated as 0-based, end-exclusive).

    Performance notes (v3):
    - Avoid per-row Python `apply` for region filtering by precomputing `region_l` and masks.
    - Add transcript-level progress bar and timing.
    """
    t0 = time.time()

    # Work on a shallow copy so we can add helper columns without mutating caller DataFrame
    df = df.copy()

    # Per-row mean over replicates (vectorized)
    df["avg_count"] = df[signal_cols].mean(axis=1).astype(float)

    # Precompute region masks (vectorized)
    region_l = df["region"].astype(str).str.lower()
    df["_is5"] = region_l.isin({"5utr", "utr5", "5'utr", "5_utr"})
    df["_is3"] = region_l.isin({"3utr", "utr3", "3'utr", "3_utr"})
    df["_iscds"] = region_l.eq("cds")

    transcripts_data: Dict[str, dict] = {}

    grouped = df.groupby("transcript", sort=False)
    for transcript, group in tqdm(grouped, total=grouped.ngroups, desc="Building transcripts"):
        group = group.sort_values(by=["start", "end"], kind="mergesort")

        utr5 = group[group["_is5"]]
        cds = group[group["_iscds"]]
        utr3 = group[group["_is3"]]

        sequence = fasta_dict.get(str(transcript), "")

        utr5_start, utr5_end = _bounds(utr5)
        cds_start, cds_end = _bounds(cds)
        utr3_start, utr3_end = _bounds(utr3)

        utr5_seq = _safe_slice(sequence, utr5_start, utr5_end)
        cds_seq = _safe_slice(sequence, cds_start, cds_end)
        utr3_seq = _safe_slice(sequence, utr3_start, utr3_end)

        # cds_length for normalization
        cds_length = 1
        if not cds.empty:
            try:
                cds_length = int(cds_end) - int(cds_start)
                if cds_length <= 0:
                    cds_length = 1
            except Exception:
                cds_length = 1

        utr5_cov = utr5["avg_count"].astype(float).tolist()
        cds_cov = cds["avg_count"].astype(float).tolist()
        cds_cov_norm = (cds["avg_count"] / cds_length).astype(float).tolist()
        utr3_cov = utr3["avg_count"].astype(float).tolist()

        transcripts_data[str(transcript)] = {
            "5utr": {
                "avg_count": utr5_cov,
                "start": None if utr5.empty else int(utr5_start),
                "end": None if utr5.empty else int(utr5_end),
                "sequence": utr5_seq,
            },
            "cds": {
                "avg_count": cds_cov,
                "start": None if cds.empty else int(cds_start),
                "end": None if cds.empty else int(cds_end),
                "sequence": cds_seq,
                "avg_count_norm": cds_cov_norm,
            },
            "3utr": {
                "avg_count": utr3_cov,
                "start": None if utr3.empty else int(utr3_start),
                "end": None if utr3.empty else int(utr3_end),
                "sequence": utr3_seq,
            },
        }

    print(f"[TIME] build_npz_for_one_signal finished in {time.time() - t0:.2f} s "
          f"({len(transcripts_data):,} transcripts)")

    return transcripts_data

def group_signal_columns(sample_cols: List[str]) -> Tuple[str, Dict[str, List[str]]]:
    """Group columns into signals.

    - new format: group by removing replicate field (averaging replicates)
    - legacy: group by prefix before _repN

    Returns (format, {label: [columns...]})
    """
    fmt = detect_format(sample_cols)

    if fmt == "new":
        groups: Dict[str, List[str]] = {}
        for c in sample_cols:
            sc = parse_new_sample_col(c)
            groups.setdefault(sc.group_label, []).append(c)
        return fmt, groups

    groups = {}
    for c in sample_cols:
        m = LEGACY_REP_RE.match(c)
        assert m is not None
        prefix = m.group("prefix")
        groups.setdefault(prefix, []).append(c)
    return fmt, groups


def csv_to_npz_dir(
    csv_path: str,
    fasta_path: str,
    out_dir: str,
    fasta_cache: Optional[str] = None,
    workers: int = 1,
    chunksize: int = 5_000_000,
) -> List[str]:
    """Convert ONE CSV into multiple NPZs using chunk-based streaming.

    Designed for very large CSVs (>=1e8 rows) with bounded memory.
    Semantic outputs are identical to the non-streaming version.
    """
    t_all = time.time()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    print(f"[PREPROCESS] CSV streaming mode (chunksize={chunksize:,})")

    # Load FASTA once
    t_fa = time.time()
    fasta_dict = load_fasta_dict(fasta_path, cache_path=fasta_cache)
    print(f"[TIME] FASTA load: {time.time()-t_fa:.2f} s | transcripts={len(fasta_dict):,}")
 
    # First pass: infer schema and signal groups from header only
    hdr = pd.read_csv(csv_path, nrows=0)
    sample_cols = validate_csv_columns(hdr.columns)
    fmt, groups = group_signal_columns(sample_cols)
    print(f"[PREPROCESS] Signal grouping: format={fmt} | groups={len(groups)}")

    written: List[str] = []
    stem = Path(csv_path).stem

    # Process each signal group independently to keep memory bounded
    for label in tqdm(sorted(groups.keys()), desc="Signal groups"):
        cols = groups[label]
        acc: Dict[str, dict] = {}
        rows_seen = 0
        chunk_i = 0

        usecols = ["transcript","region","start","end"] + cols
        t_sig = time.time()

        for chunk in tqdm(pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize),
                          desc=f"Chunks [{label}]", leave=False):
            rows_seen += len(chunk)
            chunk_i += 1
            if chunk_i % 5 == 0:
                print(f"[PREPROCESS] rows={rows_seen:,} | accumulated transcripts={len(acc):,}")

            # vectorized avg_count
            chunk["avg_count"] = chunk[cols].mean(axis=1).astype(float)

            # region masks
            region_l = chunk["region"].astype(str).str.lower()
            is5 = region_l.isin({"5utr","utr5","5'utr","5_utr"})
            is3 = region_l.isin({"3utr","utr3","3'utr","3_utr"})
            iscds = region_l.eq("cds")

            for tid, g in chunk.groupby("transcript", sort=False):
                # keep transcript ids consistent with FASTA keys
                tid = str(tid).strip()

                rec = acc.get(tid)
                if rec is None:
                    rec = {
                        "5utr":{"avg_count":[],"start":None,"end":None},
                        "cds":{"avg_count":[],"avg_count_norm":[],"start":None,"end":None},
                        "3utr":{"avg_count":[],"start":None,"end":None},
                    }
                    acc[tid]=rec

                # Keep per-transcript ordering consistent with the original
                # implementation (sort by start/end within transcript).
                g = g.sort_values(by=["start", "end"], kind="mergesort")

                # append coverage
                g5 = g[is5.loc[g.index]]
                g3 = g[is3.loc[g.index]]
                gc = g[iscds.loc[g.index]]

                if not g5.empty:
                    rec["5utr"]["avg_count"].extend(g5["avg_count"].tolist())
                    s5 = int(g5["start"].min())
                    e5 = int(g5["end"].max())
                    # IMPORTANT: do NOT overwrite bounds per chunk.
                    # We must keep global min/max across all chunks.
                    rec["5utr"]["start"] = s5 if rec["5utr"]["start"] is None else min(rec["5utr"]["start"], s5)
                    rec["5utr"]["end"]   = e5 if rec["5utr"]["end"] is None else max(rec["5utr"]["end"], e5)
                if not g3.empty:
                    rec["3utr"]["avg_count"].extend(g3["avg_count"].tolist())
                    s3 = int(g3["start"].min())
                    e3 = int(g3["end"].max())
                    rec["3utr"]["start"] = s3 if rec["3utr"]["start"] is None else min(rec["3utr"]["start"], s3)
                    rec["3utr"]["end"]   = e3 if rec["3utr"]["end"] is None else max(rec["3utr"]["end"], e3)
                if not gc.empty:
                    rec["cds"]["avg_count"].extend(gc["avg_count"].tolist())
                    sc = int(gc["start"].min())
                    ec = int(gc["end"].max())
                    rec["cds"]["start"] = sc if rec["cds"]["start"] is None else min(rec["cds"]["start"], sc)
                    rec["cds"]["end"]   = ec if rec["cds"]["end"] is None else max(rec["cds"]["end"], ec)

        # finalize sequences and normalization
        out_npz = {}
        for tid, rec in tqdm(acc.items(), desc=f"Finalize [{label}]", leave=False):
            seq = fasta_dict.get(tid, "")
            # sequences
            def sl(s,e):
                if s is None or e is None or e<=s:
                    return ""
                # slicing is safe even if e > len(seq)
                return seq[int(s):int(e)]
            rec["5utr"]["sequence"] = sl(rec["5utr"]["start"], rec["5utr"]["end"])
            rec["3utr"]["sequence"] = sl(rec["3utr"]["start"], rec["3utr"]["end"])
            cds_s, cds_e = rec["cds"]["start"], rec["cds"]["end"]
            rec["cds"]["sequence"] = sl(cds_s, cds_e)
            cds_len = max(1, (cds_e-cds_s) if cds_s is not None and cds_e is not None else 1)
            rec["cds"]["avg_count_norm"] = [v/cds_len for v in rec["cds"]["avg_count"]]
            out_npz[tid]=rec

        parts = str(label).split("|")
        safe_label = "_".join([p for p in parts if p])
        out_file = outp / f"{stem}__{safe_label}.npz"
        np.savez_compressed(out_file, **out_npz)
        print(f"[TIME] Saved {out_file.name} | rows={rows_seen:,} | transcripts={len(out_npz):,} "
              f"| signal_time={time.time()-t_sig:.2f}s")
        written.append(str(out_file))

    print(f"[TIME] TOTAL csv_to_npz_dir: {time.time()-t_all:.2f} s")
    return written
