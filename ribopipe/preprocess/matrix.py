from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _cds_length_from_item(item: dict) -> int:
    try:
        s = int(item.get("start", 0))
        e = int(item.get("end", 0))
        L = e - s
        return L if L > 0 else 1
    except Exception:
        return 1


def coverage_metric_from_cds(cds_dict: dict) -> float:
    if not cds_dict:
        return np.nan
    avg_norm = cds_dict.get("avg_count_norm", None)
    if avg_norm is not None and len(avg_norm) > 0:
        return float(np.sum(np.asarray(avg_norm, dtype=float)))

    avg = cds_dict.get("avg_count", None)
    if avg is not None and len(avg) > 0:
        L = _cds_length_from_item(cds_dict)
        return float(np.sum(np.asarray(avg, dtype=float))) / float(L)

    return np.nan


def load_npz_coverage_table(npz_path: Path) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=True)
    rows = []
    for tid in data.files:
        try:
            obj = data[tid].item()
        except Exception:
            continue
        cds = obj.get("cds", {})
        val = coverage_metric_from_cds(cds)
        if not np.isnan(val):
            rows.append({"transcript": tid, "metric": val})
    return pd.DataFrame(rows)


def label_from_filename(f: Path) -> str:
    return f.stem


def build_coverage_matrix(npz_dir: str, out_matrix_csv: str, write_sorted_tables: bool = True) -> str:
    d = Path(npz_dir)
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {npz_dir}")

    sample_series: Dict[str, pd.Series] = {}

    for f in files:
        df = load_npz_coverage_table(f)
        if df.empty:
            continue
        df_sorted = df.sort_values("metric", ascending=False).reset_index(drop=True)
        label = label_from_filename(f)
        if write_sorted_tables:
            out_file = d / f"{label}__coverage_sorted.csv"
            df_sorted.to_csv(out_file, index=False)
        s = pd.Series(df_sorted["metric"].values, index=df_sorted["transcript"].values, name=label)
        sample_series[label] = s

    if not sample_series:
        raise RuntimeError("No valid samples were produced; cannot assemble matrix")

    all_tids = sorted({tid for s in sample_series.values() for tid in s.index})
    labels = [label_from_filename(f) for f in files if label_from_filename(f) in sample_series]

    matrix = pd.DataFrame(index=all_tids, columns=labels, dtype=float)
    for lab in labels:
        col = sample_series[lab]
        matrix.loc[col.index, lab] = col.values

    out_path = Path(out_matrix_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_path)
    return str(out_path)
