from __future__ import annotations

import os
import pickle
from typing import Dict, Optional

from Bio import SeqIO


def load_fasta_dict(fasta_path: str, cache_path: Optional[str] = None) -> Dict[str, str]:
    import pickle
    from pathlib import Path

    if cache_path and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    fasta_dict = {}

    with open(fasta_path) as f:
        header = None
        seq_lines = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header:
                    seq = "".join(seq_lines)
                    full_id = header.split("|")[0]
                    base_id = full_id.split(".")[0]

                    # 存完整版本
                    fasta_dict[full_id] = seq

                    # 如果还没存 base_id，就存 base_id
                    if base_id not in fasta_dict:
                        fasta_dict[base_id] = seq

                raw_id = line[1:].split()[0]
                header = raw_id
                seq_lines = []
            else:
                seq_lines.append(line.upper())

        if header:
            seq = "".join(seq_lines)
            full_id = header
            base_id = full_id.split(".")[0]

            fasta_dict[full_id] = seq
            if base_id not in fasta_dict:
                fasta_dict[base_id] = seq

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(fasta_dict, f)

    return fasta_dict
