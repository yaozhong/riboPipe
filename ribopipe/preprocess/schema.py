"""CSV schema validation (defensive).

New (recommended) sample column naming rule (STRICT):
  Celltype|genotype|treatment|replicate|type|remarks

Legacy (supported) sample column naming rule:
  <condition>_repN  (N is integer)

Relaxed fallback mode:
  - Auto-detect delimiter among ["|", "｜", "_", "-"]
  - Only require last token to be repN
  - Other missing fields filled with "x"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

# ---- Allowed values (edit to match your lab's conventions) ----
CELLTYPES  = {"HEK293T", "U2OS"}
TREATMENTS = {"NT", "DMSO", "ANS", "STM", "CC885"}
TYPES      = {"mono", "diso"}
GENOTYPES  = {"WT", "ZKO", "RNFKO", "NEMFKO"}

REPLICATE_RE = re.compile(r"^rep\d+$")
LEGACY_REP_RE = re.compile(r"^(?P<prefix>.+)_rep(?P<rep>\d+)$")

REQUIRED_COLS = {
    "transcript",
    "start",
    "end",
    "from_cds_start",
    "from_cds_stop",
    "region",
}

POSSIBLE_DELIMS = ["|", "｜", "_", "-"]


@dataclass(frozen=True)
class SampleColumn:
    raw: str
    celltype: str
    genotype: str
    treatment: str
    replicate: str
    sample_type: str
    remarks: str

    @property
    def group_key(self) -> Tuple[str, str, str, str, str]:
        return (self.celltype, self.genotype, self.treatment, self.sample_type, self.remarks)

    @property
    def group_label(self) -> str:
        return f"{self.celltype}|{self.genotype}|{self.treatment}|{self.sample_type}|{self.remarks}"


# ---------------- STRICT PARSER ----------------

def parse_new_sample_col(col: str) -> SampleColumn:
    parts = col.split("|")
    if len(parts) != 6:
        raise ValueError

    cell, geno, treat, rep, typ, remark = parts

    if cell not in CELLTYPES:
        raise ValueError
    if geno not in GENOTYPES:
        raise ValueError
    if treat not in TREATMENTS:
        raise ValueError
    if typ not in TYPES:
        raise ValueError
    if not REPLICATE_RE.fullmatch(rep):
        raise ValueError

    return SampleColumn(col, cell, geno, treat, rep, typ, remark)


# ---------------- FORMAT DETECTION ----------------

def _auto_loose_parse(sample_cols: List[str]) -> Optional[str]:
    """
    Try relaxed parsing:
    - last token must be repN
    - other fields can be anything
    """
    for delim in POSSIBLE_DELIMS:
        ok = 0
        for c in sample_cols:
            parts = c.split(delim)
            if len(parts) < 2:
                break

            rep = parts[-1]
            if not REPLICATE_RE.fullmatch(rep):
                break

            ok += 1

        if ok == len(sample_cols):
            print(f"[CSV WARNING] Auto-detected delimiter '{delim}' (relaxed mode)")
            return delim

    return None


def detect_format(sample_cols: Sequence[str]) -> str:
    if not sample_cols:
        raise ValueError("[CSV] No sample columns found")

    new_ok = 0
    legacy_ok = 0

    for c in sample_cols:
        if "|" in c:
            try:
                parse_new_sample_col(c)
                new_ok += 1
                continue
            except Exception:
                pass

        if LEGACY_REP_RE.match(c):
            legacy_ok += 1

    if new_ok == len(sample_cols):
        return "new"

    if legacy_ok == len(sample_cols):
        return "legacy"

    delim = _auto_loose_parse(list(sample_cols))
    if delim:
        return f"auto:{delim}"

    return "unknown"


# ---------------- VALIDATION ENTRY ----------------

def validate_csv_columns(columns: Iterable[str]) -> List[str]:
    columns = list(columns)

    missing = REQUIRED_COLS - set(columns)
    if missing:
        raise ValueError(f"[CSV] Missing required columns: {sorted(missing)}")

    sample_cols = [
        c for c in columns
        if c not in REQUIRED_COLS
        and not c.startswith("Unnamed")
        and c != ""
    ]

    if not sample_cols:
        raise ValueError("[CSV] No sample columns detected")

    fmt = detect_format(sample_cols)

    # ---- STRICT NEW ----
    if fmt == "new":
        for c in sample_cols:
            parse_new_sample_col(c)
        return sample_cols

    # ---- STRICT LEGACY ----
    if fmt == "legacy":
        for c in sample_cols:
            if not LEGACY_REP_RE.match(c):
                raise ValueError(f"[CSV] Legacy sample column must match '<condition>_repN': {c}")
        return sample_cols

    # ---- RELAXED AUTO MODE ----
    if fmt.startswith("auto:"):
        delim = fmt.split(":", 1)[1]
        print(f"[CSV WARNING] Using relaxed parsing with delimiter '{delim}'.")

        normalized = []

        for c in sample_cols:
            parts = c.split(delim)

            rep = parts[-1]
            if not REPLICATE_RE.fullmatch(rep):
                raise ValueError(f"[CSV] Cannot detect replicate in column '{c}'")

            core = parts[:-1]

            # ensure 5 fields before replicate
            if len(core) < 5:
                core = core + ["x"] * (5 - len(core))
            elif len(core) > 5:
                core = core[:4] + ["_".join(core[4:])]

            new_col = "|".join(core + [rep])
            normalized.append(new_col)

        return normalized

    # ---- FAILURE ----
    print("[CSV ERROR] Column names do not match expected formats.")
    for c in sample_cols[:5]:
        print("  ", c)

    raise ValueError(
        "[CSV] Unknown sample column format."
    )