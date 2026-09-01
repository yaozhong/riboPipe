from __future__ import annotations

import pandas as pd

from .schema import validate_csv_columns


def read_csv(path: str) -> pd.DataFrame:
    """Read CSV and validate its schema.

    Returns the DataFrame with no modifications.
    """
    df = pd.read_csv(path)
    validate_csv_columns(df.columns)
    return df
