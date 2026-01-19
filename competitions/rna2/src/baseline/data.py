"""Data loading utilities for baseline Phase1 implementation."""
from typing import Tuple
import pandas as pd


def _normalize_sequence_df(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names to expected `target_id` and `sequence`
    cols = {c: c for c in df.columns}
    if 'target_id' not in df.columns:
        if 'id' in df.columns:
            cols['id'] = 'target_id'
        elif 'ID' in df.columns:
            cols['ID'] = 'target_id'
    if 'sequence' not in df.columns:
        for c in df.columns:
            if c.lower() == 'sequence':
                cols[c] = 'sequence'
                break
    return df.rename(columns=cols)


def load_sequences(path: str) -> pd.DataFrame:
    """Load sequences CSV and return DataFrame with at least `target_id` and `sequence` columns."""
    df = pd.read_csv(path)
    df = _normalize_sequence_df(df)
    if 'target_id' not in df.columns or 'sequence' not in df.columns:
        raise ValueError(f"Expected columns 'target_id' and 'sequence' in {path}; got {list(df.columns)}")
    return df[['target_id', 'sequence']].copy()


def load_labels(path: str) -> pd.DataFrame:
    """Load labels / coordinates CSV. Ensures there's a `target_id` column for grouping.

    The function is tolerant: if `target_id` is missing but `ID` exists, it derives
    `target_id` by splitting `ID` on the first underscore.
    """
    df = pd.read_csv(path)
    if 'target_id' not in df.columns:
        if 'ID' in df.columns:
            df['target_id'] = df['ID'].astype(str).apply(lambda x: x.split('_')[0])
    return df
