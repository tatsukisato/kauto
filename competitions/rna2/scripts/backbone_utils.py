import re
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

_TRIPLET_RE = re.compile(r"^x_(\d+)$")

def _find_triplet_columns(df: pd.DataFrame) -> Tuple[str,str,str]:
    # prefer x_1,y_1,z_1 if present
    if {"x_1","y_1","z_1"}.issubset(df.columns):
        return "x_1","y_1","z_1"
    # otherwise pick the smallest index available
    for col in df.columns:
        m = _TRIPLET_RE.match(col)
        if m:
            idx = m.group(1)
            x = f"x_{idx}";
            y = f"y_{idx}";
            z = f"z_{idx}";
            if {x,y,z}.issubset(df.columns):
                return x,y,z
    raise ValueError("No coordinate triplet columns found in dataframe")

def extract_C1p_coords(labels_df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Extract C1' coords per target from a labels CSV DataFrame.

    Returns mapping target_id -> (coords_array (N,3), resid_list)
    """
    xcol,ycol,zcol = _find_triplet_columns(labels_df)
    # derive target_id from ID column (prefix before '_')
    if 'ID' not in labels_df.columns:
        raise ValueError("labels_df must contain 'ID' column")
    df = labels_df.copy()
    df['target_id'] = df['ID'].apply(lambda s: s.split('_')[0])
    df['resid'] = df['ID'].apply(lambda s: s.split('_')[-1])
    groups = {}
    for tid, g in df.groupby('target_id'):
        coords = g[[xcol,ycol,zcol]].replace({-1e+18: np.nan}).to_numpy(dtype=float)
        resid_list = g['resid'].astype(str).tolist()
        groups[tid] = (coords, resid_list)
    return groups

def extract_coords_from_submission(sub_df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Same as extract_C1p_coords but for submission/prediction DataFrame.

    Expects 'ID' and triplet columns.
    """
    return extract_C1p_coords(sub_df)


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({
        'ID': ['T1_1','T1_2'],
        'x_1': [1.0,2.0],'y_1':[0.0,0.0],'z_1':[0.0,0.0]
    })
    print(extract_C1p_coords(df))
