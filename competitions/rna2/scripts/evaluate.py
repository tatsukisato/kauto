"""Evaluation utilities for RNA TBM baseline.

Provides `evaluate_submission` extracted from the run_baseline notebook.
This module expects `backbone_utils` and `kabsch_utils` to be importable from the same
`competitions/rna2/scripts` package or from sys.path when used in the notebook.
"""
import os
from typing import Tuple, Optional

import pandas as pd
import numpy as np

try:
    # when running from the experiments notebook, scripts dir is on sys.path
    from backbone_utils import extract_C1p_coords, extract_coords_from_submission
except Exception:
    from competitions.rna2.scripts.backbone_utils import extract_C1p_coords, extract_coords_from_submission

try:
    from kabsch_utils import align_and_rmsd
except Exception:
    from competitions.rna2.scripts.kabsch_utils import align_and_rmsd

__all__ = ["evaluate_submission"]


def evaluate_submission(pred_df: Optional[pd.DataFrame] = None,
                        pred_csv: Optional[str] = None,
                        true_df: Optional[pd.DataFrame] = None,
                        true_csv: Optional[str] = None,
                        min_pairs: int = 3) -> Tuple[pd.DataFrame, dict]:
    """Evaluate a prediction against ground-truth labels.

    Returns (per_target_df, summary_dict). Supply either DataFrames or CSV paths.
    """
    if pred_df is None and pred_csv is None:
        raise ValueError('Provide pred_df or pred_csv')
    if true_df is None and true_csv is None:
        raise ValueError('Provide true_df or true_csv')
    if pred_df is None:
        pred_df = pd.read_csv(pred_csv)
    if true_df is None:
        true_df = pd.read_csv(true_csv)

    true_groups = extract_C1p_coords(true_df)
    pred_groups = extract_coords_from_submission(pred_df)
    results = []
    for tid, (true_coords, true_resids) in true_groups.items():
        pred_entry = pred_groups.get(tid)
        if pred_entry is None:
            results.append({'target_id': tid, 'n_matched': 0, 'mean_rmsd': None, 'median_rmsd': None, 'skipped': True})
            continue
        pred_coords, pred_resids = pred_entry
        true_map = {r: i for i, r in enumerate(true_resids)}
        pred_map = {r: i for i, r in enumerate(pred_resids)}
        common = sorted(set(true_map.keys()) & set(pred_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
        pairs = []
        for r in common:
            t_idx = true_map[r]
            p_idx = pred_map[r]
            tc = true_coords[t_idx]
            pc = pred_coords[p_idx]
            if np.any(np.isnan(tc)) or np.any(np.isnan(pc)):
                continue
            pairs.append((tc, pc))
        if len(pairs) < min_pairs:
            results.append({'target_id': tid, 'n_matched': len(pairs), 'mean_rmsd': None, 'median_rmsd': None, 'skipped': True})
            continue
        P = np.vstack([t for t, p in pairs])
        Q = np.vstack([p for t, p in pairs])
        Q_aligned, rmsd = align_and_rmsd(P, Q)
        results.append({'target_id': tid, 'n_matched': len(pairs), 'mean_rmsd': float(rmsd), 'median_rmsd': float(np.median(np.linalg.norm(P - Q_aligned, axis=1))), 'skipped': False})
    df = pd.DataFrame(results)
    summary = {
        'n_targets': int(df.shape[0]),
        'n_evaluated': int((~df['skipped']).sum()),
        'mean_rmsd': None if df['mean_rmsd'].dropna().empty else float(df['mean_rmsd'].dropna().mean()),
        'median_rmsd': None if df['mean_rmsd'].dropna().empty else float(df['mean_rmsd'].dropna().median()),
    }
    return df, summary
