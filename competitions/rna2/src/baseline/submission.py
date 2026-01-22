"""Submission helper: template serialization and post-processing utilities.

This module extracts template serialization and post-processing (gap-filling,
coverage summary) from `predict.py` to keep that file concise.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd


def prepare_templates_serial(repo) -> List[Tuple]:
    """Create the serializable templates list used by worker initializer.

    Returns a list of tuples: (tid, seq, coord_cols, coords_arr, resnames, kmer_set)
    """
    templates_serial = []
    if repo is None:
        return templates_serial
    for tid, info in getattr(repo, 'templates', {}).items():
        templates_serial.append(
            (
                tid,
                info.get('seq', ''),
                info.get('coord_cols'),
                info.get('coords_arr'),
                info.get('resnames', []),
                info.get('kmer_set', set()),
            )
        )
    return templates_serial


def postprocess_submission(out_df: pd.DataFrame, repo, n_structures: int) -> pd.DataFrame:
    """Apply gap-filling per-target/per-structure and persist coverage summary.

    Returns the modified DataFrame.
    """
    try:
        try:
            from baseline import template_utils as _tu
        except Exception:
            from competitions.rna2.src.baseline import template_utils as _tu
    except Exception:
        _tu = None

    cfg = getattr(repo, 'config', {}) if repo is not None else {}
    fill_enabled = cfg.get('fill_gaps', False) if hasattr(cfg, 'get') else False
    max_interp_length = cfg.get('max_interp_length', 5) if hasattr(cfg, 'get') else 5
    interp_method = cfg.get('interp_method', 'linear') if hasattr(cfg, 'get') else 'linear'
    neighbor_k = cfg.get('neighbor_k', 2) if hasattr(cfg, 'get') else 2

    coverage_rows = []
    if _tu is not None and not out_df.empty:
        ids = out_df['ID'].astype(str)
        target_ids = ids.apply(lambda s: s.rsplit('_', 1)[0])
        out_df['_target_id_tmp'] = target_ids
        for tid, group in out_df.groupby('_target_id_tmp'):
            idx = group.index
            stats = {}
            for si in range(n_structures):
                cols_xyz = [f'x_{si+1}', f'y_{si+1}', f'z_{si+1}']
                coords = out_df.loc[idx, cols_xyz].to_numpy(dtype=float)
                raw_missing = _tu.missing_rate(coords)
                new_coords = coords
                if fill_enabled:
                    try:
                        new_coords = _tu.fill_gaps_pipeline(
                            coords, max_interp_length=max_interp_length, interp_method=interp_method, neighbor_k=neighbor_k
                        )
                    except Exception:
                        new_coords = coords
                post_missing = _tu.missing_rate(new_coords)
                out_df.loc[idx, cols_xyz] = new_coords
                stats[f'raw_missing_{si+1}'] = raw_missing
                stats[f'post_missing_{si+1}'] = post_missing
            stats['target_id'] = tid
            coverage_rows.append(stats)
        out_df.drop(columns=['_target_id_tmp'], inplace=True)
        # persist summary for later analysis
        if coverage_rows:
            try:
                import pathlib
                import os
                summary_df = pd.DataFrame(coverage_rows)
                repo_root = pathlib.Path(__file__).resolve().parents[2]
                experiments_dir = repo_root / 'experiments'
                os.makedirs(experiments_dir, exist_ok=True)
                summary_path = experiments_dir / 'interp_coverage_summary.csv'
                summary_df.to_csv(str(summary_path), index=False)
            except Exception:
                pass

    return out_df
