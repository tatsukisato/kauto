"""Prediction / submission builder for Phase1 TBM baseline."""
from typing import Callable, Optional
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
import sys
import json
from pathlib import Path

# generator-based worker (require this module to be present)
from baseline.generator import init_pool_with_scorer as _init_pool_with_scorer, worker_process_query as _worker_process_query

# pure-Python fallback removed: PairwiseAligner is required for mapping


def _copy_and_reid(row: pd.Series, new_target_id: str) -> pd.Series:
    row = row.copy()
    if 'ID' in row.index:
        parts = str(row['ID']).split('_')
        resid = parts[-1]
        row['ID'] = f"{new_target_id}_{resid}"
    return row


def generate_submission(
    test_seq_df: pd.DataFrame,
    repo,
    scorer: Callable,
    n_structures: int = 1,
    n_jobs: int = 1,
    diagnostic_output_path: Optional[str] = None
) -> pd.DataFrame:
    """Generate a Kaggle-style submission DataFrame by transferring coordinates from best templates.

    Produces one row per residue with columns: ID,resname,resid,x_1,y_1,z_1,... up to n_structures.
    If a residue is not mapped for a given candidate, the x/y/z are left as NaN.
    n_structures: number of top templates to transfer (1..5 recommended).
    diagnostic_output_path: If provided, save diagnostic logs to this path (JSON Lines format).
    """
    return generate_submission_parallel(
        test_seq_df, repo, scorer,
        n_structures=n_structures,
        n_jobs=n_jobs,
        diagnostic_output_path=diagnostic_output_path
    )


def _init_pool(templates_serial, prefilter_k, prefilter_top_n):
    global _TEMPLATES_SERIAL, _PREFILTER_K, _PREFILTER_TOP_N
    _TEMPLATES_SERIAL = templates_serial
    _PREFILTER_K = prefilter_k
    _PREFILTER_TOP_N = prefilter_top_n


# Local fallback initializers removed; generator module provides these.


# Local worker function removed; `generator.py` provides `_worker_process_query`.


def generate_submission_parallel(
    test_seq_df: pd.DataFrame,
    repo,
    scorer: Callable,
    n_structures: int = 1,
    n_jobs: int = None,
    chunk_size: int = 5,
    diagnostic_output_path: Optional[str] = None
) -> pd.DataFrame:
    # prepare serializable templates (moved to submission helper)
    try:
        from baseline import submission as _submission
    except Exception:
        try:
            from competitions.rna2.src.baseline import submission as _submission
        except Exception:
            _submission = None

    templates_serial = _submission.prepare_templates_serial(repo) if _submission is not None else []

    # default to number of CPUs
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    # initializer sets globals in worker
    # Use 'fork' start method on Unix to avoid spawn trying to re-run '<stdin>' in interactive runs.
    try:
        ctx = multiprocessing.get_context('fork') if sys.platform != 'win32' else multiprocessing.get_context()
    except Exception:
        ctx = multiprocessing.get_context()
    # Enable diagnostic mode if output path is provided
    config = getattr(repo, 'config', {}) or {}
    if diagnostic_output_path:
        if isinstance(config, dict):
            config = config.copy()
            config['diagnostic'] = True
        else:
            # If config is not a dict, create a new dict with diagnostic flag
            import copy
            config = copy.copy(config) if hasattr(config, '__dict__') else {}
            if hasattr(config, '__setitem__'):
                config['diagnostic'] = True
            else:
                setattr(config, 'diagnostic', True)

    pool = ctx.Pool(
        processes=n_jobs,
        initializer=_init_pool_with_scorer,
        initargs=(templates_serial, config, getattr(repo, 'prefilter_k', 4), getattr(repo, 'prefilter_top_n', 200), scorer),
    )
    try:
        tasks = [(str(row['target_id']), str(row['sequence']), n_structures) for _, row in test_seq_df.iterrows()]
        results = pool.map(_worker_process_query, tasks, chunksize=chunk_size)
    finally:
        pool.close()
        pool.join()

    # Separate diagnostic logs if diagnostic mode is enabled
    diagnostics = []
    if diagnostic_output_path:
        # Results are (rows, diag) tuples
        flat = []
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                rows, diag = result
                flat.extend(rows)
                if diag:
                    diagnostics.append(diag)
            else:
                # Fallback for non-diagnostic results
                flat.extend(result if isinstance(result, list) else [result])
    else:
        # flatten normally
        flat = [r for grp in results for r in grp]
    if not flat:
        return pd.DataFrame()
    out_df = pd.DataFrame(flat)
    # postprocess (gap-filling, coverage summary) delegated to submission helper
    try:
        try:
            from baseline import submission as _submission
        except Exception:
            from competitions.rna2.src.baseline import submission as _submission
    except Exception:
        _submission = None

    if _submission is not None:
        try:
            out_df = _submission.postprocess_submission(out_df, repo, n_structures)
        except Exception:
            # keep original out_df on failure
            pass
    # ensure columns order
    cols = ['ID','resname','resid']
    for si in range(n_structures):
        cols.extend([f'x_{si+1}', f'y_{si+1}', f'z_{si+1}'])
    for c in out_df.columns:
        if c not in cols:
            cols.append(c)

    # Save diagnostic logs if path is provided
    if diagnostic_output_path and diagnostics:
        output_path = Path(diagnostic_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for diag in diagnostics:
                f.write(json.dumps(diag) + '\n')

    return out_df[cols].reset_index(drop=True)
