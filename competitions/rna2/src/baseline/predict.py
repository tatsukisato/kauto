"""Prediction / submission builder for Phase1 TBM baseline."""
from typing import Callable
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
import sys

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


def generate_submission(test_seq_df: pd.DataFrame, repo, scorer: Callable, n_structures: int = 1, n_jobs: int = 1) -> pd.DataFrame:
    """Generate a Kaggle-style submission DataFrame by transferring coordinates from best templates.

    Produces one row per residue with columns: ID,resname,resid,x_1,y_1,z_1,... up to n_structures.
    If a residue is not mapped for a given candidate, the x/y/z are left as NaN.
    n_structures: number of top templates to transfer (1..5 recommended).
    """
    return generate_submission_parallel(test_seq_df, repo, scorer, n_structures=n_structures, n_jobs=n_jobs)


def _init_pool(templates_serial, prefilter_k, prefilter_top_n):
    global _TEMPLATES_SERIAL, _PREFILTER_K, _PREFILTER_TOP_N
    _TEMPLATES_SERIAL = templates_serial
    _PREFILTER_K = prefilter_k
    _PREFILTER_TOP_N = prefilter_top_n


# Local fallback initializers removed; generator module provides these.


# Local worker function removed; `generator.py` provides `_worker_process_query`.


def generate_submission_parallel(test_seq_df: pd.DataFrame, repo, scorer: Callable, n_structures: int = 1, n_jobs: int = None, chunk_size: int = 5) -> pd.DataFrame:
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
    pool = ctx.Pool(
        processes=n_jobs,
        initializer=_init_pool_with_scorer,
        initargs=(templates_serial, getattr(repo, 'config', {}), getattr(repo, 'prefilter_k', 4), getattr(repo, 'prefilter_top_n', 200), scorer),
    )
    try:
        tasks = [(str(row['target_id']), str(row['sequence']), n_structures) for _, row in test_seq_df.iterrows()]
        results = pool.map(_worker_process_query, tasks, chunksize=chunk_size)
    finally:
        pool.close()
        pool.join()

    # flatten
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
    return out_df[cols].reset_index(drop=True)
