"""Prediction / submission builder for Phase1 TBM baseline."""
from typing import Callable
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
import sys

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


def _init_pool_with_scorer(templates_serial, prefilter_k, prefilter_top_n, scorer):
    global _TEMPLATES_SERIAL, _PREFILTER_K, _PREFILTER_TOP_N, _SCORER
    _TEMPLATES_SERIAL = templates_serial
    _PREFILTER_K = prefilter_k
    _PREFILTER_TOP_N = prefilter_top_n
    _SCORER = scorer


def _worker_process_query(qtuple):
    # qtuple: (qid, qseq, n_structures)
    # do not import Bio.pairwise2 here to avoid deprecation warnings; import only in fallback paths
    # robust import: try common package layouts used in tests and scripts
    seq_identity = None
    try:
        from baseline.search import seq_identity as _si
        seq_identity = _si
    except Exception:
        try:
            from competitions.rna2.src.baseline.search import seq_identity as _si
            seq_identity = _si
        except Exception:
            # last-resort: try dynamic import by walking up from this file
            try:
                import importlib, pathlib, sys
                repo_root = pathlib.Path(__file__).resolve().parents[4]
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                mod = importlib.import_module('competitions.rna2.src.baseline.search')
                seq_identity = getattr(mod, 'seq_identity')
            except Exception:
                raise
    qid, qseq, n_structures = qtuple
    # prepare q k-mers
    k = _PREFILTER_K
    q_kmers = set()
    if qseq:
        if len(qseq) <= k:
            q_kmers.add(qseq)
        else:
            for i in range(len(qseq) - k + 1):
                q_kmers.add(qseq[i:i+k])

    # prefilter overlap
    overlaps = []
    for tid, tpl_seq, coord_cols, coords_arr, resnames, kmer_set in _TEMPLATES_SERIAL:
        tlen = len(tpl_seq)
        qlen = len(qseq)
        if tlen == 0:
            continue
        ratio = tlen / max(1, qlen)
        if ratio < 0.7 or ratio > 1.3:
            continue
        overlap = len(q_kmers & kmer_set) if kmer_set else 0
        overlaps.append((tid, tpl_seq, overlap, coord_cols, coords_arr, resnames))
    if not overlaps:
        return []
    overlaps.sort(key=lambda x: x[2], reverse=True)
    top_n = min(_PREFILTER_TOP_N, len(overlaps))
    shortlisted = overlaps[:top_n]

    # score on shortlist (use provided scorer when available)
    scored = []
    for tid, tpl_seq, _, coord_cols, coords_arr, resnames in shortlisted:
        try:
            sc = _SCORER(qseq, tpl_seq) if ('_SCORER' in globals() and _SCORER is not None) else seq_identity(qseq, tpl_seq)
        except Exception:
            sc = seq_identity(qseq, tpl_seq)
        scored.append((tid, tpl_seq, sc, coord_cols, coords_arr, resnames))
    scored.sort(key=lambda x: x[2], reverse=True)
    best = scored[:n_structures]

    # build residue-level rows
    rows = []
    structure_maps = []
    for (tid, tpl_seq, score, coord_cols, coords_arr, resnames) in best:
        resid_map = {}
        # Prefer PairwiseAligner for mapping if available
        try:
            from Bio.Align import PairwiseAligner
            aligner = PairwiseAligner()
            aligner.match_score = 1.0
            aligner.mismatch_score = 0.0
            aligner.open_gap_score = 0.0
            aligner.extend_gap_score = 0.0
            alns = aligner.align(qseq, tpl_seq)
            # avoid calling len() on the alignment collection (can be very large)
            aln = None
            for a in alns:
                aln = a
                break
            if aln is None:
                structure_maps.append(resid_map)
                continue
            q_blocks, t_blocks = aln.aligned
            for (qs, qe), (ts, te) in zip(q_blocks, t_blocks):
                for offset in range(qe - qs):
                    q_idx = qs + offset
                    t_idx = ts + offset
                    if coords_arr is not None and t_idx < len(coords_arr):
                        x, y, z = coords_arr[t_idx]
                        resn = resnames[t_idx] if t_idx < len(resnames) else ''
                        resid_map[str(q_idx + 1)] = (x, y, z, resn)
        except Exception as exc:
            raise RuntimeError("Bio.Align.PairwiseAligner is required for mapping; please install a recent Biopython") from exc
        structure_maps.append(resid_map)

    # assemble rows per residue
    for qi in range(1, len(qseq)+1):
        if qseq[qi-1] == '-':
            continue
        row = {'ID': f"{qid}_{qi}", 'resname': '', 'resid': qi}
        for s_map in structure_maps:
            if str(qi) in s_map:
                row['resname'] = s_map[str(qi)][3] or row['resname']
                break
        for si in range(n_structures):
            xk = f'x_{si+1}'
            yk = f'y_{si+1}'
            zk = f'z_{si+1}'
            val = structure_maps[si].get(str(qi)) if si < len(structure_maps) else None
            if val:
                x, y, z, _ = val
                row[xk] = x
                row[yk] = y
                row[zk] = z
            else:
                row[xk] = np.nan
                row[yk] = np.nan
                row[zk] = np.nan
        rows.append(row)
    return rows


def generate_submission_parallel(test_seq_df: pd.DataFrame, repo, scorer: Callable, n_structures: int = 1, n_jobs: int = None, chunk_size: int = 5) -> pd.DataFrame:
    # prepare serializable templates
    templates_serial = []
    for tid, info in repo.templates.items():
        templates_serial.append((tid, info.get('seq',''), info.get('coord_cols'), info.get('coords_arr'), info.get('resnames', []), info.get('kmer_set', set())))

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
        initargs=(templates_serial, getattr(repo, 'prefilter_k', 4), getattr(repo, 'prefilter_top_n', 200), scorer),
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
    # ensure columns order
    cols = ['ID','resname','resid']
    for si in range(n_structures):
        cols.extend([f'x_{si+1}', f'y_{si+1}', f'z_{si+1}'])
    for c in out_df.columns:
        if c not in cols:
            cols.append(c)
    return out_df[cols].reset_index(drop=True)
