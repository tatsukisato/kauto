"""Prediction / submission builder for Phase1 TBM baseline."""
from typing import Callable
import pandas as pd

try:
    from Bio import pairwise2
except Exception:
    pairwise2 = None


def _copy_and_reid(row: pd.Series, new_target_id: str) -> pd.Series:
    row = row.copy()
    if 'ID' in row.index:
        parts = str(row['ID']).split('_')
        resid = parts[-1]
        row['ID'] = f"{new_target_id}_{resid}"
    return row


def generate_submission(test_seq_df: pd.DataFrame, repo, scorer: Callable, n_structures: int = 1) -> pd.DataFrame:
    """Generate a Kaggle-style submission DataFrame by transferring coordinates from best template.

    Fallback behavior when Biopython is unavailable:
    - Use simple position-wise mapping up to min(len(query), len(template)).
    """
    out_rows = []
    for _, qrow in test_seq_df.iterrows():
        qid = str(qrow['target_id'])
        qseq = str(qrow['sequence'])
        best = repo.find_best(qseq, scorer, top_k=1)
        if not best:
            continue
        tid, tpl_seq, score = best[0]
        tpl = repo.get(tid)
        if tpl is None:
            continue
        coords_df = tpl.get('coords')
        if coords_df is None or coords_df.empty:
            continue

        if pairwise2 is not None:
            aln = pairwise2.align.globalxx(qseq, tpl_seq, one_alignment_only=True)
            if not aln:
                continue
            seqA, seqB, sc, st, en = aln[0]
            t_idx = 0
            for qa, ta in zip(seqA, seqB):
                if ta != '-':
                    try:
                        tpl_row = coords_df.iloc[t_idx]
                    except Exception:
                        t_idx += 1
                        continue
                    if qa != '-' and ta != '-':
                        new_row = _copy_and_reid(tpl_row, qid)
                        out_rows.append(new_row)
                    t_idx += 1
        else:
            # fallback: naive index-wise mapping
            L = min(len(qseq), len(tpl_seq), len(coords_df))
            for i in range(L):
                if qseq[i] != '-' and tpl_seq[i] != '-':
                    tpl_row = coords_df.iloc[i]
                    new_row = _copy_and_reid(tpl_row, qid)
                    out_rows.append(new_row)

    if not out_rows:
        return pd.DataFrame()
    out_df = pd.DataFrame(out_rows)
    return out_df.reset_index(drop=True)
