"""Prediction / submission builder for Phase1 TBM baseline."""
from typing import Callable
import pandas as pd
import numpy as np

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
    """Generate a Kaggle-style submission DataFrame by transferring coordinates from best templates.

    Produces one row per residue with columns: ID,resname,resid,x_1,y_1,z_1,... up to n_structures.
    If a residue is not mapped for a given candidate, the x/y/z are left as NaN.
    n_structures: number of top templates to transfer (1..5 recommended).
    """
    if n_structures < 1:
        raise ValueError("n_structures must be >= 1")

    out_rows = []
    for _, qrow in test_seq_df.iterrows():
        qid = str(qrow['target_id'])
        qseq = str(qrow['sequence'])
        # get top templates
        best = repo.find_best(qseq, scorer, top_k=n_structures)
        if not best:
            continue

        # prepare per-structure mapping: list of dict resid->(x,y,z)
        structure_maps = []
        for (tid, tpl_seq, score) in best:
            tpl = repo.get(tid)
            if tpl is None:
                structure_maps.append({})
                continue
            coords_df = tpl.get('coords')
            coords_arr = tpl.get('coords_arr')
            resnames = tpl.get('resnames', [])
            if (coords_df is None or coords_df.empty) and coords_arr is None:
                structure_maps.append({})
                continue

            # determine coordinate columns (prefer x_1,y_1,z_1)
            coord_cols = None
            if {'x_1','y_1','z_1'}.issubset(coords_df.columns):
                coord_cols = ('x_1','y_1','z_1')
            else:
                trip = [c for c in coords_df.columns if c.startswith('x_')]
                if trip:
                    idx = trip[0].split('_')[1]
                    coord_cols = (f'x_{idx}', f'y_{idx}', f'z_{idx}')
            if coord_cols is None:
                structure_maps.append({})
                continue

            resid_map = {}
            if pairwise2 is not None:
                aln = pairwise2.align.globalxx(qseq, tpl_seq, one_alignment_only=True)
                if not aln:
                    structure_maps.append({})
                    continue
                seqA, seqB, sc, st, en = aln[0]
                t_idx = 0
                q_idx = 0
                for qa, ta in zip(seqA, seqB):
                    if qa != '-':
                        q_idx += 1
                    if ta != '-':
                        # get coords from precomputed array if available
                        if coords_arr is not None and t_idx < coords_arr.shape[0]:
                            x, y, z = coords_arr[t_idx]
                            resn = resnames[t_idx] if t_idx < len(resnames) else ''
                        else:
                            try:
                                tpl_row = coords_df.iloc[t_idx]
                                x = tpl_row.get(coord_cols[0], np.nan)
                                y = tpl_row.get(coord_cols[1], np.nan)
                                z = tpl_row.get(coord_cols[2], np.nan)
                                resn = tpl_row.get('resname', '') if 'resname' in tpl_row.index else ''
                            except Exception:
                                t_idx += 1
                                continue
                    if qa != '-' and ta != '-':
                        resid_map[str(q_idx)] = (x, y, z, resn)
                    if ta != '-':
                        t_idx += 1
                structure_maps.append(resid_map)
            else:
                # naive index-wise mapping
                if coords_arr is not None:
                    L = min(len(qseq), len(tpl_seq), coords_arr.shape[0])
                    for i in range(L):
                        if qseq[i] != '-' and tpl_seq[i] != '-':
                            x, y, z = coords_arr[i]
                            resn = resnames[i] if i < len(resnames) else ''
                            resid_map[str(i+1)] = (x, y, z, resn)
                else:
                    L = min(len(qseq), len(tpl_seq), len(coords_df))
                    for i in range(L):
                        if qseq[i] != '-' and tpl_seq[i] != '-':
                            tpl_row = coords_df.iloc[i]
                            x, y, z = tpl_row.get(coord_cols[0], np.nan), tpl_row.get(coord_cols[1], np.nan), tpl_row.get(coord_cols[2], np.nan)
                            resid_map[str(i+1)] = (x, y, z, tpl_row.get('resname','') if 'resname' in tpl_row.index else '')
                structure_maps.append(resid_map)

        # build output rows for each residue position in query sequence
        for qi in range(1, len(qseq) + 1):
            if qseq[qi-1] == '-':
                continue
            row = {'ID': f"{qid}_{qi}", 'resname': '', 'resid': qi}
            # try to fill resname from first available map
            for s_map in structure_maps:
                if str(qi) in s_map:
                    res_entry = s_map[str(qi)]
                    # res_entry is (x,y,z,resname)
                    if len(res_entry) >= 4:
                        row['resname'] = res_entry[3] or row['resname']
                    break

            # add coords for each candidate
            for si in range(n_structures):
                xk = f'x_{si+1}'
                yk = f'y_{si+1}'
                zk = f'z_{si+1}'
                val = structure_maps[si].get(str(qi)) if si < len(structure_maps) else None
                if val:
                    x, y, z, _ = val
                    row[xk] = x if not pd.isna(x) else np.nan
                    row[yk] = y if not pd.isna(y) else np.nan
                    row[zk] = z if not pd.isna(z) else np.nan
                else:
                    row[xk] = np.nan
                    row[yk] = np.nan
                    row[zk] = np.nan

            out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()
    out_df = pd.DataFrame(out_rows)
    # ensure columns order: ID,resname,resid, x_1,y_1,z_1,...
    cols = ['ID','resname','resid']
    for si in range(n_structures):
        cols.extend([f'x_{si+1}', f'y_{si+1}', f'z_{si+1}'])
    # keep other columns if present
    for c in out_df.columns:
        if c not in cols:
            cols.append(c)
    return out_df[cols].reset_index(drop=True)
