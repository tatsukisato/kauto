"""TemplateRepository: store templates and find best matches for query sequences."""
from typing import Callable, Dict, List, Tuple, Any
import pandas as pd
import numpy as np


class TemplateRepository:
    """A minimal in-memory template repository.

    templates: dict mapping target_id -> {'seq': str, 'coords': DataFrame}
    """

    def __init__(self, length_ratio_min: float = 0.7, length_ratio_max: float = 1.3):
        self.length_ratio_min = length_ratio_min
        self.length_ratio_max = length_ratio_max
        # prefilter settings
        self.prefilter_k = 4
        self.prefilter_top_n = 200
        self.templates: Dict[str, Dict[str, Any]] = {}

    def fit(self, seq_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
        """Populate templates from sequence and labels dataframes.

        seq_df: DataFrame with `target_id` and `sequence`.
        labels_df: DataFrame with `target_id` and coordinate columns; rows correspond to residues.
        """
        # ensure labels have target_id
        if 'target_id' not in labels_df.columns:
            if 'ID' in labels_df.columns:
                labels_df = labels_df.copy()
                labels_df['target_id'] = labels_df['ID'].astype(str).apply(lambda x: x.split('_')[0])

        # index labels by target_id
        grouped = {k: g.copy().reset_index(drop=True) for k, g in labels_df.groupby('target_id')}

        for _, row in seq_df.iterrows():
            tid = str(row['target_id'])
            seq = str(row['sequence'])
            coords = grouped.get(tid, pd.DataFrame())
            entry: Dict[str, Any] = {'seq': seq, 'coords': coords}
            # Precompute coordinate arrays for faster access in prediction
            if not coords.empty:
                # find coordinate triplet columns
                coord_cols = None
                if {'x_1', 'y_1', 'z_1'}.issubset(coords.columns):
                    coord_cols = ('x_1', 'y_1', 'z_1')
                else:
                    trip = [c for c in coords.columns if c.startswith('x_')]
                    if trip:
                        idx = trip[0].split('_')[1]
                        coord_cols = (f'x_{idx}', f'y_{idx}', f'z_{idx}')

                if coord_cols is not None:
                    try:
                        arr = coords.loc[:, list(coord_cols)].replace({-1e+18: np.nan}).to_numpy(dtype=float)
                    except Exception:
                        arr = None
                    entry['coord_cols'] = coord_cols
                    entry['coords_arr'] = arr
                else:
                    entry['coord_cols'] = None
                    entry['coords_arr'] = None

                # extract resids and resnames if present
                if 'ID' in coords.columns:
                    resids = coords['ID'].astype(str).apply(lambda x: x.split('_')[-1]).tolist()
                    entry['resids'] = resids
                else:
                    entry['resids'] = []
                if 'resname' in coords.columns:
                    entry['resnames'] = coords['resname'].astype(str).tolist()
                else:
                    entry['resnames'] = [''] * len(entry.get('resids', []))
            else:
                entry['coord_cols'] = None
                entry['coords_arr'] = None
                entry['resids'] = []
                entry['resnames'] = []

            # compute k-mer set for prefiltering
            k = self.prefilter_k
            seq_kmers = set()
            if seq:
                if len(seq) <= k:
                    seq_kmers.add(seq)
                else:
                    for i in range(len(seq) - k + 1):
                        seq_kmers.add(seq[i:i+k])
            entry['kmer_set'] = seq_kmers

            self.templates[tid] = entry

    def get(self, target_id: str) -> Dict[str, Any]:
        return self.templates.get(target_id)

    def find_best(self, query_seq: str, scorer: Callable[[str, str], float], top_k: int = 1) -> List[Tuple[str, str, float]]:
        """Linear scan to find top_k templates by scorer. Applies length ratio filter.

        Returns list of (target_id, template_seq, score) sorted descending by score.
        """
        qlen = len(query_seq)
        # build query k-mer set
        k = self.prefilter_k
        q_kmers = set()
        if query_seq:
            if len(query_seq) <= k:
                q_kmers.add(query_seq)
            else:
                for i in range(len(query_seq) - k + 1):
                    q_kmers.add(query_seq[i:i+k])

        # compute simple overlap score for prefilter
        candidates_overlap: List[Tuple[str, str, int]] = []
        for tid, info in self.templates.items():
            tpl_seq = info.get('seq', '')
            tlen = len(tpl_seq)
            if tlen == 0:
                continue
            ratio = tlen / max(1, qlen)
            if ratio < self.length_ratio_min or ratio > self.length_ratio_max:
                continue
            kset = info.get('kmer_set', set())
            overlap = len(q_kmers & kset) if kset else 0
            candidates_overlap.append((tid, tpl_seq, overlap))

        if not candidates_overlap:
            return []

        # shortlist by overlap
        candidates_overlap.sort(key=lambda x: x[2], reverse=True)
        top_n = min(self.prefilter_top_n, len(candidates_overlap))
        shortlisted = candidates_overlap[:top_n]

        # compute expensive scorer only on shortlisted templates
        scored: List[Tuple[str, str, float]] = []
        for tid, tpl_seq, _ in shortlisted:
            score = scorer(query_seq, tpl_seq)
            scored.append((tid, tpl_seq, float(score)))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]
