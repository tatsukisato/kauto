"""TemplateRepository: store templates and find best matches for query sequences."""
from typing import Callable, Dict, List, Tuple, Any
import pandas as pd


class TemplateRepository:
    """A minimal in-memory template repository.

    templates: dict mapping target_id -> {'seq': str, 'coords': DataFrame}
    """

    def __init__(self, length_ratio_min: float = 0.7, length_ratio_max: float = 1.3):
        self.length_ratio_min = length_ratio_min
        self.length_ratio_max = length_ratio_max
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
            # store even if coords empty; downstream will handle missing coords
            self.templates[tid] = {'seq': seq, 'coords': coords}

    def get(self, target_id: str) -> Dict[str, Any]:
        return self.templates.get(target_id)

    def find_best(self, query_seq: str, scorer: Callable[[str, str], float], top_k: int = 1) -> List[Tuple[str, str, float]]:
        """Linear scan to find top_k templates by scorer. Applies length ratio filter.

        Returns list of (target_id, template_seq, score) sorted descending by score.
        """
        cand: List[Tuple[str, str, float]] = []
        qlen = len(query_seq)
        for tid, info in self.templates.items():
            tpl_seq = info.get('seq', '')
            tlen = len(tpl_seq)
            if tlen == 0:
                continue
            ratio = tlen / max(1, qlen)
            if ratio < self.length_ratio_min or ratio > self.length_ratio_max:
                continue
            score = scorer(query_seq, tpl_seq)
            cand.append((tid, tpl_seq, float(score)))
        cand.sort(key=lambda x: x[2], reverse=True)
        return cand[:top_k]
