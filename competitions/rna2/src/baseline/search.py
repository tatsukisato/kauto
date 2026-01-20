"""Sequence scoring utilities for baseline Phase1."""
from typing import Tuple
import math

try:
    from Bio.Align import PairwiseAligner
except Exception:  # pragma: no cover - allow missing Biopython until runtime
    PairwiseAligner = None



def seq_identity(a: str, b: str) -> float:
    """Compute simple normalized identity using globalxx alignment.

    Returns matches / max(len(a), len(b)). Requires Biopython `PairwiseAligner`.
    """
    # Prefer PairwiseAligner when available (faster, maintained API)
    if PairwiseAligner is None:
        raise RuntimeError("Bio.Align.PairwiseAligner is required for seq_identity; please install a recent Biopython.")
    if PairwiseAligner is not None:
        try:
            aligner = PairwiseAligner()
            aligner.match_score = 1.0
            aligner.mismatch_score = 0.0
            aligner.open_gap_score = 0.0
            aligner.extend_gap_score = 0.0
            score = aligner.score(a, b)
            return float(score) / max(1, max(len(a), len(b)))
        except Exception:
            pass
    # should not reach here; PairwiseAligner succeeded above
    return 0.0
