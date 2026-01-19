"""Sequence scoring utilities for baseline Phase1."""
from typing import Tuple
import math

try:
    from Bio import pairwise2
except Exception:  # pragma: no cover - allow missing Biopython until runtime
    pairwise2 = None


def seq_identity(a: str, b: str) -> float:
    """Compute simple normalized identity using globalxx alignment.

    Returns matches / max(len(a), len(b)). Requires Biopython `pairwise2`.
    """
    if pairwise2 is None:
        # fallback: simple ungapped identity on prefix
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / max(1, max(len(a), len(b)))
    aln = pairwise2.align.globalxx(a, b, one_alignment_only=True)
    if not aln:
        return 0.0
    seqA, seqB, score, start, end = aln[0]
    matches = sum(1 for x, y in zip(seqA, seqB) if x == y)
    return matches / max(1, max(len(a), len(b)))
