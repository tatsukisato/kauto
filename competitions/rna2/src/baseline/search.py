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


def local_align_blocks(a: str, b: str, min_identity: float = 0.5, min_length: int = 5):
    """Find local alignment blocks between `a` (query) and `b` (template).

    Returns a list of blocks dicts: {'q_start':, 'q_end':, 't_start':, 't_end':, 'length':, 'identity':}
    Requires Biopython `PairwiseAligner`.
    """
    if PairwiseAligner is None:
        raise RuntimeError("Bio.Align.PairwiseAligner is required for local_align_blocks; please install Biopython.")
    try:
        aligner = PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        # negative gap penalties for Smith-Waterman style behaviour
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = -0.5
        alns = aligner.align(a, b)
    except Exception as exc:
        raise RuntimeError("PairwiseAligner failed for local alignment") from exc

    blocks = []
    # iterate alignments (keep first few alignments)
    for aln in alns:
        q_blocks, t_blocks = aln.aligned
        for (qs, qe), (ts, te) in zip(q_blocks, t_blocks):
            length = qe - qs
            if length <= 0:
                continue
            # compute identity within the block by direct comparison
            matches = 0
            for i in range(length):
                if a[qs + i] == b[ts + i]:
                    matches += 1
            identity = matches / float(length)
            if identity >= min_identity and length >= min_length:
                blocks.append({'q_start': qs, 'q_end': qe, 't_start': ts, 't_end': te, 'length': length, 'identity': identity})
        # do not iterate too many alignments; take candidate blocks from top alignment only
        if blocks:
            break
    return blocks


def local_align_and_map(query: str, tpl: str, tpl_coords, resnames=None, min_identity: float = 0.5, min_length: int = 5):
    """Map template coordinates to query positions using local alignment blocks.

    Returns a dict mapping 1-based query residue index (str) -> (x,y,z,resname).
    """
    mapping = {}
    if not query or not tpl or tpl_coords is None:
        return mapping
    blocks = local_align_blocks(query, tpl, min_identity=min_identity, min_length=min_length)
    for blk in blocks:
        qs, qe = blk['q_start'], blk['q_end']
        ts, te = blk['t_start'], blk['t_end']
        for offset in range(qe - qs):
            q_idx = qs + offset
            t_idx = ts + offset
            if t_idx < len(tpl_coords):
                x, y, z = tpl_coords[t_idx]
                resn = resnames[t_idx] if (resnames is not None and t_idx < len(resnames)) else ''
                # store with 1-based index string to match predict.py mapping convention
                # include t_idx for downstream transform updates
                mapping[str(q_idx + 1)] = (x, y, z, resn, t_idx)
    return mapping
