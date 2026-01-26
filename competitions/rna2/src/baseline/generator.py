"""Submission generator: encapsulates worker mapping and pool initializer.

Provides:
- `init_pool_with_scorer(templates_serial, config, prefilter_k, prefilter_top_n, scorer)`
- `worker_process_query(qtuple)` for use with multiprocessing.Pool.map

The implementation mirrors the previous logic from `predict.py` but groups
state into a class so the code is easier to test and reason about.
"""
from typing import Callable
import numpy as np


class SubmissionGenerator:
    # class-level state set by initializer (safe for forked workers)
    TEMPLATES_SERIAL = []
    PREFILTER_K = 4
    PREFILTER_TOP_N = 200
    SCORER = None
    LENGTH_RATIO_MIN = 0.7
    LENGTH_RATIO_MAX = 1.3
    CONFIG = {}
    GLOBAL_THRESHOLD = 0.6
    LOCAL_MIN_IDENTITY = 0.5
    MIN_LOCAL_LENGTH = 5
    USE_KABSCH = False
    KABSCH_MIN_PAIRS = 3
    KABSCH_MIN_COVERAGE = 0.3
    DIAGNOSTIC = False  # Enable diagnostic logging
    # Phase 3a: Local alignment activation parameters
    LARGE_SEQ_THRESHOLD = 500
    FORCE_LOCAL_FOR_LARGE = True
    MIN_COVERAGE_FOR_SKIP_LOCAL = 0.7

    @classmethod
    def init_pool_with_scorer(cls, templates_serial, config, prefilter_k, prefilter_top_n, scorer: Callable):
        cls.TEMPLATES_SERIAL = templates_serial
        cls.PREFILTER_K = prefilter_k
        cls.PREFILTER_TOP_N = prefilter_top_n
        cls.SCORER = scorer
        cls.CONFIG = config or {}

        try:
            cls.LENGTH_RATIO_MIN = config.get('length_ratio_min', 0.7) if hasattr(config, 'get') else getattr(config, 'length_ratio_min', 0.7)
            cls.LENGTH_RATIO_MAX = config.get('length_ratio_max', 1.3) if hasattr(config, 'get') else getattr(config, 'length_ratio_max', 1.3)
        except Exception:
            cls.LENGTH_RATIO_MIN = 0.7
            cls.LENGTH_RATIO_MAX = 1.3

        try:
            cls.GLOBAL_THRESHOLD = config.get('global_threshold', 0.6) if hasattr(config, 'get') else getattr(config, 'global_threshold', 0.6)
            cls.LOCAL_MIN_IDENTITY = config.get('local_min_identity', 0.5) if hasattr(config, 'get') else getattr(config, 'local_min_identity', 0.5)
            cls.MIN_LOCAL_LENGTH = config.get('min_local_length', 5) if hasattr(config, 'get') else getattr(config, 'min_local_length', 5)
        except Exception:
            cls.GLOBAL_THRESHOLD = 0.6
            cls.LOCAL_MIN_IDENTITY = 0.5
            cls.MIN_LOCAL_LENGTH = 5

        try:
            cls.USE_KABSCH = config.get('use_kabsch', False) if hasattr(config, 'get') else getattr(config, 'use_kabsch', False)
            cls.KABSCH_MIN_PAIRS = config.get('kabsch_min_pairs', 3) if hasattr(config, 'get') else getattr(config, 'kabsch_min_pairs', 3)
            cls.KABSCH_MIN_COVERAGE = config.get('kabsch_min_coverage', 0.3) if hasattr(config, 'get') else getattr(config, 'kabsch_min_coverage', 0.3)
        except Exception:
            cls.USE_KABSCH = False
            cls.KABSCH_MIN_PAIRS = 3
            cls.KABSCH_MIN_COVERAGE = 0.3

        # Phase 3a: Local alignment activation parameters
        try:
            cls.LARGE_SEQ_THRESHOLD = config.get('large_seq_threshold', 500) if hasattr(config, 'get') else getattr(config, 'large_seq_threshold', 500)
            cls.FORCE_LOCAL_FOR_LARGE = config.get('force_local_for_large', True) if hasattr(config, 'get') else getattr(config, 'force_local_for_large', True)
            cls.MIN_COVERAGE_FOR_SKIP_LOCAL = config.get('min_coverage_for_skip_local', 0.7) if hasattr(config, 'get') else getattr(config, 'min_coverage_for_skip_local', 0.7)
        except Exception:
            cls.LARGE_SEQ_THRESHOLD = 500
            cls.FORCE_LOCAL_FOR_LARGE = True
            cls.MIN_COVERAGE_FOR_SKIP_LOCAL = 0.7

        # Diagnostic flag
        try:
            cls.DIAGNOSTIC = config.get('diagnostic', False) if hasattr(config, 'get') else getattr(config, 'diagnostic', False)
        except Exception:
            cls.DIAGNOSTIC = False

    @classmethod
    def process_query(cls, qtuple):
        # qtuple: (qid, qseq, n_structures)
        # local import helpers
        seq_identity = None
        try:
            from baseline.search import seq_identity as _si
            seq_identity = _si
        except Exception:
            try:
                from competitions.rna2.src.baseline.search import seq_identity as _si
                seq_identity = _si
            except Exception:
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

        # Initialize diagnostic log if enabled
        diag = None
        if cls.DIAGNOSTIC:
            diag = {
                'query_id': qid,
                'query_length': len(qseq),
                'template_id': None,
                'seq_identity': None,
                'used_local_align': False,
                'kabsch_applied': False,
                'raw_coverage': None,
                'final_coverage': None,
                # Phase 3a: Enhanced diagnostic fields
                'n_local_added': 0,
                'coverage_before_local': None,
                'coverage_after_local': None,
                'local_trigger_reason': None,
            }
        k = cls.PREFILTER_K
        q_kmers = set()
        if qseq:
            if len(qseq) <= k:
                q_kmers.add(qseq)
            else:
                for i in range(len(qseq) - k + 1):
                    q_kmers.add(qseq[i:i+k])

        overlaps = []
        for tid, tpl_seq, coord_cols, coords_arr, resnames, kmer_set in cls.TEMPLATES_SERIAL:
            tlen = len(tpl_seq)
            qlen = len(qseq)
            if tlen == 0:
                continue
            ratio = tlen / max(1, qlen)
            if ratio < cls.LENGTH_RATIO_MIN or ratio > cls.LENGTH_RATIO_MAX:
                continue
            overlap = len(q_kmers & kmer_set) if kmer_set else 0
            overlaps.append((tid, tpl_seq, overlap, coord_cols, coords_arr, resnames))
        if not overlaps:
            return []
        overlaps.sort(key=lambda x: x[2], reverse=True)
        top_n = min(cls.PREFILTER_TOP_N, len(overlaps))
        shortlisted = overlaps[:top_n]

        scored = []
        for tid, tpl_seq, _, coord_cols, coords_arr, resnames in shortlisted:
            try:
                sc = cls.SCORER(qseq, tpl_seq) if (cls.SCORER is not None) else seq_identity(qseq, tpl_seq)
            except Exception:
                sc = seq_identity(qseq, tpl_seq)
            scored.append((tid, tpl_seq, sc, coord_cols, coords_arr, resnames))
        scored.sort(key=lambda x: x[2], reverse=True)
        best = scored[:n_structures]

        # Record best template info in diagnostic log
        if diag is not None and best:
            diag['template_id'] = best[0][0]
            diag['seq_identity'] = float(best[0][2]) if best[0][2] is not None else None

        rows = []
        structure_maps = []
        for (tid, tpl_seq, score, coord_cols, coords_arr, resnames) in best:
            resid_map = {}
            # Prefer PairwiseAligner for mapping
            try:
                from Bio.Align import PairwiseAligner
                aligner = PairwiseAligner()
                aligner.match_score = 1.0
                aligner.mismatch_score = 0.0
                aligner.open_gap_score = 0.0
                aligner.extend_gap_score = 0.0
                alns = aligner.align(qseq, tpl_seq)
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
                            resid_map[str(q_idx + 1)] = (x, y, z, resn, t_idx)
            except Exception as exc:
                raise RuntimeError("Bio.Align.PairwiseAligner is required for mapping; please install a recent Biopython") from exc

            # local fallback (Phase 3a: enhanced triggering logic)
            try:
                mapped_count = len(resid_map)
                qlen = len(qseq)
                use_local = False
                local_trigger_reason = None

                # Condition 1: Low sequence identity
                if score is not None and score < cls.GLOBAL_THRESHOLD:
                    use_local = True
                    local_trigger_reason = 'low_identity'

                # Condition 2: Low coverage after global alignment
                coverage = mapped_count / max(1, qlen)
                if coverage < cls.MIN_COVERAGE_FOR_SKIP_LOCAL:
                    use_local = True
                    local_trigger_reason = 'low_coverage'

                # Condition 3: Force local for large sequences
                if qlen >= cls.LARGE_SEQ_THRESHOLD and cls.FORCE_LOCAL_FOR_LARGE:
                    use_local = True
                    local_trigger_reason = 'large_sequence'

                if use_local:
                    # Record coverage before local alignment (for diagnostic)
                    if diag is not None and len(structure_maps) == 0:
                        diag['coverage_before_local'] = coverage
                        diag['local_trigger_reason'] = local_trigger_reason

                    try:
                        from baseline.search import local_align_and_map as _local_map
                    except Exception:
                        try:
                            from competitions.rna2.src.baseline.search import local_align_and_map as _local_map
                        except Exception:
                            _local_map = None
                    if _local_map is not None:
                        local_map = _local_map(qseq, tpl_seq, coords_arr, resnames, min_identity=cls.LOCAL_MIN_IDENTITY, min_length=cls.MIN_LOCAL_LENGTH)
                        n_added = 0
                        for k, v in local_map.items():
                            if k not in resid_map:
                                resid_map[k] = v
                                n_added += 1
                        # Record local align usage (only for first template)
                        if diag is not None and len(structure_maps) == 0:
                            if len(local_map) > 0:
                                diag['used_local_align'] = True
                            diag['n_local_added'] = n_added
                            # Coverage after local alignment
                            new_coverage = len(resid_map) / max(1, qlen)
                            diag['coverage_after_local'] = new_coverage
            except Exception:
                pass

            # Kabsch alignment against first template when enabled
            try:
                if len(structure_maps) > 0 and coords_arr is not None and cls.USE_KABSCH:
                    ref_map = structure_maps[0]
                    overlap_keys = set(ref_map.keys()) & set(resid_map.keys())
                    qlen = len(qseq)
                    if len(overlap_keys) >= cls.KABSCH_MIN_PAIRS and (len(overlap_keys) / max(1, qlen)) >= cls.KABSCH_MIN_COVERAGE:
                        P = []
                        Q = []
                        for k in overlap_keys:
                            P.append(ref_map[k][0:3])
                            Q.append(resid_map[k][0:3])
                        P = np.asarray(P, dtype=float)
                        Q = np.asarray(Q, dtype=float)
                        try:
                            from baseline.math_utils import kabsch, apply_transform
                        except Exception:
                            from competitions.rna2.src.baseline.math_utils import kabsch, apply_transform
                        R, t, rmsd_after = kabsch(P, Q)
                        coords_arr = apply_transform(coords_arr, R, t)
                        for key, val in list(resid_map.items()):
                            if len(val) >= 5:
                                t_idx = val[4]
                                if t_idx is not None and t_idx < len(coords_arr):
                                    x, y, z = coords_arr[t_idx]
                                    resn = val[3]
                                    resid_map[key] = (x, y, z, resn, t_idx)
                        # Record Kabsch usage
                        if diag is not None:
                            diag['kabsch_applied'] = True
            except Exception:
                pass

            structure_maps.append(resid_map)

        # Calculate coverage for diagnostic
        if diag is not None and structure_maps:
            first_map = structure_maps[0]
            total_residues = len(qseq)
            mapped_residues = sum(1 for qi in range(1, total_residues + 1) if str(qi) in first_map and not any(np.isnan(first_map[str(qi)][:3])))
            diag['raw_coverage'] = mapped_residues / max(1, total_residues)

        for qi in range(1, len(qseq) + 1):
            if qseq[qi - 1] == '-':
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
                    # mapping values may be (x,y,z,resname,t_idx) or similar; accept extra fields
                    try:
                        x, y, z, *rest = val
                    except Exception:
                        # unexpected format
                        continue
                    row[xk] = x
                    row[yk] = y
                    row[zk] = z
                else:
                    row[xk] = np.nan
                    row[yk] = np.nan
                    row[zk] = np.nan
            rows.append(row)

        # Calculate final coverage (after gap filling would be applied later)
        if diag is not None:
            # For now, raw_coverage = final_coverage (gap filling happens in predict.py)
            diag['final_coverage'] = diag['raw_coverage']

        # Return rows and diagnostic log if enabled
        if cls.DIAGNOSTIC:
            return rows, diag
        else:
            return rows


def init_pool_with_scorer(templates_serial, config, prefilter_k, prefilter_top_n, scorer):
    SubmissionGenerator.init_pool_with_scorer(templates_serial, config, prefilter_k, prefilter_top_n, scorer)


def worker_process_query(qtuple):
    return SubmissionGenerator.process_query(qtuple)
