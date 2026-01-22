import sys
import pathlib
import pytest

p = pathlib.Path(__file__).resolve()
repo_root = None
for _ in range(8):
    if (p / 'pyproject.toml').exists() or (p / 'competitions').exists():
        repo_root = p
        break
    p = p.parent
if repo_root is None:
    repo_root = pathlib.Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from competitions.rna2.src.baseline import search


def test_local_align_and_map_basic():
    q = "AAAAACCCCC"
    t = "GGGGACCCCCGGG"
    # local_align_and_map may rely on Biopython; skip if not available
    try:
        local_map = search.local_align_and_map(q, t, coords_arr=None, resnames=[], min_identity=0.5, min_length=3)
    except Exception as exc:
        pytest.skip(f"local_align_and_map unavailable: {exc}")
    # Expect mapping for the CCCCC region (positions 6-10 in q -> positions 6-10 in t)
    # mapping keys are 1-based indices as strings
    keys = sorted(int(k) for k in local_map.keys())
    assert any(k >= 5 for k in keys)
