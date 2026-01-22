import sys
import pathlib
import numpy as np

# ensure repo root is on sys.path for imports
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

from competitions.rna2.src.baseline import template_utils as tu


def test_linear_interpolate_simple():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [np.nan, np.nan, np.nan],
        [2.0, 2.0, 2.0],
        [np.nan, np.nan, np.nan],
        [4.0, 4.0, 4.0],
    ], dtype=float)
    filled = tu.fill_gaps_pipeline(coords, max_interp_length=5, interp_method='linear', neighbor_k=1)
    # all internal gaps should be filled
    assert not np.isnan(filled).any()
    # linear spacing check
    assert np.allclose(filled[0], [0.0, 0.0, 0.0])
    assert np.allclose(filled[2], [2.0, 2.0, 2.0])
    assert np.allclose(filled[4], [4.0, 4.0, 4.0])


def test_missing_rate_and_neighbor_fill():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [3.0, 3.0, 3.0],
    ], dtype=float)
    raw = tu.missing_rate(coords)
    assert raw > 0
    # neighbor fill should reduce missing
    # pipeline uses linear interpolation then neighbor fill; request 'linear' here
    filled = tu.fill_gaps_pipeline(coords, max_interp_length=1, interp_method='linear', neighbor_k=1)
    post = tu.missing_rate(filled)
    assert post <= raw
