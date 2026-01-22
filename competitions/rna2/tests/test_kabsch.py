import sys
import pathlib
import numpy as np

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

from competitions.rna2.src.baseline import math_utils as mu


def random_rotation_translation(seed=1):
    rng = np.random.RandomState(seed)
    A = rng.randn(3, 3)
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    t = rng.randn(3)
    return R, t


def test_kabsch_reconstruction():
    rng = np.random.RandomState(2)
    P = rng.randn(10, 3)
    R_true, t_true = random_rotation_translation(seed=3)
    Q = (P @ R_true.T) + t_true
    R, t, rmsd_after = mu.kabsch(P, Q)
    # apply transform to Q to align to P
    Qt = mu.apply_transform(Q, R, t)
    # RMSD should be near zero
    assert rmsd_after < 1e-6 or np.allclose(P, Qt, atol=1e-6)
