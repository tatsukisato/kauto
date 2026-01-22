"""Math utilities: Kabsch algorithm and RMSD helpers.

Functions:
- kabsch(P, Q): compute rotation matrix R and translation t that best aligns Q to P (least-squares)
- apply_transform(coords, R, t): apply rigid transform to coords
- rmsd(P, Q): compute RMSD between two point sets

P and Q are expected as (N,3) numpy arrays with N>=3 for stable Kabsch.
"""
import numpy as np


def _centroid(X: np.ndarray) -> np.ndarray:
    return np.mean(X, axis=0)


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD between two point sets of same shape."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    diff = P - Q
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Perform Kabsch algorithm to find rotation R and translation t that aligns Q -> P.

    Returns (R, t, rmsd_after)
    - R: (3,3) rotation matrix
    - t: (3,) translation vector
    - rmsd_after: RMSD after applying transform to Q

    Notes:
    - Requires at least 3 non-collinear points for unique solution.
    - This implementation follows standard SVD-based correction for reflection.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must be (N,3) array")
    if Q.ndim != 2 or Q.shape[1] != 3:
        raise ValueError("Q must be (N,3) array")
    if P.shape[0] != Q.shape[0]:
        raise ValueError("P and Q must have same number of points")
    n = P.shape[0]
    if n < 3:
        # allow attempt but warn: solution may be unstable
        pass

    centroid_P = _centroid(P)
    centroid_Q = _centroid(Q)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # covariance matrix
    H = Q_centered.T @ P_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # correct possible reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_P - R @ centroid_Q

    Q_transformed = (R @ Q.T).T + t
    rmsd_after = rmsd(P, Q_transformed)
    return R, t, rmsd_after


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid transform to coords: (N,3) -> (N,3)"""
    coords = np.asarray(coords, dtype=float)
    return (R @ coords.T).T + np.asarray(t, dtype=float)
