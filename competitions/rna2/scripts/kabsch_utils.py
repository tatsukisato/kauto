import numpy as np

def _centroid(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)

def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that aligns Q to P using the Kabsch algorithm.

    P, Q must be shape (N,3) and N>=1. Returns 3x3 rotation matrix R such that Q @ R approximates P (after centering).
    """
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    # center
    Pc = P - _centroid(P)
    Qc = Q - _centroid(Q)
    # covariance
    H = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # correct reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def apply_rotation(Q: np.ndarray, R: np.ndarray, translate_to: np.ndarray) -> np.ndarray:
    """Apply rotation R to Q (already in original coords) and translate so centroid matches translate_to."""
    Qc = Q - _centroid(Q)
    Qr = Qc @ R
    Qr += translate_to
    return Qr

def compute_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD between P and Q (same shape)."""
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    diff = P - Q
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))

def align_and_rmsd(P: np.ndarray, Q: np.ndarray) -> (np.ndarray, float):
    """Align Q to P using Kabsch and return (Q_aligned, rmsd).

    Both P and Q must be (N,3). Returns aligned Q and rmsd.
    """
    R = kabsch_rotation(P, Q)
    Q_aligned = apply_rotation(Q, R, _centroid(P))
    rmsd = compute_rmsd(P, Q_aligned)
    return Q_aligned, rmsd


if __name__ == "__main__":
    # simple smoke test
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    Q = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    Q_aligned, rmsd = align_and_rmsd(P, Q)
    print("aligned:\n", Q_aligned)
    print("rmsd:", rmsd)
