"""Utilities for template coordinate post-processing: gap interpolation and simple fills.

Functions:
- interpolate_linear(coords, max_gap): linearly interpolate short internal gaps (per-axis)
- neighbor_fill(coords, k): fill remaining NaNs by nearest-neighbour average within window k
- fill_gaps_pipeline(coords, max_interp_length=5, interp_method='linear', neighbor_k=2): helper pipeline
- missing_rate(coords): fraction of residues with any NaN coordinate

Designed to be small and dependency-light (numpy only).
"""
from typing import Optional
import numpy as np


def interpolate_linear(coords: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Linearly interpolate internal gaps of length <= max_gap.

    coords: (n,3) array-like with floats and np.nan for missing.
    Returns a copy with short internal gaps filled; leading/trailing NaNs are left.
    """
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("coords must be (n,3) array-like")
    out = arr.copy()
    n = out.shape[0]
    for axis in range(3):
        col = out[:, axis]
        isnan = np.isnan(col)
        if not np.any(isnan):
            continue
        good_idx = np.where(~isnan)[0]
        if good_idx.size == 0:
            continue
        # iterate gaps between consecutive good indices
        for i in range(len(good_idx) - 1):
            left = good_idx[i]
            right = good_idx[i + 1]
            gap_len = right - left - 1
            if gap_len <= 0:
                continue
            if gap_len <= max_gap:
                start = col[left]
                end = col[right]
                # interpolate gap_len values
                interp_vals = np.linspace(start, end, gap_len + 2)[1:-1]
                col[left + 1:right] = interp_vals
        out[:, axis] = col
    return out


def neighbor_fill(coords: np.ndarray, k: int = 2) -> np.ndarray:
    """Fill remaining NaNs by averaging nearest up to k neighbours on each side.

    This is a conservative fallback for boundary or long gaps.
    """
    arr = np.asarray(coords, dtype=float).copy()
    n = arr.shape[0]
    for i in range(n):
        if not np.any(np.isnan(arr[i])):
            continue
        # collect neighbor values for each axis
        vals = []
        for offset in range(1, k + 1):
            left = i - offset
            right = i + offset
            if left >= 0 and not np.any(np.isnan(arr[left])):
                vals.append(arr[left])
            if right < n and not np.any(np.isnan(arr[right])):
                vals.append(arr[right])
            if vals:
                break
        if vals:
            vals = np.vstack(vals)
            arr[i] = np.nanmean(vals, axis=0)
    return arr


def fill_gaps_pipeline(coords: np.ndarray, max_interp_length: int = 5, interp_method: str = 'linear', neighbor_k: int = 2) -> np.ndarray:
    """Run interpolation then neighbor fill to produce a filled coordinate array.

    interp_method currently supports 'linear' only.
    """
    arr = np.asarray(coords, dtype=float)
    if interp_method == 'linear':
        arr = interpolate_linear(arr, max_gap=max_interp_length)
    else:
        raise ValueError(f"unsupported interp_method: {interp_method}")
    # fill remaining NaNs conservatively
    arr = neighbor_fill(arr, k=neighbor_k)
    return arr


def missing_rate(coords: np.ndarray) -> float:
    """Return fraction of residues with any NaN coordinate."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    n = arr.shape[0]
    if n == 0:
        return 0.0
    missing = np.any(np.isnan(arr), axis=1)
    return float(np.count_nonzero(missing)) / float(n)
