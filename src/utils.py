import numpy as np
from typing import Tuple, List


def vech(M: np.ndarray) -> np.ndarray:
    """Extract lower-triangular elements (including diagonal) of symmetric matrix."""
    n = M.shape[0]
    idx = np.tril_indices(n)
    return M[idx]


def vech_to_matrix(v: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct symmetric matrix from vech vector."""
    M = np.zeros((n, n))
    idx = np.tril_indices(n)
    M[idx] = v
    M = M + M.T - np.diag(np.diag(M))
    return M


def nearest_psd(M: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Project matrix to nearest positive semi-definite matrix via eigenvalue clipping."""
    M = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def safe_logm(M: np.ndarray) -> np.ndarray:
    """Compute matrix logarithm with PSD projection for numerical stability."""
    from scipy.linalg import logm
    M_psd = nearest_psd(M)
    return logm(M_psd)


def safe_expm(M: np.ndarray) -> np.ndarray:
    """Compute matrix exponential."""
    from scipy.linalg import expm
    return expm(M)


def har_regressors(
    series: np.ndarray,
    t: int,
    day_lag: int = 1,
    week_lag: int = 5,
    month_lag: int = 22,
) -> np.ndarray:
    """Build HAR regressors [1, x_day, x_week, x_month] at time t.

    series: 1D array of length >= t
    Returns row vector of HAR regressors (scalar case).
    """
    x_day = series[t - day_lag]
    x_week = np.mean(series[t - week_lag:t])
    x_month = np.mean(series[t - month_lag:t])
    return np.array([1.0, x_day, x_week, x_month])
