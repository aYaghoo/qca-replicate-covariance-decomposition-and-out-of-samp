import numpy as np


def decompose_covariance(
    Sigma_t: np.ndarray,
    W_t: np.ndarray,
    reg_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose realized covariance matrix using factor model.

    Given:
    - Sigma_t: (N, N) realized covariance matrix
    - W_t: (K, N) factor weight matrix

    Compute:
    - Sigma_f_t = W_t @ Sigma_t @ W_t.T  (K x K factor covariance)
    - B_t = inv(Sigma_f_t) @ W_t @ Sigma_t  (K x N beta matrix)
    - Sigma_e_t = Sigma_t - B_t.T @ Sigma_f_t @ B_t  (N x N residual covariance)

    Parameters
    ----------
    Sigma_t: (N, N) realized covariance matrix
    W_t: (K, N) factor weight matrix
    reg_eps: regularization for Sigma_f inversion

    Returns
    -------
    Sigma_f_t: (K, K) factor covariance
    B_t: (K, N) factor loadings
    Sigma_e_t: (N, N) residual covariance
    """
    K = W_t.shape[0]

    # Factor covariance: K x K
    Sigma_f_t = W_t @ Sigma_t @ W_t.T
    Sigma_f_t = (Sigma_f_t + Sigma_f_t.T) / 2  # enforce symmetry

    # Regularize for inversion
    Sigma_f_reg = Sigma_f_t + reg_eps * np.eye(K)
    Sigma_f_inv = np.linalg.inv(Sigma_f_reg)

    # Factor loadings: K x N
    B_t = Sigma_f_inv @ W_t @ Sigma_t

    # Residual covariance: N x N
    Sigma_e_t = Sigma_t - B_t.T @ Sigma_f_t @ B_t
    Sigma_e_t = (Sigma_e_t + Sigma_e_t.T) / 2  # enforce symmetry

    return Sigma_f_t, B_t, Sigma_e_t


def verify_decomposition(
    Sigma_t: np.ndarray,
    Sigma_f_t: np.ndarray,
    B_t: np.ndarray,
    Sigma_e_t: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    """Verify that Sigma_t ≈ B_t.T @ Sigma_f_t @ B_t + Sigma_e_t."""
    reconstructed = B_t.T @ Sigma_f_t @ B_t + Sigma_e_t
    error = np.max(np.abs(Sigma_t - reconstructed))
    return error < tol


def extract_sector_blocks(
    Sigma_e_t: np.ndarray,
    sector_indices: list[np.ndarray],
) -> list[np.ndarray]:
    """Extract sector-diagonal blocks from residual covariance matrix.

    Parameters
    ----------
    Sigma_e_t: (N, N) residual covariance matrix
    sector_indices: list of index arrays, one per sector

    Returns
    -------
    blocks: list of (N_s, N_s) block matrices
    """
    blocks = []
    for idx in sector_indices:
        block = Sigma_e_t[np.ix_(idx, idx)]
        block = (block + block.T) / 2
        blocks.append(block)
    return blocks


def assemble_from_sector_blocks(
    blocks: list[np.ndarray],
    sector_indices: list[np.ndarray],
    N: int,
) -> np.ndarray:
    """Assemble full N x N block-diagonal matrix from sector blocks.

    Cross-sector entries are set to zero.

    Parameters
    ----------
    blocks: list of (N_s, N_s) block matrices
    sector_indices: list of index arrays, one per sector
    N: total number of assets

    Returns
    -------
    Sigma_e: (N, N) block-diagonal matrix
    """
    Sigma_e = np.zeros((N, N))
    for block, idx in zip(blocks, sector_indices):
        Sigma_e[np.ix_(idx, idx)] = block
    return Sigma_e
