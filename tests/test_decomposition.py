"""Tests for src/decomposition.py — factor covariance decomposition."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decomposition import decompose_covariance, extract_sector_blocks, assemble_from_sector_blocks
from src.utils import vech, vech_to_matrix


def make_random_psd(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    M = A @ A.T + np.eye(n) * n * 0.1
    return (M + M.T) / 2


def make_weight_matrix(K: int, N: int, seed: int = 0) -> np.ndarray:
    """Full-rank factor weight matrix W of shape (K, N)."""
    rng = np.random.default_rng(seed)
    W_raw = rng.standard_normal((K, N))
    # Orthonormalize rows to ensure W has rank K
    W, _ = np.linalg.qr(W_raw.T)
    return W.T[:K]


# ──────────────────────────────────────────────────────────────────────────────
# decompose_covariance tests
# ──────────────────────────────────────────────────────────────────────────────

def test_decompose_output_shapes():
    """Factor cov (K×K), betas (K×N), and residual (N×N) have correct shapes."""
    N, K = 10, 3
    Sigma = make_random_psd(N)
    W = make_weight_matrix(K, N)
    Sf, Bt, Se = decompose_covariance(Sigma, W)
    assert Sf.shape == (K, K), f"Factor cov shape mismatch: {Sf.shape}"
    assert Bt.shape == (K, N), f"Beta shape mismatch: {Bt.shape}"
    assert Se.shape == (N, N), f"Residual shape mismatch: {Se.shape}"


def test_decompose_reconstruction_identity():
    """B'Σ_f B + Σ_e should equal Σ_t up to numerical precision."""
    N, K = 12, 3
    Sigma = make_random_psd(N)
    W = make_weight_matrix(K, N)
    Sf, Bt, Se = decompose_covariance(Sigma, W)
    reconstructed = Bt.T @ Sf @ Bt + Se
    np.testing.assert_allclose(Sigma, reconstructed, atol=1e-8,
        err_msg="Decomposition identity Sigma = B'Σ_f B + Σ_e failed")


def test_factor_cov_is_symmetric():
    """Factor covariance Σ_f,t is symmetric."""
    N, K = 8, 2
    Sigma = make_random_psd(N)
    W = make_weight_matrix(K, N)
    Sf, _, _ = decompose_covariance(Sigma, W)
    np.testing.assert_allclose(Sf, Sf.T, atol=1e-12)


def test_factor_cov_is_psd():
    """Factor covariance Σ_f,t has non-negative eigenvalues."""
    N, K = 10, 3
    Sigma = make_random_psd(N)
    W = make_weight_matrix(K, N)
    Sf, _, _ = decompose_covariance(Sigma, W)
    eigvals = np.linalg.eigvalsh(Sf)
    assert np.all(eigvals >= -1e-8), f"Σ_f not PSD: min eigenvalue = {eigvals.min():.2e}"


def test_residual_cov_is_symmetric():
    """Residual covariance Σ_e,t is symmetric."""
    N, K = 10, 2
    Sigma = make_random_psd(N)
    W = make_weight_matrix(K, N)
    _, _, Se = decompose_covariance(Sigma, W)
    np.testing.assert_allclose(Se, Se.T, atol=1e-12)


def test_k1_factor_spec():
    """K=1 (market-only factor) decomposition is valid."""
    N = 8
    Sigma = make_random_psd(N)
    W = make_weight_matrix(1, N)
    Sf, Bt, Se = decompose_covariance(Sigma, W)
    assert Sf.shape == (1, 1)
    assert Bt.shape == (1, N)
    reconstructed = Bt.T @ Sf @ Bt + Se
    np.testing.assert_allclose(Sigma, reconstructed, atol=1e-8)


def test_k7_factor_spec():
    """K=7 factor decomposition is valid for N=30."""
    N, K = 30, 7
    Sigma = make_random_psd(N)
    W = make_weight_matrix(K, N)
    Sf, Bt, Se = decompose_covariance(Sigma, W)
    assert Sf.shape == (K, K)
    reconstructed = Bt.T @ Sf @ Bt + Se
    np.testing.assert_allclose(Sigma, reconstructed, atol=1e-7)


def test_different_asset_sizes():
    """Decomposition works for different values of N."""
    for N, K in [(5, 1), (20, 3), (50, 5)]:
        Sigma = make_random_psd(N)
        W = make_weight_matrix(K, N)
        Sf, Bt, Se = decompose_covariance(Sigma, W)
        reconstructed = Bt.T @ Sf @ Bt + Se
        np.testing.assert_allclose(Sigma, reconstructed, atol=1e-6,
            err_msg=f"Reconstruction failed for N={N}, K={K}")


# ──────────────────────────────────────────────────────────────────────────────
# extract_sector_blocks / assemble_from_sector_blocks tests
# ──────────────────────────────────────────────────────────────────────────────

def make_sector_indices(N: int, S: int = 4) -> list:
    """Partition N assets into S sectors of roughly equal size."""
    base = N // S
    indices = []
    start = 0
    for s in range(S):
        end = start + base + (1 if s < N % S else 0)
        indices.append(np.arange(start, end))
        start = end
    return indices


def test_extract_sector_blocks_shape():
    """Sector blocks have correct (N^s, N^s) shapes."""
    N = 20
    Se = make_random_psd(N)
    sector_indices = make_sector_indices(N, S=5)
    blocks = extract_sector_blocks(Se, sector_indices)
    assert len(blocks) == 5
    for idx, block in zip(sector_indices, blocks):
        assert block.shape == (len(idx), len(idx)), f"Sector block shape mismatch"


def test_assemble_block_diagonal_structure():
    """Assembled block-diagonal matrix has zeros in off-diagonal blocks."""
    N = 16
    Se = make_random_psd(N)
    sector_indices = make_sector_indices(N, S=4)
    blocks = extract_sector_blocks(Se, sector_indices)
    assembled = assemble_from_sector_blocks(blocks, sector_indices, N)

    for s1, idx1 in enumerate(sector_indices):
        for s2, idx2 in enumerate(sector_indices):
            if s1 != s2:
                cross_block = assembled[np.ix_(idx1, idx2)]
                np.testing.assert_allclose(cross_block, 0, atol=1e-14,
                    err_msg=f"Cross-sector block ({s1},{s2}) is not zero")


def test_assemble_diagonal_blocks_match_input():
    """Diagonal blocks in assembled matrix match extracted blocks."""
    N = 12
    Se = make_random_psd(N)
    sector_indices = make_sector_indices(N, S=3)
    blocks = extract_sector_blocks(Se, sector_indices)
    assembled = assemble_from_sector_blocks(blocks, sector_indices, N)

    for idx, block in zip(sector_indices, blocks):
        assembled_block = assembled[np.ix_(idx, idx)]
        np.testing.assert_allclose(assembled_block, block, atol=1e-14)


def test_assemble_is_symmetric():
    """Assembled block-diagonal matrix is symmetric."""
    N = 15
    Se = make_random_psd(N)
    sector_indices = make_sector_indices(N, S=5)
    blocks = extract_sector_blocks(Se, sector_indices)
    assembled = assemble_from_sector_blocks(blocks, sector_indices, N)
    np.testing.assert_allclose(assembled, assembled.T, atol=1e-14)


def test_assemble_output_shape():
    """Assembled matrix has shape (N, N)."""
    N = 18
    Se = make_random_psd(N)
    sector_indices = make_sector_indices(N, S=6)
    blocks = extract_sector_blocks(Se, sector_indices)
    assembled = assemble_from_sector_blocks(blocks, sector_indices, N)
    assert assembled.shape == (N, N)


# ──────────────────────────────────────────────────────────────────────────────
# vech / vech_to_matrix round-trip tests
# ──────────────────────────────────────────────────────────────────────────────

def test_vech_length():
    """vech of (K×K) symmetric matrix has K*(K+1)/2 elements."""
    for K in [1, 3, 5, 7]:
        M = make_random_psd(K)
        v = vech(M)
        assert len(v) == K * (K + 1) // 2, f"vech length wrong for K={K}"


def test_vech_round_trip():
    """vech followed by vech_to_matrix returns the original symmetric matrix."""
    for n in [2, 4, 6, 10]:
        M = make_random_psd(n)
        M_rec = vech_to_matrix(vech(M), n)
        np.testing.assert_allclose(M_rec, M, atol=1e-14,
            err_msg=f"vech round-trip failed for n={n}")


def test_vech_to_matrix_is_symmetric():
    """Reconstructed matrix is exactly symmetric."""
    M = make_random_psd(5)
    M_rec = vech_to_matrix(vech(M), 5)
    np.testing.assert_allclose(M_rec, M_rec.T, atol=1e-15)
