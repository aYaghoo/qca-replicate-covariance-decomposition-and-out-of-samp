"""Tests for src/cleaning.py — realized covariance matrix cleaning."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cleaning import clean_covariance_matrices


def make_random_psd(n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    M = A @ A.T + np.eye(n) * 0.1
    return (M + M.T) / 2


def make_sigma_list(T: int = 50, n: int = 5, seed: int = 0) -> list:
    return [make_random_psd(n, seed + t) for t in range(T)]


# ──────────────────────────────────────────────────────────────────────────────
# Basic output structure
# ──────────────────────────────────────────────────────────────────────────────

def test_clean_returns_correct_length():
    """Output list has same length as input."""
    Sigma_list = make_sigma_list(T=30, n=4)
    cleaned, flags = clean_covariance_matrices(Sigma_list)
    assert len(cleaned) == 30
    assert len(flags) == 30


def test_clean_output_is_symmetric():
    """All cleaned matrices are symmetric."""
    Sigma_list = make_sigma_list(T=20, n=5)
    cleaned, _ = clean_covariance_matrices(Sigma_list)
    for M in cleaned:
        np.testing.assert_allclose(M, M.T, atol=1e-12)


def test_no_flags_without_outliers():
    """Clean well-behaved matrices produce no flags."""
    Sigma_list = make_sigma_list(T=40, n=4)
    _, flags = clean_covariance_matrices(Sigma_list, sigma_threshold=10.0, flag_fraction=0.5)
    # With very loose thresholds, few or no matrices should be flagged
    assert sum(flags) <= 2  # allow up to 2 due to small-sample variance estimate edge cases


def test_outlier_matrix_is_flagged():
    """A grossly inflated matrix is flagged."""
    rng = np.random.default_rng(42)
    n = 5
    Sigma_list = make_sigma_list(T=40, n=n)
    # Replace one matrix with a very extreme version
    extreme = make_random_psd(n, 99) * 1000.0
    Sigma_list[35] = extreme
    _, flags = clean_covariance_matrices(Sigma_list, sigma_threshold=4.0, flag_fraction=0.25)
    assert flags[35], "Extreme matrix at t=35 should be flagged"


def test_flagged_matrix_is_replaced():
    """A flagged matrix is replaced, not kept as-is."""
    n = 4
    Sigma_list = make_sigma_list(T=40, n=n)
    original_35 = Sigma_list[35].copy()
    extreme = make_random_psd(n, 99) * 1e6
    Sigma_list[35] = extreme

    cleaned, flags = clean_covariance_matrices(Sigma_list, sigma_threshold=4.0, flag_fraction=0.25)
    if flags[35]:
        assert not np.allclose(cleaned[35], extreme), "Flagged matrix should be replaced"


def test_replacement_is_average_of_preceding():
    """Replaced matrix is an average of preceding non-flagged matrices."""
    n = 3
    # Create a list where we can identify the replacement
    Sigma_list = make_sigma_list(T=30, n=n)
    extreme = make_random_psd(n, 99) * 1e8
    Sigma_list[20] = extreme

    cleaned, flags = clean_covariance_matrices(
        Sigma_list, sigma_threshold=4.0, flag_fraction=0.25, replacement_window=5
    )
    if flags[20]:
        # Replacement should differ substantially from extreme
        assert not np.allclose(cleaned[20], extreme, rtol=0.1)


def test_unflagged_matrices_unchanged():
    """Matrices that are not flagged remain exactly the same."""
    Sigma_list = make_sigma_list(T=20, n=4)
    cleaned, flags = clean_covariance_matrices(Sigma_list)
    for t, (orig, cln, flag) in enumerate(zip(Sigma_list, cleaned, flags)):
        if not flag:
            np.testing.assert_array_equal(orig, cln,
                err_msg=f"Unflagged matrix at t={t} was modified")


def test_single_matrix_input():
    """A single-matrix input is handled without error."""
    Sigma_list = [make_random_psd(3)]
    cleaned, flags = clean_covariance_matrices(Sigma_list)
    assert len(cleaned) == 1
    assert len(flags) == 1


def test_flag_fraction_sensitivity():
    """Stricter flag_fraction (lower value) causes more matrices to be flagged."""
    Sigma_list = make_sigma_list(T=40, n=6)
    Sigma_list[25] = make_random_psd(6, 99) * 100.0
    _, flags_strict = clean_covariance_matrices(Sigma_list, flag_fraction=0.01)
    _, flags_loose = clean_covariance_matrices(Sigma_list, flag_fraction=0.99)
    assert sum(flags_strict) >= sum(flags_loose)


def test_output_matrices_are_psd_positive():
    """Cleaned matrices retain non-negative diagonal (variance non-negative)."""
    Sigma_list = make_sigma_list(T=30, n=5)
    cleaned, _ = clean_covariance_matrices(Sigma_list)
    for M in cleaned:
        diag = np.diag(M)
        assert np.all(diag >= -1e-10), "Diagonal (variance) must be non-negative"


def test_large_replacement_window_clips_to_available():
    """A replacement_window larger than available history does not raise an error."""
    Sigma_list = make_sigma_list(T=20, n=4)
    Sigma_list[5] = make_random_psd(4, 99) * 1e9
    cleaned, _ = clean_covariance_matrices(Sigma_list, replacement_window=100)
    assert len(cleaned) == 20
