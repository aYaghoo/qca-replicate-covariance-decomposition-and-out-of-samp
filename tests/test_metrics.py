"""Tests for src/metrics.py — forecast evaluation and portfolio performance metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import (
    average_l2_forecast_error,
    compute_portfolio_metrics,
    lower_partial_std,
    random_walk_forecast,
)
from src.utils import vech


def make_psd(n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    M = A @ A.T + np.eye(n) * 0.5
    return (M + M.T) / 2


# ──────────────────────────────────────────────────────────────────────────────
# average_l2_forecast_error tests
# ──────────────────────────────────────────────────────────────────────────────


def test_l2_error_identical_forecasts():
    """Perfect forecasts produce zero average L2 error."""
    T = 10
    matrices = [make_psd(5, t) for t in range(T)]
    err = average_l2_forecast_error(matrices, matrices)
    assert err == pytest.approx(0.0, abs=1e-12), f"Error for identical matrices: {err}"


def test_l2_error_non_negative():
    """Average L2 error is always non-negative."""
    T = 15
    forecasts = [make_psd(5, t) for t in range(T)]
    actuals = [make_psd(5, t + 100) for t in range(T)]
    err = average_l2_forecast_error(forecasts, actuals)
    assert err >= 0.0


def test_l2_error_known_value():
    """L2 error equals expected value for a known pair."""
    n = 3
    A = np.eye(n)  # forecast
    B = 2 * np.eye(n)  # actual: differs by eye(n)
    diff = A - B  # = -eye(n)
    expected = float(np.linalg.norm(vech(diff)))
    err = average_l2_forecast_error([A], [B])
    assert err == pytest.approx(expected, rel=1e-10)


def test_l2_error_symmetric_input():
    """L2 error is consistent when swapping forecast and actual (not necessarily 0 but finite)."""
    T = 5
    forecasts = [make_psd(4, t) for t in range(T)]
    actuals = [make_psd(4, t + 10) for t in range(T)]
    err1 = average_l2_forecast_error(forecasts, actuals)
    err2 = average_l2_forecast_error(actuals, forecasts)
    assert err1 == pytest.approx(err2, rel=1e-10), "L2 error should be symmetric"


def test_l2_error_single_matrix():
    """L2 error works for a single matrix pair."""
    A = make_psd(4, 0)
    B = make_psd(4, 1)
    err = average_l2_forecast_error([A], [B])
    expected = float(np.linalg.norm(vech(A - B)))
    assert err == pytest.approx(expected, rel=1e-10)


def test_l2_error_length_mismatch_raises():
    """Mismatched list lengths raise an error."""
    with pytest.raises((ValueError, AssertionError)):
        average_l2_forecast_error([make_psd(3)], [make_psd(3), make_psd(3)])


# ──────────────────────────────────────────────────────────────────────────────
# random_walk_forecast tests
# ──────────────────────────────────────────────────────────────────────────────


def test_rw_forecast_length():
    """Random walk forecast list has T - t_oos_start elements."""
    T, t_start = 50, 30
    Sigma_list = [make_psd(4, t) for t in range(T)]
    rw = random_walk_forecast(Sigma_list, t_start)
    assert len(rw) == T - t_start


def test_rw_forecast_uses_lag():
    """Random walk forecast at step i equals Sigma_list[t_oos_start + i - 1]."""
    T, t_start = 20, 10
    Sigma_list = [make_psd(4, t) for t in range(T)]
    rw = random_walk_forecast(Sigma_list, t_start)
    # rw[0] should be Sigma_list[t_start - 1] (naive: yesterday's realized)
    np.testing.assert_array_equal(rw[0], Sigma_list[t_start - 1])


def test_rw_forecast_shape_consistency():
    """All random walk forecasts have the same shape as the input matrices."""
    T, t_start, n = 30, 15, 6
    Sigma_list = [make_psd(n, t) for t in range(T)]
    rw = random_walk_forecast(Sigma_list, t_start)
    for M in rw:
        assert M.shape == (n, n)


# ──────────────────────────────────────────────────────────────────────────────
# lower_partial_std tests
# ──────────────────────────────────────────────────────────────────────────────


def test_lower_partial_std_all_positive():
    """If all returns are positive, lower partial std is 0."""
    r = np.array([0.01, 0.02, 0.005, 0.03])
    lpstd = lower_partial_std(r)
    assert lpstd == pytest.approx(0.0, abs=1e-12)


def test_lower_partial_std_all_negative():
    """If all returns are negative, lower partial std equals std of all returns."""
    r = np.array([-0.01, -0.02, -0.005])
    lpstd = lower_partial_std(r)
    expected = float(np.std(r, ddof=1))
    assert lpstd == pytest.approx(expected, rel=1e-8)


def test_lower_partial_std_non_negative():
    """Lower partial std is always non-negative."""
    rng = np.random.default_rng(42)
    r = rng.standard_normal(100) * 0.01
    lpstd = lower_partial_std(r)
    assert lpstd >= 0.0


def test_lower_partial_std_uses_centered():
    """Lower partial std uses centered returns (below mean)."""
    r = np.array([-0.1, 0.0, 0.1, -0.05, 0.05])
    r_centered = r - r.mean()
    neg_centered = r_centered[r_centered < 0]
    expected = float(np.std(neg_centered, ddof=1)) if len(neg_centered) > 1 else 0.0
    lpstd = lower_partial_std(r)
    assert lpstd == pytest.approx(expected, rel=1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# compute_portfolio_metrics tests
# ──────────────────────────────────────────────────────────────────────────────


def make_returns_and_weights(T: int = 50, N: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((T, N)) * 0.01
    # Random long-only weights
    W = rng.dirichlet(np.ones(N), size=T)
    Sigma_list = [make_psd(N, t) for t in range(T)]
    return returns, W, Sigma_list


def test_metrics_output_keys():
    """compute_portfolio_metrics returns all required metric keys."""
    required = {
        "std_ann",
        "lpstd_ann",
        "excess_kurtosis",
        "skewness",
        "avg_diversification_ratio",
        "avg_max_weight",
        "avg_min_weight",
        "avg_gross_leverage",
        "prop_negative",
        "avg_turnover",
        "port_returns",
    }
    returns, W, Sigma_list = make_returns_and_weights()
    port_returns = np.sum(returns * W, axis=1)
    metrics = compute_portfolio_metrics(
        port_returns=port_returns,
        weights=W,
        returns=returns,
        Sigma_hat_list=Sigma_list,
    )
    for key in required:
        assert key in metrics, f"Missing metric key: {key}"


def test_metrics_std_non_negative():
    """Annualized standard deviation is non-negative."""
    returns, W, Sigma_list = make_returns_and_weights()
    port_returns = np.sum(returns * W, axis=1)
    metrics = compute_portfolio_metrics(port_returns, W, returns, Sigma_list)
    assert metrics["std_ann"] >= 0.0


def test_metrics_gross_leverage_positive():
    """Average gross leverage is positive."""
    returns, W, Sigma_list = make_returns_and_weights()
    port_returns = np.sum(returns * W, axis=1)
    metrics = compute_portfolio_metrics(port_returns, W, returns, Sigma_list)
    assert metrics["avg_gross_leverage"] > 0.0


def test_metrics_prop_negative_in_01():
    """Proportion of negative positions is in [0, 1]."""
    returns, W, Sigma_list = make_returns_and_weights()
    port_returns = np.sum(returns * W, axis=1)
    metrics = compute_portfolio_metrics(port_returns, W, returns, Sigma_list)
    assert 0.0 <= metrics["prop_negative"] <= 1.0


def test_metrics_turnover_non_negative():
    """Average turnover is non-negative."""
    returns, W, Sigma_list = make_returns_and_weights()
    port_returns = np.sum(returns * W, axis=1)
    metrics = compute_portfolio_metrics(port_returns, W, returns, Sigma_list)
    assert metrics["avg_turnover"] >= 0.0


def test_metrics_annualization_factor():
    """Std is approximately equal to daily std × sqrt(252)."""
    rng = np.random.default_rng(0)
    T, N = 100, 5
    returns = rng.standard_normal((T, N)) * 0.01
    W = np.tile(np.ones(N) / N, (T, 1))
    Sigma_list = [make_psd(N, t) for t in range(T)]
    port_returns = np.sum(returns * W, axis=1)
    metrics = compute_portfolio_metrics(port_returns, W, returns, Sigma_list)
    daily_std = np.std(port_returns, ddof=1)
    expected_ann = daily_std * np.sqrt(252)
    assert metrics["std_ann"] == pytest.approx(expected_ann, rel=1e-6)
