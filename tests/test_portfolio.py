"""Tests for src/portfolio.py — minimum-variance portfolio optimization."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.portfolio import (
    min_variance_long_only,
    min_variance_restricted,
    min_variance_unconstrained,
    run_portfolio_experiment,
)

SHORT_CAP = 0.30
MAX_W = 0.20


def make_psd(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    M = A @ A.T + np.eye(n) * n * 0.5
    return (M + M.T) / 2


# ──────────────────────────────────────────────────────────────────────────────
# Unconstrained min-variance
# ──────────────────────────────────────────────────────────────────────────────


def test_unconstrained_weights_sum_to_one():
    """Unconstrained weights sum to 1."""
    Sigma = make_psd(10)
    w = min_variance_unconstrained(Sigma)
    assert abs(np.sum(w) - 1.0) < 1e-6, f"Weights sum = {np.sum(w):.6f}"


def test_unconstrained_weights_shape():
    """Unconstrained weights have shape (N,)."""
    N = 15
    Sigma = make_psd(N)
    w = min_variance_unconstrained(Sigma)
    assert w.shape == (N,), f"Shape mismatch: {w.shape}"


def test_unconstrained_portfolio_variance_minimal():
    """Unconstrained min-var portfolio has lower variance than equal weights."""
    N = 10
    Sigma = make_psd(N)
    w_mv = min_variance_unconstrained(Sigma)
    w_eq = np.ones(N) / N
    var_mv = float(w_mv @ Sigma @ w_mv)
    var_eq = float(w_eq @ Sigma @ w_eq)
    assert var_mv <= var_eq + 1e-6, f"Min-var ({var_mv:.6f}) > equal-weight ({var_eq:.6f})"


def test_unconstrained_different_sizes():
    """Unconstrained optimizer works for various N."""
    for N in [5, 10, 20, 50]:
        Sigma = make_psd(N)
        w = min_variance_unconstrained(Sigma)
        assert abs(np.sum(w) - 1.0) < 1e-5


# ──────────────────────────────────────────────────────────────────────────────
# Restricted min-variance (short leverage ≤ 30%, |w_i| ≤ 20%)
# ──────────────────────────────────────────────────────────────────────────────


def test_restricted_weights_sum_to_one():
    """Restricted weights sum to 1."""
    N = 10
    Sigma = make_psd(N)
    w = min_variance_restricted(Sigma, short_leverage_cap=SHORT_CAP, max_weight=MAX_W)
    assert abs(np.sum(w) - 1.0) < 1e-4, f"Weights sum = {np.sum(w):.6f}"


def test_restricted_max_weight_constraint():
    """All |w_i| ≤ max_weight in restricted portfolio."""
    N = 15
    Sigma = make_psd(N)
    w = min_variance_restricted(Sigma, short_leverage_cap=SHORT_CAP, max_weight=MAX_W)
    assert np.all(np.abs(w) <= MAX_W + 1e-4), (
        f"Max |w_i| = {np.max(np.abs(w)):.4f} exceeds cap {MAX_W}"
    )


def test_restricted_short_leverage_constraint():
    """Sum of absolute short positions ≤ short_leverage_cap."""
    N = 20
    Sigma = make_psd(N)
    w = min_variance_restricted(Sigma, short_leverage_cap=SHORT_CAP, max_weight=MAX_W)
    short_leverage = float(np.sum(np.abs(w[w < 0])))
    assert short_leverage <= SHORT_CAP + 1e-4, (
        f"Short leverage {short_leverage:.4f} exceeds cap {SHORT_CAP}"
    )


def test_restricted_portfolio_shape():
    """Restricted weights have shape (N,)."""
    N = 12
    Sigma = make_psd(N)
    w = min_variance_restricted(Sigma, short_leverage_cap=SHORT_CAP, max_weight=MAX_W)
    assert w.shape == (N,)


def test_restricted_variance_feasible():
    """Restricted portfolio variance is positive."""
    N = 10
    Sigma = make_psd(N)
    w = min_variance_restricted(Sigma, short_leverage_cap=SHORT_CAP, max_weight=MAX_W)
    var = float(w @ Sigma @ w)
    assert var > 0.0, f"Portfolio variance should be positive, got {var}"


# ──────────────────────────────────────────────────────────────────────────────
# Long-only min-variance (0 ≤ w_i ≤ 20%)
# ──────────────────────────────────────────────────────────────────────────────


def test_long_only_weights_sum_to_one():
    """Long-only weights sum to 1."""
    N = 10
    Sigma = make_psd(N)
    w = min_variance_long_only(Sigma, max_weight=MAX_W)
    assert abs(np.sum(w) - 1.0) < 1e-4


def test_long_only_no_negative_weights():
    """Long-only portfolio has no negative weights."""
    N = 15
    Sigma = make_psd(N)
    w = min_variance_long_only(Sigma, max_weight=MAX_W)
    assert np.all(w >= -1e-5), f"Negative weight found: min={w.min():.6f}"


def test_long_only_max_weight_constraint():
    """All weights ≤ max_weight in long-only portfolio."""
    N = 12
    Sigma = make_psd(N)
    w = min_variance_long_only(Sigma, max_weight=MAX_W)
    assert np.all(w <= MAX_W + 1e-4), f"Max weight {w.max():.4f} exceeds cap {MAX_W}"


def test_long_only_shape():
    """Long-only weights have shape (N,)."""
    N = 8
    Sigma = make_psd(N)
    w = min_variance_long_only(Sigma, max_weight=MAX_W)
    assert w.shape == (N,)


def test_long_only_variance_minimal_vs_equal():
    """Long-only min-var portfolio variance ≤ equal-weight variance."""
    N = 10
    Sigma = make_psd(N)
    w_mv = min_variance_long_only(Sigma, max_weight=MAX_W)
    w_eq = np.ones(N) / N
    var_mv = float(w_mv @ Sigma @ w_mv)
    var_eq = float(w_eq @ Sigma @ w_eq)
    # Equal-weight satisfies constraints, so min-var should be ≤
    assert var_mv <= var_eq + 1e-5, f"min-var variance {var_mv:.6f} > equal-weight {var_eq:.6f}"


# ──────────────────────────────────────────────────────────────────────────────
# run_portfolio_experiment integration tests
# ──────────────────────────────────────────────────────────────────────────────


def make_forecast_series(T_oos: int = 30, N: int = 10, seed: int = 0):
    """Generate list of T_oos forecast covariance matrices and returns."""
    Sigma_hat_list = [make_psd(N, seed + t) for t in range(T_oos)]
    rng = np.random.default_rng(seed + 99)
    returns = rng.standard_normal((T_oos, N)) * 0.01
    return Sigma_hat_list, returns


def test_run_portfolio_experiment_returns_length():
    """Portfolio experiment returns T_oos realized portfolio returns."""
    T_oos, N = 20, 10
    Sigma_hat_list, returns = make_forecast_series(T_oos, N)
    res = run_portfolio_experiment(
        Sigma_hat_list,
        returns,
        constraint_type="unconstrained",
        short_leverage_cap=SHORT_CAP,
        max_weight=MAX_W,
    )
    assert len(res["metrics"]["port_returns"]) == T_oos


def test_run_portfolio_experiment_weights_shape():
    """Portfolio weights array has shape (T_oos, N)."""
    T_oos, N = 20, 10
    Sigma_hat_list, returns = make_forecast_series(T_oos, N)
    res = run_portfolio_experiment(
        Sigma_hat_list,
        returns,
        constraint_type="long_only",
        short_leverage_cap=SHORT_CAP,
        max_weight=MAX_W,
    )
    assert res["weights"].shape == (T_oos, N)


def test_run_portfolio_experiment_all_constraints():
    """run_portfolio_experiment runs without error for all constraint types."""
    T_oos, N = 15, 8
    Sigma_hat_list, returns = make_forecast_series(T_oos, N)
    for ctype in ["unconstrained", "restricted", "long_only"]:
        res = run_portfolio_experiment(
            Sigma_hat_list,
            returns,
            constraint_type=ctype,
            short_leverage_cap=SHORT_CAP,
            max_weight=MAX_W,
        )
        assert "metrics" in res, f"No metrics for {ctype}"
        assert "port_returns" in res["metrics"]


def test_run_portfolio_std_positive():
    """Realized portfolio standard deviation is positive."""
    T_oos, N = 30, 10
    Sigma_hat_list, returns = make_forecast_series(T_oos, N, seed=13)
    for ctype in ["unconstrained", "restricted", "long_only"]:
        res = run_portfolio_experiment(
            Sigma_hat_list,
            returns,
            constraint_type=ctype,
            short_leverage_cap=SHORT_CAP,
            max_weight=MAX_W,
        )
        std = res["metrics"]["std_ann"]
        assert std >= 0.0, f"Non-positive std for {ctype}: {std}"


def test_run_portfolio_metrics_keys():
    """Portfolio metrics dictionary contains all required keys."""
    required_keys = {
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
    T_oos, N = 20, 10
    Sigma_hat_list, returns = make_forecast_series(T_oos, N)
    res = run_portfolio_experiment(
        Sigma_hat_list,
        returns,
        constraint_type="restricted",
        short_leverage_cap=SHORT_CAP,
        max_weight=MAX_W,
    )
    for key in required_keys:
        assert key in res["metrics"], f"Missing metric: {key}"


def test_turnover_non_negative():
    """Average turnover is non-negative."""
    T_oos, N = 25, 10
    Sigma_hat_list, returns = make_forecast_series(T_oos, N, seed=7)
    res = run_portfolio_experiment(
        Sigma_hat_list,
        returns,
        constraint_type="long_only",
        short_leverage_cap=SHORT_CAP,
        max_weight=MAX_W,
    )
    assert res["metrics"]["avg_turnover"] >= 0.0


def test_diversification_ratio_positive():
    """Diversification ratio is positive."""
    T_oos, N = 20, 10
    Sigma_hat_list, returns = make_forecast_series(T_oos, N)
    res = run_portfolio_experiment(
        Sigma_hat_list,
        returns,
        constraint_type="long_only",
        short_leverage_cap=SHORT_CAP,
        max_weight=MAX_W,
    )
    assert res["metrics"]["avg_diversification_ratio"] > 0.0


def test_restricted_gross_leverage_bounded():
    """Restricted portfolio gross leverage respects short leverage cap (+ 1 for long)."""
    T_oos, N = 20, 10
    Sigma_hat_list, returns = make_forecast_series(T_oos, N, seed=3)
    res = run_portfolio_experiment(
        Sigma_hat_list,
        returns,
        constraint_type="restricted",
        short_leverage_cap=SHORT_CAP,
        max_weight=MAX_W,
    )
    avg_gl = res["metrics"]["avg_gross_leverage"]
    # Gross leverage = sum |w_i|; long positions sum >= 1-0.3=0.7, short <= 0.3 → total ≤ 1.6
    assert avg_gl <= 1.0 + SHORT_CAP + 1e-3, (
        f"Gross leverage {avg_gl:.4f} exceeds expected max {1 + SHORT_CAP}"
    )
