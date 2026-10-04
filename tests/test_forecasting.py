"""Tests for src/lasso_har.py and src/forecasting.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.forecasting import (
    build_factor_cov_har_matrix,
    forecast_betas,
    forecast_factor_covariance,
    forecast_residual_blocks,
    rolling_forecast_pipeline,
)
from src.lasso_har import bic_score, fit_har_lasso_equation, har_design_matrix, lasso_bic


def make_psd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    M = A @ A.T + np.eye(n) * n * 0.1
    return (M + M.T) / 2


def make_sigma_list(T: int = 80, n: int = 10, seed: int = 0) -> list:
    return [make_psd(n, seed + t) for t in range(T)]


def make_sigma_arr(T: int, n: int, seed: int = 0) -> np.ndarray:
    return np.array([make_psd(n, seed + t) for t in range(T)])


def make_W(K: int, N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W_raw = rng.standard_normal((K, N))
    Q, _ = np.linalg.qr(W_raw.T)
    return Q.T[:K]


def make_sector_indices(N: int, S: int = 4) -> list:
    base = N // S
    start = 0
    idxs = []
    for s in range(S):
        end = start + base + (1 if s < N % S else 0)
        idxs.append(np.arange(start, end))
        start = end
    return idxs


def build_B_Se_arrays(T: int, N: int, K: int, seed: int = 0):
    W = make_W(K, N, seed)
    Sigma_list = make_sigma_list(T, N, seed)
    B_arr = np.zeros((T, K, N))
    Sigma_e_arr = np.zeros((T, N, N))
    for t_i, Sigma in enumerate(Sigma_list):
        Sf = W @ Sigma @ W.T
        Sf_inv = np.linalg.inv(Sf + np.eye(K) * 1e-8)
        Bt = Sf_inv @ W @ Sigma
        Se = Sigma - Bt.T @ Sf @ Bt
        B_arr[t_i] = Bt
        Sigma_e_arr[t_i] = (Se + Se.T) / 2
    return W, B_arr, Sigma_e_arr


# bic_score tests


def test_bic_perfect_fit_is_low():
    T = 50
    y = np.random.default_rng(0).standard_normal(T)
    assert bic_score(y, y, n_nonzero=3) < bic_score(y, np.zeros(T), n_nonzero=0)


def test_bic_penalizes_complexity():
    T = 40
    y = np.zeros(T)
    assert bic_score(y + 1e-6, y, n_nonzero=10) > bic_score(y + 1e-6, y, n_nonzero=1)


# lasso_bic tests


def test_lasso_bic_output_shapes():
    T, p = 40, 5
    rng = np.random.default_rng(42)
    X = rng.standard_normal((T, p))
    y = rng.standard_normal(T)
    coef, intercept, best_alpha = lasso_bic(X, y, n_alphas=3)
    assert coef.shape == (p,)
    assert best_alpha > 0.0


def test_lasso_bic_prediction_finite():
    T, p = 30, 4
    rng = np.random.default_rng(7)
    X = rng.standard_normal((T, p))
    y = rng.standard_normal(T)
    coef, intercept, _ = lasso_bic(X, y, n_alphas=3)
    assert np.isfinite(intercept + X[-1] @ coef)


def test_lasso_bic_noiseless_signal():
    T, p = 80, 3
    rng = np.random.default_rng(5)
    beta_true = np.array([2.0, -1.5, 0.0])
    X = rng.standard_normal((T, p))
    y = X @ beta_true
    coef, intercept, _ = lasso_bic(X, y, n_alphas=10)
    rmse = float(np.sqrt(np.mean((X @ coef + intercept - y) ** 2)))
    assert rmse < 0.5 * np.std(y) + 0.01


# har_design_matrix tests


def test_har_design_matrix_returns_row():
    T, K = 60, 3
    M = K * (K + 1) // 2
    series_2d = np.random.default_rng(0).standard_normal((T, M))
    row = har_design_matrix(series_2d, t=25, window=22)
    assert row.shape == (1 + 3 * M,)


def test_har_design_matrix_intercept_is_one():
    T, M = 50, 3
    series_2d = np.random.default_rng(1).standard_normal((T, M))
    row = har_design_matrix(series_2d, t=25, window=22)
    assert row[0] == pytest.approx(1.0)


def test_har_design_matrix_day_lag():
    T, M = 50, 2
    series_2d = np.random.default_rng(2).standard_normal((T, M))
    t = 30
    row = har_design_matrix(series_2d, t=t, window=22)
    np.testing.assert_allclose(row[1 : 1 + M], series_2d[t - 1, :], atol=1e-14)


# fit_har_lasso_equation tests


def test_fit_har_lasso_returns_coefficient_vector():
    T, p = 50, 6
    rng = np.random.default_rng(42)
    Z = np.column_stack([np.ones(T), rng.standard_normal((T, p))])
    y_i = rng.standard_normal(T)
    coef = fit_har_lasso_equation(Z, y_i, use_adaptive=False, n_alphas=3)
    assert len(coef) == p + 1


def test_fit_har_lasso_adaptive_no_crash():
    T, p = 40, 4
    rng = np.random.default_rng(7)
    Z = np.column_stack([np.ones(T), rng.standard_normal((T, p))])
    y_i = rng.standard_normal(T)
    coef = fit_har_lasso_equation(Z, y_i, use_adaptive=True, n_alphas=3)
    assert len(coef) == p + 1
    assert np.all(np.isfinite(coef))


def test_fit_har_lasso_prediction_finite():
    T, p = 40, 3
    rng = np.random.default_rng(11)
    Z = np.column_stack([np.ones(T), rng.standard_normal((T, p))])
    y_i = rng.standard_normal(T)
    coef = fit_har_lasso_equation(Z, y_i, use_adaptive=False, n_alphas=3)
    assert np.isfinite(coef[0] + Z[-1, 1:] @ coef[1:])


# build_factor_cov_har_matrix tests


def test_build_factor_cov_har_matrix_row_shape():
    T, K = 60, 3
    M = K * (K + 1) // 2
    Sigma_f_series = make_sigma_arr(T, K)
    row = build_factor_cov_har_matrix(Sigma_f_series, t=25, use_log=False)
    assert row.shape == (1 + 3 * M,)


def test_build_factor_cov_har_matrix_log_shape():
    T, K = 60, 2
    Sigma_f_series = make_sigma_arr(T, K)
    row_log = build_factor_cov_har_matrix(Sigma_f_series, t=25, use_log=True)
    assert row_log.shape == (1 + 3 * K * (K + 1) // 2,)


def test_build_factor_cov_har_matrix_intercept():
    T, K = 50, 3
    Sigma_f_series = make_sigma_arr(T, K)
    row = build_factor_cov_har_matrix(Sigma_f_series, t=25, use_log=False)
    assert row[0] == pytest.approx(1.0)


# forecast_factor_covariance tests


def test_forecast_factor_cov_output_shape():
    T, K = 50, 3
    Sf_hat = forecast_factor_covariance(
        make_sigma_arr(T, K),
        train_start=0,
        train_end=45,
        use_log=False,
        use_adaptive=False,
        n_alphas=3,
    )
    assert Sf_hat.shape == (K, K)


def test_forecast_factor_cov_is_symmetric():
    T, K = 50, 3
    Sf_hat = forecast_factor_covariance(
        make_sigma_arr(T, K),
        train_start=0,
        train_end=45,
        use_log=False,
        use_adaptive=False,
        n_alphas=3,
    )
    np.testing.assert_allclose(Sf_hat, Sf_hat.T, atol=1e-10)


def test_forecast_factor_cov_log_symmetric():
    T, K = 50, 2
    Sf_hat = forecast_factor_covariance(
        make_sigma_arr(T, K),
        train_start=0,
        train_end=45,
        use_log=True,
        use_adaptive=False,
        n_alphas=3,
    )
    np.testing.assert_allclose(Sf_hat, Sf_hat.T, atol=1e-10)


def test_forecast_factor_cov_is_psd():
    T, K = 50, 3
    Sf_hat = forecast_factor_covariance(
        make_sigma_arr(T, K),
        train_start=0,
        train_end=45,
        use_log=False,
        use_adaptive=False,
        n_alphas=3,
    )
    assert np.all(np.linalg.eigvalsh(Sf_hat) >= -1e-7)


# forecast_betas tests


def test_forecast_betas_shape():
    T, K, N = 60, 3, 10
    _, B_arr, _ = build_B_Se_arrays(T, N, K)
    assert forecast_betas(B_arr, train_start=0, train_end=50).shape == (K, N)


def test_forecast_betas_finite():
    T, K, N = 60, 2, 8
    _, B_arr, _ = build_B_Se_arrays(T, N, K)
    assert np.all(np.isfinite(forecast_betas(B_arr, train_start=0, train_end=50)))


# forecast_residual_blocks tests


def test_forecast_residual_blocks_shape():
    N, K, S, T = 20, 3, 4, 50
    sector_indices = make_sector_indices(N, S)
    _, _, Sigma_e_arr = build_B_Se_arrays(T, N, K)
    Se_hat = forecast_residual_blocks(Sigma_e_arr, sector_indices, 0, 45, n_alphas=3)
    assert Se_hat.shape == (N, N)


def test_forecast_residual_blocks_is_block_diagonal():
    N, K, S, T = 20, 3, 4, 50
    sector_indices = make_sector_indices(N, S)
    _, _, Sigma_e_arr = build_B_Se_arrays(T, N, K)
    Se_hat = forecast_residual_blocks(Sigma_e_arr, sector_indices, 0, 45, n_alphas=3)
    for s1, idx1 in enumerate(sector_indices):
        for s2, idx2 in enumerate(sector_indices):
            if s1 != s2:
                np.testing.assert_allclose(Se_hat[np.ix_(idx1, idx2)], 0, atol=1e-14)


def test_forecast_residual_blocks_symmetric():
    N, K, S, T = 16, 2, 4, 50
    sector_indices = make_sector_indices(N, S)
    _, _, Sigma_e_arr = build_B_Se_arrays(T, N, K)
    Se_hat = forecast_residual_blocks(Sigma_e_arr, sector_indices, 0, 45, n_alphas=3)
    np.testing.assert_allclose(Se_hat, Se_hat.T, atol=1e-14)


# rolling_forecast_pipeline integration tests


def test_rolling_forecast_pipeline_oos_count():
    N, K, T, win = 10, 3, 120, 60
    W = make_W(K, N)
    sector_indices = make_sector_indices(N, 3)
    Sigma_list = make_sigma_list(T, N)
    res = rolling_forecast_pipeline(
        Sigma_list=Sigma_list,
        W_t=W,
        sector_indices=sector_indices,
        K=K,
        rolling_window=win,
        n_alphas=3,
        use_log=False,
        use_adaptive=False,
        verbose=False,
    )
    assert len(res["l2_errors"]) == T - (win + 22)
    assert len(res["Sigma_hat_list"]) == T - (win + 22)


def test_rolling_forecast_pipeline_l2_non_negative():
    N, K, T, win = 10, 2, 100, 50
    W = make_W(K, N)
    Sigma_list = make_sigma_list(T, N)
    res = rolling_forecast_pipeline(
        Sigma_list=Sigma_list,
        W_t=W,
        sector_indices=make_sector_indices(N, 3),
        K=K,
        rolling_window=win,
        n_alphas=3,
        use_log=False,
        use_adaptive=False,
        verbose=False,
    )
    assert all(e >= 0 for e in res["l2_errors"])


def test_rolling_forecast_pipeline_sigma_shape():
    N, K, T, win = 10, 2, 90, 45
    W = make_W(K, N)
    Sigma_list = make_sigma_list(T, N)
    res = rolling_forecast_pipeline(
        Sigma_list=Sigma_list,
        W_t=W,
        sector_indices=make_sector_indices(N, 3),
        K=K,
        rolling_window=win,
        n_alphas=3,
        use_log=False,
        use_adaptive=False,
        verbose=False,
    )
    assert all(Sh.shape == (N, N) for Sh in res["Sigma_hat_list"])


def test_rolling_forecast_log_vs_nolog_same_length():
    N, K, T, win = 10, 2, 90, 45
    W = make_W(K, N)
    Sigma_list = make_sigma_list(T, N)
    kwargs = dict(
        Sigma_list=Sigma_list,
        W_t=W,
        sector_indices=make_sector_indices(N, 3),
        K=K,
        rolling_window=win,
        n_alphas=3,
        use_adaptive=False,
        verbose=False,
    )
    r1 = rolling_forecast_pipeline(**kwargs, use_log=False)
    r2 = rolling_forecast_pipeline(**kwargs, use_log=True)
    assert len(r1["l2_errors"]) == len(r2["l2_errors"])
