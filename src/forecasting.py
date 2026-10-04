import logging
from typing import Any, Optional

import numpy as np

from src.lasso_har import HAR_MONTH, HAR_WEEK, adaptive_lasso_bic, lasso_bic

from .decomposition import decompose_covariance
from .utils import nearest_psd, safe_expm, safe_logm, vech, vech_to_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-computed HAR feature cache (eliminates repeated logm calls)
# ---------------------------------------------------------------------------


class _HARCache:
    """O(1) rolling-mean access via pre-computed cumulative sums.

    After a single pass computing vech (or log-vech) for every time step,
    rolling sums at any window width are returned without re-iterating.
    """

    def __init__(self, Sigma_f_arr: np.ndarray, use_log: bool = False) -> None:
        T, K, _ = Sigma_f_arr.shape
        M = K * (K + 1) // 2

        # Vectorise: compute all log-matrices (or plain matrices) at once
        if use_log:
            vech_mat = np.array([vech(safe_logm(Sigma_f_arr[t]).real) for t in range(T)])  # (T, M)
        else:
            tril_r, tril_c = np.tril_indices(K)
            vech_mat = Sigma_f_arr[:, tril_r, tril_c]  # (T, M) – no loop

        self.vech_mat = vech_mat  # (T, M)
        self.cumsum = np.cumsum(vech_mat, axis=0)  # (T, M) prefix sums
        self.T = T
        self.M = M

    # ------------------------------------------------------------------
    def rolling_mean(self, t_excl_end: int, window: int) -> np.ndarray:
        """Mean of vech_mat[t_excl_end-window : t_excl_end] in O(1)."""
        end = min(t_excl_end, self.T)
        start = max(0, end - window)
        n = end - start
        if n <= 0:
            return np.zeros(self.M)
        total = self.cumsum[end - 1]
        if start > 0:
            total = total - self.cumsum[start - 1]
        return total / n

    # ------------------------------------------------------------------
    def har_row(self, t: int) -> np.ndarray:
        """Build HAR regressor row [1, day_lag, week_lag, month_lag] at t.

        Regression target is Sigma_f_t; regressors use t-1 lags per paper.
        """
        v_day = self.vech_mat[t - 1] if t >= 1 else np.zeros(self.M)
        v_week = self.rolling_mean(t, HAR_WEEK)  # mean of [t-5 … t-1]
        v_month = self.rolling_mean(t, HAR_MONTH)  # mean of [t-22 … t-1]
        return np.concatenate([[1.0], v_day, v_week, v_month])

    # ------------------------------------------------------------------
    def build_design_matrix(self, t_first: int, t_last: int) -> tuple[np.ndarray, np.ndarray]:
        """Build Z (T_reg, 1+3M) and Y (T_reg, M) for the training window."""
        T_reg = t_last - t_first + 1
        Z = np.zeros((T_reg, 1 + 3 * self.M))
        for i, t in enumerate(range(t_first, t_last + 1)):
            Z[i] = self.har_row(t)
        Y = self.vech_mat[t_first : t_last + 1]  # (T_reg, M)
        return Z, Y


# ---------------------------------------------------------------------------
# Public API: build_factor_cov_har_matrix (kept for test compatibility)
# ---------------------------------------------------------------------------


def build_factor_cov_har_matrix(
    Sigma_f_series: np.ndarray,
    t: int,
    use_log: bool = False,
) -> np.ndarray:
    """Build a single HAR regressor row for factor-covariance forecasting at t.

    This thin wrapper is retained for backward compatibility with tests.
    Production code uses :class:`_HARCache` for bulk construction.

    Parameters
    ----------
    Sigma_f_series : (T, K, K) array of factor covariance matrices
    t              : time index for regressor construction (predicts t+1)
    use_log        : if True, work in matrix-log space

    Returns
    -------
    row : (1 + 3*M,) regressor row, M = K(K+1)/2
    """
    cache = _HARCache(Sigma_f_series, use_log=use_log)
    return cache.har_row(t)


# ---------------------------------------------------------------------------
# Factor covariance forecasting
# ---------------------------------------------------------------------------


def forecast_factor_covariance(
    Sigma_f_series: np.ndarray,
    train_start: int,
    train_end: int,
    use_log: bool = False,
    use_adaptive: bool = False,
    n_alphas: int = 20,
    _cache: Optional["_HARCache"] = None,
) -> np.ndarray:
    """Forecast factor covariance matrix at train_end+1 using HAR/VHAR-LASSO.

    Fits equation-by-equation for each of M = K(K+1)/2 unique entries.

    Parameters
    ----------
    Sigma_f_series : (T_total, K, K) factor covariance matrices
    train_start    : start index of rolling training window
    train_end      : end index of rolling training window (inclusive)
    use_log        : use matrix-log transformation
    use_adaptive   : use adaptive LASSO (two-stage)
    n_alphas       : number of alpha values for lambda grid
    _cache         : pre-built _HARCache (avoids rebuilding per OOS step)

    Returns
    -------
    Sigma_f_hat : (K, K) forecast factor covariance matrix
    """
    K = Sigma_f_series.shape[1]
    M = K * (K + 1) // 2

    cache = _cache if _cache is not None else _HARCache(Sigma_f_series, use_log)

    # First observation with full month window
    t_first = train_start + HAR_MONTH
    t_last = train_end

    T_reg = t_last - t_first + 1
    if T_reg <= 0:
        raise ValueError(f"Insufficient data for HAR regression: T_reg={T_reg}")

    Z, Y_raw = cache.build_design_matrix(t_first, t_last)
    z_pred = cache.har_row(train_end + 1)  # predict at train_end+1

    # Equation-by-equation LASSO
    pred_vech = np.zeros(M)
    X = Z[:, 1:]  # strip intercept column; lasso_bic handles intercept
    x_pred = z_pred[1:]

    for i in range(M):
        y_i = Y_raw[:, i]
        coef, intercept, _ = lasso_bic(X, y_i, n_alphas=n_alphas)
        if use_adaptive:
            coef, intercept, _ = adaptive_lasso_bic(X, y_i, initial_coef=coef, n_alphas=n_alphas)
        pred_vech[i] = intercept + x_pred @ coef

    # Reconstruct matrix
    if use_log:
        Omega_hat = vech_to_matrix(pred_vech, K)
        Sigma_f_hat = safe_expm(Omega_hat).real
    else:
        Sigma_f_hat = vech_to_matrix(pred_vech, K)

    return nearest_psd(Sigma_f_hat)


# ---------------------------------------------------------------------------
# Beta forecasting
# ---------------------------------------------------------------------------


def _build_har_matrix_1d(series: np.ndarray, t_first: int, t_last: int) -> np.ndarray:
    """Build (T_reg, 4) HAR design matrix [1, day, week, month] for a 1-D series.

    Uses cumulative sums for O(1) rolling means instead of inner loops.
    """
    T_reg = t_last - t_first + 1
    # Pre-compute cumsum for O(1) rolling means
    cs = np.cumsum(series)

    def _roll(t: int, w: int) -> float:
        end = t  # exclusive end = t; includes [t-w, t-1]
        start = max(0, end - w)
        n = end - start
        if n == 0:
            return 0.0
        total = cs[end - 1] - (cs[start - 1] if start > 0 else 0.0)
        return total / n

    Z = np.empty((T_reg, 4))
    ts = np.arange(t_first, t_last + 1)
    Z[:, 0] = 1.0
    Z[:, 1] = series[ts - 1]  # day lag: series[t-1]
    for j, t in enumerate(ts):
        Z[j, 2] = _roll(t, HAR_WEEK)  # week mean
        Z[j, 3] = _roll(t, HAR_MONTH)  # month mean
    return Z


def forecast_betas(
    B_series: np.ndarray,
    train_start: int,
    train_end: int,
) -> np.ndarray:
    """Forecast beta matrix B at train_end+1 using HAR-OLS (equation by equation).

    For each (k, i) element of B, estimate:
      beta_{k,i,t} = omega + phi_day * beta_{k,i,t-1}
                   + phi_week * beta_{k,i,week,t-1}
                   + phi_month * beta_{k,i,month,t-1} + error
    Coefficients selected by OLS; prediction at train_end+1.

    Parameters
    ----------
    B_series   : (T_total, K, N) beta matrices
    train_start: start of rolling window
    train_end  : end of rolling window (inclusive)

    Returns
    -------
    B_hat : (K, N) forecast beta matrix
    """
    T_total, K, N = B_series.shape
    t_first = train_start + HAR_MONTH
    t_last = train_end
    T_reg = t_last - t_first + 1

    B_flat = B_series.reshape(T_total, K * N)  # (T, K*N)
    B_hat_flat = np.zeros(K * N)

    # Pre-build prediction regressors once
    t_pred = train_end + 1
    cs_all = np.cumsum(B_flat, axis=0)  # (T, K*N) prefix sums

    def _pred_row(col: int) -> np.ndarray:
        s = B_flat[:, col]
        cs = cs_all[:, col]

        def _roll(t: int, w: int) -> float:
            end = t
            start = max(0, end - w)
            n = end - start
            if n == 0:
                return 0.0
            return (cs[end - 1] - (cs[start - 1] if start > 0 else 0.0)) / n

        return np.array([1.0, s[t_pred - 1], _roll(t_pred, HAR_WEEK), _roll(t_pred, HAR_MONTH)])

    for col in range(K * N):
        series = B_flat[:, col]
        Z = _build_har_matrix_1d(series, t_first, t_last)
        y = series[t_first : t_last + 1]

        try:
            coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
        except np.linalg.LinAlgError:
            coef = np.array([series[train_end], 0.0, 0.0, 0.0])

        z_pred = _pred_row(col)
        B_hat_flat[col] = z_pred @ coef

    return B_hat_flat.reshape(K, N)


# ---------------------------------------------------------------------------
# Residual block forecasting
# ---------------------------------------------------------------------------


def forecast_residual_blocks(
    Sigma_e_series: np.ndarray,
    sector_indices: list[np.ndarray],
    train_start: int,
    train_end: int,
    use_adaptive: bool = False,
    n_alphas: int = 20,
) -> np.ndarray:
    """Forecast block-diagonal residual covariance at train_end+1.

    For each sector s with N_s assets:
    - Response  : vech(Sigma_e_t^s), each unique entry separately
    - Regressors: diag(Sigma_e_{t-1}^s)  (within-sector variances only)
    - Estimation: LASSO / adaptive LASSO with lambda by BIC

    Parameters
    ----------
    Sigma_e_series  : (T_total, N, N) residual covariance matrices
    sector_indices  : list of S index arrays
    train_start     : start of rolling window
    train_end       : end of rolling window (inclusive)
    use_adaptive    : use adaptive LASSO
    n_alphas        : number of alpha values

    Returns
    -------
    Sigma_e_hat : (N, N) block-diagonal residual covariance forecast
    """
    N = Sigma_e_series.shape[1]
    t_first = train_start + 1
    t_last = train_end
    T_reg = t_last - t_first + 1

    Sigma_e_hat = np.zeros((N, N))

    for sector_idx in sector_indices:
        ns = len(sector_idx)
        Ms = ns * (ns + 1) // 2
        idx_2d = np.ix_(sector_idx, sector_idx)
        tril_r, tril_c = np.tril_indices(ns)

        # Vectorised extraction: (T_total, Ms) response and (T_total-1, ns) regressors
        blocks = Sigma_e_series[:, sector_idx[:, None], sector_idx[None, :]]  # (T, ns, ns)

        Y_s = blocks[t_first : t_last + 1, tril_r, tril_c]  # (T_reg, Ms)
        X_s = np.array([np.diag(blocks[t - 1]) for t in range(t_first, t_last + 1)])  # (T_reg, ns)
        x_pred = np.diag(blocks[train_end])  # lag for t+1

        pred_vech_s = np.zeros(Ms)
        for m in range(Ms):
            y_m = Y_s[:, m]
            coef, intercept, _ = lasso_bic(X_s, y_m, n_alphas=n_alphas)
            if use_adaptive:
                coef, intercept, _ = adaptive_lasso_bic(
                    X_s, y_m, initial_coef=coef, n_alphas=n_alphas
                )
            pred_vech_s[m] = intercept + x_pred @ coef

        block_hat = np.zeros((ns, ns))
        block_hat[tril_r, tril_c] = pred_vech_s
        block_hat += block_hat.T - np.diag(np.diag(block_hat))  # symmetrise
        block_hat = nearest_psd(block_hat)
        Sigma_e_hat[np.ix_(sector_idx, sector_idx)] = block_hat

    return Sigma_e_hat


# ---------------------------------------------------------------------------
# Rolling-window pipeline
# ---------------------------------------------------------------------------


def _validate_pipeline_inputs(
    Sigma_list: list[np.ndarray],
    W_t: np.ndarray,
    sector_indices: list[np.ndarray],
    K: int,
    rolling_window: int,
) -> list[np.ndarray]:
    """Check pipeline inputs and return the sector indices as integer arrays.

    Parameters
    ----------
    Sigma_list
        Realized covariance matrices, each of shape (N, N).
    W_t
        Factor weight matrix, shape (K, N).
    sector_indices
        One index sequence per sector. Lists of integers are converted.
    K
        Number of factors; must equal the number of rows of ``W_t``.
    rolling_window
        Estimation window length; must be positive.

    Returns
    -------
    list[np.ndarray]
        The sector indices, each as a 1-D integer array.

    Raises
    ------
    ValueError
        If ``rolling_window`` is not positive, ``K`` does not match ``W_t``,
        a covariance matrix contains NaN or inf, or a sector index is not a
        1-D array of integers.
    """
    if rolling_window <= 0:
        raise ValueError(f"rolling_window must be positive, got {rolling_window}")
    if W_t.shape[0] != K:
        raise ValueError(f"K={K} does not match W_t with {W_t.shape[0]} rows")
    for t, Sigma in enumerate(Sigma_list):
        if not np.isfinite(Sigma).all():
            raise ValueError(f"Sigma_list[{t}] contains NaN or inf (first bad index: {t})")
    checked = []
    for s, idx in enumerate(sector_indices):
        arr = np.asarray(idx)
        if arr.ndim != 1 or arr.dtype.kind not in "iu":
            raise ValueError(
                f"sector_indices[{s}] must be a 1-D integer array, "
                f"got dtype {arr.dtype} with shape {arr.shape}"
            )
        checked.append(arr)
    return checked


def _decompose_all(
    Sigma_list: list[np.ndarray], W_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose every matrix in ``Sigma_list`` into factor, beta, and residual parts.

    Parameters
    ----------
    Sigma_list
        T realized covariance matrices, each of shape (N, N).
    W_t
        Factor weight matrix, shape (K, N).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Factor covariances (T, K, K), betas (T, K, N), residual covariances (T, N, N).
    """
    T = len(Sigma_list)
    K, N = W_t.shape
    Sigma_f_arr = np.zeros((T, K, K))
    B_arr = np.zeros((T, K, N))
    Sigma_e_arr = np.zeros((T, N, N))
    for t in range(T):
        Sigma_f_arr[t], B_arr[t], Sigma_e_arr[t] = decompose_covariance(Sigma_list[t], W_t)
    return Sigma_f_arr, B_arr, Sigma_e_arr


def _recombine(B_hat: np.ndarray, Sigma_f_hat: np.ndarray, Sigma_e_hat: np.ndarray) -> np.ndarray:
    """Recombine component forecasts into a symmetric PSD covariance forecast.

    Parameters
    ----------
    B_hat
        Beta forecast, shape (K, N).
    Sigma_f_hat
        Factor covariance forecast, shape (K, K).
    Sigma_e_hat
        Residual covariance forecast, shape (N, N).

    Returns
    -------
    np.ndarray
        ``nearest_psd`` of the symmetrized ``B_hat.T @ Sigma_f_hat @ B_hat + Sigma_e_hat``.
    """
    Sigma_hat = B_hat.T @ Sigma_f_hat @ B_hat + Sigma_e_hat
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2
    return nearest_psd(Sigma_hat)


def _random_walk_step(
    Sigma_list: list[np.ndarray],
    Sigma_f_arr: np.ndarray,
    B_arr: np.ndarray,
    Sigma_e_arr: np.ndarray,
    t_pred: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the random-walk forecast for ``t_pred``: copies of day ``t_pred - 1``.

    Parameters
    ----------
    Sigma_list
        Realized covariance matrices.
    Sigma_f_arr
        Factor covariances, shape (T, K, K).
    B_arr
        Betas, shape (T, K, N).
    Sigma_e_arr
        Residual covariances, shape (T, N, N).
    t_pred
        Index of the day being forecast.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Full covariance, factor covariance, betas, and residual covariance of day ``t_pred - 1``.
    """
    prev = t_pred - 1
    return (
        Sigma_list[prev].copy(),
        Sigma_f_arr[prev].copy(),
        B_arr[prev].copy(),
        Sigma_e_arr[prev].copy(),
    )


def rolling_forecast_pipeline(
    Sigma_list: list[np.ndarray],
    W_t: np.ndarray,
    sector_indices: list[np.ndarray],
    K: int,
    rolling_window: int = 1000,
    use_log: bool = False,
    use_adaptive: bool = False,
    n_alphas: int = 20,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full rolling-window one-step-ahead covariance forecasting pipeline.

    At each forecast origin t:
    1. Decompose all T matrices → Sigma_f_arr, B_arr, Sigma_e_arr  (done once)
    2. For each OOS step: forecast Sigma_f, B, Sigma_e and recombine
       Sigma_hat = B_hat.T @ Sigma_f_hat @ B_hat + Sigma_e_hat

    Performance: All logm / vech computations are pre-computed once via
    _HARCache, so rolling-mean construction is O(1) per step.

    If a step raises ``np.linalg.LinAlgError``, that step falls back to the
    random-walk forecast (the previous day's matrices), a warning is logged,
    and the step is recorded in ``fallback_steps``. Any other exception
    propagates.

    Parameters
    ----------
    Sigma_list    : list of T (N, N) realized covariance matrices
    W_t           : (K, N) factor weight matrix (time-invariant)
    sector_indices: list of S index arrays
    K             : number of factors
    rolling_window: estimation window length W (paper: 1 000)
    use_log       : matrix-log transform for factor covariance
    use_adaptive  : adaptive LASSO
    n_alphas      : lambda grid size
    verbose       : log progress at INFO if True, at DEBUG if False.
                    Fallback warnings are logged either way.

    Returns
    -------
    dict with keys:
      Sigma_hat_list   – list of T_oos (N, N) full forecasts
      Sigma_f_hat_list – list of T_oos (K, K) factor covariance forecasts
      B_hat_list       – list of T_oos (K, N) beta forecasts
      Sigma_e_hat_list – list of T_oos (N, N) residual covariance forecasts
      l2_errors        – list of T_oos scalar L2 errors (full covariance)
      l2_factor_errors – list of T_oos scalar L2 errors (factor covariance)
      n_oos            – number of out-of-sample forecasts
      t_oos_start      – first OOS index
      K                – factor count
      n_fallback       – number of steps that fell back to the random walk
      fallback_steps   – OOS step indices (positions in the lists above) that fell back

    Raises
    ------
    ValueError
        If the inputs fail validation (see ``_validate_pipeline_inputs``).
    """
    sector_indices = _validate_pipeline_inputs(Sigma_list, W_t, sector_indices, K, rolling_window)
    progress_level = logging.INFO if verbose else logging.DEBUG
    T = len(Sigma_list)

    # ── 1. Decompose all T matrices (single pass) ─────────────────────────
    logger.log(progress_level, "Decomposing all covariance matrices")
    Sigma_f_arr, B_arr, Sigma_e_arr = _decompose_all(Sigma_list, W_t)

    # ── 2. Pre-build HAR cache (single logm pass if use_log) ─────────────
    har_cache = _HARCache(Sigma_f_arr, use_log=use_log)

    # ── 3. OOS window ─────────────────────────────────────────────────────
    t_oos_start = rolling_window + HAR_MONTH
    t_oos_end = T - 1
    n_oos = t_oos_end - t_oos_start + 1

    logger.log(
        progress_level,
        "Running %d rolling forecasts (t=%d to %d)",
        n_oos,
        t_oos_start,
        t_oos_end,
    )

    Sigma_hat_list: list[np.ndarray] = []
    Sigma_f_hat_list: list[np.ndarray] = []
    B_hat_list: list[np.ndarray] = []
    Sigma_e_hat_list: list[np.ndarray] = []
    l2_errors: list[float] = []
    l2_factor_errors: list[float] = []
    fallback_steps: list[int] = []

    for step, t_pred in enumerate(range(t_oos_start, t_oos_end + 1)):
        train_end = t_pred - 1
        train_start = max(0, train_end - rolling_window - HAR_MONTH + 1)

        if step % 50 == 0:
            logger.log(progress_level, "step %d/%d: forecasting t=%d", step + 1, n_oos, t_pred)

        try:
            # 2a. Factor covariance (uses pre-built cache – no redundant logm)
            Sigma_f_hat = forecast_factor_covariance(
                Sigma_f_arr,
                train_start,
                train_end,
                use_log=use_log,
                use_adaptive=use_adaptive,
                n_alphas=n_alphas,
                _cache=har_cache,
            )
            # 2b. Betas
            B_hat = forecast_betas(B_arr, train_start, train_end)

            # 2c. Residual blocks
            Sigma_e_hat = forecast_residual_blocks(
                Sigma_e_arr,
                sector_indices,
                train_start,
                train_end,
                use_adaptive=use_adaptive,
                n_alphas=n_alphas,
            )

            # 2d. Recombine
            Sigma_hat = _recombine(B_hat, Sigma_f_hat, Sigma_e_hat)

        except np.linalg.LinAlgError as exc:
            logger.warning(
                "Forecast failed at step %d (t=%d): %s; using random walk", step, t_pred, exc
            )
            fallback_steps.append(step)
            Sigma_hat, Sigma_f_hat, B_hat, Sigma_e_hat = _random_walk_step(
                Sigma_list, Sigma_f_arr, B_arr, Sigma_e_arr, t_pred
            )

        Sigma_hat_list.append(Sigma_hat)
        Sigma_f_hat_list.append(Sigma_f_hat)
        B_hat_list.append(B_hat)
        Sigma_e_hat_list.append(Sigma_e_hat)

        # ── L2 errors ──────────────────────────────────────────────────────
        Sigma_true = Sigma_list[t_pred]
        l2_errors.append(float(np.linalg.norm(vech(Sigma_hat - Sigma_true))))
        l2_factor_errors.append(float(np.linalg.norm(vech(Sigma_f_hat - Sigma_f_arr[t_pred]))))

    return {
        "Sigma_hat_list": Sigma_hat_list,
        "Sigma_f_hat_list": Sigma_f_hat_list,
        "B_hat_list": B_hat_list,
        "Sigma_e_hat_list": Sigma_e_hat_list,
        "l2_errors": l2_errors,
        "l2_factor_errors": l2_factor_errors,
        "n_oos": n_oos,
        "t_oos_start": t_oos_start,
        "K": K,
        "n_fallback": len(fallback_steps),
        "fallback_steps": fallback_steps,
    }
