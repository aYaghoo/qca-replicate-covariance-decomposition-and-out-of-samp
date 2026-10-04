import numpy as np
from sklearn.linear_model import Lasso

HAR_WEEK = 5  # weekly HAR horizon, in days
HAR_MONTH = 22  # monthly HAR horizon, in days; the longest lag a HAR row needs


def bic_score(y: np.ndarray, y_hat: np.ndarray, n_nonzero: int) -> float:
    """Compute BIC for LASSO selection.

    BIC = T * log(RSS/T) + n_nonzero * log(T)

    Parameters
    ----------
    y: (T,) true values
    y_hat: (T,) predicted values
    n_nonzero: number of non-zero coefficients (excluding intercept)

    Returns
    -------
    bic: scalar BIC value (lower is better)
    """
    T = len(y)
    residuals = y - y_hat
    rss = np.sum(residuals**2)
    if rss <= 0:
        rss = 1e-300
    bic = T * np.log(rss / T) + n_nonzero * np.log(T)
    return bic


def lasso_bic(
    X: np.ndarray,
    y: np.ndarray,
    n_alphas: int = 30,
    alpha_min_ratio: float = 1e-4,
    standardize: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Fit LASSO with BIC-selected lambda.

    Objective: (1/T)||y - Z gamma||^2 + 2 * lambda * ||beta||_1
    where beta excludes the intercept.

    Parameters
    ----------
    X: (T, p) design matrix (including intercept column or not)
    y: (T,) response
    n_alphas: number of lambda values to try
    alpha_min_ratio: ratio of lambda_min to lambda_max
    standardize: whether to standardize predictors

    Returns
    -------
    coef: (p,) estimated coefficients
    intercept: scalar intercept
    best_alpha: selected lambda value
    """
    T, p = X.shape

    # Standardize predictors
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0

    if standardize:
        X_s = (X - X_mean) / X_std
    else:
        X_s = X.copy()

    y_mean = y.mean()
    y_c = y - y_mean

    # Lambda grid: from lambda_max down by ratio
    # lambda_max = ||X^T y||_inf / T (roughly)
    lambda_max = np.max(np.abs(X_s.T @ y_c)) / T
    if lambda_max < 1e-10:
        lambda_max = 1.0
    lambda_min = alpha_min_ratio * lambda_max
    alphas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), n_alphas))

    best_bic = np.inf
    best_coef = np.zeros(p)
    best_intercept = y_mean
    best_alpha = alphas[0]

    for alpha in alphas:
        # sklearn Lasso: objective = (1/(2*T)) * ||y - Xw||^2 + alpha * ||w||_1
        # Paper: (1/T)||y - Z*gamma||^2 + 2*lambda*||beta||_1
        # Mapping: alpha_sklearn = lambda_paper (factor of 2 absorbed in convention)
        lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000, tol=1e-4)
        lasso.fit(X_s, y_c)

        coef_s = lasso.coef_
        y_hat = X_s @ coef_s + lasso.intercept_ + y_mean
        n_nonzero = np.sum(np.abs(coef_s) > 1e-10)

        bic = bic_score(y, y_hat, n_nonzero)
        if bic < best_bic:
            best_bic = bic
            # Transform back to original scale
            if standardize:
                coef_orig = coef_s / X_std
                intercept_orig = y_mean + lasso.intercept_ - X_mean @ coef_orig
            else:
                coef_orig = coef_s
                intercept_orig = y_mean + lasso.intercept_
            best_coef = coef_orig
            best_intercept = intercept_orig
            best_alpha = alpha

    return best_coef, best_intercept, best_alpha


def adaptive_lasso_bic(
    X: np.ndarray,
    y: np.ndarray,
    initial_coef: np.ndarray,
    n_alphas: int = 30,
    alpha_min_ratio: float = 1e-4,
    standardize: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Adaptive LASSO with BIC-selected lambda.

    Uses initial LASSO coefficients as adaptive weights.
    For each predictor j: weight_j = 1 / |initial_coef_j| (or a large value if zero).
    Rescale predictor j by weight_j before fitting standard LASSO.

    Parameters
    ----------
    X: (T, p) design matrix
    y: (T,) response
    initial_coef: (p,) initial LASSO coefficients
    n_alphas: number of lambda values
    alpha_min_ratio: ratio of lambda_min to lambda_max
    standardize: whether to standardize predictors

    Returns
    -------
    coef: (p,) adaptive LASSO coefficients
    intercept: scalar
    best_alpha: selected alpha
    """
    # Adaptive weights: 1 / |initial_coef_j|
    ada_weights = 1.0 / np.maximum(np.abs(initial_coef), 1e-10)

    # Rescale design matrix: X_ada[:, j] = X[:, j] / ada_weights[j]
    # so that LASSO penalty becomes sum_j ada_weights[j] * |coef_j|
    X_ada = X / ada_weights[np.newaxis, :]  # (T, p)

    # Fit standard LASSO on rescaled X
    coef_ada_s, intercept, best_alpha = lasso_bic(X_ada, y, n_alphas, alpha_min_ratio, standardize)

    # Transform back: original coef_j = coef_ada_s_j / ada_weights_j
    coef_orig = coef_ada_s / ada_weights

    return coef_orig, intercept, best_alpha


def har_design_matrix(series_2d: np.ndarray, t: int) -> np.ndarray:
    """Build HAR design matrix row at position t for a multi-dimensional series.

    For a (T, M) series, builds row: [1, series_day(t-1), series_week(t-1), series_month(t-1)]
    where day = series[t-1], week = mean(series[t-HAR_WEEK:t]),
    month = mean(series[t-HAR_MONTH:t]).

    Parameters
    ----------
    series_2d: (T, M) array
    t: current time index (0-based, predicting t from t-1); must be >= HAR_MONTH

    Returns
    -------
    row: (1 + 3*M,) regressor vector

    Raises
    ------
    ValueError
        If ``t < HAR_MONTH``, which would give a partial monthly average.
    """
    if t < HAR_MONTH:
        raise ValueError(f"t={t} < HAR_MONTH={HAR_MONTH}: monthly average would be partial")
    day_lag = series_2d[t - 1, :]  # (M,)
    week_lag = series_2d[t - HAR_WEEK : t, :].mean(axis=0)  # (M,)
    month_lag = series_2d[t - HAR_MONTH : t, :].mean(axis=0)  # (M,)
    return np.concatenate([[1.0], day_lag, week_lag, month_lag])


def fit_har_lasso_equation(
    Z: np.ndarray,
    y_i: np.ndarray,
    use_adaptive: bool = False,
    n_alphas: int = 30,
) -> np.ndarray:
    """Fit a single HAR equation with LASSO or adaptive LASSO.

    Parameters
    ----------
    Z: (T, 1 + 3*M) design matrix (intercept + HAR regressors)
    y_i: (T,) response for equation i
    use_adaptive: if True, run two-stage adaptive LASSO
    n_alphas: number of alpha values in grid

    Returns
    -------
    coef: (1 + 3*M,) coefficient vector [intercept, slopes...]; coef[0] is the intercept
    """
    # Design matrix without intercept (intercept handled by lasso_bic)
    X = Z[:, 1:]  # (T, 3*M)
    y = y_i

    coef, intercept, _ = lasso_bic(X, y, n_alphas=n_alphas)

    if use_adaptive:
        coef, intercept, _ = adaptive_lasso_bic(X, y, initial_coef=coef, n_alphas=n_alphas)

    # Return full coefficient vector including intercept
    return np.concatenate([[intercept], coef])
