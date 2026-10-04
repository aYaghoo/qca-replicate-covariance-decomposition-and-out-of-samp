import numpy as np
from scipy import stats as spstats

from .utils import vech


def average_l2_forecast_error(
    Sigma_hat_list: list[np.ndarray],
    Sigma_true_list: list[np.ndarray],
) -> float:
    """Compute average L2 forecast error over out-of-sample period.

    Metric: mean_t ||vech(Sigma_hat_t - Sigma_t)||_2

    Parameters
    ----------
    Sigma_hat_list: list of T_oos (N, N) forecast matrices
    Sigma_true_list: list of T_oos (N, N) realized matrices

    Returns:
    -------
    mean_l2: scalar average L2 error

    Raises:
    ------
    ValueError: if input lists have different lengths
    """
    if len(Sigma_hat_list) != len(Sigma_true_list):
        raise ValueError(
            f"Forecast and realized lists must have equal length; "
            f"got {len(Sigma_hat_list)} vs {len(Sigma_true_list)}."
        )
    errors = []
    for Sigma_hat, Sigma_true in zip(Sigma_hat_list, Sigma_true_list):
        diff = vech(Sigma_hat - Sigma_true)
        errors.append(np.linalg.norm(diff))
    return float(np.mean(errors))


def random_walk_forecast(
    Sigma_list: list[np.ndarray],
    t_oos_start: int,
) -> list[np.ndarray]:
    """Generate random walk (naive) forecasts: Sigma_hat_{t+1} = Sigma_t.

    Parameters
    ----------
    Sigma_list: full list of T realized covariance matrices
    t_oos_start: first out-of-sample index

    Returns:
    -------
    list of random walk forecasts aligned to OOS period
    """
    T = len(Sigma_list)
    rw_forecasts = [Sigma_list[t - 1] for t in range(t_oos_start, T)]
    return rw_forecasts


def compile_results_table(
    results: dict[str, dict],
    Sigma_true_list: list[np.ndarray],
    rw_l2: float,
) -> dict:
    """Compile comparison table of L2 errors across all models.

    Parameters
    ----------
    results: dict mapping model_name -> result dict with 'l2_errors'
    Sigma_true_list: list of true realized covariance matrices
    rw_l2: random walk L2 error

    Returns:
    -------
    table: dict mapping model_name -> {'mean_l2', 'std_l2', 'relative_to_rw'}
    """
    table = {}
    table["Random Walk"] = {
        "mean_l2": rw_l2,
        "std_l2": np.nan,
        "relative_to_rw": 1.0,
    }
    for name, res in results.items():
        l2_errors = res["l2_errors"]
        mean_l2 = float(np.mean(l2_errors))
        std_l2 = float(np.std(l2_errors))
        table[name] = {
            "mean_l2": mean_l2,
            "std_l2": std_l2,
            "relative_to_rw": mean_l2 / rw_l2 if rw_l2 > 0 else np.nan,
        }
    return table


def lower_partial_std(returns: np.ndarray) -> float:
    """Compute lower partial standard deviation from array of returns.

    Lower partial std = std(r, ddof=1) for all r < 0 (returns below zero).
    Returns 0.0 when fewer than 2 negative returns exist.

    Parameters
    ----------
    returns: 1D array of portfolio returns

    Returns:
    -------
    lpstd: scalar lower partial standard deviation
    """
    neg = returns[returns < 0]
    if len(neg) < 2:
        return 0.0
    return float(np.std(neg, ddof=1))


def compute_portfolio_metrics(
    port_returns: np.ndarray,
    weights: np.ndarray,
    returns: np.ndarray,
    Sigma_hat_list: list[np.ndarray],
) -> dict:
    """Compute portfolio performance metrics per paper specification.

    Parameters
    ----------
    port_returns: (T_oos,) pre-computed portfolio returns
    weights: (T_oos, N) portfolio weights at each rebalancing
    returns: (T_oos, N) realized stock returns
    Sigma_hat_list: list of T_oos (N, N) forecast covariance matrices

    Returns:
    -------
    dict with keys: std, std_ann, lpstd, lpstd_ann, excess_kurtosis, skewness,
    avg_diversification_ratio, avg_max_weight, avg_min_weight,
    avg_gross_leverage, prop_negative, avg_turnover, port_returns
    """
    T_oos = len(port_returns)

    std = float(np.std(port_returns, ddof=1))
    std_ann = std * np.sqrt(252)

    lpstd_val = lower_partial_std(port_returns)
    lpstd_ann = lpstd_val * np.sqrt(252)

    excess_kurtosis = float(spstats.kurtosis(port_returns, fisher=True))
    skewness = float(spstats.skew(port_returns))

    # Diversification ratio: sum_i |w_i| sigma_i / sqrt(w' Sigma w)
    div_ratios = []
    for t in range(T_oos):
        w = weights[t]
        Sigma = Sigma_hat_list[t]
        port_var = float(w @ Sigma @ w)
        if port_var > 1e-12:
            asset_vols = np.sqrt(np.diag(Sigma))
            div_ratios.append(np.sum(np.abs(w) * asset_vols) / np.sqrt(port_var))
    avg_div_ratio = float(np.mean(div_ratios)) if div_ratios else 0.0

    avg_max_weight = float(np.mean(np.max(weights, axis=1)))
    avg_min_weight = float(np.mean(np.min(weights, axis=1)))
    avg_gross_leverage = float(np.mean(np.sum(np.abs(weights), axis=1)))
    prop_negative = float(np.mean(weights < 0))

    turnovers = []
    for t in range(1, T_oos):
        r_p_prev = port_returns[t - 1]
        w_prev = weights[t - 1]
        denom = 1.0 + r_p_prev
        if abs(denom) > 1e-10:
            w_hold = w_prev * (1.0 + returns[t - 1]) / denom
        else:
            w_hold = w_prev
        turnovers.append(float(np.mean(np.abs(weights[t] - w_hold))))
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return {
        "std": std,
        "std_ann": std_ann,
        "lpstd": lpstd_val,
        "lpstd_ann": lpstd_ann,
        "excess_kurtosis": excess_kurtosis,
        "skewness": skewness,
        "avg_diversification_ratio": avg_div_ratio,
        "avg_max_weight": avg_max_weight,
        "avg_min_weight": avg_min_weight,
        "avg_gross_leverage": avg_gross_leverage,
        "prop_negative": prop_negative,
        "avg_turnover": avg_turnover,
        "port_returns": port_returns,
    }
