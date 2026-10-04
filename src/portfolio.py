import cvxpy as cp
import numpy as np
from scipy import stats as spstats

from .utils import nearest_psd


def min_variance_unconstrained(
    Sigma: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Unconstrained minimum-variance portfolio.

    Solve: min w'Σw s.t. 1'w = 1
    Closed form: w = Σ^{-1}1 / (1'Σ^{-1}1)

    Parameters
    ----------
    Sigma: (N, N) covariance matrix
    eps: ridge regularization for inversion

    Returns:
    -------
    w: (N,) portfolio weights
    """
    N = Sigma.shape[0]
    Sigma_reg = Sigma + eps * np.eye(N)
    ones = np.ones(N)
    try:
        Sigma_inv_ones = np.linalg.solve(Sigma_reg, ones)
        w = Sigma_inv_ones / (ones @ Sigma_inv_ones)
    except np.linalg.LinAlgError:
        w = np.ones(N) / N
    return w


def min_variance_restricted(
    Sigma: np.ndarray,
    short_leverage_cap: float = 0.30,
    max_weight: float = 0.20,
    eps: float = 1e-8,
) -> np.ndarray:
    """Restricted minimum-variance portfolio with short leverage and weight caps.

    Solve: min w'Σw
    s.t. 1'w = 1
         sum_i w_minus_i <= short_leverage_cap  (short leverage cap)
         w_plus_i + w_minus_i <= max_weight for all i  (absolute weight cap)
         w = w_plus - w_minus, w_plus >= 0, w_minus >= 0

    Parameters
    ----------
    Sigma: (N, N) covariance matrix
    short_leverage_cap: maximum total short position (e.g., 0.30 = 30%)
    max_weight: maximum absolute position in any single stock
    eps: regularization for PSD projection

    Returns:
    -------
    w: (N,) portfolio weights
    """
    N = Sigma.shape[0]
    Sigma_psd = nearest_psd(Sigma, epsilon=eps)

    w_plus = cp.Variable(N, nonneg=True)
    w_minus = cp.Variable(N, nonneg=True)
    w = w_plus - w_minus

    objective = cp.Minimize(cp.quad_form(w, Sigma_psd))
    constraints = [
        cp.sum(w_plus - w_minus) == 1,  # budget constraint
        cp.sum(w_minus) <= short_leverage_cap,  # short leverage cap
        w_plus + w_minus <= max_weight,  # absolute weight cap
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            return w.value
    except Exception:
        pass

    # Fallback: equal-weight
    return np.ones(N) / N


def min_variance_long_only(
    Sigma: np.ndarray,
    max_weight: float = 0.20,
    eps: float = 1e-8,
) -> np.ndarray:
    """Long-only minimum-variance portfolio with individual weight cap.

    Solve: min w'Σw s.t. 1'w = 1, 0 <= w_i <= max_weight for all i

    Parameters
    ----------
    Sigma: (N, N) covariance matrix
    max_weight: maximum weight per asset
    eps: regularization

    Returns:
    -------
    w: (N,) portfolio weights
    """
    N = Sigma.shape[0]
    Sigma_psd = nearest_psd(Sigma, epsilon=eps)

    w = cp.Variable(N, nonneg=True)
    objective = cp.Minimize(cp.quad_form(w, Sigma_psd))
    constraints = [
        cp.sum(w) == 1,
        w <= max_weight,
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            return np.maximum(w.value, 0)
    except Exception:
        pass

    return np.ones(N) / N


def compute_portfolio_metrics(
    weights_history: np.ndarray,
    returns: np.ndarray,
    Sigma_hat_list: list[np.ndarray],
) -> dict:
    """Compute portfolio performance metrics per paper specification.

    Parameters
    ----------
    weights_history: (T_oos, N) portfolio weights at each rebalancing
    returns: (T_oos, N) realized stock returns
    Sigma_hat_list: list of T_oos (N, N) forecast covariance matrices

    Returns:
    -------
    dict with performance metrics:
      std: ex-post realized standard deviation
      lpstd: lower partial standard deviation
      excess_kurtosis, skewness
      avg_diversification_ratio, avg_max_weight, avg_min_weight
      avg_gross_leverage, prop_negative, avg_turnover
    """

    T_oos = weights_history.shape[0]
    N = weights_history.shape[1]

    # Portfolio returns
    port_returns = np.sum(weights_history * returns, axis=1)  # (T_oos,)

    # Centered returns
    r_centered = port_returns - port_returns.mean()

    # Standard deviation
    std = np.std(r_centered, ddof=1)

    # Annualized std (x sqrt(252))
    std_ann = std * np.sqrt(252)

    # Lower partial standard deviation (only negative centered returns)
    neg_mask = r_centered < 0
    lpstd = np.sqrt(np.mean(r_centered[neg_mask] ** 2)) if neg_mask.sum() > 0 else 0.0
    lpstd_ann = lpstd * np.sqrt(252)

    # Higher moments
    excess_kurtosis = float(spstats.kurtosis(r_centered, fisher=True))
    skewness = float(spstats.skew(r_centered))

    # Diversification ratio: sum_i w_i sigma_i / sqrt(w' Sigma w)
    # Paper formula: DR = sum_i |w_i| sigma_i / sqrt(w' Sigma w)
    div_ratios = []
    for t in range(T_oos):
        w = weights_history[t]
        Sigma = Sigma_hat_list[t]
        port_var = float(w @ Sigma @ w)
        if port_var > 1e-12:
            asset_vols = np.sqrt(np.diag(Sigma))
            numerator = np.sum(np.abs(w) * asset_vols)
            div_ratios.append(numerator / np.sqrt(port_var))
    avg_div_ratio = np.mean(div_ratios) if div_ratios else 0.0

    # Weight statistics
    avg_max_weight = np.mean(np.max(weights_history, axis=1))
    avg_min_weight = np.mean(np.min(weights_history, axis=1))
    avg_gross_leverage = np.mean(np.sum(np.abs(weights_history), axis=1))
    prop_negative = np.mean(weights_history < 0)

    # Turnover
    turnovers = []
    for t in range(1, T_oos):
        r_prev = returns[t - 1, :]
        r_p_prev = port_returns[t - 1]
        w_prev = weights_history[t - 1]
        w_hold = w_prev * (1 + r_prev) / (1 + r_p_prev) if abs(1 + r_p_prev) > 1e-10 else w_prev
        turnover_t = np.mean(np.abs(weights_history[t] - w_hold))
        turnovers.append(turnover_t)
    avg_turnover = np.mean(turnovers) if turnovers else 0.0

    return {
        "std": std,
        "std_ann": std_ann,
        "lpstd": lpstd,
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


def run_portfolio_experiment(
    Sigma_hat_list: list[np.ndarray],
    returns: np.ndarray,
    constraint_type: str = "unconstrained",
    short_leverage_cap: float = 0.30,
    max_weight: float = 0.20,
) -> dict:
    """Run full portfolio experiment for a given constraint regime.

    Parameters
    ----------
    Sigma_hat_list: list of T_oos (N, N) forecast covariance matrices
    returns: (T_oos, N) realized stock returns
    constraint_type: 'unconstrained', 'restricted', or 'long_only'
    short_leverage_cap: for restricted
    max_weight: for restricted and long_only

    Returns:
    -------
    dict with weights, metrics, and returns
    """
    T_oos = len(Sigma_hat_list)
    N = Sigma_hat_list[0].shape[0]
    weights_history = np.zeros((T_oos, N))

    for t, Sigma_hat in enumerate(Sigma_hat_list):
        if constraint_type == "unconstrained":
            w = min_variance_unconstrained(Sigma_hat)
        elif constraint_type == "restricted":
            w = min_variance_restricted(Sigma_hat, short_leverage_cap, max_weight)
        elif constraint_type == "long_only":
            w = min_variance_long_only(Sigma_hat, max_weight)
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")
        weights_history[t] = w

    metrics = compute_portfolio_metrics(weights_history, returns, Sigma_hat_list)

    return {
        "weights": weights_history,
        "constraint_type": constraint_type,
        "metrics": metrics,
    }
