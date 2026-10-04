"""Minimum-variance portfolio optimizers and portfolio performance metrics.

Units: every function here expects returns in decimals (0.01 = 1%) and
covariance matrices of decimal returns. Metrics come back in the same units;
turnover in particular uses ``1 + r`` and is wrong for percent returns.
"""

import logging
from typing import Any

import cvxpy as cp
import numpy as np
from scipy import stats as spstats

from .utils import nearest_psd

logger = logging.getLogger(__name__)

SOLVER = cp.CLARABEL
ACCEPTED_STATUSES = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


class OptimizationError(RuntimeError):
    """A portfolio optimizer failed to produce an optimal solution."""


def check_solver_installed(solver: str | None = None) -> None:
    """Raise if the configured cvxpy solver is not installed.

    Parameters
    ----------
    solver
        Name of the cvxpy solver to check. Defaults to the module's ``SOLVER``.

    Raises
    ------
    RuntimeError
        If ``solver`` is not in ``cvxpy.installed_solvers()``.
    """
    solver = SOLVER if solver is None else solver
    if solver not in cp.installed_solvers():
        raise RuntimeError(
            f"cvxpy solver {solver!r} is not installed; installed: {cp.installed_solvers()}"
        )


def check_weight_cap_feasible(N: int, max_weight: float) -> None:
    """Raise if no fully invested portfolio of ``N`` assets satisfies the weight cap.

    Parameters
    ----------
    N
        Number of assets.
    max_weight
        Maximum absolute weight per asset.

    Raises
    ------
    ValueError
        If ``max_weight * N < 1``.
    """
    if max_weight * N < 1:
        raise ValueError(f"Weight cap infeasible: max_weight * N = {max_weight} * {N} < 1")


def _solve(prob: cp.Problem) -> None:
    """Solve ``prob`` with ``SOLVER`` and raise unless it reaches an accepted status.

    Parameters
    ----------
    prob
        The cvxpy problem to solve in place.

    Raises
    ------
    OptimizationError
        If the solver raises ``cvxpy.error.SolverError`` or ends with a status
        not in ``ACCEPTED_STATUSES``.
    """
    try:
        prob.solve(solver=SOLVER, verbose=False)
    except cp.error.SolverError as exc:
        raise OptimizationError(f"Solver {SOLVER} failed: {exc}") from exc
    if prob.status not in ACCEPTED_STATUSES:
        raise OptimizationError(f"Solver {SOLVER} returned status {prob.status!r}")


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

    Returns
    -------
    w: (N,) portfolio weights

    Raises
    ------
    OptimizationError
        If the regularized covariance matrix is singular.
    """
    N = Sigma.shape[0]
    Sigma_reg = Sigma + eps * np.eye(N)
    ones = np.ones(N)
    try:
        Sigma_inv_ones = np.linalg.solve(Sigma_reg, ones)
    except np.linalg.LinAlgError as exc:
        raise OptimizationError(f"Covariance matrix is singular: {exc}") from exc
    return Sigma_inv_ones / (ones @ Sigma_inv_ones)


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

    Returns
    -------
    w: (N,) portfolio weights

    Notes
    -----
    Raises ``ValueError`` (from ``check_weight_cap_feasible``) if the weight cap
    is infeasible, and ``OptimizationError`` (from ``_solve``) if the solver
    fails or does not reach an accepted status.
    """
    N = Sigma.shape[0]
    check_weight_cap_feasible(N, max_weight)
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
    _solve(prob)
    return np.asarray(w.value, dtype=float)


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

    Returns
    -------
    w: (N,) portfolio weights

    Notes
    -----
    Raises ``ValueError`` (from ``check_weight_cap_feasible``) if the weight cap
    is infeasible, and ``OptimizationError`` (from ``_solve``) if the solver
    fails or does not reach an accepted status.
    """
    N = Sigma.shape[0]
    check_weight_cap_feasible(N, max_weight)
    Sigma_psd = nearest_psd(Sigma, epsilon=eps)

    w = cp.Variable(N, nonneg=True)
    objective = cp.Minimize(cp.quad_form(w, Sigma_psd))
    constraints = [
        cp.sum(w) == 1,
        w <= max_weight,
    ]

    prob = cp.Problem(objective, constraints)
    _solve(prob)
    return np.maximum(np.asarray(w.value, dtype=float), 0)


def compute_portfolio_metrics(
    weights_history: np.ndarray,
    returns: np.ndarray,
    Sigma_hat_list: list[np.ndarray],
) -> dict:
    """Compute portfolio performance metrics per paper specification.

    Parameters
    ----------
    weights_history: (T_oos, N) portfolio weights at each rebalancing
    returns: (T_oos, N) realized stock returns, in decimals (0.01 = 1%)
    Sigma_hat_list: list of T_oos (N, N) forecast covariance matrices of decimal returns

    Returns
    -------
    dict with performance metrics:
      std: ex-post realized standard deviation
      lpstd: lower partial standard deviation
      excess_kurtosis, skewness
      avg_diversification_ratio, avg_max_weight, avg_min_weight
      avg_gross_leverage, prop_negative, avg_turnover
    """
    T_oos = weights_history.shape[0]

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


def _optimize(
    Sigma_hat: np.ndarray,
    constraint_type: str,
    short_leverage_cap: float,
    max_weight: float,
) -> np.ndarray:
    """Dispatch to the optimizer for ``constraint_type``.

    Parameters
    ----------
    Sigma_hat
        Forecast covariance matrix, shape (N, N).
    constraint_type
        One of ``CONSTRAINT_TYPES``.
    short_leverage_cap
        Short leverage cap for the restricted regime.
    max_weight
        Per-asset weight cap for the restricted and long-only regimes.

    Returns
    -------
    np.ndarray
        Portfolio weights, shape (N,).
    """
    if constraint_type == "unconstrained":
        return min_variance_unconstrained(Sigma_hat)
    if constraint_type == "restricted":
        return min_variance_restricted(Sigma_hat, short_leverage_cap, max_weight)
    return min_variance_long_only(Sigma_hat, max_weight)


CONSTRAINT_TYPES = ("unconstrained", "restricted", "long_only")


def run_portfolio_experiment(
    Sigma_hat_list: list[np.ndarray],
    returns: np.ndarray,
    constraint_type: str = "unconstrained",
    short_leverage_cap: float = 0.30,
    max_weight: float = 0.20,
) -> dict[str, Any]:
    """Run full portfolio experiment for a given constraint regime.

    On a day where the optimizer raises ``OptimizationError``, the portfolio
    falls back to equal weights, a warning is logged, and the day is counted
    in ``metrics["n_fallback"]`` and listed in ``fallback_steps``.

    Parameters
    ----------
    Sigma_hat_list: list of T_oos (N, N) forecast covariance matrices of decimal returns
    returns: (T_oos, N) realized stock returns, in decimals (0.01 = 1%)
    constraint_type: 'unconstrained', 'restricted', or 'long_only'
    short_leverage_cap: for restricted
    max_weight: for restricted and long_only

    Returns
    -------
    dict with weights, constraint_type, metrics, and fallback_steps

    Raises
    ------
    ValueError
        If ``constraint_type`` is unknown or the weight cap is infeasible.
    """
    if constraint_type not in CONSTRAINT_TYPES:
        raise ValueError(f"Unknown constraint_type: {constraint_type}")
    T_oos = len(Sigma_hat_list)
    N = Sigma_hat_list[0].shape[0]
    if constraint_type != "unconstrained":
        check_weight_cap_feasible(N, max_weight)
        check_solver_installed()
    weights_history = np.zeros((T_oos, N))
    fallback_steps: list[int] = []

    for t, Sigma_hat in enumerate(Sigma_hat_list):
        try:
            weights_history[t] = _optimize(
                Sigma_hat, constraint_type, short_leverage_cap, max_weight
            )
        except OptimizationError as exc:
            logger.warning(
                "%s optimization failed on day %d: %s; using equal weights",
                constraint_type,
                t,
                exc,
            )
            fallback_steps.append(t)
            weights_history[t] = np.ones(N) / N

    metrics = compute_portfolio_metrics(weights_history, returns, Sigma_hat_list)
    metrics["n_fallback"] = len(fallback_steps)

    return {
        "weights": weights_history,
        "constraint_type": constraint_type,
        "metrics": metrics,
        "fallback_steps": fallback_steps,
    }
