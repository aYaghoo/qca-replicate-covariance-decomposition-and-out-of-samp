"""Experiment 2: Minimum-Variance Portfolio Evaluation

Uses one-step-ahead covariance forecasts from Exp 1 to form daily
minimum-variance portfolios under three constraint regimes:
  - Unconstrained
  - Restricted (short leverage ≤ 30%, |w_i| ≤ 20%)
  - Long-only (0 ≤ w_i ≤ 20%)

Evaluates ex-post risk and portfolio characteristics for each model.

Units: the simulator and the forecasts are in percent (as in Exp 1). The
portfolio code expects decimals, so `to_decimal_units` converts once, after
forecasting: returns / 100, covariance matrices (observed and forecast) / 100**2.

Usage: python -m exp.exp2_portfolio_evaluation
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import clean_covariance_matrices
from src.data_simulation import SimulatedMarketData
from src.factors import build_factor_weight_matrix
from src.forecasting import rolling_forecast_pipeline
from src.metrics import random_walk_forecast
from src.portfolio import check_solver_installed, run_portfolio_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (must match Exp 1 scale parameters)
# ──────────────────────────────────────────────────────────────────────────────
CFG = dict(
    N=20,
    K_max=7,
    T=250,
    rolling_window=100,
    n_alphas=3,
    seed=42,
)

# Subset of models for portfolio evaluation (reduced for speed)
PORTFOLIO_MODELS = [
    dict(K=1, use_log=False, use_adaptive=False, tag="K1_LASSO"),
    dict(K=3, use_log=False, use_adaptive=False, tag="K3_LASSO"),
    dict(K=3, use_log=False, use_adaptive=True, tag="K3_AdaLASSO"),
    dict(K=3, use_log=True, use_adaptive=False, tag="K3_Log_LASSO"),
    dict(K=5, use_log=False, use_adaptive=False, tag="K5_LASSO"),
    dict(K=7, use_log=False, use_adaptive=False, tag="K7_LASSO"),
    dict(K=7, use_log=False, use_adaptive=True, tag="K7_AdaLASSO"),
]

CONSTRAINT_TYPES = ["unconstrained", "restricted", "long_only"]
PERCENT = 100.0  # simulated returns are in percent; divide by this to get decimals
SHORT_LEVERAGE_CAP = 0.30
MAX_WEIGHT = 0.20


def generate_data(
    cfg: dict[str, Any],
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], pd.Series, pd.Series, pd.DataFrame]:
    """Re-generate simulation data (must use same seed as Exp 1)."""
    sim = SimulatedMarketData(N=cfg["N"], K=cfg["K_max"], S=10, T=cfg["T"], seed=cfg["seed"])
    Sigma_list, _, stock_returns = sim.simulate_realized_covariances()
    market_caps = sim.generate_market_caps()
    bm_ratios = sim.generate_bm_ratios()
    accounting = sim.generate_accounting_data()
    cleaned, _ = clean_covariance_matrices(Sigma_list)
    return cleaned, stock_returns, sim.sector_indices, market_caps, bm_ratios, accounting


def get_forecast_results(
    Sigma_list: list[np.ndarray],
    sector_indices: list[np.ndarray],
    market_caps,
    bm_ratios,
    accounting,
    cfg: dict,
) -> dict[str, dict]:
    """Run rolling forecasts for all portfolio models."""
    results = {}
    for mspec in PORTFOLIO_MODELS:
        K = mspec["K"]
        tag = mspec["tag"]
        W = build_factor_weight_matrix(
            K=K,
            market_caps=market_caps,
            bm_ratios=bm_ratios,
            accounting=accounting,
        )
        logger.info("Forecasting %s...", tag)
        res = rolling_forecast_pipeline(
            Sigma_list=Sigma_list,
            W_t=W,
            sector_indices=sector_indices,
            K=K,
            rolling_window=cfg["rolling_window"],
            use_log=mspec["use_log"],
            use_adaptive=mspec["use_adaptive"],
            n_alphas=cfg["n_alphas"],
            verbose=False,
        )
        results[tag] = res
    return results


def to_decimal_units(
    stock_returns: np.ndarray,
    rw_forecasts: list[np.ndarray],
    forecast_results: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, dict[str, Any]]]:
    """Convert percent-unit inputs to the decimal units the portfolio code expects.

    Parameters
    ----------
    stock_returns
        Daily stock returns in percent, shape (T, N).
    rw_forecasts
        Random-walk forecasts (observed covariance matrices) of percent returns.
    forecast_results
        Pipeline results by model tag; ``Sigma_hat_list`` is in percent squared.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], dict[str, dict[str, Any]]]
        Returns / 100, RW forecasts / 100**2, and per model the forecast
        ``Sigma_hat_list`` / 100**2 with its forecast ``n_fallback``.
    """
    cov_scale = PERCENT**2
    return (
        stock_returns / PERCENT,
        [S / cov_scale for S in rw_forecasts],
        {
            tag: {
                "Sigma_hat_list": [S / cov_scale for S in res["Sigma_hat_list"]],
                "n_fallback": res["n_fallback"],
            }
            for tag, res in forecast_results.items()
        },
    )


def get_rw_forecasts(
    Sigma_list: list[np.ndarray],
    t_oos_start: int,
) -> list[np.ndarray]:
    """Naive random-walk forecasts."""
    return random_walk_forecast(Sigma_list, t_oos_start)


def run_all_portfolios(
    forecast_results: dict[str, dict],
    rw_forecasts: list[np.ndarray],
    stock_returns: np.ndarray,
    t_oos_start: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Run portfolio optimization for all models and constraint types."""
    n_oos_max = min(
        len(list(forecast_results.values())[0]["Sigma_hat_list"]),
        len(rw_forecasts),
    )
    oos_returns = stock_returns[t_oos_start : t_oos_start + n_oos_max]

    all_results = {}

    # Random walk portfolio
    for ctype in CONSTRAINT_TYPES:
        key = f"RandomWalk_{ctype}"
        logger.info("Portfolio: %s", key)
        res = run_portfolio_experiment(
            Sigma_hat_list=rw_forecasts[:n_oos_max],
            returns=oos_returns,
            constraint_type=ctype,
            short_leverage_cap=SHORT_LEVERAGE_CAP,
            max_weight=MAX_WEIGHT,
        )
        res["n_forecast_fallback"] = 0
        all_results[key] = res

    # Model-based portfolios
    for tag, fc_res in forecast_results.items():
        Sigma_hat_list = fc_res["Sigma_hat_list"][:n_oos_max]
        for ctype in CONSTRAINT_TYPES:
            key = f"{tag}_{ctype}"
            logger.info("Portfolio: %s", key)
            res = run_portfolio_experiment(
                Sigma_hat_list=Sigma_hat_list,
                returns=oos_returns,
                constraint_type=ctype,
                short_leverage_cap=SHORT_LEVERAGE_CAP,
                max_weight=MAX_WEIGHT,
            )
            res["n_forecast_fallback"] = fc_res["n_fallback"]
            all_results[key] = res

    return all_results, n_oos_max


def build_metrics_table(portfolio_results: dict) -> pd.DataFrame:
    """Assemble performance metrics into a comparison table."""
    rows = []
    for key, res in portfolio_results.items():
        m = res["metrics"]
        # Strip constraint suffix correctly — "long_only" contains an underscore,
        # so we must match full suffix strings rather than use rsplit("_", 1).
        ctype = next(ct for ct in CONSTRAINT_TYPES if key.endswith(f"_{ct}"))
        model = key[: -(len(ctype) + 1)]
        rows.append(
            {
                "model": model,
                "constraint": ctype,
                "std_ann": m["std_ann"] * 100,  # convert to %
                "lpstd_ann": m["lpstd_ann"] * 100,
                "excess_kurtosis": m["excess_kurtosis"],
                "skewness": m["skewness"],
                "avg_div_ratio": m["avg_diversification_ratio"],
                "avg_max_weight": m["avg_max_weight"],
                "avg_min_weight": m["avg_min_weight"],
                "avg_gross_leverage": m["avg_gross_leverage"],
                "prop_negative": m["prop_negative"],
                "avg_turnover": m["avg_turnover"],
                "n_fallback": m["n_fallback"],
                "n_forecast_fallback": res["n_forecast_fallback"],
            }
        )
    df = pd.DataFrame(rows)
    return df


def save_metrics(df: pd.DataFrame, path: Path) -> None:
    """Save metrics table to CSV."""
    df.to_csv(path / "exp2_portfolio_metrics.csv", index=False)
    logger.info("Portfolio metrics saved to %s", path / "exp2_portfolio_metrics.csv")


def plot_std_comparison(df: pd.DataFrame, path: Path) -> None:
    """Bar chart of annualized realized standard deviation by constraint type."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, ctype in zip(axes, CONSTRAINT_TYPES):
        subset = df[df["constraint"] == ctype].copy()
        subset = subset.sort_values("std_ann")
        colors = [
            "steelblue" if "RandomWalk" in m else "tomato" if "Log" in m else "seagreen"
            for m in subset["model"]
        ]
        ax.barh(subset["model"], subset["std_ann"], color=colors, alpha=0.85)
        ax.set_xlabel("Annualized Std Dev (%)")
        ax.set_title(f"Constraint: {ctype}")
        rw_rows = subset[subset["model"] == "RandomWalk"]["std_ann"].values
        if len(rw_rows):
            ax.axvline(rw_rows[0], color="navy", linestyle="--", alpha=0.6, label="RW")
        ax.legend(fontsize=7)

    plt.suptitle("Exp 2: Realized Portfolio Volatility by Model and Constraint", fontsize=12)
    plt.tight_layout()
    fig.savefig(path / "exp2_std_comparison.png", dpi=120)
    plt.close(fig)
    logger.info("Saved portfolio std comparison plot.")


def plot_cumulative_returns(
    portfolio_results: dict,
    path: Path,
    constraint_type: str = "restricted",
    n_show: int = 6,
) -> None:
    """Cumulative return paths for selected models under one constraint type."""
    fig, ax = plt.subplots(figsize=(12, 5))
    shown = 0
    for key, res in portfolio_results.items():
        if constraint_type not in key:
            continue
        if shown >= n_show:
            break
        port_returns = res["metrics"]["port_returns"]
        cum_ret = np.cumprod(1 + port_returns) - 1
        label = key.replace(f"_{constraint_type}", "")
        ax.plot(cum_ret, label=label, lw=1.2, alpha=0.85)
        shown += 1

    ax.set_xlabel("Out-of-sample day")
    ax.set_ylabel("Cumulative return")
    ax.set_title(f"Exp 2: Cumulative Returns ({constraint_type} portfolios)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(path / f"exp2_cumret_{constraint_type}.png", dpi=120)
    plt.close(fig)
    logger.info("Saved cumulative return plot (%s).", constraint_type)


def plot_weight_heatmap(
    portfolio_results: dict,
    path: Path,
    model_key: str = "K3_LASSO_restricted",
) -> None:
    """Heatmap of portfolio weights over time for one model."""
    if model_key not in portfolio_results:
        logger.warning("Key %s not found, skipping weight heatmap.", model_key)
        return

    W = portfolio_results[model_key]["weights"]  # (T_oos, N)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(W.T, aspect="auto", cmap="RdBu_r", vmin=-MAX_WEIGHT, vmax=MAX_WEIGHT)
    ax.set_xlabel("Out-of-sample day")
    ax.set_ylabel("Asset index")
    ax.set_title(f"Portfolio Weights Over Time: {model_key}")
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    fig.savefig(path / "exp2_weight_heatmap.png", dpi=120)
    plt.close(fig)
    logger.info("Saved weight heatmap.")


def plot_turnover(df: pd.DataFrame, path: Path) -> None:
    """Average portfolio turnover by constraint type."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot(index="model", columns="constraint", values="avg_turnover")
    pivot.plot(kind="bar", ax=ax, alpha=0.8)
    ax.set_ylabel("Average Daily Turnover")
    ax.set_title("Exp 2: Average Portfolio Turnover by Model and Constraint")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    fig.savefig(path / "exp2_turnover.png", dpi=120)
    plt.close(fig)
    logger.info("Saved turnover plot.")


def run_experiment() -> dict:
    """Main experiment runner for Exp 2."""
    logger.info("=" * 60)
    logger.info("Experiment 2: Minimum-Variance Portfolio Evaluation")
    logger.info("=" * 60)
    logger.info("Config: %s", CFG)
    check_solver_installed()

    # 1. Generate data (same seed as Exp 1)
    Sigma_list, stock_returns, sector_indices, market_caps, bm_ratios, accounting = generate_data(
        CFG
    )
    t_oos_start = CFG["rolling_window"] + 22
    oos_stock_returns = stock_returns[t_oos_start:]

    logger.info("OOS start: %d | OOS returns shape: %s", t_oos_start, oos_stock_returns.shape)

    # 2. Run rolling forecasts for portfolio models
    forecast_results = get_forecast_results(
        Sigma_list, sector_indices, market_caps, bm_ratios, accounting, CFG
    )

    # 3. Random walk forecasts
    rw_forecasts = get_rw_forecasts(Sigma_list, t_oos_start)

    # 4. Convert once from percent to the decimal units the portfolio code expects
    stock_returns_dec, rw_forecasts_dec, portfolio_inputs = to_decimal_units(
        stock_returns, rw_forecasts, forecast_results
    )
    logger.info("Converted returns (/%g) and covariances (/%g) to decimals", PERCENT, PERCENT**2)

    # 5. Portfolio optimization for all models × constraints
    portfolio_results, n_oos = run_all_portfolios(
        portfolio_inputs, rw_forecasts_dec, stock_returns_dec, t_oos_start
    )
    logger.info("Portfolio results computed for %d OOS days.", n_oos)

    # 6. Build metrics table
    df = build_metrics_table(portfolio_results)
    save_metrics(df, RESULTS_DIR)

    # 7. Summary: best model per constraint
    logger.info("\n=== Portfolio Performance Summary ===")
    for ctype in CONSTRAINT_TYPES:
        sub = df[df["constraint"] == ctype].sort_values("std_ann")
        logger.info("\nConstraint: %s", ctype)
        logger.info(
            sub[["model", "std_ann", "lpstd_ann", "avg_turnover", "n_fallback"]].to_string(
                index=False
            )
        )

    # 8. Plots
    plot_std_comparison(df, RESULTS_DIR)
    for ctype in CONSTRAINT_TYPES:
        plot_cumulative_returns(portfolio_results, RESULTS_DIR, constraint_type=ctype, n_show=5)
    plot_weight_heatmap(portfolio_results, RESULTS_DIR, model_key="K3_LASSO_restricted")
    plot_turnover(df, RESULTS_DIR)

    # 9. Save JSON summary
    best_restricted = df[df["constraint"] == "restricted"].sort_values("std_ann").iloc[0]
    best_long_only = df[df["constraint"] == "long_only"].sort_values("std_ann").iloc[0]

    summary = {
        "config": CFG,
        "units": "decimal returns; std_ann_pct and constraint_summary in annualized percent",
        "n_oos": n_oos,
        "best_restricted": {
            "model": best_restricted["model"],
            "std_ann_pct": float(best_restricted["std_ann"]),
        },
        "best_long_only": {
            "model": best_long_only["model"],
            "std_ann_pct": float(best_long_only["std_ann"]),
        },
        "n_fallback_total": int(df["n_fallback"].sum()),
        "n_forecast_fallback_by_model": {
            tag: fc["n_fallback"] for tag, fc in forecast_results.items()
        },
        "constraint_summary": df.groupby("constraint")["std_ann"]
        .agg(["mean", "min", "max"])
        .to_dict(),
    }
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Experiment 2 complete. Results saved to %s", RESULTS_DIR)
    return summary


if __name__ == "__main__":
    run_experiment()
