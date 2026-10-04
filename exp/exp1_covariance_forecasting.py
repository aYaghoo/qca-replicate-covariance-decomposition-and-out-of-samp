"""Experiment 1: Covariance Decomposition and Out-of-Sample Forecasting

Implements the paper's factor-based decomposition and rolling one-step-ahead
forecasting pipeline. Tests factor specifications K ∈ {1, 3, 5, 7}, LASSO vs
adaptive LASSO, and log vs non-log matrix transformation.

Note on scale: The paper uses N=430 stocks, T=1,495 days, rolling_window=1,000,
producing 473 out-of-sample forecasts. This script demonstrates the methodology
on simulated data with N=30, T=500, rolling_window=200 (giving ~278 OOS forecasts)
to achieve tractable run times. The code is fully general and scales to N=430.

Usage: python -m exp.exp1_covariance_forecasting
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import clean_covariance_matrices
from src.data_simulation import SimulatedMarketData
from src.decomposition import decompose_covariance
from src.factors import build_factor_weight_matrix
from src.forecasting import rolling_forecast_pipeline
from src.metrics import (
    average_l2_forecast_error,
    random_walk_forecast,
)
from src.utils import vech

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (demonstration scale — see module docstring for paper scale)
# ──────────────────────────────────────────────────────────────────────────────
CFG = dict(
    N=20,  # number of stocks (paper: 430)
    K_max=7,  # highest factor count
    T=250,  # total days (paper: 1,495)
    rolling_window=100,  # estimation window (paper: 1,000)
    n_alphas=3,  # lambda grid points (coarser grid for speed)
    seed=42,
)

FACTOR_SPECS = [1, 3, 5, 7]
METHODS = [
    dict(use_log=False, use_adaptive=False, tag="LASSO"),
    dict(use_log=False, use_adaptive=True, tag="AdaLASSO"),
    dict(use_log=True, use_adaptive=False, tag="Log_LASSO"),
    dict(use_log=True, use_adaptive=True, tag="Log_AdaLASSO"),
]


def simulate_data(cfg: dict) -> dict:
    """Generate all simulation inputs."""
    logger.info("Simulating market data (N=%d, T=%d)...", cfg["N"], cfg["T"])
    sim = SimulatedMarketData(N=cfg["N"], K=cfg["K_max"], S=10, T=cfg["T"], seed=cfg["seed"])
    Sigma_list, factor_returns, stock_returns = sim.simulate_realized_covariances()
    market_caps = sim.generate_market_caps()
    bm_ratios = sim.generate_bm_ratios()
    accounting = sim.generate_accounting_data()
    return dict(
        Sigma_list=Sigma_list,
        stock_returns=stock_returns,
        factor_returns=factor_returns,
        sector_indices=sim.sector_indices,
        market_caps=market_caps,
        bm_ratios=bm_ratios,
        accounting=accounting,
        sim=sim,
    )


def clean_matrices(Sigma_list: list[np.ndarray]) -> list[np.ndarray]:
    """Apply 4-sigma outlier detection and replacement (paper Section 2)."""
    logger.info("Cleaning %d covariance matrices...", len(Sigma_list))
    cleaned, flags = clean_covariance_matrices(
        Sigma_list,
        sigma_threshold=4.0,
        flag_fraction=0.25,
        replacement_window=10,
    )
    n_flagged = sum(flags)
    logger.info(
        "Flagged and replaced %d matrices (%.1f%%)", n_flagged, 100 * n_flagged / len(Sigma_list)
    )
    return cleaned


def build_weight_matrix(K: int, data: dict) -> np.ndarray:
    """Build K×N factor weight matrix using the paper's factor definitions."""
    return build_factor_weight_matrix(
        K=K,
        market_caps=data["market_caps"],
        bm_ratios=data["bm_ratios"],
        accounting=data["accounting"],
    )


def verify_all_decompositions(
    Sigma_list: list[np.ndarray],
    W: np.ndarray,
    K: int,
    n_check: int = 20,
) -> float:
    """Spot-check decomposition identity on random sample of days."""
    T = len(Sigma_list)
    indices = np.random.default_rng(0).choice(T, size=min(n_check, T), replace=False)
    errors = []
    for t in indices:
        Sf, Bt, Se = decompose_covariance(Sigma_list[t], W)
        reconstructed = Bt.T @ Sf @ Bt + Se
        max_abs = np.max(np.abs(Sigma_list[t] - reconstructed))
        errors.append(max_abs)
    mean_err = float(np.mean(errors))
    logger.info("K=%d decomposition check (n=%d): max-abs error = %.2e", K, n_check, mean_err)
    return mean_err


def run_all_models(
    Sigma_list: list[np.ndarray],
    sector_indices: list[np.ndarray],
    cfg: dict,
    data: dict,
) -> dict:
    """Run rolling forecast pipeline for all factor specs and method variants."""
    T = len(Sigma_list)
    t_oos_start = cfg["rolling_window"] + 22
    n_oos = T - t_oos_start
    logger.info(
        "Total days: %d | rolling_window: %d | OOS start: %d | n_oos: %d",
        T,
        cfg["rolling_window"],
        t_oos_start,
        n_oos,
    )

    results = {}
    for K in FACTOR_SPECS:
        W = build_weight_matrix(K, data)
        _ = verify_all_decompositions(Sigma_list, W, K, n_check=10)

        for mspec in METHODS:
            model_name = f"K{K}_{mspec['tag']}"
            logger.info("Running model: %s", model_name)
            t0 = time.time()
            try:
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
                elapsed = time.time() - t0
                mean_l2 = float(np.mean(res["l2_errors"]))
                mean_l2_f = float(np.mean(res["l2_factor_errors"]))
                logger.info(
                    "  %s: mean_l2=%.4f, mean_l2_factor=%.4f, n_fallback=%d, elapsed=%.1fs",
                    model_name,
                    mean_l2,
                    mean_l2_f,
                    res["n_fallback"],
                    elapsed,
                )
                results[model_name] = res
            except Exception as exc:
                logger.warning("Model %s failed: %s", model_name, exc)

    return results


def compute_random_walk_baseline(
    Sigma_list: list[np.ndarray],
    t_oos_start: int,
) -> dict:
    """Compute random-walk benchmark L2 errors."""
    T = len(Sigma_list)
    rw_forecasts = random_walk_forecast(Sigma_list, t_oos_start)
    true_matrices = [Sigma_list[t] for t in range(t_oos_start, T)]
    rw_l2 = average_l2_forecast_error(rw_forecasts, true_matrices)
    rw_errors = [
        float(np.linalg.norm(vech(rw_forecasts[i] - true_matrices[i])))
        for i in range(len(rw_forecasts))
    ]
    return {"l2_errors": rw_errors, "mean_l2": rw_l2}


def save_results_table(
    results: dict,
    rw_baseline: dict,
    cfg: dict,
    path: Path,
) -> pd.DataFrame:
    """Compile and save comparison table."""
    rows = []

    # Random walk
    rows.append(
        {
            "model": "RandomWalk",
            "K": "-",
            "method": "-",
            "log": "-",
            "mean_l2_full": rw_baseline["mean_l2"],
            "std_l2_full": float(np.std(rw_baseline["l2_errors"])),
            "mean_l2_factor": float("nan"),
            "rel_to_rw": 1.0,
            "n_fallback": 0,
        }
    )

    for name, res in results.items():
        parts = name.split("_", 1)
        K = int(parts[0][1:])
        tag = parts[1]
        use_log = tag.startswith("Log")
        use_ada = "Ada" in tag

        mean_l2 = float(np.mean(res["l2_errors"]))
        std_l2 = float(np.std(res["l2_errors"]))
        mean_l2_f = float(np.mean(res["l2_factor_errors"]))

        rows.append(
            {
                "model": name,
                "K": K,
                "method": "AdaLASSO" if use_ada else "LASSO",
                "log": use_log,
                "mean_l2_full": mean_l2,
                "std_l2_full": std_l2,
                "mean_l2_factor": mean_l2_f,
                "rel_to_rw": mean_l2 / rw_baseline["mean_l2"],
                "n_fallback": res["n_fallback"],
            }
        )

    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(path / "exp1_l2_errors.csv")
    logger.info("Saved L2 error table to %s", path / "exp1_l2_errors.csv")
    return df


def plot_l2_errors(results: dict, rw_baseline: dict, path: Path) -> None:
    """Plot mean L2 errors across model specifications."""
    sns.set_theme(style="whitegrid")

    model_names = ["RandomWalk"] + list(results.keys())
    means = [rw_baseline["mean_l2"]] + [float(np.mean(r["l2_errors"])) for r in results.values()]
    stds = [float(np.std(rw_baseline["l2_errors"]))] + [
        float(np.std(r["l2_errors"])) for r in results.values()
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 0.7), 5))
    x = np.arange(len(model_names))
    colors = [
        "steelblue" if n == "RandomWalk" else "tomato" if "Log" in n else "seagreen"
        for n in model_names
    ]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean L2 Forecast Error")
    ax.set_title("Exp 1: Mean L2 Covariance Forecast Errors by Model")
    ax.axhline(
        rw_baseline["mean_l2"], color="navy", linestyle="--", alpha=0.6, label="Random Walk"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(path / "exp1_l2_comparison.png", dpi=120)
    plt.close(fig)
    logger.info("Saved L2 comparison plot.")


def plot_l2_timeseries(results: dict, rw_baseline: dict, path: Path, n_show: int = 4) -> None:
    """Plot L2 error time series for a subset of models."""
    selected = list(results.keys())[:n_show]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rw_baseline["l2_errors"], color="grey", alpha=0.6, label="Random Walk", lw=1)
    for name in selected:
        ax.plot(results[name]["l2_errors"], alpha=0.8, label=name, lw=1)
    ax.set_xlabel("Out-of-sample step")
    ax.set_ylabel("L2 Forecast Error")
    ax.set_title("Exp 1: L2 Forecast Error Over OOS Period")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path / "exp1_l2_timeseries.png", dpi=120)
    plt.close(fig)
    logger.info("Saved L2 timeseries plot.")


def plot_factor_cov_l2(results: dict, path: Path) -> None:
    """Bar chart of factor covariance L2 errors by model."""
    model_names = list(results.keys())
    means = [float(np.mean(r["l2_factor_errors"])) for r in results.values()]

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 0.7), 4))
    colors = ["tomato" if "Log" in n else "seagreen" for n in model_names]
    ax.bar(model_names, means, color=colors, alpha=0.8)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean L2 Error (Factor Cov)")
    ax.set_title("Exp 1: Factor Covariance L2 Errors")
    plt.tight_layout()
    fig.savefig(path / "exp1_factor_cov_l2.png", dpi=120)
    plt.close(fig)
    logger.info("Saved factor covariance L2 plot.")


def plot_decomposition_check(
    Sigma_list: list[np.ndarray],
    data: dict,
    path: Path,
) -> None:
    """Visualize decomposition accuracy for K=3."""
    W = build_weight_matrix(3, data)
    T = len(Sigma_list)
    sample_t = min(50, T - 1)
    Sf, Bt, Se = decompose_covariance(Sigma_list[sample_t], W)
    reconstructed = Bt.T @ Sf @ Bt + Se

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmax = np.percentile(np.abs(Sigma_list[sample_t]), 95)
    for ax, M, title in zip(
        axes,
        [Sigma_list[sample_t], reconstructed, np.abs(Sigma_list[sample_t] - reconstructed)],
        ["Σ_t (realized)", "B'Σ_f B + Σ_e (recon)", "|Error|"],
    ):
        im = ax.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"Covariance Decomposition Check (t={sample_t}, K=3)", fontsize=11)
    plt.tight_layout()
    fig.savefig(path / "exp1_decomposition_check.png", dpi=120)
    plt.close(fig)
    logger.info("Saved decomposition check plot.")


def run_experiment() -> dict[str, Any]:
    """Main experiment runner for Exp 1."""
    logger.info("=" * 60)
    logger.info("Experiment 1: Covariance Decomposition and Forecasting")
    logger.info("=" * 60)
    logger.info("Config: %s", CFG)

    # 1. Simulate data
    data = simulate_data(CFG)
    Sigma_list = data["Sigma_list"]

    # 2. Clean matrices
    Sigma_clean = clean_matrices(Sigma_list)

    # 3. Decomposition sanity check
    plot_decomposition_check(Sigma_clean, data, RESULTS_DIR)

    # 4. Run all models
    results = run_all_models(Sigma_clean, data["sector_indices"], CFG, data)

    # 5. Random walk baseline
    t_oos_start = CFG["rolling_window"] + 22
    rw_baseline = compute_random_walk_baseline(Sigma_clean, t_oos_start)
    logger.info("Random Walk mean L2 = %.4f", rw_baseline["mean_l2"])

    # 6. Save results
    df = save_results_table(results, rw_baseline, CFG, RESULTS_DIR)
    logger.info(
        "\n%s", df[["K", "method", "log", "mean_l2_full", "rel_to_rw", "n_fallback"]].to_string()
    )

    # 7. Plots
    plot_l2_errors(results, rw_baseline, RESULTS_DIR)
    plot_l2_timeseries(results, rw_baseline, RESULTS_DIR)
    plot_factor_cov_l2(results, RESULTS_DIR)

    # 8. Save summary JSON for RESULTS.md
    summary = {
        "config": CFG,
        "n_oos": len(list(results.values())[0]["l2_errors"]) if results else 0,
        "random_walk_mean_l2": rw_baseline["mean_l2"],
        "models": {
            name: {
                "mean_l2": float(np.mean(r["l2_errors"])),
                "std_l2": float(np.std(r["l2_errors"])),
                "mean_l2_factor": float(np.mean(r["l2_factor_errors"])),
                "rel_to_rw": float(np.mean(r["l2_errors"])) / rw_baseline["mean_l2"],
                "n_fallback": r["n_fallback"],
                "fallback_steps": r["fallback_steps"],
            }
            for name, r in results.items()
        },
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Experiment 1 complete. Results saved to %s", RESULTS_DIR)
    return summary


if __name__ == "__main__":
    run_experiment()
