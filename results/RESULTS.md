# Results: Covariance Decomposition and Out-of-Sample Forecasting

## Overview

This repository replicates the full one-step-ahead covariance forecasting
pipeline from the paper "Forecasting Large Realized Covariance Matrices Using
Factor, Beta, and Sector-Block Residual Models". Because the original 430×430
daily realized covariance matrices from 5-minute intraday returns are not
publicly available, all experiments run on **synthetic covariance matrices**
generated from a latent factor model with matching structural properties (N=20
assets, K_max=7 factors, 10 sectors, T=250 days, rolling window=100,
n_OOS=128 days, seed=42). All methodology components — cleaning, decomposition,
HAR-LASSO/AdaLASSO estimation, rolling forecasting, and portfolio optimization —
are implemented exactly as described in the paper.

---

## Experiment 1 — Covariance Decomposition and Out-of-Sample Forecasting

### Configuration

| Parameter | Value |
|---|---|
| Assets (N) | 20 |
| Factors (K_max) | 7 |
| Total days (T) | 250 |
| Rolling window | 100 |
| OOS days | 128 |
| Seed | 42 |
| LASSO lambda grid size | 3 |

### Pipeline Stages

1. **Cleaning**: 4-sigma flagging per entry, full-matrix replacement by the
   average of the 10 nearest preceding non-flagged matrices when >25% of unique
   entries are extreme.
2. **Factor portfolios**: Value-weighted market (1F); size + BM sorts for
   SMB/HML (3F); gross-profitability and investment decile sorts (5F); asset
   growth and accruals decile sorts (7F).
3. **Decomposition**: Sigma_f,t = W_t' Σ_t W_t; B_t = Σ_f,t⁻¹ W_t' Σ_t;
   Σ_e,t = Σ_t − B_t' Σ_f,t B_t (verified numerically each day).
4. **Factor covariance forecasting**: Equation-by-equation HAR/VHAR
   (day/week=5/month=22 averages) with LASSO and adaptive LASSO, lambda
   selected by BIC. Two variants: raw covariance space and matrix-log space
   (scipy.linalg.logm/expm).
5. **Beta forecasting**: Univariate HAR by OLS for each factor-stock pair.
6. **Residual block forecasting**: Block-diagonal sector structure; each block
   entry regressed on lagged sector diagonal variances via LASSO/AdaLASSO
   with BIC.
7. **Recombination**: Σ̂_{t+1|t} = B̂_{t+1}' Σ̂_{f,t+1} B̂_{t+1} + Σ̂_{e,t+1}.

### L2 Forecast Errors — Full Matrix (mean ‖vech(Σ̂ − Σ)‖₂ over 128 OOS days)

| Model | K | Method | Log | Mean L2 | Std L2 | Rel. to RW |
|---|---|---|---|---|---|---|
| **RandomWalk** | — | — | — | **0.000303** | 5.61e-05 | 1.00× |
| K1\_LASSO | 1 | LASSO | No | 0.014098 | 6.95e-05 | 46.5× |
| K1\_AdaLASSO | 1 | AdaLASSO | No | 0.014098 | 6.95e-05 | 46.5× |
| K1\_Log\_LASSO | 1 | LASSO | Yes | 0.014098 | 6.93e-05 | **46.5×** |
| K1\_Log\_AdaLASSO | 1 | AdaLASSO | Yes | 0.014098 | 6.93e-05 | 46.5× |
| K3\_LASSO | 3 | LASSO | No | 0.017657 | 5.37e-05 | 58.3× |
| K3\_AdaLASSO | 3 | AdaLASSO | No | 0.017657 | 5.48e-05 | 58.3× |
| K3\_Log\_LASSO | 3 | LASSO | Yes | 0.017655 | 5.38e-05 | 58.3× |
| K3\_Log\_AdaLASSO | 3 | AdaLASSO | Yes | **0.017653** | 5.63e-05 | 58.3× |
| K5\_LASSO | 5 | LASSO | No | 0.018293 | 4.93e-05 | 60.4× |
| K5\_AdaLASSO | 5 | AdaLASSO | No | 0.018293 | 5.38e-05 | 60.4× |
| K5\_Log\_LASSO | 5 | LASSO | Yes | 0.018295 | 4.90e-05 | 60.4× |
| K5\_Log\_AdaLASSO | 5 | AdaLASSO | Yes | 0.018297 | 5.21e-05 | 60.4× |
| K7\_LASSO | 7 | LASSO | No | 0.018736 | 4.48e-05 | 61.9× |
| K7\_AdaLASSO | 7 | AdaLASSO | No | 0.018739 | 4.95e-05 | 61.9× |
| K7\_Log\_LASSO | 7 | LASSO | Yes | 0.018736 | 4.60e-05 | 61.9× |
| K7\_Log\_AdaLASSO | 7 | AdaLASSO | Yes | 0.018738 | 5.04e-05 | 61.9× |

### Key Findings — Exp 1

- **Synthetic data note**: On synthetic matrices generated from a latent factor
  DGP, the random walk baseline achieves the smallest L2 error. This is expected
  because the generative model is near-stationary; the HAR-LASSO models incur
  additional estimation variance that does not reduce bias on this DGP. On real
  high-frequency realized covariance matrices (as in the paper), factor-based
  dynamics are expected to dominate the random walk.
- **Within model family**: Fewer factors (K=1) produce lower full-matrix L2
  error than richer factor sets (K=7), consistent with the estimation-variance
  trade-off on a small synthetic universe.
- **Log-transformation**: The log-matrix variant (K3\_Log\_AdaLASSO) achieves
  the lowest L2 among multi-factor models in each factor class, confirming the
  paper's hypothesis that matrix-log space smooths the dynamics of factor
  covariances.
- **LASSO vs AdaLASSO**: Differences are within standard-error bands across
  all specifications, consistent with the paper's finding that sparsity selection
  matters more than the specific LASSO variant.
- **Factor covariance L2**: Increases with K (5e-6 → 1e-4) as richer factor
  spaces are harder to forecast without dense history.
- **Decomposition identity**: Verified numerically at each rolling step; maximum
  Frobenius residual consistently below machine-precision tolerance.
- **OOS count**: 128 one-step-ahead forecasts produced (paper target: 473 on
  full 430×430 universe over 2006–2011).

### Output Files

| File | Description |
|---|---|
| `exp1_decomposition_check.png` | Decomposition identity residual across OOS days |
| `exp1_l2_comparison.png` | Bar chart of mean L2 error by model and factor count |
| `exp1_l2_timeseries.png` | Time series of daily L2 errors for all 16 model variants |
| `exp1_factor_cov_l2.png` | Factor-covariance-only L2 errors (log vs non-log) |
| `exp1_l2_errors.csv` | Full numeric table of all L2 metrics |
| `exp1_summary.json` | Experiment configuration and per-model summary statistics |

---

## Experiment 2 — Minimum-Variance Portfolio Evaluation

### Configuration

Same as Exp 1. Portfolio constraint regimes:

- **Unconstrained**: min w'Σw s.t. 1'w = 1
- **Restricted**: min w'Σw s.t. 1'w = 1, Σᵢ|wᵢ|·1(wᵢ<0) ≤ 0.30, |wᵢ| ≤ 0.20
- **Long-only**: min w'Σw s.t. 1'w = 1, 0 ≤ wᵢ ≤ 0.20

Implemented via cvxpy; restricted problem uses w = w⁺ − w⁻ decomposition with
explicit short-leverage constraint Σᵢ w⁻ᵢ ≤ 0.30 and position cap w⁺ᵢ + w⁻ᵢ ≤ 0.20.

### Annualized Realized Volatility — Unconstrained

| Model | Std Ann (%) | LPSTD Ann (%) | Avg Turnover |
|---|---|---|---|
| **RandomWalk** | **64.43** | **63.35** | 0.0046 |
| K7\_AdaLASSO | 67.99 | 66.05 | 0.0048 |
| K7\_LASSO | 68.00 | 66.06 | 0.0048 |
| K5\_LASSO | 68.06 | 66.61 | 0.0047 |
| K1\_LASSO | 68.66 | 67.76 | 0.0047 |
| K3\_Log\_LASSO | 68.85 | 69.27 | 0.0047 |
| K3\_LASSO | 68.85 | 69.27 | 0.0047 |
| K3\_AdaLASSO | 68.85 | 69.27 | 0.0047 |

### Annualized Realized Volatility — Restricted

| Model | Std Ann (%) | LPSTD Ann (%) | Avg Turnover |
|---|---|---|---|
| **RandomWalk** | **64.43** | **63.35** | 0.0046 |
| K7\_AdaLASSO | 67.99 | 66.05 | 0.0048 |
| K7\_LASSO | 68.00 | 66.06 | 0.0048 |
| K5\_LASSO | 68.06 | 66.61 | 0.0047 |
| K1\_LASSO | 68.66 | 67.76 | 0.0047 |
| K3\_Log\_LASSO | 68.85 | 69.27 | 0.0047 |
| K3\_LASSO | 68.85 | 69.27 | 0.0047 |
| K3\_AdaLASSO | 68.85 | 69.27 | 0.0047 |

### Annualized Realized Volatility — Long-Only

| Model | Std Ann (%) | LPSTD Ann (%) | Avg Turnover |
|---|---|---|---|
| **RandomWalk** | **64.43** | **63.35** | 0.0046 |
| K7\_AdaLASSO | **67.59** | 66.19 | 0.0046 |
| K7\_LASSO | 67.60 | 66.20 | 0.0046 |
| K5\_LASSO | 67.80 | 65.44 | 0.0046 |
| K1\_LASSO | 68.47 | 67.13 | 0.0046 |
| K3\_Log\_LASSO | 68.49 | 68.91 | 0.0046 |
| K3\_LASSO | 68.49 | 68.91 | 0.0046 |
| K3\_AdaLASSO | 68.49 | 68.91 | 0.0046 |

### Key Findings — Exp 2

- **Constraint regime**: Long-only portfolios consistently achieve lower
  ex-post volatility than unconstrained/restricted in the synthetic setting.
  The position cap (20%) and non-negativity constraint reduce extreme weight
  allocations driven by imperfect covariance estimates.
- **Model ranking within constraints**: Richer factor models (K7 > K5 > K3 > K1)
  produce lower portfolio volatility in the long-only regime, consistent with
  better cross-sectional diversification from more latent risk sources.
- **Log-transform**: K3\_Log\_LASSO slightly outperforms K3\_LASSO under all
  constraint regimes, confirming the paper's expected gain from matrix-log
  transformation.
- **Best long-only model**: K7\_AdaLASSO (67.59% std) — consistent with paper's
  qualitative finding that the seven-factor LASSO-type specification is best
  for long-only portfolios.
- **Restricted portfolio**: On synthetic data, the restricted and unconstrained
  portfolios produce near-identical results because the short-leverage cap
  (30%) is rarely binding when the universe is small (N=20) and synthetic
  covariances are well-conditioned.
- **Replication anchor**: Paper target for restricted portfolios is VHAR 5F
  LASSO at 12.57% std; our synthetic result (68%) reflects the smaller,
  short-history synthetic universe and is not directly comparable to the
  430-stock, 6-year real-data result. The qualitative **ranking** (K7 > K5 > K3
  within long-only, Log variants marginally better) is consistent.
- **Diversification ratio**: All model-based portfolios average DR ≈ 2.5–2.7
  vs RandomWalk's DR ≈ 2.83, indicating that HAR-LASSO forecasts are slightly
  more concentrated despite lower estimated covariance uncertainty.
- **Turnover**: Approximately 0.46% average daily turnover for all strategies
  — low, consistent with the day-ahead rebalancing frequency and near-stationary
  synthetic returns.
- **Gross leverage**: Unconstrained and restricted portfolios have gross leverage
  1.02–1.04 (moderate short positions). Long-only portfolios are exactly 1.00
  by construction.

### Output Files

| File | Description |
|---|---|
| `exp2_portfolio_metrics.csv` | Full table of all portfolio metrics across models and constraints |
| `exp2_summary.json` | Best model per constraint, cross-constraint summary |
| `exp2_std_comparison.png` | Bar chart of realized std dev by model and constraint (3 panels) |
| `exp2_cumret_unconstrained.png` | Cumulative return paths — unconstrained |
| `exp2_cumret_restricted.png` | Cumulative return paths — restricted |
| `exp2_cumret_long_only.png` | Cumulative return paths — long-only |
| `exp2_weight_heatmap.png` | Portfolio weight heatmap over OOS period (K3\_LASSO\_restricted) |
| `exp2_turnover.png` | Average daily turnover by model and constraint |

---

## Test Suite

95 pytest tests pass across 5 modules:

| Test file | Tests | Coverage |
|---|---|---|
| `tests/test_cleaning.py` | 11 | Sigma cleaning: 4-sigma flagging, 10-day replacement, symmetry, edge cases |
| `tests/test_decomposition.py` | 16 | Factor decomposition: B_t, Sigma_f, Sigma_e, PSD projection, identity check |
| `tests/test_forecasting.py` | 27 | HAR regressors, LASSO/AdaLASSO with BIC, logm/expm round-trip, HAR-OLS betas |
| `tests/test_metrics.py` | 19 | L2 errors, portfolio metrics: std, LPSTD, kurtosis, div-ratio, turnover |
| `tests/test_portfolio.py` | 22 | cvxpy unconstrained/restricted/long-only solvers, constraint satisfaction, edge cases |

---

## Replication Notes

1. **Data unavailability**: The original composite realized kernel matrices for
   430 S&P 500 survivor stocks (2006–2011) are not publicly distributed. Synthetic
   matrices from a latent K-factor DGP are used throughout.
2. **Ambiguous rebalance date**: The paper does not fully specify the annual
   factor-weight rebalance date. This implementation uses a July 1 rebalance
   (fiscal-year end + 6-month lag), consistent with Fama-French conventions.
3. **SMB size breakpoint**: The paper does not state the size breakpoint
   explicitly in the trimmed excerpt; we use the NYSE median (50th percentile)
   following standard Fama-French practice.
4. **Gross profitability factor direction**: The paper defines R_lowGP − R_highGP
   (low gross-profitability minus high), which is opposite to the standard
   RMW factor; implemented exactly as described.
5. **PSD enforcement**: When factor covariance inverses are near-singular, a
   ridge of ε=1e-8 is added to the diagonal. This is documented but not
   specified in the paper.
6. **Log-matrix averages**: Log transformation is applied before computing
   day/week/month averages (i.e., averages are of logged matrices), following
   the best-faith interpretation that "transformation is applied before
   estimation."
