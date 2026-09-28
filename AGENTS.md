# Repository: Realized Covariance Matrix Forecasting Pipeline

## Project Structure

```
src/
  __init__.py          # empty package init
  utils.py             # vech, vech_to_matrix, nearest_psd, safe_logm, safe_expm, har_regressors
  cleaning.py          # clean_covariance_matrices (sigma-threshold based flagging + replacement)
  decomposition.py     # decompose_covariance, extract_sector_blocks, assemble_from_sector_blocks
  factors.py           # build_factor_weight_matrix (market, SMB/HML, GP, CMA, AG, Accruals)
  lasso_har.py         # lasso_bic, adaptive_lasso_bic, har_design_matrix, fit_har_lasso_equation
  data_simulation.py   # SimulatedMarketData (N stocks, K factors, S sectors, T days)
  forecasting.py       # rolling_forecast_pipeline + per-component forecasting functions
  portfolio.py         # min-variance solvers (unconstrained/restricted/long-only) + metrics
  metrics.py           # average_l2_forecast_error, random_walk_forecast, compile_results_table
```

## Key Design Patterns

### Factor Model Convention
- **B is (K, N)**: factor loadings where B[k, i] = loading of factor k on asset i
- **Sigma_t = B.T @ Sigma_f_t @ B + Sigma_e_t** (N×N = (N×K)(K×K)(K×N) + N×N)
- **W_t is (K, N)**: factor weight matrix (rows = factor portfolios)
- **Sigma_f_t = W_t @ Sigma_t @ W_t.T** (K×K)
- **B_t = inv(Sigma_f_t) @ W_t @ Sigma_t** (K×N, from OLS projection)

### LASSO with BIC
- `lasso_bic(X, y)` standardizes X internally and returns unstandardized (coef, intercept, alpha)
- `adaptive_lasso_bic` rescales X by 1/|initial_coef| then calls `lasso_bic`
- BIC = T * log(RSS/T) + n_nonzero * log(T)

### HAR Structure
- Design row: [1, x_{t-1}, mean(x_{t-5:t}), mean(x_{t-22:t})]
- Requires at least 22 lags → `t_first = train_start + 22`
- Applied equation-by-equation to vech of factor covariance matrices

### Rolling Window
- Default: `rolling_window = 1000`, first OOS at index `rolling_window + 22`
- `train_end = t_pred - 1`, `train_start = max(0, train_end - rolling_window - 22 + 1)`

### Sector Structure
- 10 sectors (hardcoded), sized proportionally to paper's 430-stock distribution
- `N >= 20` required for non-degenerate sector sizes; N=50 is the documented default

## Package Dependencies
- numpy, pandas, scipy — core numerics
- scikit-learn 1.5.1 — Lasso solver
- cvxpy 1.7.5 with CLARABEL solver — portfolio optimization

## Convergence Warnings
Sklearn `ConvergenceWarning` messages during LASSO fits on small datasets (T_reg < 100)
are expected and non-critical; duality gaps are sub-1e-9 and below machine precision.

## Testing
Run `python -c "from src.<module> import ..."` from `/workspace/project` for quick checks.
Full integration test requires N>=20, T>=100, rolling_window such that T > rolling_window+22.
