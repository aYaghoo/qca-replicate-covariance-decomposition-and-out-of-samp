# Covariance Decomposition and Out-of-Sample Forecasting

Replication of the factor-based realized covariance forecasting pipeline from
the paper *"Forecasting Large Realized Covariance Matrices Using Factor, Beta,
and Sector-Block Residual Models"*, extended with minimum-variance portfolio
evaluation.

## Structure

```
.
├── data/           # Data loading utilities and (optionally) realized cov matrices
├── src/            # Core implementation modules
│   ├── cleaning.py          # 4-sigma outlier detection and matrix replacement
│   ├── data_simulation.py   # Synthetic covariance matrix generator (DGP)
│   ├── decomposition.py     # Factor/beta/residual covariance decomposition
│   ├── factors.py           # Factor portfolio weight construction (1F–7F)
│   ├── forecasting.py       # HAR-LASSO/AdaLASSO factor cov & beta forecasting
│   ├── lasso_har.py         # LASSO/AdaLASSO with BIC lambda selection
│   ├── metrics.py           # L2 forecast error and portfolio performance metrics
│   ├── portfolio.py         # cvxpy min-variance solver (3 constraint regimes)
│   └── utils.py             # vech/unvech, PSD projection, symmetry enforcement
├── exp/            # Experiment scripts
│   ├── exp1_covariance_forecasting.py   # Full rolling HAR-LASSO pipeline
│   └── exp2_portfolio_evaluation.py     # Min-variance portfolio evaluation
├── tests/          # pytest test suite (95 tests)
├── results/        # Output metrics, plots, and RESULTS.md
└── AGENTS.md       # Repository skill / memory for OpenHands
```

## Methodology

### Experiment 1 — Covariance Forecasting

1. **Clean** realized covariance matrices using historical entry statistics
   (flag entries > 4σ; replace full matrix if > 25% entries flagged, using
   average of 10 nearest preceding clean matrices).
2. **Decompose** each daily Σₜ into:
   - Factor covariance: Σ_{f,t} = Wₜ' Σₜ Wₜ
   - Time-varying loadings: Bₜ = Σ_{f,t}⁻¹ Wₜ' Σₜ
   - Residual (sector-block): Σ_{e,t} = Σₜ − Bₜ' Σ_{f,t} Bₜ
3. **Forecast** each component via HAR/VHAR (1-day / 5-day / 22-day averages):
   - Factor covariance: equation-by-equation LASSO / adaptive LASSO with
     BIC-selected lambda; optionally in matrix-log space.
   - Beta loadings: univariate HAR-OLS per factor-stock pair.
   - Residual blocks: LASSO / adaptive LASSO using lagged sector variances.
4. **Recombine**: Σ̂_{t+1|t} = B̂_{t+1}' Σ̂_{f,t+1} B̂_{t+1} + Σ̂_{e,t+1}.
5. **Evaluate** on 128 OOS days using average ‖vech(Σ̂ − Σ)‖₂.

Factor specifications: K ∈ {1, 3, 5, 7}  
Estimation variants: LASSO, AdaLASSO, Log-LASSO, Log-AdaLASSO  
Rolling window: 100 days

### Experiment 2 — Portfolio Evaluation

Uses Exp 1 forecast matrices as inputs to daily minimum-variance optimization:

- **Unconstrained**: standard closed-form solution.
- **Restricted**: 30% max short-side leverage + 20% max |wᵢ|.
- **Long-only**: non-negative weights + 20% max wᵢ.

All solved with cvxpy. Metrics: realized σ, LPSTD, kurtosis, skewness,
diversification ratio, average weights, turnover.

## Quick Start

```bash
pip install -r requirements.txt

# Run covariance forecasting experiment
python exp/exp1_covariance_forecasting.py

# Run portfolio evaluation
python exp/exp2_portfolio_evaluation.py

# Run tests
pytest tests/ -v
```

## Data

The paper's original 430×430 daily realized covariance matrices (5-minute
composite realized kernels) are not publicly available. The synthetic data
generator in `src/data_simulation.py` creates structurally equivalent matrices
from a latent K-factor DGP for full-pipeline testing. To use real data, provide
a numpy array of shape `(T, N, N)` to the cleaning and decomposition modules.

Real stock data can be fetched via the Massive REST API:

```python
import os
from massive import RESTClient

client = RESTClient(api_key=os.getenv("MASSIVE_API_KEY"))
```

## Dependencies

See `requirements.txt`. Key packages:
- `numpy`, `scipy` — linear algebra, matrix log/exp
- `pandas` — indexed data handling
- `scikit-learn` — LASSO fitting
- `cvxpy` — constrained quadratic portfolio optimization
- `matplotlib`, `seaborn` — visualization
- `pytest` — testing

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for full experiment results,
tables, and interpretation.
