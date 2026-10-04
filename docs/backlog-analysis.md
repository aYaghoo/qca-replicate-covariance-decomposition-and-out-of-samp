# Lint and Type-Check Backlog: Analysis and Recommendations

Branch `add-harness-policy` at `6d508fb`. Analysis only. No existing file was modified.

## How this was produced

Tool output at the start (from the project `.venv`):

| Tool | Findings |
|---|---|
| `ruff check . --output-format concise` (0.16.10) | 141 findings across 33 rules. **0 safe fixes, 61 unsafe fixes** |
| `mypy src data exp tests` | 94 errors in 16 files |
| `pydoclint src data exp` | 46 violations (31 × DOC111) |
| `pytest` | 95 passed |

All experiments ran in a scratch copy at `/tmp/rcff-scratch/repo`. The instrumentation lives
next to it in `/tmp/rcff-scratch/`, and none of it touches repo source:

- `instrument.py` monkeypatches four things from outside: `src.lasso_har.Lasso` (counts fits,
  exceptions, and `ConvergenceWarning`s), the three `forecast_*` stages, and
  `cvxpy.Problem.solve` (records solver status). It runs the pipeline with `verbose=True` and
  counts the fallback messages.
- `run_exp1.py` and `run_exp2.py` reproduce the exact `CFG`, `METHODS`, and `PORTFOLIO_MODELS`
  from `exp/`, calling the experiment modules' own helpers.
- `probe*.py` hold the targeted exception and edge-case probes. `probe_loader.py` mocks
  `RESTClient`, so the live API was never called.
- `bias_demo.py` injects failures to measure how fallbacks bias the comparison.
- `fixrepo/` is a second scratch copy where candidate fixes were checked against mypy and pytest.

Installed versions: numpy 2.5.3, scikit-learn 1.9.1, cvxpy 1.9.3 (CLARABEL 0.11.1), pandas
3.0.6. `AGENTS.md` still names scikit-learn 1.5.1 and cvxpy 1.7.5. The lockfile pins the
newer versions.

---

## Summary

### Tier 1 items

| # | Item | Classification | Impact on current results | Confidence |
|---|---|---|---|---|
| 1a | `rolling_forecast_pipeline` falls back to random walk on any `Exception` | Missing check. Latent runtime-bug risk | **None measured**: 0 of 2,944 OOS steps fell back under the `exp/` configs. Easy inputs trigger 100% silent fallback, and `exp/` sets `verbose=False`, so nobody would see it | High (measured) |
| 1b | `min_variance_restricted` / `_long_only` fall back to equal weights | Missing check. Latent runtime-bug risk | **None measured**: all 2,048 solves returned `optimal`, 0 fallbacks. The fallback **violates the 20% cap** when N < 5. A missing CLARABEL turns every portfolio into equal weights | High (measured) |
| 1c | `lasso_bic` skips penalties whose fit raises | Missing check (input validation) | **None measured**: 0 exceptions in 644,352 `Lasso.fit` calls. Only non-finite input reaches the handler, and the result is then a silent mean-only or NaN forecast | High (measured) |
| 2 | `har_design_matrix` ignores `window` | Dead code (unused parameter in a function used only by tests) | None. Production uses other HAR code, and no partial monthly averages occur (min `t` = 22, measured) | High |
| 3a | `T_reg` unused in `forecast_betas` / `forecast_residual_blocks` | Leftover that marks a **missing guard** | None via the pipeline, because the factor stage raises first. Called directly, `forecast_betas` silently returns an all-zero B when `T_reg = 0` | High |
| 3b | `idx_2d` unused in `forecast_residual_blocks` | Dead code (recomputed a few lines later) | None | High |
| 3c | `Sigma_true_list` unused in `compile_results_table` | Dead code in a function nothing calls | None | High |
| 3d | `original_35` unused in `tests/test_cleaning.py` | Dead code. The test is weaker than intended | None on results. It is a test-coverage gap | High that the test is weak. Intent is inferred |
| 4a | `fit_har_lasso_equation` annotated as returning a tuple, returns an array | Wrong annotation | None. The function is unused in production | High |
| 4b | `exp1.run_experiment` and `exp2.run_all_portfolios` return values that contradict their annotations | Wrong annotation | None | High |
| 4c | `.dropna()` on an "array" in `data/loader.py` | Wrong (stub-inferred) type. The code works | None on current results, since nothing calls the loader. The live path is **untested** and has several real defects (see 4c) | High (mocked) |
| 4d | `None` passed for `returns: pd.DataFrame` in `src/factors.py` | Unused parameter (wrong signature). Related latent bug: K ∉ {1,3,5,7} is silently accepted | None for `exp/` (K ∈ {1,3,5,7}) | High |

### Other findings that affect results (not flagged by any tool)

I found these while verifying Tier 1. They matter more for the research than anything in
Tier 1.

| # | Finding | Impact | Confidence |
|---|---|---|---|
| A | **`results/` is stale.** It was generated in `bd70bdd` and the DGP was replaced in `52035fa`. Re-running Exp 1 with the current code gives RW mean L2 = 8.78, with models at **0.83–0.90 × RW**. `RESULTS.md` reports RW = 0.0003 and models at **46–62 × worse** than RW | **High**: the headline conclusion in `RESULTS.md` is reversed | High (re-ran the same functions with the same `CFG`) |
| B | **Return units mismatch.** The simulator produces returns in percent (daily sd ≈ 1.8), but `portfolio.compute_portfolio_metrics` and `exp2` treat them as decimals. `std_ann * 100` reports 1,496 "%". Turnover uses `1 + r`. The cumulative-return plot uses `cumprod(1 + r)` and collapses to 1e-18 | **High** for turnover, the cumulative-return plots, and reported σ levels. Rankings by σ are unaffected because σ scales uniformly | High (measured) |
| C | **Two `compute_portfolio_metrics` with different LPSTD definitions.** `portfolio.py` uses centered RMS of negative returns. `metrics.py` uses ddof=1 std of raw negative returns. Exp 2 uses the `portfolio.py` one. The values differ (0.965 vs 0.607 on the same returns). `test_lower_partial_std_uses_centered` is vacuous because its data has mean 0 | Medium. LPSTD as reported depends on which copy is called | High |
| D | `exp1.run_all_models` also catches `Exception` (BLE001) and **drops failed models from the table** without a trace beyond one log line | Medium (selection bias if it fires). Did not fire | High |
| E | `clean_covariance_matrices` has no minimum history. Day 2 is flagged in the `exp/` config because its "historical std" comes from 2 samples | Low. Day 2 is inside the first training windows | High |

---

## Tier 1: detailed findings

### 1a. Random-walk fallback in `rolling_forecast_pipeline`

**Location.** `src/forecasting.py`, `rolling_forecast_pipeline`, the `try`/`except Exception`
around stages 2a–2d (about L434–469).

**What the tool reports.** `BLE001` (blind `except Exception`) and `T201` (the `print` in the
handler). `PLR0913`/`PLR0917`/`PLR0915` on the function.

**What is actually happening.**

```python
        except Exception as exc:
            if verbose:
                print(f"  warning at step {step}: {exc} — using random walk")
            Sigma_hat = Sigma_list[t_pred - 1].copy()
            Sigma_f_hat = Sigma_f_arr[t_pred - 1].copy()
            ...
```

Both `exp/` scripts call the pipeline with `verbose=False`, so a fallback leaves no trace at
all. The returned dict has no fallback count.

*Does it fire under the `exp/` configurations?* Measured with `run_exp1.py` and `run_exp2.py`:

| Configuration | Model runs × OOS steps | Fallback steps | `Lasso.fit` exceptions | `ConvergenceWarning`s |
|---|---|---|---|---|
| Exp 1: N=20, T=250, window=100, n_alphas=3, K ∈ {1,3,5,7} × 4 methods | 16 × 128 = 2,048 | **0** | 0 / 465,408 | 14,719 (K=7 only) |
| Exp 2: same `CFG`, 7 `PORTFOLIO_MODELS` | 7 × 128 = 896 | **0** | 0 / 178,944 | 7,493 |

*Which exceptions can reach the handler?* Probed in `probe.py` (N=20, K=3, T=170,
window=40, 108 OOS steps):

| Input | Exception reaching the handler | Fallback steps |
|---|---|---|
| Baseline | none | 0 / 108 |
| `sector_indices` as Python lists instead of arrays | `TypeError: list indices must be integers or slices, not tuple`, from `sector_idx[:, None]` in `forecast_residual_blocks` | **108 / 108** |
| `rolling_window=0` | `ValueError: Insufficient data for HAR regression: T_reg=0` | **148 / 148** |
| One NaN (or inf) entry in `Sigma_list[100]` | `LinAlgError: Eigenvalues did not converge`, from `nearest_psd` | **69 / 108**: every step from the first window containing day 100 to the end of the sample |
| A zero row in `W`, log and non-log | none | 0 |
| Σ scaled by 1e6, log | none | 0 |

Two of these deserve comment:

- **The NaN case contaminates permanently.** `_HARCache` stores prefix sums
  (`np.cumsum(vech_mat)`), so a single NaN day poisons every later rolling mean. The damage is
  not limited to windows that contain the bad day.
- **The handler also hides programming errors.** The list-vs-array `TypeError` is a caller
  bug, yet it turns the model into the benchmark. The model's L2 then equals RW's exactly, so
  `rel_to_rw` = 1.000. Real-data callers are the likely ones to pass lists, because
  `load_or_simulate` returns `None` for `sector_indices` and the caller has to build them.

Exceptions that can reach the handler in principle:

- `LinAlgError` from `eigh` (non-finite input), `logm`, or `expm`
- `ValueError` from the `T_reg` guard
- `TypeError` or `IndexError` from bad argument types or shapes

Lasso input-validation errors never reach it, because `lasso_bic` swallows them (see 1c).
`KeyboardInterrupt` is not caught, since it is a `BaseException`.

*How a fallback biases the comparison.* On a fallback step the model's forecast **is** the RW
forecast, so the step's L2 error equals RW's exactly. Mean L2 is linear in per-step errors, so
`rel_to_rw` moves toward 1 in proportion to the fallback rate. Measured in `bias_demo.py`
(K=1 LASSO, exp1 config, failures injected into `forecast_betas`):

| Injected failure rate | `rel_to_rw` |
|---|---|
| 0% | 0.8464 |
| 10% | 0.8536 |
| 25% | 0.8736 |
| 50% | 0.9144 |

The direction depends on the model:

- A model that beats RW looks worse.
- A model that loses to RW looks better.
- Paired per-step differences on fallback steps are exactly zero, which understates the
  variance in any Diebold–Mariano-style test.
- `l2_factor_errors` is affected the same way.

Exp 2 inherits this: a fallback step's portfolio is built from Σ_{t-1}.

**Classification and impact.** A missing check, and a latent risk of hidden runtime bugs. It
has **no effect on current numbers**, because it never fired. The risk is high, because
plausible inputs make the model silently identical to the benchmark.

**Recommended fix.** Validate inputs up front, catch narrowly, make the fallback policy
explicit with `"raise"` as the research default, and record every event. A sketch (the config
and result types are defined in Tier 2):

```python
class ForecastFailure(RuntimeError):
    """A forecast stage failed and the policy is to raise."""


def _validate_inputs(Sigma_arr: np.ndarray, sector_indices: list[np.ndarray], window: int) -> None:
    if window < 1:
        raise ValueError(f"rolling_window must be >= 1, got {window}")
    if not np.isfinite(Sigma_arr).all():
        bad = np.unique(np.argwhere(~np.isfinite(Sigma_arr))[:, 0])
        raise ValueError(f"non-finite covariance entries on days {bad.tolist()}")
    ...  # sector_indices: coerce with np.asarray(idx, dtype=np.intp) and check it is a partition of range(N)

...
        try:
            Sigma_f_hat = forecast_factor_covariance(...)
            ...
        except (np.linalg.LinAlgError, ValueError) as exc:
            if config.on_failure == "raise":
                raise ForecastFailure(f"forecast failed at t={t_pred}") from exc
            logger.warning("Forecast failed at t=%d (%s); using random walk", t_pred, exc)
            fallbacks.append(FallbackEvent(step=step, t_pred=t_pred, error=repr(exc)))
            ...
```

**Confidence.** High. Verified: the zero-fallback counts, each probe row, and the bias table.
Inferred: that real data would produce NaN days. That is typical of realized-kernel panels
with missing intraday data, but I could not check it here.

**Questions for the author.**

- Should a failed step abort the run, or be recorded and excluded from the comparison? Or
  recorded and included (the current behavior, minus the silence)?
- Should the `TypeError` class of failures (caller bugs) ever fall back?

---

### 1b. Equal-weight fallback in `min_variance_restricted` / `min_variance_long_only`

**Location.** `src/portfolio.py`, `min_variance_restricted` (about L77–85) and
`min_variance_long_only` (about L118–125).

**What the tool reports.** `BLE001` and `S110` (`try`/`except`/`pass`) on both. `ERA001` on the
comment `# Fallback: equal-weight`. That one is a false positive: Ruff parses the prose as an
annotation. mypy reports `no-any-return` on `w.value`.

**What is actually happening.**

```python
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            return w.value
    except Exception:
        pass
    return np.ones(N) / N
```

*Under the `exp/` configuration* (`run_exp2.py`, RW plus 7 models × 128 days × 2 cvxpy
regimes), all **2,048** solves returned `optimal`, and **0** returned equal weights. The
unconstrained solver's own `LinAlgError` fallback also returned equal weights 0 of 1,024
times.

*What can reach the handler* (`probe.py`, `probe2.py`):

| Condition | Outcome |
|---|---|
| CLARABEL not installed (simulated by pointing `cp.CLARABEL` at a bogus name) | `cvxpy.error.SolverError` is caught, and **every** portfolio becomes equal-weight. An environment problem silently becomes a result |
| N = 4 with `max_weight = 0.2` (infeasible, since N·0.2 < 1) | status `infeasible`, so equal weights of **0.25, which violates the 20% cap** the regime is supposed to enforce |
| Σ scaled by ≥ 1e8 (restricted) | `SolverError: Solver 'CLARABEL' failed`, so equal weights |
| Σ scaled by 1e12 (long-only) | CLARABEL reports a **feasible** problem as `infeasible`, so equal weights |
| Σ scaled by 1e-8 … 1e6 | `optimal` |
| NaN in Σ | **Not** handled. `nearest_psd` runs *before* the `try` and raises `LinAlgError`. The unconstrained solver returns NaN weights without raising |

At the simulator's scale (variances O(1), percent units) and at decimal-return scale (about
1e-4), the solver is well inside its working range.

*Bias.* A fallback replaces the model portfolio with 1/N. On a day it fires for the model
but not for RW, or the reverse, the comparison is between a model and 1/N. Equal-weight
portfolios have higher σ than min-variance ones, so fallbacks inflate the affected model's σ.
They also distort turnover (a jump to and from 1/N), the diversification ratio, and the weight
statistics. Accepting `optimal_inaccurate` also lets slightly infeasible weights through
without a flag.

**Classification and impact.** A missing check. **No effect on current numbers**, but there
are two real defects:

- the fallback violates the constraint set (N < 5);
- an environment failure is indistinguishable from a result.

**Recommended fix.**

```python
@dataclass(frozen=True)
class PortfolioSolution:
    weights: np.ndarray
    status: str          # cvxpy status, or "solver_error"
    fallback: bool


def min_variance_long_only(Sigma: np.ndarray, max_weight: float = 0.20, eps: float = 1e-8) -> PortfolioSolution:
    N = Sigma.shape[0]
    if N * max_weight < 1:
        raise ValueError(f"infeasible: N * max_weight = {N * max_weight:.2f} < 1")
    ...
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except cp.error.SolverError as exc:
        logger.warning("CLARABEL failed (%s); equal weights", exc)
        return PortfolioSolution(np.full(N, 1.0 / N), "solver_error", fallback=True)
    if prob.status != cp.OPTIMAL or w.value is None:
        logger.warning("solver status %s; equal weights", prob.status)
        return PortfolioSolution(np.full(N, 1.0 / N), prob.status, fallback=True)
    return PortfolioSolution(np.asarray(w.value, dtype=float), prob.status, fallback=False)
```

Separately, check `cp.CLARABEL in cp.installed_solvers()` once at experiment start and fail
fast if it is missing. `run_portfolio_experiment` should then report `n_fallback` and a count
per status.

**Confidence.** High (measured).

**Questions for the author.**

- Is `optimal_inaccurate` acceptable, or should it count as a fallback?
- On a solver failure, is 1/N the right substitute? Alternatives: the previous day's weights,
  which is a cleaner "no trade" counterfactual, or excluding the day from the metrics.

---

### 1c. `lasso_bic` skips penalty values whose fit raises

**Location.** `src/lasso_har.py`, `lasso_bic`, the loop over `alphas` (about L83–91).

**What the tool reports.** `BLE001` and `S112` (`try`/`except`/`continue`).

**What is actually happening.** `sklearn.linear_model.Lasso.fit` raises only on invalid input:
`ValueError` for NaN or inf, or for zero samples. A `ConvergenceWarning` is a warning, not an
exception, and is unaffected. Measured:

- **0 exceptions in 644,352 fits** across both `exp/` configurations.
- If every alpha fails, the function silently returns its initial values: `coef = 0`,
  `intercept = y_mean`, and `best_alpha` (`probe.py`):

| Input | Returned |
|---|---|
| clean | `coef=[0.976, 0, 0, 2.002]`, a sensible fit |
| NaN in `y` | `coef=0`, `intercept=nan`, `alpha=nan` |
| inf or NaN in `X` | `coef=0`, `intercept=0.204` (the sample mean), `alpha=nan`. That is **an unconditional-mean forecast presented as a fitted model** |
| `n_alphas=0` | `IndexError` (escapes) |

In the pipeline, a NaN in `y` becomes a NaN forecast. That makes `nearest_psd` raise, and 1a
turns the result into a random walk. An inf in `X` with a finite `y` would give a mean-only
forecast with no signal at all.

**Classification and impact.** A missing check (input validation). **No effect on current
results.** The handler only ever converts bad input into a plausible-looking wrong answer.

**Recommended fix.** Validate once and drop the `try`.

```python
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("lasso_bic: X and y must be finite")
    if n_alphas < 1:
        raise ValueError("n_alphas must be >= 1")
    ...
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000, tol=1e-4)
        lasso.fit(X_s, y_c)
        ...
```

A more useful diagnostic than the `except` would be to flag when BIC selects an alpha at the
edge of the grid. With `n_alphas=3` in both `exp/` configs, the grid is {λmax, 10⁻²λmax,
10⁻⁴λmax}.

**Confidence.** High (measured).

**Questions for the author.** Is `n_alphas=3` deliberate (for speed), or a leftover? The
library default is 20 or 30, and BIC selection over 3 points is coarse.

---

### 2. `har_design_matrix` ignores `window`

**Location.** `src/lasso_har.py`, `har_design_matrix`.

**What the tool reports.** `ARG001` (unused `window`), `RET504`, and pydoclint `DOC111`.

**What is actually happening.**

```python
def har_design_matrix(series_2d: np.ndarray, t: int, window: int) -> np.ndarray:
    ...
    window: rolling estimation window size
    ...
    day_lag = series_2d[t - 1, :]
    week_lag = series_2d[max(0, t - 5) : t, :].mean(axis=0)
    month_lag = series_2d[max(0, t - 22) : t, :].mean(axis=0)
```

- The output is identical for `window=1` and `window=1000` (verified).
- **Call sites:** only `tests/test_forecasting.py`: `test_har_design_matrix_returns_row`
  (t=25), `_intercept_is_one` (t=25), and `_day_lag` (t=30). All three pass `window=22`.
  Production code never calls it.
- The repo has **five** HAR-row implementations:
  - `utils.har_regressors`: unused. It has parameterized lags, but with `t < 22` its negative
    slice start produces a wrong window.
  - `lasso_har.har_design_matrix`: tests only.
  - `forecasting._HARCache.har_row` / `rolling_mean`: production (factor covariance).
  - `forecasting._build_har_matrix_1d`: production (betas).
  - `forecast_betas._pred_row`: production (beta prediction row).
- On the same data, `har_design_matrix(series, t, …)` equals `_HARCache.har_row(t)` for
  t ∈ {22, 50, 200} (verified), so the copies agree today.

*Partial monthly averages.* `har_design_matrix` produces partial averages for `t < 22`:
- `t=1` averages 1 observation, `t=5` averages 5, and `t=21` averages 21.
- `t=0` reads `series[-1]` (the last row) and averages an empty slice, giving NaN.

But the production equivalents never see `t < 22`. Instrumenting the pipeline in the exp
config found:

- min `t` passed to `_HARCache.rolling_mean` = **22**, with 0 partial windows;
- min `t_first` for the beta HAR = **22**.

This follows from the indexing: `t_first = train_start + 22`, `train_start ≥ 0`, and the
prediction row is at `train_end + 1 > t_first`.

*What was `window` for?* The docstring says "rolling estimation window size". That is the
`rolling_window` concept, which has no role in building a single row. The other plausible
intent is the monthly horizon, since the tests pass 22. The code cannot settle which.

**Classification and impact.** Dead code: an unused parameter in a function used only by tests.
**No effect on results.**

**Recommended fix.** Delete `har_design_matrix` and `utils.har_regressors`, and point the
three tests at `_HARCache.har_row`, which is what production uses. If the function is kept
as a public helper, make the lags explicit and reject short histories:

```python
def har_design_matrix(series_2d: np.ndarray, t: int, lags: tuple[int, int, int] = (1, 5, 22)) -> np.ndarray:
    day, week, month = lags
    if t < month:
        raise ValueError(f"t={t} < monthly lag {month}: partial averages")
    ...
```

Then add a test that it equals `_HARCache.har_row(t)`.

**Confidence.** High on the behavior and call sites. The intent of `window` is unknown.

**Questions for the author.** Was `window` meant to be the monthly horizon, or a leftover
from a version that built the whole design matrix for a window? Is `har_design_matrix` part of
any public API that needs to stay?

---

### 3a. `T_reg` unused in `forecast_betas` and `forecast_residual_blocks`

**Location.** `src/forecasting.py`, `forecast_betas` (`T_reg = t_last - t_first + 1`) and
`forecast_residual_blocks` (same expression).

**What the tool reports.** `F841`.

**What is actually happening.** `forecast_factor_covariance` computes the same quantity and
guards on it:

```python
    T_reg = t_last - t_first + 1
    if T_reg <= 0:
        raise ValueError(f"Insufficient data for HAR regression: T_reg={T_reg}")
```

The other two stages compute `T_reg` and drop the guard. It reads as a copy-paste in which
the check was lost. Behavior without it (`probe3d.py`):

| Call | Result |
|---|---|
| `forecast_betas(..., 0, 21)` (T_reg = 0) | **all-zero B̂, returned silently** |
| `forecast_betas(..., 0, 22)` (T_reg = 1, 4 parameters) | min-norm `lstsq` solution of an underdetermined system, returned silently |
| `forecast_residual_blocks(..., T_reg=0)` | raises a cryptic `ValueError: not enough values to unpack` inside `lasso_bic` |
| `forecast_residual_blocks(..., T_reg=1)` | silently fits on one observation |

Inside `rolling_forecast_pipeline`, the factor stage runs first and raises for `T_reg ≤ 0`, so
these paths are reachable only by calling the stages directly (as the tests do). Even then the
raise is converted into a random walk (1a). In the `exp/` configs, `T_reg` = `rolling_window`
= 100.

**Classification and impact.** A leftover that points to a **missing check**. No effect on
current results.

**Recommended fix.** One shared helper, used by all three stages, with a minimum that fits
each regression:

```python
def _regression_span(train_start: int, train_end: int, max_lag: int, min_obs: int) -> tuple[int, int]:
    t_first = train_start + max_lag
    n_obs = train_end - t_first + 1
    if n_obs < min_obs:
        raise ValueError(f"need >= {min_obs} regression rows, got {n_obs}")
    return t_first, train_end
```

Suggested arguments: `max_lag=22, min_obs=5` for betas (4 parameters) and `max_lag=1` for
residuals.

**Confidence.** High.

---

### 3b. `idx_2d` unused in `forecast_residual_blocks`

**Location.** `src/forecasting.py`, `forecast_residual_blocks`, loop body.

**What the tool reports.** `F841`.

**What is actually happening.** `idx_2d = np.ix_(sector_idx, sector_idx)` is computed and
then recomputed verbatim at the end of the loop:
`Sigma_e_hat[np.ix_(sector_idx, sector_idx)] = block_hat`. The block extraction uses
`sector_idx[:, None], sector_idx[None, :]`, which is equivalent fancy indexing. Nothing is
being checked.

**Classification and impact.** Dead code (a duplicate). None.

**Recommended fix.** Use `idx_2d` in both places, block extraction included
(`blocks = Sigma_e_series[:, *idx_2d]`), and delete the recomputation. Using `np.ix_` also
removes the list-vs-array `TypeError` from 1a.

**Confidence.** High.

---

### 3c. `Sigma_true_list` unused in `compile_results_table`

**Location.** `src/metrics.py`, `compile_results_table`.

**What the tool reports.** `ARG001`. mypy reports `type-arg` on the bare `dict`.

**What is actually happening.** The **function is never called**. `exp1` builds its own table in
`save_results_table`. The function takes a precomputed `rw_l2` and reads only
`res["l2_errors"]`. `Sigma_true_list` was probably meant either for computing `rw_l2`
internally, or for checking that every model's error series is aligned with the truth.
Neither happens. There is no alignment check anywhere today: `exp1` assumes every model has
`T - t_oos_start` errors, which holds by construction.

**Classification and impact.** Dead code. It would be evidence of a missing alignment check
if the function were used. No effect on results.

**Recommended fix.** Delete `compile_results_table`. Otherwise, make it the single source of
the results table and use the argument:

```python
def compile_results_table(
    results: dict[str, ForecastResult], Sigma_true_list: list[np.ndarray], rw_forecasts: list[np.ndarray]
) -> dict[str, dict[str, float]]:
    rw_l2 = average_l2_forecast_error(rw_forecasts, Sigma_true_list)
    for name, res in results.items():
        if len(res.l2_errors) != len(Sigma_true_list):
            raise ValueError(f"{name}: {len(res.l2_errors)} errors vs {len(Sigma_true_list)} targets")
    ...
```

**Confidence.** High.

**Questions for the author.** Should Exp 1's table come from `src/metrics.py` (one definition)
or stay in `exp1`?

---

### 3d. `original_35` (and `rng`) unused in `tests/test_cleaning.py`

**Location.** `test_flagged_matrix_is_replaced` (`original_35`) and
`test_outlier_matrix_is_flagged` (`rng`).

**What the tool reports.** `F841` ×2.

**What the test verifies now.**

```python
    original_35 = Sigma_list[35].copy()
    extreme = make_random_psd(n, 99) * 1e6
    Sigma_list[35] = extreme
    cleaned, flags = clean_covariance_matrices(...)
    if flags[35]:
        assert not np.allclose(cleaned[35], extreme), "Flagged matrix should be replaced"
```

It checks only that the output is not the outlier, and only *if* the outlier was flagged. The
`if` makes the test vacuous should flagging stop working. With this data `flags[35]` is
`True` (along with days 2 and 3, which come from finding E). So the assertion runs today, but
it would also pass if the cleaner replaced the matrix with zeros, the identity, or anything
else.

**What it appears to have been meant to verify.** Saving the pre-corruption matrix only makes
sense for comparing it with the replacement. The likely intent: "the replacement is restored
to the normal scale, close to the original and not to the outlier." The docstring says
"replaced, not kept as-is". The sibling test, `test_replacement_is_average_of_preceding`,
promises "average of preceding" in its docstring, but it only checks that the output is not
the outlier. I verified the replacement *is* the mean of the 10 preceding unflagged matrices
in both tests, so a stronger assertion would pass today. `rng` is a leftover; the test draws
nothing from it.

**Classification and impact.** Dead code, and a test weaker than intended. No effect on
results. It is a coverage gap for the paper's cleaning rule.

**Recommended fix.**

```python
def test_flagged_matrix_is_replaced():
    n = 4
    Sigma_list = make_sigma_list(T=40, n=n)
    original_35 = Sigma_list[35].copy()
    extreme = make_random_psd(n, 99) * 1e6
    Sigma_list[35] = extreme
    cleaned, flags = clean_covariance_matrices(Sigma_list, sigma_threshold=4.0, flag_fraction=0.25)
    assert flags[35]
    preceding = [i for i in range(35) if not flags[i]][-10:]
    np.testing.assert_allclose(cleaned[35], np.mean([cleaned[i] for i in preceding], axis=0))
    assert np.linalg.norm(cleaned[35] - original_35) < np.linalg.norm(cleaned[35] - extreme)
```

Apply the same unconditional `assert flags[20]` and average check to
`test_replacement_is_average_of_preceding`, and delete `rng`.

**Confidence.** High on what the test does. Its intent is inferred.

**Questions for the author.** Was `original_35` meant for a "closer to original than to
outlier" check, or for something else, such as checking that unflagged neighbours are
untouched?

---

### 4a. `fit_har_lasso_equation` annotation vs. return value

**Location.** `src/lasso_har.py`, `fit_har_lasso_equation`. Callers are in
`tests/test_forecasting.py`.

**What the tool reports.** mypy `return-value` at the `return`, plus `arg-type` at
`tests/test_forecasting.py:161` (`np.isfinite(coef)` on a "tuple").

**What is actually happening.**

```python
) -> tuple[np.ndarray, float]:
    """...
    Returns
    -------
    coef: (1 + 3*M,) coefficient vector [intercept, slopes]
    intercept: scalar (already embedded in coef[0])
    """
    ...
    return np.concatenate([[intercept], coef])
```

All three tests treat the result as an array: `len(coef) == p + 1`,
`np.isfinite(coef)`, and `coef[0] + Z[-1, 1:] @ coef[1:]`. **Production never calls it.**
`forecast_factor_covariance` inlines the same `lasso_bic` → optional `adaptive_lasso_bic`
sequence, then computes `intercept + x_pred @ coef`. That equals
`z_pred @ fit_har_lasso_equation(...)`, because `z_pred[0] = 1`.

**Classification and impact.** Wrong annotation: the code and every caller agree on an array.
None.

**Recommended fix.** Annotate `-> np.ndarray` and document a single return,
`coef`, shape (1 + 3M,), `[intercept, slopes...]`. Then have `forecast_factor_covariance`
call it, which removes the duplicated logic:
`pred_vech[i] = z_pred @ fit_har_lasso_equation(Z, Y_raw[:, i], use_adaptive, n_alphas)`.
In `fixrepo`, the annotation change alone clears both mypy errors and all 95 tests pass.

**Confidence.** High (verified).

---

### 4b. Return annotations in `exp/`

**Location and evidence.**

| Function | Annotation | Actually returns | Caller | Verdict |
|---|---|---|---|---|
| `exp2.run_all_portfolios` | `-> dict` | `(all_results, n_oos_max)` | `run_experiment` unpacks `portfolio_results, n_oos = run_all_portfolios(...)` | Annotation wrong |
| `exp2.get_forecast_results` | `-> dict[str, dict]` | `dict[object, dict]` per mypy, because `tag` comes from `dict(K=1, ..., tag="...")`, which is typed `dict[str, object]` | `run_all_portfolios` | The code is right. The untyped config is the cause (Tier 2) |
| `exp2.generate_data` | none (`ANN201`) | a 6-tuple | `run_experiment` unpacks 6 values | Missing annotation |
| `exp1.run_experiment` | `-> None` | `summary` dict | `__main__` ignores it | Annotation wrong (exp2's twin returns `dict`) |

**Classification and impact.** Wrong annotations. None.

**Recommended fix.**

- `run_all_portfolios -> tuple[dict[str, PortfolioRun], int]`
- `exp1.run_experiment -> dict[str, object]`, or better, an `ExperimentSummary` `TypedDict`
  shared with exp2.
- `generate_data -> SimulatedInputs`, a small frozen dataclass rather than a 6-tuple.

In `fixrepo`, the tuple and dict annotations clear their mypy errors.

**Confidence.** High.

---

### 4c. `data/loader.py`: `.dropna()` type error and the live-data path

**Location.** `data/loader.py`, `fetch_daily_returns` and `load_or_simulate`.

**What the tool reports.**

- mypy: `attr-defined` (`ndarray has no attribute "dropna"`), `no-any-return`, and a bare
  `tuple`.
- Ruff: `BLE001`, `PERF402`, `PLR0913`/`0917`, `E501`, and `ERA001` on the comment
  `# Log-returns`, which is a false positive: Ruff reads it as `Log - returns`.
- pydoclint: missing `Raises`, and grouped parameters.

**What is actually happening.** At runtime `np.log(DataFrame)` returns a `DataFrame` via
`__array_ufunc__` (verified: `type(...) = DataFrame`), so `.dropna()` works. The error comes
from the numpy and pandas stubs typing the ufunc result as `ndarray`. A stub-friendly
equivalent:

```python
    log_prices = prices_df.apply(np.log)
    returns = log_prices.diff().dropna()
```

It gives the same values (verified with `np.allclose`) and clears both mypy errors on that
function in `fixrepo`.

*Test coverage.* **None.** No test imports `data.loader`. Nothing in `exp/` calls it either.

*Would it crash with real data?* I exercised it with a mocked `RESTClient` that yields
`massive.rest.models.Agg` objects (`probe_loader.py`):

| Scenario | Result |
|---|---|
| 3 tickers, same dates | Works: a `DataFrame` (9 × 3) of log returns in **decimal** units |
| One ticker lists 5 days late (an IPO in the sample) | **The whole panel is truncated** to the late ticker's history (9 → 4 rows), because the final `.dropna()` drops any row with any NaN |
| One ticker delisted mid-sample | `ffill()` carries the last price forward, giving **6 fabricated zero returns** |
| One ticker's request raises (e.g. HTTP 429) | Logged at WARNING, then the **column is silently dropped**. Columns no longer match `tickers` positionally, despite the docstring promise "columns ordered as `tickers`" |
| A bar with `close=None` | **Crashes**: `TypeError: float() argument must be ... not 'NoneType'`. This is outside the `try` |
| `load_or_simulate(use_simulation=False, ...)` | Returns `(None, None, DataFrame, None, None, None, None)`. No caller can use it as a drop-in for the simulated tuple. It also cannot accept an API key argument, so it reads only `MASSIVE_TOKEN` |

Also:

- `README.md` tells users to set **`MASSIVE_API_KEY`**, but the loader reads
  **`MASSIVE_TOKEN`**.
- The loader returns decimal returns, the simulator returns percent (finding B), and the
  portfolio code assumes decimals.

**Classification and impact.** Wrong (stub-inferred) annotation for the mypy item. The live
path has several **runtime defects**: one crash, silent truncation, fabricated returns, and
silent column loss. None of this affects current results, because the path is unused. It
would corrupt any real-data run.

**Recommended fix.**

- Use the `apply(np.log).diff()` form above.
- Build the panel with `pd.DataFrame(close_prices).reindex(columns=tickers)`, then apply an
  explicit coverage policy (a `min_coverage` parameter, and a report of dropped or short
  tickers) rather than `ffill().dropna()`.
- Skip bars where `close is None`, with a warning.
- Keep the iteration inside the `try`. The paginator is lazy, which also bears on the
  `PERF402` fix in Tier 3.
- Unify the env var name.
- Add a mocked-client test that covers the scenarios in the table above.

**Confidence.** High. Mocked, not live: the `Agg` field names come from the installed
`massive` 2.8.0. Real API pagination and timestamp conventions were not exercised.

**Questions for the author.**

- What should happen with tickers that IPO or delist during the sample? Options: drop them,
  require full coverage, or allow NaN through to a pairwise realized-covariance estimator.
- Is `load_or_simulate`'s live branch meant to feed the pipeline, and if so, where do the
  realized covariances, sectors, and characteristics come from?

---

### 4d. `None` passed to `double_sort_smb_hml(returns: pd.DataFrame)`

**Location.** `src/factors.py`, the call in `build_factor_weight_matrix`. The definition is
`double_sort_smb_hml`.

**What the tool reports.** mypy `arg-type` at the call, and Ruff `ARG001` (`returns` unused) at
the definition.

**What is actually happening.** `double_sort_smb_hml` never reads `returns`. Value-weighted
portfolio *weights* need only market caps and BM, while *returns* would be needed only to form
factor return series, which this function does not do. The only call site passes `None`. The
annotation is not the problem: the parameter should not exist.

Related mypy errors in the same function:
- **18 `arg-type` / `union-attr` / `operator` errors** come from `.values` returning
  `ExtensionArray | ndarray`. Replacing `.values` with `.to_numpy()` clears them (verified in
  `fixrepo`, tests pass).
- One `union-attr` is on `bm_ratios.values` in the `K >= 5` branch. It is safe at runtime,
  because K ≥ 5 implies the K ≥ 3 branch already raised on `None`, but mypy cannot see that.
  Validate both optional inputs once at the top.

A **latent runtime bug** in the same function: K ∉ {1, 3, 5, 7} is not rejected.
- K = 2 yields a 1-row `W`, K = 4 a 3-row `W`, and K = 6 a 5-row `W` (verified).
- `rolling_forecast_pipeline(K=2)` with that 1-row `W` then **runs without error**. The
  (1,1) `Sigma_f` and (1,N) `B` broadcast into the pre-allocated (2,2) and (2,N) arrays.
- Every factor-covariance forecast becomes a 2×2 matrix of identical entries (observed
  `[[0.8053, 0.8053], [0.8053, 0.8053]]`).
- The `exp/` scripts use only K ∈ {1, 3, 5, 7}, so this is not triggered today.

**Classification and impact.** Unused parameter (wrong signature), plus a latent runtime bug
for unsupported K. None on current results.

**Recommended fix.**

```python
FACTOR_COUNTS = (1, 3, 5, 7)

def double_sort_smb_hml(market_caps: pd.Series, bm_ratios: pd.Series, size_breakpoint: float = 0.5, ...) -> ...

def build_factor_weight_matrix(K: int, market_caps: pd.Series, bm_ratios: pd.Series | None, accounting: pd.DataFrame | None) -> np.ndarray:
    if K not in FACTOR_COUNTS:
        raise ValueError(f"K must be one of {FACTOR_COUNTS}, got {K}")
    if K >= 3 and bm_ratios is None: ...
    if K >= 5 and accounting is None: ...
```

In the pipeline, derive `K = W_t.shape[0]` instead of taking it as a separate argument. That
removes the broadcast hazard entirely.

**Confidence.** High (verified).

---

## Tier 2: design proposals

### T2.1 Configuration objects

**The problem.**

- `PLR0913`/`PLR0917` fire on 7 functions:
  - `rolling_forecast_pipeline` (9 args, and `PLR0915` with 49 statements)
  - `forecast_factor_covariance` and `forecast_residual_blocks` (6 each)
  - `adaptive_lasso_bic` (6)
  - `SimulatedMarketData.__init__` (14)
  - `exp2.get_forecast_results` (6)
  - `load_or_simulate` (8)
- The `dict(...)` configs (`CFG`, `METHODS`, `PORTFOLIO_MODELS`, and the test `kwargs`) are
  typed `dict[str, object]`. That causes **14 `arg-type` errors** (exp1 2, exp2 4,
  tests 8), the `dict[object, …]` return in 4b, and most of the 15 `C408` findings.
- `CFG` is duplicated between exp1 and exp2, with a comment saying they "must match".

**Proposed types** in `src/config.py`, all frozen:

```python
@dataclass(frozen=True, slots=True)
class LassoConfig:
    n_alphas: int = 20
    alpha_min_ratio: float = 1e-4
    standardize: bool = True
    adaptive: bool = False


@dataclass(frozen=True, slots=True)
class HARLags:
    day: int = 1
    week: int = 5
    month: int = 22          # replaces the hard-coded 22 in five places


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    rolling_window: int = 1000
    use_log: bool = False
    lasso: LassoConfig = LassoConfig()
    lags: HARLags = HARLags()
    on_failure: Literal["raise", "random_walk"] = "raise"

    def __post_init__(self) -> None:
        if self.rolling_window < 1:
            raise ValueError("rolling_window must be >= 1")


@dataclass(frozen=True, slots=True)
class TrainWindow:
    start: int
    end: int             # inclusive

    @classmethod
    def for_origin(cls, t_pred: int, cfg: ForecastConfig) -> "TrainWindow": ...
```

In `exp/config.py`, shared by both experiments:

```python
@dataclass(frozen=True, slots=True)
class SimulationConfig:
    N: int = 20
    K_max: int = 7
    S: int = 10
    T: int = 250
    seed: int = 42


@dataclass(frozen=True, slots=True)
class ModelSpec:
    K: int
    use_log: bool
    adaptive: bool

    @property
    def tag(self) -> str: ...   # "K3_Log_AdaLASSO"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    sim: SimulationConfig = SimulationConfig()
    rolling_window: int = 100
    n_alphas: int = 3
    def forecast_config(self, spec: ModelSpec) -> ForecastConfig: ...
```

**Where each is constructed.**

- `ExperimentConfig()` is built once in each `exp/*.py` `run_experiment`, from the shared
  module.
- `ForecastConfig` is derived per `ModelSpec` via `forecast_config()`.
- Tests build `ForecastConfig(rolling_window=45, lasso=LassoConfig(n_alphas=3))` directly and
  use `dataclasses.replace(cfg, use_log=True)` for the log/no-log test.
- `dataclasses.asdict(cfg)` replaces the raw `CFG` in the JSON summaries.

**Signature changes.**

| Before | After |
|---|---|
| `rolling_forecast_pipeline(Sigma_list, W_t, sector_indices, K, rolling_window, use_log, use_adaptive, n_alphas, verbose) -> dict` | `rolling_forecast_pipeline(Sigma_list, W_t, sector_indices, config: ForecastConfig) -> ForecastResult`. K comes from `W_t.shape[0]` (fixes 4d). `verbose` is removed (T2.2). The body splits into `_decompose_all`, `_forecast_step`, and `_score_step` (fixes `PLR0915`) |
| `forecast_factor_covariance(Sigma_f_series, train_start, train_end, use_log, use_adaptive, n_alphas, _cache)` | `forecast_factor_covariance(Sigma_f_series, window: TrainWindow, config: ForecastConfig, *, cache: _HARCache \| None = None)` |
| `forecast_residual_blocks(Sigma_e_series, sector_indices, train_start, train_end, use_adaptive, n_alphas)` | `forecast_residual_blocks(Sigma_e_series, sector_indices, window: TrainWindow, lasso: LassoConfig)` |
| `forecast_betas(B_series, train_start, train_end)` | `forecast_betas(B_series, window: TrainWindow, lags: HARLags = HARLags())` |
| `lasso_bic(X, y, n_alphas, alpha_min_ratio, standardize)` / `adaptive_lasso_bic(X, y, initial_coef, n_alphas, alpha_min_ratio, standardize)` | `lasso_bic(X, y, config: LassoConfig = LassoConfig())` / `adaptive_lasso_bic(X, y, initial_coef, config: LassoConfig = LassoConfig())` |
| `SimulatedMarketData.__init__(N, K, S, T, seed, M, alignment, burn_in, har_coefs, ...)` (14) | `SimulatedMarketData(sim: SimulationConfig, dgp: DGPParams = DGPParams())`. `DGPParams` is a frozen dataclass holding the 9 DGP knobs |
| `exp2.get_forecast_results(Sigma_list, sector_indices, market_caps, bm_ratios, accounting, cfg)` | `get_forecast_results(inputs: SimulatedInputs, cfg: ExperimentConfig, specs: Sequence[ModelSpec])` |
| `load_or_simulate(use_simulation, N, K, T, seed, tickers, start_date, end_date) -> tuple` | Split in two: `simulate_inputs(sim: SimulationConfig) -> SimulatedInputs` and `fetch_daily_returns(tickers, period: DateRange, api_key=None)`. The live branch's 7-tuple of `None`s goes away |

A `ForecastResult` dataclass replaces the result `dict`. It fixes most of the 28 bare-`dict`
`type-arg` errors in `exp/`:

```python
@dataclass
class ForecastResult:
    Sigma_hat: list[np.ndarray]
    Sigma_f_hat: list[np.ndarray]
    B_hat: list[np.ndarray]
    Sigma_e_hat: list[np.ndarray]
    l2_errors: list[float]
    l2_factor_errors: list[float]
    t_oos_start: int
    K: int
    config: ForecastConfig
    fallbacks: list[FallbackEvent] = field(default_factory=list)

    @property
    def n_oos(self) -> int: ...
```

### T2.2 `print` → `logging` in library code

The only library `print`s are the 4 in `rolling_forecast_pipeline` (`T201`). The `exp/`
scripts and `data/loader.py` already use `logging`.

The proposal:

- Add `logger = logging.getLogger(__name__)` in `src/forecasting.py`, and in `src/portfolio.py`
  and `src/lasso_har.py` once they log fallbacks.
- Level mapping:
  - "Decomposing …" and "Running n forecasts" → `logger.info`
  - per-step progress (every 50 steps) → `logger.debug`
  - each fallback → `logger.warning`, with `exc_info=True` at DEBUG
- Use %-style arguments, not f-strings (the `G` rules).
- **Remove the `verbose` parameter.** Callers control verbosity through logging config. The
  `exp/` scripts already call `logging.basicConfig(level=INFO)`, and can silence progress with
  `logging.getLogger("src.forecasting").setLevel(logging.WARNING)`.
- The key behavior change: **fallback warnings are no longer suppressible by
  `verbose=False`.** That flag currently hides all fallbacks in `exp/`.
- If backward compatibility matters, keep `verbose` for one release, emit a
  `DeprecationWarning`, and map it to the logger level.
- **Do not** apply Ruff's unsafe `T201` fix. It replaces each `print` with `pass`, including
  the fallback warning, which makes the silent fallback completely silent (verified with
  `--diff`).

### T2.3 Recording and reporting fallbacks

1. **Event type.**
   `FallbackEvent(step: int, t_pred: int, stage: str, error: str)` for forecasts, and
   `PortfolioSolution.status` / `.fallback` for portfolios (1b).
2. **Policy.** `ForecastConfig.on_failure` defaults to `"raise"`. Research runs must opt in to
   `"random_walk"`. Catch only `np.linalg.LinAlgError` and `ValueError`. Programming errors
   (`TypeError`, `IndexError`) always propagate.
3. **Results.**
   - `ForecastResult.fallbacks` is stored.
   - The `exp1_l2_errors.csv` table gains `n_fallback`, `fallback_rate`, and
     `mean_l2_excl_fallback`. That last column is computed on steps where neither the model
     nor any compared model fell back, so the comparison stays paired.
   - `exp2_portfolio_metrics.csv` gains `n_fallback` and `n_inaccurate` for each
     model × constraint.
   - Both JSON summaries carry the counts.
4. **Gates.** `run_experiment` raises if any `fallback_rate` exceeds a configured threshold
   (default 0), unless that is explicitly allowed. `exp1.run_all_models` stops swallowing
   exceptions (finding D). A failed model is either a hard failure or appears in the table as
   a row marked `failed`, never a silently missing row.
5. **Lasso diagnostics.** Instead of counting skipped penalties (they cannot occur once input
   is validated), record how often BIC picks the first or last grid point. That shows whether
   `n_alphas` / `alpha_min_ratio` constrain the fit.

---

## Tier 3: mechanical findings

All 61 Ruff fixes are **unsafe-only**, so the pre-commit `ruff-check --fix` fixes nothing.
Run `--unsafe-fixes` **per rule** (`--select RULE`), never repo-wide. Two rules in the set
(`T201`, `F841`) produce harmful edits.

### Ruff

| Rule | Count | Files | Fix pattern | Auto-fix | Care needed |
|---|---|---|---|---|---|
| C408 `dict()` call | 15 | exp1 (6), exp2 (8), test_forecasting (1) | literal `{...}` | unsafe | Most go away when configs become dataclasses (T2.1). Do that instead |
| PLR2004 magic value | 13 | cleaning, factors (4: `2`, `3`, `5`, `7`), lasso_har (3), metrics (3), portfolio (2) | named module constants | no | `1e-10` plays **three different roles** in `lasso_har` (std floor, λmax floor, non-zero threshold). Give each its own name rather than one shared `EPS`. Factor counts become `FACTOR_COUNTS` (4d) |
| PD011 `.values` | 10 | factors (9), exp2 (1) | `.to_numpy()` | no | Also clears 18 mypy errors in `factors.py` (verified). Numerically identical for float Series |
| TID252 relative import | 10 | forecasting (8), metrics, portfolio | `from src.x import ...` | unsafe | Safe here: tests and `exp/` already import `src.*` |
| D100 / D104 missing docstring | 9 / 3 | all `src` modules; `src`, `data`, `exp` `__init__.py` | one-line module docstrings | no | |
| PLR0913 / PLR0917 | 7 / 7 | see T2.1 | config objects | no | Tier 2 |
| RET504 unnecessary assign | 7 | lasso_har (2), factors (2), metrics, utils, exp2 | return the expression | unsafe | Harmless |
| B905 `zip` without `strict` | 7 | decomposition, metrics, exp1, exp2, tests (3) | add `strict=` | unsafe (inserts `strict=False`) | Choose **`strict=True`** deliberately where lengths must match: `assemble_from_sector_blocks`, `average_l2_forecast_error`, `test_unflagged_matrices_unchanged`. The auto-fix only preserves today's silent truncation |
| BLE001 / S110 / S112 | 6 / 2 / 1 | forecasting, portfolio (2), lasso_har, exp1, loader | narrow the exception and record it | no | Tier 1 (1a–1c, D, 4c) |
| F841 unused variable | 6 | forecasting (3), portfolio (`N`), test_cleaning (2) | delete, or use (3b) | unsafe | **The unsafe fix leaves the right-hand side as a bare expression statement** (e.g. `t_last - t_first + 1`, `Sigma_list[35].copy()`). Fix by hand after deciding on 3a/3b/3d |
| ANN001 / ANN201 / ANN204 | 4 / 1 / 1 | exp2 (`get_forecast_results`, `generate_data`), data_simulation (`_rank_score(x)`, `__init__`) | add annotations | ANN204 unsafe | `generate_data` → a `SimulatedInputs` dataclass |
| T201 `print` | 4 | forecasting | logging | unsafe | **Do not auto-fix**: it rewrites prints as `pass` (T2.2) |
| ARG001 unused argument | 4 | exp1 `save_results_table(cfg)`, factors `returns`, lasso_har `window`, metrics `Sigma_true_list` | remove or use | no | Tier 1 items 2, 3c, 4d. For `cfg`, either drop it or write the config into the CSV |
| PTH123 `open()` | 2 | exp1, exp2 | `path.open("w")` | no | |
| ERA001 commented-out code | 2 | loader (`# Log-returns`), portfolio (`# Fallback: equal-weight`) | reword the comment | no | **False positives**: prose parsed as code. Rephrase (e.g. "Compute log returns"); no suppression |
| SIM108 ternary | 2 | lasso_har, metrics | ternary | unsafe | Harmless |
| N802 function name | 2 | tests (`make_W`, `build_B_Se_arrays`) | rename (e.g. `make_weight_matrix`) | no | Only N803/N806 are ignored for math notation, and the config is settled, so rename |
| E501 line too long | 2 | cleaning, loader (docstrings) | wrap | no | |
| D400 / D401 | 2 / 2 | exp1 and exp2 module docstrings; "Main experiment runner…" | add period; "Run Experiment 1." | D400 unsafe | |
| ICN001 | 2 | exp1, exp2 | `import matplotlib as mpl` | unsafe | Also rewrites `matplotlib.use` → `mpl.use`. Fine |
| RUF015 | 2 | exp1, exp2 | `next(iter(d.values()))` | unsafe | |
| RUF059 unused unpacked | 2 | decomposition (`N`), test_forecasting (`intercept`) | `K = W_t.shape[0]`; `_` | unsafe (renames to `_N`) | Prefer the explicit form |
| RUF005 | 1 | exp1 | `["RandomWalk", *results]` | unsafe | |
| PERF402 | 1 | loader | `aggs = list(client.list_aggs(...))` | no | Keep it **inside the `try`**: the paginator is lazy and raises during iteration |
| PD010 `.pivot` | 1 | exp2 `plot_turnover` | `pivot_table` | no | `pivot_table` silently averages duplicate keys where `pivot` raises. Keys are unique today, so the output is the same. Add `assert not df.duplicated(["model", "constraint"]).any()` before it to keep the safety `pivot` gave |
| PLR0915 | 1 | `rolling_forecast_pipeline` | split | no | T2.1 |

### mypy (94)

| Code | Count | Where | Fix pattern |
|---|---|---|---|
| `type-arg` (bare `dict` ×28, `list` ×4, `tuple` ×1) | 33 | exp1 13, exp2 9, src 6, tests 4, loader 1 | Result dicts → `ForecastResult` / `PortfolioRun` dataclasses or `TypedDict` (T2.1). `list` → `list[np.ndarray]`. `tuple` → a dataclass |
| `arg-type` | 31 | factors (pandas `ExtensionArray`), configs typed `dict[str, object]` (14), forecasting `_roll(np.signedinteger)` ×2, factors `None` (4d), test_forecasting:161 (4a) | `.to_numpy()`; dataclass configs; `int(t)` or iterate over `range`; Tier 1 |
| `no-any-return` | 14 | decomposition, utils, factors (3), portfolio (4), lasso_har, forecasting (3), loader | Wrap untyped numpy/scipy/cvxpy results: `bool(...)`, `float(...)`, `np.asarray(x, dtype=np.float64)`. cvxpy is skipped, so `w.value` is `Any` |
| `no-untyped-def` | 6 | exp2 (2), data_simulation `_rank_score`, test helpers (3) | annotate. The three test helpers have annotated parameters but no return type. That trips `disallow_incomplete_defs`, which `--strict` keeps on even though `tests.*` sets `disallow_untyped_defs = false` |
| `union-attr` / `operator` | 4 / 1 | factors | `.to_numpy()`; narrow `bm_ratios` once at the top |
| `return-value` / `attr-defined` | 4 / 1 | Tier 1 (4a, 4b, 4c) | |

### pydoclint (46)

| Code | Count | Fix pattern |
|---|---|---|
| DOC111 types in docstring | 31 | Every documented function uses `name: (shape) description` or `name : description`. Convert each to NumPy format: parameter name **alone** on its line, description indented below, with shapes kept in the prose |
| DOC501 / DOC503 `Raises` | 5 / 6 | Add `Raises` to `build_factor_weight_matrix`, `forecast_factor_covariance`, `run_portfolio_experiment`, `fetch_daily_returns` (`OSError`, `RuntimeError`), and `load_or_simulate`. Reformat `average_l2_forecast_error`'s Google-style `ValueError: if ...` |
| DOC101 / DOC103 | 2 / 2 | `investment_factor_weights` and `load_or_simulate` group parameters (`size_char, bm_char, inv_char:`). **pydoclint rejects grouped NumPy parameters** (verified), so list each separately |

The verified accepted form under this config:

```python
    """Decompose a realized covariance matrix.

    Parameters
    ----------
    Sigma_t
        Realized covariance matrix, shape (N, N).
    W_t
        Factor weight matrix, shape (K, N).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Factor covariance (K, K), loadings (K, N), residual covariance (N, N).

    Raises
    ------
    ValueError
        If ``W_t`` and ``Sigma_t`` have incompatible shapes.
    """
```

Raises sections should follow the Tier 1 and Tier 2 changes, since those add new `raise`
statements.

---

## Recommended order of work

1. **Decide on results policy first** (questions Q1–Q3). Findings A and B mean the published
   numbers are wrong today, regardless of lint.
2. **Units (B), then regenerate `results/` (A).** Pick percent or decimal end to end, fix
   `compute_portfolio_metrics`, `exp2`'s `* 100`, and the cumulative-return plot, then re-run
   both experiments and rewrite `RESULTS.md`.
3. **Fallbacks (1a, 1b, 1c, D) together with T2.3 and T2.2.** Add input validation, narrow the
   exceptions, add the `on_failure` policy, record events, add the result columns, and switch
   to logging. Do this before step 4, because it touches the same signatures.
4. **Config objects (T2.1).** This clears most `PLR0913`/`PLR0917`, the 14 config `arg-type`
   errors, most `type-arg` errors, and `C408`. Derive K from `W_t` (fixes 4d's broadcast bug).
5. **Small Tier 1 fixes:**
   - the `_regression_span` guard (3a) and the `idx_2d` reuse (3b);
   - delete or merge the dead helpers (2, 3c, `utils.har_regressors`, the duplicate
     `compute_portfolio_metrics` from C);
   - the `fit_har_lasso_equation` annotation and its reuse (4a), the `exp/` annotations (4b),
     and the `factors` signature (4d);
   - strengthen the cleaning tests (3d) and the vacuous LPSTD test (C).
6. **Loader (4c)** if real data is planned: the coverage policy, the `None` guard, the env
   var, and a mocked-client test.
7. **Tier 3 mechanics**, one rule at a time: `.to_numpy()` first (it clears 18 mypy errors),
   then the safe-in-practice unsafe fixes (`RET504`, `SIM108`, `RUF005/015/059`, `ICN001`,
   `TID252`, `D400`). Then `B905` with `strict=True` chosen by hand, the manual `PLR2004`
   constants, and the docstring conversion last.

---

## Questions needing the author's judgment

1. **Failure policy for forecasts (1a):** on a failed step, abort, exclude the step from both
   sides of the comparison, or substitute the random walk and report the rate?
2. **Failure policy for portfolios (1b):** on a solver failure, use equal weights, the
   previous day's weights, or exclude the day? Is `optimal_inaccurate` acceptable?
3. **Units (B):** should the pipeline work in percent (as the simulator does) or in decimals
   (as the portfolio code and the loader assume)?
4. **`n_alphas=3` (1c):** a deliberate speed setting, or a leftover? Should BIC selections at
   the grid boundary be reported?
5. **`har_design_matrix` (2):** was `window` meant to be the monthly horizon? Can the function
   (and `utils.har_regressors`) be deleted, or is it public API?
6. **`compile_results_table` (3c):** delete it, or make it the single source of Exp 1's table?
7. **`test_flagged_matrix_is_replaced` (3d):** was `original_35` meant for a "replacement is
   closer to the original than to the outlier" check?
8. **Live data (4c):** how should mid-sample IPOs and delistings be handled? Is the live branch
   of `load_or_simulate` meant to feed the pipeline, and where would realized covariances,
   sectors, and characteristics come from? Which env var name is canonical, `MASSIVE_TOKEN` or
   `MASSIVE_API_KEY`?
9. **LPSTD definition (C):** centered RMS of negative deviations (`portfolio.py`), or ddof=1 std
   of raw negative returns (`metrics.py`)? Which one matches the paper?
10. **Cleaning warm-up (E):** should there be a minimum history before a day can be flagged?
11. **`verbose` (T2.2):** remove it outright, or deprecate it for one release?
12. **Unsupported K (4d):** is anything beyond {1, 3, 5, 7} planned? If not, reject other
    values.
