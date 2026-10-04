"""Data loading utilities for the realized covariance forecasting pipeline.

Primary data source: Massive REST API (daily stock returns).
For the full paper replication, 5-minute realized covariance matrices
are required. Since these cannot be fetched directly from the Massive REST
API (which provides bar-level aggregates, not pre-computed realized kernels),
the pipeline falls back to simulated data or user-supplied matrices.

Environment variable: MASSIVE_TOKEN (API key)
"""

import logging
import os

import numpy as np
import pandas as pd
from massive import RESTClient
from src.data_simulation import SimulatedMarketData

logger = logging.getLogger(__name__)


def fetch_daily_returns(
    tickers: list[str],
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch daily adjusted closing returns for a list of tickers from the Massive API.

    Parameters
    ----------
    tickers : list of ticker symbols
    start_date : ISO date string, e.g. '2006-01-01'
    end_date   : ISO date string, e.g. '2011-12-31'
    api_key    : Massive API key; if None, reads MASSIVE_TOKEN env var

    Returns
    -------
    returns : DataFrame (T, N) of daily log-returns, indexed by date,
              columns ordered as `tickers`.  Missing values are forward-filled
              then dropped.
    """
    api_key = api_key or os.getenv("MASSIVE_TOKEN")
    if not api_key:
        raise OSError("No Massive API key found. Set MASSIVE_TOKEN environment variable.")


    client = RESTClient(api_key=api_key)
    close_prices: dict[str, pd.Series] = {}

    for ticker in tickers:
        aggs = []
        try:
            for a in client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date,
                limit=50_000,
                adjusted=True,
            ):
                aggs.append(a)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", ticker, exc)
            continue

        if not aggs:
            logger.warning("No data returned for %s", ticker)
            continue

        timestamps = []
        closes = []
        for a in aggs:
            ts = pd.to_datetime(a.timestamp, unit="ms", utc=True).normalize()
            timestamps.append(ts)
            closes.append(float(a.close))

        series = pd.Series(closes, index=timestamps, name=ticker)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        close_prices[ticker] = series

    if not close_prices:
        raise RuntimeError("No price data retrieved from Massive API.")

    prices_df = pd.DataFrame(close_prices)
    prices_df = prices_df.ffill().dropna(how="all")

    # Log-returns
    returns = np.log(prices_df / prices_df.shift(1)).dropna()
    logger.info(
        "Fetched returns: shape %s, date range %s to %s",
        returns.shape,
        returns.index[0],
        returns.index[-1],
    )
    return returns


def load_or_simulate(
    use_simulation: bool = True,
    N: int = 30,
    K: int = 7,
    T: int = 500,
    seed: int = 42,
    tickers: list[str] | None = None,
    start_date: str = "2006-01-01",
    end_date: str = "2011-12-31",
) -> tuple:
    """Load market data or fall back to simulation.

    Parameters
    ----------
    use_simulation : if True, use simulated data (default for reproducibility)
    N, K, T, seed  : simulation parameters
    tickers        : list of tickers for live data fetch
    start_date, end_date : date range for live data

    Returns
    -------
    tuple: (Sigma_list, factor_returns, stock_returns, sector_indices, market_caps, bm_ratios, accounting)
    """
    if use_simulation:

        sim = SimulatedMarketData(N=N, K=K, S=10, T=T, seed=seed)
        Sigma_list, factor_returns, stock_returns = sim.simulate_realized_covariances()
        market_caps = sim.generate_market_caps()
        bm_ratios = sim.generate_bm_ratios()
        accounting = sim.generate_accounting_data()
        sector_indices = sim.sector_indices
        logger.info(
            "Simulation: N=%d, K=%d, T=%d, sectors=%s",
            N,
            K,
            T,
            [len(s) for s in sector_indices],
        )
        return (
            Sigma_list,
            factor_returns,
            stock_returns,
            sector_indices,
            market_caps,
            bm_ratios,
            accounting,
        )

    # Live data path
    if tickers is None:
        raise ValueError("tickers must be provided when use_simulation=False.")
    returns_df = fetch_daily_returns(tickers, start_date, end_date)
    logger.info("Live data fetched: %s", returns_df.shape)
    # Return without pre-built Sigma_list (caller must supply realized covariance matrices)
    return None, None, returns_df, None, None, None, None
