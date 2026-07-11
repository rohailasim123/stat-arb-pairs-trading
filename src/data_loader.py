"""
Handles data acquisition for the stat-arb pipeline.

Two modes:
1. `fetch_prices()`     -> pulls real daily adjusted-close data via yfinance.
2. `synthetic_pair()`   -> generates a synthetic cointegrated pair (and a
                           universe of non-cointegrated "noise" assets) using
                           a controlled DGP. This lets the entire pipeline be
                           unit-tested and demoed deterministically without
                           depending on network access or a specific market
                           regime -- useful for CI and for reviewers who want
                           to reproduce results exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_prices(tickers: list[str], start: str, end: str | None = None,
                  interval: str = "1d") -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers via yfinance.

    Returns a DataFrame indexed by date, one column per ticker.
    Requires network access to Yahoo Finance (not available in sandboxed
    execution environments -- see README for how to run this locally).
    """
    import yfinance as yf

    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)
    raw = yf.download(tickers, start=start, end=end, interval=interval,
                       auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices = prices.dropna(how="all").ffill().dropna()
    return prices


@dataclass
class SyntheticConfig:
    n_days: int = 1500
    seed: int = 42
    coint_pairs: int = 3          # number of true cointegrated pairs to embed
    noise_assets: int = 12        # number of unrelated random-walk assets
    mean_reversion_speed: float = 0.03   # theta in OU process for the spread
    spread_vol: float = 0.5
    drift_vol: float = 0.015      # daily vol of the common stochastic trend


def synthetic_universe(cfg: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, dict]:
    """
    Generates a synthetic price universe containing:
      - `cfg.coint_pairs` pairs of series that are cointegrated by
        construction: Y_t = alpha + beta * X_t + spread_t, where spread_t
        follows a mean-reverting Ornstein-Uhlenbeck process, and X_t is a
        common stochastic trend (so each leg individually is non-stationary,
        i.e. I(1), while the spread is stationary, i.e. I(0)).
      - `cfg.noise_assets` independent geometric random walks with no
        cointegrating relationship to anything else (negative controls --
        a correct pipeline should reject these as candidate pairs).

    Returns
    -------
    prices : DataFrame of price levels, columns are tickers like 'A0','B0',...
    truth  : dict mapping pair name -> {'beta': ..., 'legs': (colX, colY)}
             ground truth used by tests/ to check recovery accuracy.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_days
    dates = pd.bdate_range("2015-01-02", periods=n)

    data = {}
    truth = {}

    for i in range(cfg.coint_pairs):
        beta_true = rng.uniform(0.5, 2.5)
        alpha_true = rng.uniform(-5, 5)

        # common stochastic trend (I(1))
        trend = np.cumsum(rng.normal(0, cfg.drift_vol, n))
        x = 50 + trend * 10 + rng.normal(0, 0.3, n).cumsum() * 0.1
        x = np.maximum(x, 1.0)  # keep positive

        # mean-reverting spread (OU process), stationary by construction
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = spread[t - 1] + cfg.mean_reversion_speed * (0 - spread[t - 1]) \
                        + rng.normal(0, cfg.spread_vol)

        y = alpha_true + beta_true * x + spread
        y = np.maximum(y, 1.0)

        colx, coly = f"A{i}", f"B{i}"
        data[colx] = x
        data[coly] = y
        truth[f"pair_{i}"] = {"beta": beta_true, "alpha": alpha_true, "legs": (colx, coly)}

    for j in range(cfg.noise_assets):
        walk = 50 + np.cumsum(rng.normal(0, 1.0, n))
        walk = np.maximum(walk, 1.0)
        data[f"N{j}"] = walk

    prices = pd.DataFrame(data, index=dates)
    return prices, truth


if __name__ == "__main__":
    prices, truth = synthetic_universe()
    print(prices.head())
    print("Ground truth pairs:", truth)
