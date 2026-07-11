"""
backtest.py
===========
Event-driven-ish (but vectorized for speed) backtest engine for a single
pairs-trading strategy, plus a walk-forward harness that re-fits pair
selection / hedge ratio on rolling windows to avoid look-ahead bias.

  - Trades execute at the NEXT bar's price after a signal is generated
    (no same-bar execution -> no look-ahead).
  - Transaction costs charged in bps on each leg's notional, on every
    change in position (entry, exit, AND rebalancing due to the hedge
    ratio drifting between rebalances).
  - Optional slippage modeled as additional bps cost proportional to
    recent realized volatility (wider spreads costs more to cross in
    volatile regimes).
  - Position sizing is dollar-neutral: at entry, $1 of capital is split
    so that the y-leg and beta*x-leg notionals are equal and offsetting.
  - Walk-forward split: hedge ratio / half-life / z-score parameters are
    always estimated on a trailing IN-SAMPLE window and applied only to
    the OUT-OF-SAMPLE window that follows
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    tc_bps: float = 5.0          # transaction cost, bps of notional per leg per trade
    slippage_bps: float = 2.0    # additional slippage, bps
    capital: float = 1_000_000.0
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_holding_days: int | None = 30


def run_backtest(
    y: pd.Series,
    x: pd.Series,
    beta: pd.Series,
    z: pd.Series,
    positions: pd.Series,
    cfg: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """
    Simulate PnL for a single pair given precomputed hedge ratio (beta_t),
    z-score (z_t) and discrete positions (pos_t in {-1,0,1}).

    All of y, x, beta, z, positions must share the same DatetimeIndex.

    Returns a DataFrame with per-day PnL, cumulative equity, turnover,
    and cost drag, indexed one bar later than the signal (signals at t
    are executed at t+1's return).
    """
    idx = y.index
    ret_y = y.pct_change().fillna(0.0)
    ret_x = x.pct_change().fillna(0.0)

    # execute at t+1: shift signal forward by one bar
    exec_pos = positions.shift(1).fillna(0.0)
    exec_beta = beta.shift(1).bfill()

    # dollar-neutral leg weights: w_y = +/-0.5 * capital, w_x = -sign * 0.5*beta*capital
    # normalized so gross exposure per leg pair sums to `capital`
    denom = (1 + exec_beta.abs())
    w_y = exec_pos * (1.0 / denom)
    w_x = -exec_pos * (exec_beta / denom)

    strat_ret = w_y * ret_y + w_x * ret_x

    # transaction costs: charged on turnover of each leg's weight
    turnover_y = w_y.diff().abs().fillna(w_y.abs())
    turnover_x = w_x.diff().abs().fillna(w_x.abs())
    total_bps = (cfg.tc_bps + cfg.slippage_bps) / 10_000.0
    costs = (turnover_y + turnover_x) * total_bps

    net_ret = strat_ret - costs
    equity = cfg.capital * (1 + net_ret).cumprod()

    out = pd.DataFrame({
        "position": exec_pos,
        "beta": exec_beta,
        "z": z,
        "gross_return": strat_ret,
        "cost_drag": costs,
        "net_return": net_ret,
        "equity": equity,
        "turnover": turnover_y + turnover_x,
    }, index=idx)
    return out


def performance_stats(bt: pd.DataFrame, periods_per_year: int = 252) -> dict:
    r = bt["net_return"]
    n = len(r)
    ann_ret = (1 + r).prod() ** (periods_per_year / n) - 1 if n > 0 else np.nan
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0]
    ann_downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino = ann_ret / ann_downside_vol if ann_downside_vol and ann_downside_vol > 0 else np.nan

    equity = bt["equity"]
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    total_cost_drag = bt["cost_drag"].sum()
    avg_turnover = bt["turnover"].mean()
    n_trades = (bt["position"].diff().fillna(0) != 0).sum()
    win_rate = (r[r != 0] > 0).mean() if (r != 0).any() else np.nan

    return {
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "total_cost_drag": total_cost_drag,
        "avg_daily_turnover": avg_turnover,
        "n_position_changes": int(n_trades),
        "win_rate": win_rate,
    }


def walk_forward_split(index: pd.DatetimeIndex, train_days: int, test_days: int, step_days: int | None = None):
    """
    Yields (train_slice, test_slice) index pairs for rolling walk-forward
    validation. step_days defaults to test_days (i.e. non-overlapping test
    windows that tile the full sample).
    """
    step_days = step_days or test_days
    n = len(index)
    start = 0
    while start + train_days + test_days <= n:
        train_idx = index[start: start + train_days]
        test_idx = index[start + train_days: start + train_days + test_days]
        yield train_idx, test_idx
        start += step_days
