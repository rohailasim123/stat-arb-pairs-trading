"""
signals.py
==========
Converts a spread z-score series into discrete trading signals / positions.

Rule (standard stat-arb bands, e.g. Gatev, Goetzmann & Rouwenhorst 2006 /
Chan 2013):
    z >  entry_z   -> spread is "too high": short the spread
                       (short y, long beta*x)
    z < -entry_z   -> spread is "too low": long the spread
                       (long y, short beta*x)
    |z| <  exit_z  -> flatten (mean reversion has occurred / signal decayed)
    |z| >  stop_z  -> stop-loss: flatten regardless of direction
                       (protects against structural breaks in the
                       cointegrating relationship, e.g. M&A, index removal)

Positions are held constant between entry and exit (no re-averaging), which
keeps turnover and transaction costs realistic and auditable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_positions(
    z: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    max_holding_days: int | None = 30,
) -> pd.Series:
    """
    Returns a position series in {-1, 0, +1} representing the spread
    position (not the individual leg weights -- those are derived in
    backtest.py using the hedge ratio at trade entry).

    +1 = long the spread (long y, short beta*x)
    -1 = short the spread (short y, long beta*x)
     0 = flat
    """
    pos = np.zeros(len(z))
    state = 0
    days_in_trade = 0
    z_vals = z.values

    for t in range(len(z_vals)):
        zt = z_vals[t]
        if np.isnan(zt):
            pos[t] = state
            continue

        if state == 0:
            if zt > entry_z:
                state = -1
                days_in_trade = 0
            elif zt < -entry_z:
                state = 1
                days_in_trade = 0
        else:
            days_in_trade += 1
            hit_stop = abs(zt) > stop_z
            hit_exit = abs(zt) < exit_z
            hit_time = max_holding_days is not None and days_in_trade >= max_holding_days
            if hit_stop or hit_exit or hit_time:
                state = 0
                days_in_trade = 0

        pos[t] = state

    return pd.Series(pos, index=z.index, name="position")
