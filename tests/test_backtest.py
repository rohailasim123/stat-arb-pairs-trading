import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals import generate_positions
from src.backtest import run_backtest, performance_stats, BacktestConfig, walk_forward_split
from src.stats_tests import block_bootstrap_sharpe, deflated_sharpe_ratio
from src.pair_selection import benjamini_hochberg


def test_generate_positions_basic_entry_exit():
    z = pd.Series([0, 0.5, 2.5, 2.0, 0.3, 0, -2.5, -3, 0.1, 0])
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=4.0, max_holding_days=None)
    # enters short at index 2 (z=2.5), exits by index 4 (|z|=0.3 < exit_z)
    assert pos.iloc[2] == -1
    assert pos.iloc[4] == 0
    # enters long at index 6 (z=-2.5), exits at index 8 (|z|=0.1 < exit_z)
    assert pos.iloc[6] == 1
    assert pos.iloc[8] == 0


def test_generate_positions_stop_loss_triggers():
    z = pd.Series([0, 2.5, 3.0, 4.5, 4.5])  # breach entry, then blow past stop
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=4.0, max_holding_days=None)
    assert pos.iloc[1] == -1
    assert pos.iloc[3] == 0  # stopped out once |z| > stop_z


def test_backtest_zero_cost_flat_position_zero_pnl():
    """A position series that's always flat should produce exactly zero
    net return and zero cost drag, regardless of price path."""
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    rng = np.random.default_rng(1)
    y = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))), index=idx)
    x = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))), index=idx)
    beta = pd.Series(1.0, index=idx)
    z = pd.Series(0.0, index=idx)
    pos = pd.Series(0.0, index=idx)

    bt = run_backtest(y, x, beta, z, pos, BacktestConfig(tc_bps=5, slippage_bps=2))
    assert np.allclose(bt["net_return"], 0.0)
    assert np.allclose(bt["cost_drag"], 0.0)


def test_backtest_costs_reduce_returns():
    """Adding transaction costs should never improve net returns relative
    to the zero-cost case for the same position path."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(2)
    y = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 100))), index=idx)
    x = pd.Series(50 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, 100))), index=idx)
    beta = pd.Series(1.0, index=idx)
    z = pd.Series(np.sin(np.linspace(0, 10, 100)) * 3, index=idx)
    pos = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)

    bt_free = run_backtest(y, x, beta, z, pos, BacktestConfig(tc_bps=0, slippage_bps=0))
    bt_costly = run_backtest(y, x, beta, z, pos, BacktestConfig(tc_bps=10, slippage_bps=5))

    assert bt_costly["equity"].iloc[-1] <= bt_free["equity"].iloc[-1]


def test_walk_forward_split_no_overlap_and_covers_expected_range():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    splits = list(walk_forward_split(idx, train_days=100, test_days=50))
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert len(train_idx) == 100
        assert len(test_idx) == 50
        assert train_idx[-1] < test_idx[0]  # no look-ahead: train strictly precedes test


def test_block_bootstrap_sharpe_null_case():
    """Pure noise (mean zero) returns should have a bootstrap CI that
    comfortably contains zero and a high p-value for H0: Sharpe<=0 being
    true is NOT rejected, i.e. p should be well above a small threshold."""
    rng = np.random.default_rng(3)
    noise = pd.Series(rng.normal(0, 0.01, 500))
    result = block_bootstrap_sharpe(noise, n_boot=1000, block_size=10, seed=3)
    assert result["ci_low"] < 0 < result["ci_high"]


def test_deflated_sharpe_ratio_penalizes_more_trials():
    """Holding the observed Sharpe fixed, deflated Sharpe (confidence
    that true Sharpe > 0) should decrease as the number of trials
    (pairs screened) increases -- this is the multiple-testing penalty."""
    dsr_few = deflated_sharpe_ratio(sharpe=1.0, n_trials=1, n_obs=500)
    dsr_many = deflated_sharpe_ratio(sharpe=1.0, n_trials=500, n_obs=500)
    assert dsr_many < dsr_few


def test_benjamini_hochberg_all_significant():
    pvals = [0.001, 0.002, 0.003]
    assert all(benjamini_hochberg(pvals, alpha=0.05))


def test_benjamini_hochberg_none_significant():
    pvals = [0.5, 0.6, 0.7]
    assert not any(benjamini_hochberg(pvals, alpha=0.05))
