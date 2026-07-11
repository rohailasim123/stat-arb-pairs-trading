#!/usr/bin/env python3
"""
run_backtest.py
================
End-to-end demo of the full pipeline:

  1. Load data (synthetic universe by default; pass --tickers for real
     data via yfinance).
  2. Screen all pairs for cointegration (correlation filter -> Engle-Granger
     -> Johansen -> half-life filter, with FDR correction).
  3. For the best candidate pair, estimate a dynamic hedge ratio with a
     Kalman filter.
  4. Generate z-score based entry/exit signals.
  5. Walk-forward backtest with realistic transaction costs.
  6. Statistical significance testing (block bootstrap Sharpe CI,
     deflated Sharpe ratio accounting for the number of pairs screened).
  7. Save plots + a results table to results/.

Usage:
    python scripts/run_backtest.py                     
    python scripts/run_backtest.py --tickers XOM CVX --start 2018-01-01
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import fetch_prices, synthetic_universe, SyntheticConfig
from src.pair_selection import find_candidate_pairs, summarize
from src.kalman_hedge import KalmanHedgeRatio
from src.signals import generate_positions
from src.backtest import run_backtest, performance_stats, BacktestConfig, walk_forward_split
from src.stats_tests import block_bootstrap_sharpe, deflated_sharpe_ratio

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=None,
                     help="Real tickers to fetch via yfinance. Omit for synthetic demo data.")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--train-days", type=int, default=500)
    ap.add_argument("--test-days", type=int, default=125)
    args = ap.parse_args()

    if args.tickers:
        print(f"Fetching real data for {args.tickers} ...")
        prices = fetch_prices(args.tickers, args.start, args.end)
        n_trials_for_dsr = len(args.tickers) * (len(args.tickers) - 1) // 2
    else:
        print("No --tickers passed, using synthetic cointegrated universe for demo/reproducibility.")
        prices, truth = synthetic_universe(SyntheticConfig())
        print("Ground truth embedded pairs:", {k: round(v["beta"], 3) for k, v in truth.items()})
        n_trials_for_dsr = prices.shape[1] * (prices.shape[1] - 1) // 2

    print(f"Universe: {prices.shape[1]} assets, {prices.shape[0]} days")

    # ---- Step 1: pair screening ----
    print("\nScreening candidate pairs (correlation -> Engle-Granger -> Johansen -> half-life)...")
    candidates = find_candidate_pairs(prices, min_correlation=0.80)
    table = summarize(candidates)
    table.to_csv(RESULTS_DIR / "pair_screening_results.csv", index=False)
    print(table.head(10).to_string(index=False))

    passers = [c for c in candidates if c.passes]
    if not passers:
        print("No pairs passed all filters. Try lowering --min-correlation or check data.")
        return
    best = passers[0]
    print(f"\nSelected pair: {best.asset_y} ~ {best.asset_x} "
          f"(EG p={best.eg_pvalue:.4f}, half-life={best.half_life_days:.1f}d, "
          f"static OLS beta={best.hedge_ratio_ols:.3f})")

    y = np.log(prices[best.asset_y])
    x = np.log(prices[best.asset_x])

    # ---- Step 2: Kalman dynamic hedge ratio ----
    print("\nFitting Kalman filter for dynamic hedge ratio...")
    kf = KalmanHedgeRatio(delta=1e-4, obs_var=np.var(y - best.hedge_ratio_ols * x) * 0.1)
    kf_out = kf.filter(y, x)

    # ---- Step 3: signals ----
    positions = generate_positions(kf_out["spread_z"], entry_z=2.0, exit_z=0.5, stop_z=4.0)

    # ---- Step 4: walk-forward backtest ----
    print("\nRunning walk-forward backtest...")
    cfg = BacktestConfig()
    all_segments = []
    price_y_lvl = prices[best.asset_y]
    price_x_lvl = prices[best.asset_x]

    for train_idx, test_idx in walk_forward_split(prices.index, args.train_days, args.test_days):
        # re-fit Kalman filter fresh on train+test data but we only *use*
        # (score) the test segment; the filter's online nature means beta_t
        # at time t only ever depends on data up to t, so this is naturally
        # walk-forward as long as we score out-of-sample segments only.
        seg_idx = train_idx.append(test_idx)
        seg_y, seg_x = y.loc[seg_idx], x.loc[seg_idx]
        kf_seg = KalmanHedgeRatio(delta=1e-4, obs_var=kf.obs_var).filter(seg_y, seg_x)
        pos_seg = generate_positions(kf_seg["spread_z"], entry_z=2.0, exit_z=0.5, stop_z=4.0)

        bt_seg = run_backtest(
            price_y_lvl.loc[seg_idx], price_x_lvl.loc[seg_idx],
            kf_seg["beta"], kf_seg["spread_z"], pos_seg, cfg,
        )
        all_segments.append(bt_seg.loc[test_idx])

    bt_oos = pd.concat(all_segments).sort_index()
    bt_oos = bt_oos[~bt_oos.index.duplicated(keep="first")]
    bt_oos["net_return"] = bt_oos["net_return"].fillna(0.0)
    bt_oos["equity"] = cfg.capital * (1 + bt_oos["net_return"]).cumprod()

    stats = performance_stats(bt_oos)
    print("\nOut-of-sample walk-forward performance:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- Step 5: statistical significance ----
    print("\nRunning block-bootstrap significance test (5000 resamples, block=10d)...")
    boot = block_bootstrap_sharpe(bt_oos["net_return"], n_boot=5000, block_size=10)
    print(f"  Sharpe (point est.): {boot['sharpe_hat']:.3f}")
    print(f"  95% bootstrap CI:   [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]")
    print(f"  P(Sharpe <= 0):      {boot['p_value_le_zero']:.4f}")

    dsr = deflated_sharpe_ratio(stats["sharpe"], n_trials=max(n_trials_for_dsr, 1),
                                 n_obs=len(bt_oos))
    print(f"  Deflated Sharpe Ratio (P[true SR > 0], corrected for "
          f"{n_trials_for_dsr} pairs screened): {dsr:.4f}")

    # ---- Step 5b: cost sensitivity -- how much of the gross edge survives costs? ----
    print("\nCost sensitivity: Sharpe as a function of round-trip transaction cost (bps)...")
    cost_sweep_bps = [0, 1, 2, 3, 5, 7, 10, 15, 20]
    cost_sweep_sharpe = []
    for bps in cost_sweep_bps:
        cfg_sweep = BacktestConfig(tc_bps=bps, slippage_bps=0)
        bt_full = run_backtest(price_y_lvl, price_x_lvl, kf_out["beta"], kf_out["spread_z"],
                                positions, cfg_sweep)
        s = performance_stats(bt_full)["sharpe"]
        cost_sweep_sharpe.append(s)
        print(f"  {bps:>3} bps -> Sharpe {s:.3f}")
    breakeven_bps = None
    for i in range(1, len(cost_sweep_bps)):
        if cost_sweep_sharpe[i - 1] > 0 >= cost_sweep_sharpe[i]:
            breakeven_bps = np.interp(0, [cost_sweep_sharpe[i], cost_sweep_sharpe[i - 1]],
                                       [cost_sweep_bps[i], cost_sweep_bps[i - 1]])
    if breakeven_bps is not None:
        print(f"  Estimated breakeven cost: ~{breakeven_bps:.1f} bps round-trip "
              f"(strategy needs costs below this to be net profitable in-sample)")

    # ---- Step 6: plots ----
    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=False)

    axes[0].plot(prices.index, np.log(prices[best.asset_y]), label=best.asset_y)
    axes[0].plot(prices.index, kf_out["beta"] * np.log(prices[best.asset_x]) + kf_out["alpha"],
                 label=f"{best.hedge_ratio_ols:.2f}*{best.asset_x} (Kalman-fitted)", alpha=0.8)
    axes[0].set_title(f"Pair: {best.asset_y} vs {best.asset_x} (log price)")
    axes[0].legend()

    axes[1].plot(kf_out.index, kf_out["beta"], color="darkorange")
    axes[1].axhline(best.hedge_ratio_ols, color="gray", linestyle="--", label="static OLS beta")
    axes[1].set_title("Kalman-filtered dynamic hedge ratio (beta_t)")
    axes[1].legend()

    axes[2].plot(kf_out.index, kf_out["spread_z"], color="purple", linewidth=0.8)
    axes[2].axhline(2.0, color="red", linestyle="--", linewidth=0.8)
    axes[2].axhline(-2.0, color="red", linestyle="--", linewidth=0.8)
    axes[2].axhline(0.5, color="green", linestyle=":", linewidth=0.8)
    axes[2].axhline(-0.5, color="green", linestyle=":", linewidth=0.8)
    axes[2].set_title("Spread z-score with entry/exit bands")

    axes[3].plot(bt_oos.index, bt_oos["equity"], color="black")
    axes[3].set_title(f"Out-of-sample walk-forward equity curve "
                       f"(Sharpe={stats['sharpe']:.2f}, MaxDD={stats['max_drawdown']:.1%})")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "backtest_summary.png", dpi=130)
    print(f"\nSaved plots to {RESULTS_DIR / 'backtest_summary.png'}")

    # cost sensitivity plot
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(cost_sweep_bps, cost_sweep_sharpe, marker="o", color="darkred")
    ax3.axhline(0, color="gray", linestyle="--")
    if breakeven_bps is not None:
        ax3.axvline(breakeven_bps, color="green", linestyle=":",
                    label=f"breakeven ~{breakeven_bps:.1f} bps")
        ax3.legend()
    ax3.set_xlabel("Round-trip transaction cost (bps)")
    ax3.set_ylabel("Full-sample Sharpe ratio")
    ax3.set_title("Cost sensitivity: how much of the gross edge survives realistic costs?")
    fig3.tight_layout()
    fig3.savefig(RESULTS_DIR / "cost_sensitivity.png", dpi=130)

    # bootstrap distribution plot
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(boot["boot_dist"], bins=60, color="steelblue", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", label="H0: Sharpe = 0")
    ax2.axvline(boot["sharpe_hat"], color="black", label="point estimate")
    ax2.set_title("Block-bootstrap distribution of annualized Sharpe (OOS)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "bootstrap_sharpe_distribution.png", dpi=130)

    # save results table
    results_summary = {**stats, "bootstrap_sharpe_ci_low": boot["ci_low"],
                        "bootstrap_sharpe_ci_high": boot["ci_high"],
                        "bootstrap_p_value_le_zero": boot["p_value_le_zero"],
                        "deflated_sharpe_ratio": dsr,
                        "pair": f"{best.asset_y}~{best.asset_x}"}
    pd.Series(results_summary).to_csv(RESULTS_DIR / "performance_summary.csv")
    bt_oos.to_csv(RESULTS_DIR / "oos_backtest_daily.csv")
    print(f"Saved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
