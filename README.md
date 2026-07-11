# Statistical Arbitrage: Cointegration-Based Pairs Trading with Dynamic Hedging

A from-scratch statistical arbitrage research pipeline: candidate pair discovery via
cointegration testing, dynamic hedge-ratio estimation with a Kalman filter, a
realistic-cost backtest engine with walk-forward validation, and rigorous
statistical significance testing of the results (block bootstrap, deflated
Sharpe ratio).

## Result

On the synthetic demo universe (see [Why synthetic data](#why-synthetic-data-by-default)),
the pipeline correctly identifies the embedded cointegrated pair, estimates a
dynamic hedge ratio, and finds a **gross Sharpe ratio of ~0.68** that decays to
**roughly break-even around 4bps of round-trip transaction cost** and is
**negative after realistic 7bps costs out-of-sample**. A block-bootstrap test
confirms the out-of-sample Sharpe is not statistically distinguishable from
zero (p = 0.98 for H0: Sharpe ≤ 0), and the deflated Sharpe ratio which
penalizes for the ~150 candidate pairs screened confirms this isn't just an
underpowered test.


## Architecture

```
data_loader.py       -> price data (real via yfinance, or synthetic w/ known ground truth)
        |
        v
pair_selection.py    -> correlation filter -> Engle-Granger -> Benjamini-Hochberg FDR
        |                -> Johansen confirmation -> half-life filter
        v
kalman_hedge.py       -> online Kalman filter for time-varying hedge ratio beta_t
        |                (random-walk state-space model, hand-implemented)
        v
signals.py             -> z-score entry/exit/stop-loss rules on the spread
        |
        v
backtest.py            -> walk-forward simulation, next-bar execution,
        |                  transaction costs + slippage, dollar-neutral sizing
        v
stats_tests.py          -> block bootstrap Sharpe CI, deflated Sharpe ratio
        |                  (multiple-testing correction)
        v
regime.py (optional)     -> HMM volatility-regime overlay for position sizing
```

## Quickstart

```bash
git clone https://github.com/rohailasim123/stat-arb-pairs-trading.git && cd stat-arb-pairs-trading
pip install -r requirements.txt

# synthetic demo (deterministic, no network required, ships with known ground truth)
python scripts/run_backtest.py

# real data (requires network access to Yahoo Finance)
python scripts/run_backtest.py --tickers XOM CVX --start 2018-01-01

# run the test suite
pytest tests/ -v
```

Outputs land in `results/`: a pair-screening audit table (`pair_screening_results.csv`),
the OOS daily backtest (`oos_backtest_daily.csv`), a performance summary
(`performance_summary.csv`), and three plots (see below).

## Why synthetic data by default

`yfinance` requires outbound network access to Yahoo Finance, which isn't
available in every execution environment (CI runners, sandboxed notebooks,
etc.), and real market data is regime-dependent. A backtest run today vs.
in six months on the same tickers can tell a very different story, which
makes results hard to reproduce and review.

`data_loader.synthetic_universe()` generates a controlled panel where a known
number of pairs are cointegrated **by construction** (common stochastic trend
+ a stationary Ornstein-Uhlenbeck spread with a known true hedge ratio and
mean-reversion speed) alongside a larger set of independent random walks that
should **not** be flagged as cointegrated. This lets `tests/test_pair_selection.py`
assert the pipeline actually recovers the true pairs and mostly rejects the
noise assets.

Pass `--tickers` to run on real data.

## Interpreting the results

Three plots are generated in `results/`:

1. **`backtest_summary.png`** — the pair's log-price series with the
   Kalman-fitted relationship overlaid, the time-varying hedge ratio vs. the
   static OLS estimate, the z-score with entry/exit bands, and the
   out-of-sample walk-forward equity curve.
2. **`bootstrap_sharpe_distribution.png`** — the block-bootstrap distribution
   of the annualized OOS Sharpe ratio, with the null (Sharpe = 0) marked.
3. **`cost_sensitivity.png`** — Sharpe ratio vs. assumed round-trip
   transaction cost, with the estimated break-even cost marked.

There is a real, statistically detectable mean-reverting relationship in the 
synthetic pair (by construction), and the pipeline finds it but the edge is 
thin enough that it doesn't clear realistic trading frictions out-of-sample.
