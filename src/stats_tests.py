"""
Statistical significance testing for backtested strategy returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_sharpe(
    returns: pd.Series,
    n_boot: int = 5000,
    block_size: int = 10,
    periods_per_year: int = 252,
    seed: int = 7,
) -> dict:
    """
    Bootstrap the annualized Sharpe ratio of `returns` using the moving
    block bootstrap, and report a percentile confidence interval plus a
    one-sided p-value for H0: Sharpe <= 0.

    Returns dict with: sharpe_hat, ci_low, ci_high, p_value, boot_dist
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    n = len(r)
    if n < block_size * 2:
        raise ValueError("Series too short for requested block size")

    n_blocks = int(np.ceil(n / block_size))
    boot_sharpes = np.empty(n_boot)

    max_start = n - block_size
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([r[s: s + block_size] for s in starts])[:n]
        mu = sample.mean() * periods_per_year
        sigma = sample.std() * np.sqrt(periods_per_year)
        boot_sharpes[b] = mu / sigma if sigma > 0 else 0.0

    sharpe_hat = (r.mean() * periods_per_year) / (r.std() * np.sqrt(periods_per_year))
    ci_low, ci_high = np.percentile(boot_sharpes, [2.5, 97.5])
    p_value = float((boot_sharpes <= 0).mean())  # fraction of bootstrap draws at/below 0

    return {
        "sharpe_hat": float(sharpe_hat),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value_le_zero": p_value,
        "boot_dist": boot_sharpes,
    }


def block_bootstrap_mean_diff(
    returns_a: pd.Series,
    returns_b: pd.Series,
    n_boot: int = 5000,
    block_size: int = 10,
    seed: int = 7,
) -> dict:
    """
    Bootstrapped test of whether the mean daily return of strategy A is
    different from strategy B (e.g. "RL-selected pairs" vs "static
    cointegration pairs", or "with Kalman hedge" vs "static OLS hedge"),
    analogous to the bootstrapped Welch's t-test methodology used to
    compare model performance distributions with unequal variance.
    """
    rng = np.random.default_rng(seed)
    a = returns_a.dropna().values
    b = returns_b.dropna().values
    obs_diff = a.mean() - b.mean()

    def block_resample(arr, n_boot_local):
        n = len(arr)
        n_blocks = int(np.ceil(n / block_size))
        max_start = n - block_size
        out = np.empty(n_boot_local)
        for i in range(n_boot_local):
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            sample = np.concatenate([arr[s: s + block_size] for s in starts])[:n]
            out[i] = sample.mean()
        return out

    boot_a = block_resample(a, n_boot)
    boot_b = block_resample(b, n_boot)
    boot_diff = boot_a - boot_b
    ci_low, ci_high = np.percentile(boot_diff, [2.5, 97.5])
    p_value = float(2 * min((boot_diff <= 0).mean(), (boot_diff >= 0).mean()))

    return {
        "observed_diff": float(obs_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value_two_sided": p_value,
    }


def deflated_sharpe_ratio(sharpe: float, n_trials: int, n_obs: int,
                           skew: float = 0.0, kurt: float = 3.0,
                           periods_per_year: int = 252) -> float:
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio: adjusts the
    probability that the observed Sharpe is genuinely positive for the
    fact that many pair/parameter combinations were likely trialed before
    selecting this one (multiple-testing / selection bias correction --
    directly relevant since pair_selection.py screens many candidate pairs).

    Returns the probability (0-1) that the true Sharpe ratio exceeds 0,
    after deflating for `n_trials` independent trials.
    """
    from scipy.stats import norm

    sr = sharpe / np.sqrt(periods_per_year)  # per-period Sharpe
    euler_gamma = 0.5772156649
    # expected max Sharpe under n_trials independent N(0,1) trials
    if n_trials > 1:
        e_max = (1 - euler_gamma) * norm.ppf(1 - 1.0 / n_trials) + \
                euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
    else:
        e_max = 0.0

    sr0 = e_max / np.sqrt(n_obs)  # benchmark SR to beat, in per-period units
    numerator = (sr - sr0) * np.sqrt(n_obs - 1)
    denom = np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr ** 2)
    dsr_stat = numerator / denom if denom > 0 else np.nan
    return float(norm.cdf(dsr_stat))
