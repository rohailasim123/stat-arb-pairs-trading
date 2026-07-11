"""
Candidate pair discovery for statistical arbitrage.

Pipeline: correlation pre-filter -> Engle-Granger cointegration test
-> Johansen confirmation -> half-life of mean reversion filter.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class PairCandidate:
    asset_x: str
    asset_y: str
    correlation: float
    eg_pvalue: float
    eg_pvalue_adj: float | None = None
    johansen_trace_stat: float | None = None
    johansen_crit_95: float | None = None
    hedge_ratio_ols: float | None = None
    half_life_days: float | None = None
    passes: bool = False
    reasons: list[str] = field(default_factory=list)


def _ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    x_ = np.vstack([x.values, np.ones(len(x))]).T
    beta, _ = np.linalg.lstsq(x_, y.values, rcond=None)[0]
    return float(beta)


def _half_life(spread: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion by fitting
        delta_spread_t = theta * spread_{t-1} + eps_t
    (discretized OU process) via OLS, then half-life = -ln(2)/theta.
    Returns np.inf if theta >= 0 (no mean reversion detected).
    """
    spread_lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    spread_lag = spread_lag.loc[delta.index]
    x_ = np.vstack([spread_lag.values, np.ones(len(spread_lag))]).T
    theta, _ = np.linalg.lstsq(x_, delta.values, rcond=None)[0]
    if theta >= 0:
        return float("inf")
    return float(-np.log(2) / theta)


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Standard BH step-up procedure. Returns boolean list of which
    hypotheses are rejected (i.e. declared significant) at level `alpha`."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = np.array(pvalues)[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = sorted_p <= thresh
    if not below.any():
        return [False] * m
    max_i = np.max(np.where(below)[0])
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[: max_i + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject.tolist()


def find_candidate_pairs(
    prices: pd.DataFrame,
    min_correlation: float = 0.80,
    eg_alpha: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 60.0,
    fdr_alpha: float = 0.05,
) -> list[PairCandidate]:
    """
    Run the full candidate-pair discovery pipeline over all pairs in
    `prices` (columns = tickers, values = price levels).
    """
    log_prices = np.log(prices)
    tickers = list(prices.columns)
    corr = log_prices.corr()

    stage1: list[PairCandidate] = []
    for a, b in itertools.combinations(tickers, 2):
        rho = corr.loc[a, b]
        if abs(rho) < min_correlation:
            continue
        stage1.append(PairCandidate(asset_x=a, asset_y=b, correlation=float(rho),
                                     eg_pvalue=np.nan))

    if not stage1:
        return []

    # Engle-Granger on the correlation-filtered survivors
    for cand in stage1:
        y = log_prices[cand.asset_y]
        x = log_prices[cand.asset_x]
        _, pvalue, _ = coint(y, x)
        cand.eg_pvalue = float(pvalue)

    pvals = [c.eg_pvalue for c in stage1]
    bh_reject = benjamini_hochberg(pvals, alpha=fdr_alpha)
    for cand, rej in zip(stage1, bh_reject):
        cand.eg_pvalue_adj = cand.eg_pvalue  # BH doesn't rescale p, just flags rejection
        if not rej:
            cand.reasons.append("fails FDR-adjusted Engle-Granger significance")

    survivors = [c for c in stage1 if c.eg_pvalue < eg_alpha and "fails FDR-adjusted Engle-Granger significance" not in c.reasons]

    # Johansen confirmation + hedge ratio + half-life
    for cand in survivors:
        y = log_prices[cand.asset_y]
        x = log_prices[cand.asset_x]
        pair_mat = np.column_stack([y.values, x.values])
        try:
            jres = coint_johansen(pair_mat, det_order=0, k_ar_diff=1)
            trace_stat = float(jres.lr1[0])
            crit_95 = float(jres.cvt[0, 1])
        except Exception:
            trace_stat, crit_95 = np.nan, np.nan
        cand.johansen_trace_stat = trace_stat
        cand.johansen_crit_95 = crit_95

        beta = _ols_hedge_ratio(y, x)
        cand.hedge_ratio_ols = beta
        spread = y - beta * x
        hl = _half_life(spread)
        cand.half_life_days = hl

        passes = True
        if not (trace_stat > crit_95):
            passes = False
            cand.reasons.append("fails Johansen trace test at 95%")
        if not (min_half_life <= hl <= max_half_life):
            passes = False
            cand.reasons.append(f"half-life {hl:.1f}d outside [{min_half_life},{max_half_life}]")
        cand.passes = passes

    # keep all stage1 candidates (even rejected ones) so callers can audit
    # the funnel, but sort passers first. Among passers, rank by Johansen
    # trace-stat margin over its 95% critical value (how far the
    # relationship sits above the borderline) rather than by EG p-value
    # alone -- a single p-value is more prone to rewarding a lucky
    # spurious correlation among the many pairs screened, whereas the
    # Johansen margin is more robust to that.
    def _rank_key(c):
        margin = (c.johansen_trace_stat - c.johansen_crit_95) if c.passes else -np.inf
        return (not c.passes, -margin)

    all_candidates = stage1
    all_candidates.sort(key=_rank_key)
    return all_candidates


def summarize(candidates: list[PairCandidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        rows.append({
            "pair": f"{c.asset_y}~{c.asset_x}",
            "correlation": round(c.correlation, 3),
            "eg_pvalue": round(c.eg_pvalue, 4) if c.eg_pvalue is not None and not np.isnan(c.eg_pvalue) else None,
            "johansen_trace": round(c.johansen_trace_stat, 2) if c.johansen_trace_stat else None,
            "johansen_crit95": round(c.johansen_crit_95, 2) if c.johansen_crit_95 else None,
            "hedge_ratio": round(c.hedge_ratio_ols, 3) if c.hedge_ratio_ols else None,
            "half_life_days": round(c.half_life_days, 1) if c.half_life_days not in (None, float("inf")) else None,
            "passes": c.passes,
            "reasons": "; ".join(c.reasons) if c.reasons else "",
        })
    return pd.DataFrame(rows)
