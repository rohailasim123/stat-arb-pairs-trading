"""
Optional volatility-regime detection via a Gaussian Hidden Markov Model,
fit on the spread's rolling volatility and z-score dynamics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_regime_hmm(z: pd.Series, window: int = 20, n_states: int = 2, seed: int = 0):
    """
    Fits a Gaussian HMM with `n_states` on features [rolling_vol(z), |z|].
    Returns (model, regime_series) where regime_series holds the most
    likely state index per timestep, and states are relabeled so that
    state 0 = lower volatility regime, state 1 = higher volatility regime.
    """
    roll_vol = z.rolling(window).std()
    feats = pd.DataFrame({"roll_vol": roll_vol, "abs_z": z.abs()}).dropna()

    X = feats.values
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                         n_iter=200, random_state=seed)
    model.fit(X)
    states = model.predict(X)

    # relabel by mean roll_vol so state 0 = calm, state 1 = turbulent
    means = [X[states == s, 0].mean() for s in range(n_states)]
    order = np.argsort(means)
    relabel = {old: new for new, old in enumerate(order)}
    states_relabeled = np.array([relabel[s] for s in states])

    regime = pd.Series(states_relabeled, index=feats.index, name="regime")
    regime = regime.reindex(z.index).ffill().fillna(0).astype(int)
    return model, regime


def regime_position_scalar(regime: pd.Series, scale_calm: float = 1.0,
                            scale_turbulent: float = 0.3) -> pd.Series:
    """Maps regime state -> position size multiplier."""
    return regime.map({0: scale_calm, 1: scale_turbulent}).fillna(scale_calm)
