"""
Dynamic hedge-ratio estimation via a Kalman filter, following the
state-space formulation popularized by Ernest Chan ("Algorithmic Trading",
Ch. 3) and standard in the stat-arb literature.

State-space model:
    Observation:  y_t = [x_t, 1] @ [beta_t, alpha_t]' + epsilon_t   (obs noise)
    State:        [beta_t, alpha_t]' = [beta_{t-1}, alpha_{t-1}]' + eta_t (process noise)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class KalmanHedgeRatio:
    """
    Online estimator of a time-varying hedge ratio (beta_t) and intercept
    (alpha_t) between two price series using a 2D Kalman filter.

    Parameters
    ----------
    delta : float
        Controls how quickly beta/alpha are allowed to drift. Smaller delta
        -> smoother, slower-moving beta. This maps to the process noise
        covariance Q = delta/(1-delta) * I.
    obs_var : float
        Observation noise variance (R). Can be estimated from residuals of
        a static OLS fit on a warm-up window, or left as a tunable constant.
    """

    def __init__(self, delta: float = 1e-4, obs_var: float = 1e-3):
        self.delta = delta
        self.obs_var = obs_var
        self.Q = (delta / (1 - delta)) * np.eye(2)  # process noise cov
        self.state = None       # [beta, alpha]
        self.P = np.eye(2) * 1.0  # state covariance

    def filter(self, y: pd.Series, x: pd.Series, z_window: int = 20) -> pd.DataFrame:
        """
        Run the filter forward over the full series. Returns a DataFrame
        with columns:
          beta, alpha  : latent state estimates
          spread       : one-step-ahead prediction error
                         (y_t - beta_{t|t-1}*x_t - alpha_{t|t-1}), i.e. the
                         tradeable mean-reverting residual
          spread_var   : Kalman *filter* uncertainty of the spread
                         (innovation variance) -- reflects how confident
                         the filter is in beta_t, not the spread's
                         historical trading range, so kept as a diagnostic
          spread_z     : z-score of `spread` normalized by its own trailing
                         `z_window`-day rolling std. This is the practical
                         signal used for entries/exits -- normalizing by
                         the filter's innovation variance instead tends to
                         be overly conservative before the filter has
                         converged and understates genuine opportunities.
        """
        n = len(y)
        betas = np.zeros(n)
        alphas = np.zeros(n)
        spreads = np.zeros(n)
        spread_vars = np.zeros(n)

        state = np.array([1.0, 0.0])  # init beta=1, alpha=0
        P = np.eye(2) * 1.0

        y_vals = y.values
        x_vals = x.values

        for t in range(n):
            # ---- predict ----
            state_pred = state  # random walk: F = I
            P_pred = P + self.Q

            H = np.array([x_vals[t], 1.0])  # observation matrix row
            y_hat = H @ state_pred
            S = H @ P_pred @ H.T + self.obs_var  # innovation variance
            e = y_vals[t] - y_hat                 # innovation (= spread estimate)

            # ---- update ----
            K = P_pred @ H / S  # Kalman gain (2,)
            state = state_pred + K * e
            P = P_pred - np.outer(K, H) @ P_pred

            betas[t] = state[0]
            alphas[t] = state[1]
            spreads[t] = e
            spread_vars[t] = S

        out = pd.DataFrame({
            "beta": betas,
            "alpha": alphas,
            "spread": spreads,
            "spread_var": spread_vars,
        }, index=y.index)
        roll_std = out["spread"].rolling(z_window, min_periods=z_window // 2).std()
        out["spread_z"] = out["spread"] / roll_std
        return out
