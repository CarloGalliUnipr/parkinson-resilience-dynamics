"""Optional fast Monte Carlo implementation using Numba.

This module is not required for reproducibility, but it is useful for generating
Figure 5 efficiently with 1000 trajectories per lambda.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install numba to use pd_resilience_model.numba_mc") from exc


A_M = 0.5
B_M = 0.4
M_REST = 1.0
A_S = 0.3
B_S = 0.6


@njit
def equilibrium_numba(lam: float):
    M_star = M_REST - (B_M * A_S / (A_M * B_S)) * lam**2
    if M_star <= 0.0:
        return np.nan, np.nan
    S_star = (A_S * lam) / (B_S * M_star)
    return M_star, S_star


@njit
def single_escape_time_numba(lam: float, T: float, dt: float, S_c: float, M_c: float):
    M, S = equilibrium_numba(lam)
    if np.isnan(M) or np.isnan(S):
        return np.nan

    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    for k in range(n_steps):
        dM = A_M * (M_REST - M) - B_M * lam * S * M
        dS = A_S * lam - B_S * M * S

        sigmaM = 0.003
        sigmaS = 0.015 + 0.03 * lam * max(S, 0.0)

        M_new = M + dM * dt + sigmaM * sqrt_dt * np.random.randn()
        S_new = S + dS * dt + sigmaS * sqrt_dt * np.random.randn()

        M = -M_new if M_new < 0.0 else M_new
        S = -S_new if S_new < 0.0 else S_new

        if (S > S_c) and (M < M_c):
            return (k + 1) * dt

    return np.nan


@njit(parallel=True)
def monte_carlo_escape_times_numba(
    lambda_values: np.ndarray,
    n_reps: int,
    T: float,
    dt: float,
    S_c: float,
    M_c: float,
):
    escape_times = np.empty((len(lambda_values), n_reps))
    for i in prange(len(lambda_values)):
        for r in range(n_reps):
            escape_times[i, r] = single_escape_time_numba(lambda_values[i], T, dt, S_c, M_c)
    return escape_times
