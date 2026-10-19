"""Analysis functions for the minimal resilience-stress model."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.integrate import solve_ivp

from .model import ModelParams, drift, dominant_eigenvalue, equilibrium, first_passage_time


def relaxation_time(
    lam: float,
    delta0: float = 1e-2,
    t_max: float = 500.0,
    n_steps: int = 5000,
    p: ModelParams = ModelParams(),
) -> float:
    """Finite-amplitude relaxation time to e^-1 of the initial perturbation."""
    x_star = equilibrium(lam, p)
    if np.any(np.isnan(x_star)):
        return np.nan

    x0 = x_star + delta0 * np.array([1.0, 0.0])
    t_eval = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(
        lambda t, x: drift(t, x, lam, p), (0.0, t_max), x0, t_eval=t_eval, rtol=1e-8, atol=1e-10
    )

    d0 = norm(x0 - x_star)
    d_t = norm(sol.y.T - x_star, axis=1)
    hits = np.where(d_t <= np.exp(-1) * d0)[0]
    return float(sol.t[hits[0]]) if len(hits) else np.nan


def deterministic_summary(lambda_values: Iterable[float], p: ModelParams = ModelParams()) -> pd.DataFrame:
    """Compute equilibrium, dominant eigenvalue, linear relaxation scale, and relaxation time."""
    rows = []
    for lam in lambda_values:
        x_star = equilibrium(float(lam), p)
        rows.append(
            {
                "lambda": float(lam),
                "M_star": x_star[0],
                "S_star": x_star[1],
                "mu_max": dominant_eigenvalue(float(lam), p),
                "linear_relaxation_scale": -1.0 / dominant_eigenvalue(float(lam), p),
                "relaxation_time": relaxation_time(float(lam), p=p),
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_escape_statistics(
    lambda_values: Iterable[float],
    n_reps: int = 1000,
    T: float = 500.0,
    dt: float = 0.02,
    S_c: float = 3.5,
    M_c: float = 0.15,
    base_seed: int = 20240529,
    p: ModelParams = ModelParams(),
) -> Tuple[dict[float, np.ndarray], pd.DataFrame]:
    """Run Monte Carlo first-passage simulations and summarize escape statistics."""
    escape_times: dict[float, np.ndarray] = {}
    rows = []

    for i, lam_raw in enumerate(lambda_values):
        lam = float(lam_raw)
        times = np.empty(n_reps, dtype=float)
        for r in range(n_reps):
            times[r] = first_passage_time(
                lam=lam,
                T=T,
                dt=dt,
                S_c=S_c,
                M_c=M_c,
                seed=base_seed + i * n_reps + r,
                p=p,
            )

        escape_times[lam] = times
        escaped = ~np.isnan(times)
        n_escaped = int(np.sum(escaped))

        rows.append(
            {
                "lambda": lam,
                "n_reps": n_reps,
                "T_max": T,
                "dt": dt,
                "M_c": M_c,
                "S_c": S_c,
                "n_escaped": n_escaped,
                "escape_probability": n_escaped / n_reps,
                "mean_escape_time_uncensored": np.nanmean(times) if n_escaped else np.nan,
                "median_escape_time_uncensored": np.nanmedian(times) if n_escaped else np.nan,
                "variance_escape_time_uncensored": np.nanvar(times, ddof=1) if n_escaped > 1 else np.nan,
                "sd_escape_time_uncensored": np.nanstd(times, ddof=1) if n_escaped > 1 else np.nan,
            }
        )

    return escape_times, pd.DataFrame(rows)


def empirical_survival_curve(times: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Empirical survival curve P(tau > t), treating NaN as right-censored at infinity."""
    observed = np.array(times, copy=True)
    observed[np.isnan(observed)] = np.inf
    return np.array([np.mean(observed > t) for t in t_grid])


def monte_carlo_first_entry_times(
    lambda_values: Iterable[float],
    n_reps: int = 1000,
    T: float = 500.0,
    dt: float = 0.02,
    S_c: float = 3.5,
    M_c: float = 0.15,
    seed: int = 20240529,
    p: ModelParams = ModelParams(),
) -> np.ndarray:
    """Vectorized, seeded Euler-Maruyama first-entry times into F = {S > S_c and M < M_c}.

    All lambda values and replicates are integrated simultaneously (Ito interpretation,
    reflecting boundaries at M = 0 and S = 0). Entry into F is recorded once and the
    trajectory is then frozen: F is treated as absorbing by assumption (commitment point),
    not because the drift prevents return. Returns an array of shape
    (len(lambda_values), n_reps) with NaN for trajectories that did not enter F by T.
    """
    lams = np.asarray(list(lambda_values), dtype=float)
    rng = np.random.default_rng(seed)
    M_star = p.M_rest - (p.b_M * p.a_S / (p.a_M * p.b_S)) * lams**2
    S_star = np.where(M_star > 0, p.a_S * lams / (p.b_S * np.where(M_star > 0, M_star, 1.0)), np.nan)
    lam = np.repeat(lams[:, None], n_reps, axis=1)
    M = np.repeat(M_star[:, None], n_reps, axis=1)
    S = np.repeat(S_star[:, None], n_reps, axis=1)
    tau = np.full(M.shape, np.nan)
    alive = np.isfinite(M) & np.isfinite(S) & (M > 0)
    sqrt_dt = np.sqrt(dt)
    for k in range(int(round(T / dt))):
        if not alive.any():
            break
        dM = p.a_M * (p.M_rest - M) - p.b_M * lam * S * M
        dS = p.a_S * lam - p.b_S * M * S
        sS = p.sigma_S0 + p.sigma_S1 * lam * np.maximum(S, 0.0)
        M_new = np.abs(M + dM * dt + p.sigma_M_const * sqrt_dt * rng.standard_normal(M.shape))
        S_new = np.abs(S + dS * dt + sS * sqrt_dt * rng.standard_normal(S.shape))
        M = np.where(alive, M_new, M)
        S = np.where(alive, S_new, S)
        hit = alive & (S > S_c) & (M < M_c)
        tau[hit] = (k + 1) * dt
        alive &= ~hit
    return tau
