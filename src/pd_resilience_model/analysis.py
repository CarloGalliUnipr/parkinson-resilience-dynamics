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
    sol = solve_ivp(lambda t, x: drift(t, x, lam, p), (0.0, t_max), x0, t_eval=t_eval)

    d0 = norm(x0 - x_star)
    d_t = norm(sol.y.T - x_star, axis=1)
    hits = np.where(d_t <= np.exp(-1) * d0)[0]
    return float(sol.t[hits[0]]) if len(hits) else np.nan


def finite_time_separation_rate(
    lam: float,
    delta0: float = 1e-4,
    T: float = 200.0,
    n_steps: int = 4000,
    p: ModelParams = ModelParams(),
) -> float:
    """Finite-window separation diagnostic for two nearby deterministic trajectories."""
    x_star = equilibrium(lam, p)
    if np.any(np.isnan(x_star)):
        return np.nan

    x1_0 = x_star
    x2_0 = x_star + delta0 * np.array([1.0, 0.0])
    y0 = np.concatenate([x1_0, x2_0])

    def pair_drift(t: float, y: np.ndarray) -> np.ndarray:
        return np.concatenate([drift(t, y[:2], lam, p), drift(t, y[2:], lam, p)])

    sol = solve_ivp(pair_drift, (0.0, T), y0, t_eval=np.linspace(0.0, T, n_steps))
    d0 = norm(x2_0 - x1_0)
    dT = norm(sol.y[2:, -1] - sol.y[:2, -1])
    return float((1.0 / T) * np.log(dT / d0)) if d0 > 0 and dT > 0 else np.nan


def deterministic_summary(lambda_values: Iterable[float], p: ModelParams = ModelParams()) -> pd.DataFrame:
    """Compute equilibrium, dominant eigenvalue, relaxation time, and separation rate."""
    rows = []
    for lam in lambda_values:
        x_star = equilibrium(float(lam), p)
        rows.append(
            {
                "lambda": float(lam),
                "M_star": x_star[0],
                "S_star": x_star[1],
                "mu_max": dominant_eigenvalue(float(lam), p),
                "relaxation_time": relaxation_time(float(lam), p=p),
                "finite_time_separation_rate": finite_time_separation_rate(float(lam), p=p),
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
