"""Minimal resilience-stress model for dopaminergic neuron vulnerability.

This module contains the deterministic drift, analytic equilibrium, Jacobian,
noise amplitudes, and stochastic first-passage utilities used to generate the
figures in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import eigvals


@dataclass(frozen=True)
class ModelParams:
    """Parameter set for the minimal two-variable model."""

    a_M: float = 0.5
    b_M: float = 0.4
    M_rest: float = 1.0
    a_S: float = 0.3
    b_S: float = 0.6
    sigma_M_const: float = 0.003
    sigma_S0: float = 0.015
    sigma_S1: float = 0.03


def drift(t: float, x: np.ndarray, lam: float, p: ModelParams = ModelParams()) -> np.ndarray:
    """Deterministic drift F(M,S;lambda), G(M,S;lambda)."""
    M, S = x
    dM = p.a_M * (p.M_rest - M) - p.b_M * lam * S * M
    dS = p.a_S * lam - p.b_S * M * S
    return np.array([dM, dS], dtype=float)


def equilibrium(lam: float, p: ModelParams = ModelParams()) -> np.ndarray:
    """Analytic positive survival equilibrium for a given chronic load lambda."""
    M_star = p.M_rest - (p.b_M * p.a_S / (p.a_M * p.b_S)) * lam**2
    if M_star <= 0:
        return np.array([np.nan, np.nan], dtype=float)
    S_star = (p.a_S * lam) / (p.b_S * M_star)
    return np.array([M_star, S_star], dtype=float)


def jacobian(M: float, S: float, lam: float, p: ModelParams = ModelParams()) -> np.ndarray:
    """Analytic Jacobian of the deterministic drift."""
    return np.array(
        [
            [-p.a_M - p.b_M * lam * S, -p.b_M * lam * M],
            [-p.b_S * S, -p.b_S * M],
        ],
        dtype=float,
    )


def dominant_eigenvalue(lam: float, p: ModelParams = ModelParams()) -> float:
    """Largest real part of the Jacobian eigenvalues at equilibrium."""
    x_star = equilibrium(lam, p)
    if np.any(np.isnan(x_star)):
        return np.nan
    M_star, S_star = x_star
    return float(np.max(np.real(eigvals(jacobian(M_star, S_star, lam, p)))))


def sigma_M(M: float, S: float, lam: float, p: ModelParams = ModelParams()) -> float:
    """Noise amplitude for mitochondrial reserve."""
    return p.sigma_M_const


def sigma_S(M: float, S: float, lam: float, p: ModelParams = ModelParams()) -> float:
    """State- and load-dependent noise amplitude for composite stress."""
    return p.sigma_S0 + p.sigma_S1 * lam * max(S, 0.0)


def simulate_stochastic(
    lam: float,
    x0: np.ndarray,
    T: float = 200.0,
    dt: float = 0.02,
    seed: Optional[int] = None,
    p: ModelParams = ModelParams(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one stochastic trajectory with Euler-Maruyama integration."""
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t = np.linspace(0.0, T, n_steps + 1)
    M = np.empty(n_steps + 1, dtype=float)
    S = np.empty(n_steps + 1, dtype=float)
    M[0], S[0] = x0

    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        Mk, Sk = M[k], S[k]
        dM, dS = drift(t[k], np.array([Mk, Sk]), lam, p)
        M_new = Mk + dM * dt + sigma_M(Mk, Sk, lam, p) * sqrt_dt * rng.normal()
        S_new = Sk + dS * dt + sigma_S(Mk, Sk, lam, p) * sqrt_dt * rng.normal()

        # Reflecting boundaries prevent unphysical negative values.
        M[k + 1] = -M_new if M_new < 0.0 else M_new
        S[k + 1] = -S_new if S_new < 0.0 else S_new

    return t, M, S


def first_passage_time(
    lam: float,
    T: float = 500.0,
    dt: float = 0.02,
    S_c: float = 3.5,
    M_c: float = 0.15,
    seed: Optional[int] = None,
    p: ModelParams = ModelParams(),
) -> float:
    """Return first-passage time into F={S>S_c and M<M_c}; NaN if no escape."""
    rng = np.random.default_rng(seed)
    x0 = equilibrium(lam, p)
    if np.any(np.isnan(x0)):
        return np.nan

    M, S = float(x0[0]), float(x0[1])
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    for k in range(n_steps):
        dM, dS = drift(k * dt, np.array([M, S]), lam, p)
        M_new = M + dM * dt + sigma_M(M, S, lam, p) * sqrt_dt * rng.normal()
        S_new = S + dS * dt + sigma_S(M, S, lam, p) * sqrt_dt * rng.normal()

        M = -M_new if M_new < 0.0 else M_new
        S = -S_new if S_new < 0.0 else S_new

        if (S > S_c) and (M < M_c):
            return (k + 1) * dt

    return np.nan
