"""Minimal stochastic resilience-stress model package."""

__version__ = "0.2.0"

from .model import ModelParams, drift, equilibrium, jacobian, dominant_eigenvalue, simulate_stochastic, first_passage_time

__all__ = [
    "ModelParams",
    "drift",
    "equilibrium",
    "jacobian",
    "dominant_eigenvalue",
    "simulate_stochastic",
    "first_passage_time",
]
