#!/usr/bin/env python3
"""Generate manuscript figures and numerical summaries.

Examples
--------
Quick test run:
    python scripts/make_figures.py --out outputs --n-reps 100 --no-numba

Paper-style run:
    python scripts/make_figures.py --out outputs --n-reps 1000 --use-numba
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from pd_resilience_model.analysis import (
    deterministic_summary,
    empirical_survival_curve,
    monte_carlo_escape_statistics,
)
from pd_resilience_model.model import drift, equilibrium, simulate_stochastic


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def make_figure2(out: Path, lambda_grid: np.ndarray) -> pd.DataFrame:
    summary = deterministic_summary(lambda_grid)

    lam_low = 1.0
    x_star_low = equilibrium(lam_low)
    x0 = np.array([0.5, 0.8])
    t_eval = np.linspace(0, 100, 2001)
    sol = solve_ivp(lambda t, x: drift(t, x, lam_low), (0, 100), x0, t_eval=t_eval)

    fig, axes = plt.subplots(3, 1, figsize=(7, 10))

    axes[0].plot(sol.t, sol.y[0], label="M (mitochondrial reserve)")
    axes[0].plot(sol.t, sol.y[1], label="S (stress burden)")
    axes[0].axhline(x_star_low[0], linestyle="--", linewidth=1)
    axes[0].axhline(x_star_low[1], linestyle="--", linewidth=1)
    axes[0].set_xlabel("Time (a.u.)")
    axes[0].set_ylabel("State variables")
    axes[0].set_title("A) Convergence to survival equilibrium (λ = 1.00)")
    axes[0].legend(loc="best")

    axes[1].plot(summary["lambda"], summary["M_star"], marker="o", ms=3, label=r"$M^*$")
    axes[1].plot(summary["lambda"], summary["S_star"], marker="o", ms=3, label=r"$S^*$")
    axes[1].set_xlabel("λ (chronic stress)")
    axes[1].set_ylabel("Equilibrium values")
    axes[1].set_title("B) Equilibrium deformation under chronic stress")
    axes[1].legend(loc="best")

    axes[2].plot(summary["lambda"], summary["mu_max"], marker="o", ms=3)
    axes[2].axhline(0, linestyle="--", linewidth=1)
    axes[2].set_xlabel("λ (chronic stress)")
    axes[2].set_ylabel(r"$\mu_{\max}(\lambda)$")
    axes[2].set_title("C) Dominant Jacobian eigenvalue")

    fig.tight_layout()
    fig.savefig(out / "Figure2_deterministic.png", bbox_inches="tight")
    fig.savefig(out / "Figure2_deterministic.pdf", bbox_inches="tight")
    plt.close(fig)

    summary.to_csv(out / "deterministic_summary.csv", index=False)
    return summary


def make_figure3(out: Path, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(summary["lambda"], summary["relaxation_time"], marker="o", ms=4)
    axes[0].set_xlabel("λ (chronic stress)")
    axes[0].set_ylabel("Relaxation time (a.u.)")
    axes[0].set_title("A) Relaxation time")

    axes[1].plot(summary["lambda"], summary["finite_time_separation_rate"], marker="o", ms=4)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("λ (chronic stress)")
    axes[1].set_ylabel("Finite-time separation rate")
    axes[1].set_title("B) Finite-time contraction / separation")

    fig.tight_layout()
    fig.savefig(out / "Figure3_relaxation_separation.png", bbox_inches="tight")
    fig.savefig(out / "Figure3_relaxation_separation.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figure4(out: Path, S_c: float, M_c: float) -> None:
    lambda_low = 1.0
    lambda_high = 1.40
    x0_low = equilibrium(lambda_low)
    x0_high = equilibrium(lambda_high)

    t_low, M_low, S_low = simulate_stochastic(lambda_low, x0_low, T=200, dt=0.02, seed=1)
    t_high, M_high, S_high = simulate_stochastic(lambda_high, x0_high, T=200, dt=0.02, seed=2)

    escape_idx = np.where((S_high > S_c) & (M_high < M_c))[0]
    t_escape = t_high[escape_idx[0]] if len(escape_idx) else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(t_low, M_low, label="M (mitochondrial reserve)")
    axes[0].plot(t_low, S_low, label="S (stress burden)")
    axes[0].axhline(x0_low[0], linestyle="--", linewidth=1)
    axes[0].axhline(x0_low[1], linestyle="--", linewidth=1)
    axes[0].set_xlabel("Time (a.u.)")
    axes[0].set_ylabel("State variables")
    axes[0].set_title("A) Confined fluctuations (λ = 1.00)")
    axes[0].legend(loc="best")

    axes[1].plot(t_high, M_high, label="M (mitochondrial reserve)")
    axes[1].plot(t_high, S_high, label="S (stress burden)")
    axes[1].axhline(S_c, linestyle="--", linewidth=1, label=r"$S=S_c$")
    axes[1].axhline(M_c, linestyle=":", linewidth=1, label=r"$M=M_c$")
    if t_escape is not None:
        axes[1].axvline(t_escape, linestyle="-.", linewidth=1, label="First passage")
    axes[1].set_xlabel("Time (a.u.)")
    axes[1].set_title("B) Large excursions near weak stability (λ = 1.40)")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out / "Figure4_stochastic_trajectories.png", bbox_inches="tight")
    fig.savefig(out / "Figure4_stochastic_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def run_monte_carlo(lambda_values: np.ndarray, n_reps: int, T: float, dt: float, S_c: float, M_c: float, use_numba: bool):
    if use_numba:
        from pd_resilience_model.numba_mc import monte_carlo_escape_times_numba

        escape_times_matrix = monte_carlo_escape_times_numba(lambda_values, n_reps, T, dt, S_c, M_c)
        rows = []
        for i, lam in enumerate(lambda_values):
            times = escape_times_matrix[i, :]
            escaped = ~np.isnan(times)
            n_escaped = int(np.sum(escaped))
            rows.append(
                {
                    "lambda": float(lam),
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
        escape_times = {float(lambda_values[i]): escape_times_matrix[i, :] for i in range(len(lambda_values))}
        return escape_times, pd.DataFrame(rows)

    return monte_carlo_escape_statistics(lambda_values, n_reps=n_reps, T=T, dt=dt, S_c=S_c, M_c=M_c)


def make_figure5(out: Path, lambda_values: np.ndarray, n_reps: int, T: float, dt: float, S_c: float, M_c: float, use_numba: bool) -> None:
    escape_times, summary = run_monte_carlo(lambda_values, n_reps, T, dt, S_c, M_c, use_numba)
    summary.to_csv(out / "monte_carlo_escape_summary.csv", index=False)

    selected_lambdas = [1.37, 1.39, 1.41, 1.43, 1.45]
    t_grid = np.linspace(0, T, 800)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for lam_sel in selected_lambdas:
        lam = float(lambda_values[np.argmin(np.abs(lambda_values - lam_sel))])
        surv = empirical_survival_curve(escape_times[lam], t_grid)
        axes[0].plot(t_grid, surv, linewidth=2, label=fr"$\lambda = {lam:.2f}$")

    axes[0].set_xlabel("Time (a.u.)")
    axes[0].set_ylabel("Survival probability")
    axes[0].set_title("A) Survival curves")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="best", frameon=True)

    plot_df = summary[(summary["lambda"] >= 1.25) & (summary["lambda"] <= 1.45)]
    axes[1].plot(plot_df["lambda"], plot_df["escape_probability"], marker="o", linewidth=2)
    axes[1].set_xlabel(r"$\lambda$ (chronic load)")
    axes[1].set_ylabel(fr"Escape probability by $T = {T:.0f}$")
    axes[1].set_title("B) Escape probability")
    axes[1].set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(out / "Figure5_monte_carlo_survival_analysis.png", bbox_inches="tight")
    fig.savefig(out / "Figure5_monte_carlo_survival_analysis.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--n-reps", type=int, default=1000, help="Monte Carlo trajectories per lambda")
    parser.add_argument("--T-survival", type=float, default=500.0, help="Monte Carlo time horizon")
    parser.add_argument("--dt", type=float, default=0.02, help="Euler-Maruyama time step")
    parser.add_argument("--S-c", type=float, default=3.5, help="Stress threshold")
    parser.add_argument("--M-c", type=float, default=0.15, help="Mitochondrial reserve threshold")
    parser.add_argument("--use-numba", action="store_true", help="Use Numba-accelerated Monte Carlo")
    parser.add_argument("--no-numba", action="store_false", dest="use_numba")
    parser.set_defaults(use_numba=True)
    args = parser.parse_args()

    configure_plots()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lambda_grid = np.linspace(0.6, 1.45, 22)
    survival_grid = np.linspace(1.25, 1.50, 26)

    print("Generating Figure 2 and deterministic summary...")
    summary = make_figure2(out, lambda_grid)
    print("Generating Figure 3...")
    make_figure3(out, summary)
    print("Generating Figure 4...")
    make_figure4(out, args.S_c, args.M_c)
    print("Generating Figure 5 and Monte Carlo summary...")
    make_figure5(out, survival_grid, args.n_reps, args.T_survival, args.dt, args.S_c, args.M_c, args.use_numba)
    print(f"Done. Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
