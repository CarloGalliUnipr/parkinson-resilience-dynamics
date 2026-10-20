#!/usr/bin/env python3
"""Generate manuscript figures, supplementary figures and numerical summaries.

Examples
--------
Quick test run:
    python scripts/make_figures.py --out outputs_test --n-reps 100

Paper-style run:
    python scripts/make_figures.py --out outputs --n-reps 1000

Random numbers: NumPy ``default_rng`` (PCG64). Figure 4 uses seeds 1 (lambda = 1.00) and
2 (lambda = 1.40); Figures 3B-C, 5 and S1 use the base seed (default 20240529), one stream per
call shared by all loads; Figure 6B uses the base seed + 0, 1, 2 for lambda = 1.00, 1.37, 1.43.
Each S1 configuration reuses the base seed (common random numbers across configurations).

All Monte Carlo simulations use the seeded, vectorized Euler-Maruyama integrator in
``pd_resilience_model.analysis.monte_carlo_first_entry_times`` (Ito interpretation), so
results are reproducible bit-for-bit for a given seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp

from pd_resilience_model.analysis import (
    deterministic_summary,
    lna_fluctuations,
    monte_carlo_first_entry_times,
    simulated_fluctuations,
)
from pd_resilience_model.model import ModelParams, drift, equilibrium, jacobian, simulate_stochastic

# Categorical slots (fixed order) and a single-hue sequential ramp for ordered lambda values.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
F_FILL = "#e6e5e1"

LOAD_LABEL = r"$\lambda$ (sustained load)"


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
            "axes.prop_cycle": cycler(color=[BLUE, ORANGE, AQUA]),
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def first_entry_summary(lams: np.ndarray, tau: np.ndarray, T: float, **meta) -> pd.DataFrame:
    rows = []
    for lam, t in zip(lams, tau):
        entered = ~np.isnan(t)
        n = t.size
        p = entered.mean()
        rows.append(
            {
                "lambda": float(lam),
                "n_reps": n,
                "T": T,
                **meta,
                "n_entered": int(entered.sum()),
                "p_entry": p,
                # Normal-approximation (Wald) binomial 95% interval, as shown in Figure 5B.
                "p_entry_ci95_halfwidth": 1.96 * np.sqrt(p * (1 - p) / n),
                "median_entry_time": np.nanmedian(t) if entered.any() else np.nan,
                "mean_entry_time_given_entry": np.nanmean(t) if entered.any() else np.nan,
                "cv_entry_time_given_entry": (np.nanstd(t, ddof=1) / np.nanmean(t)) if entered.sum() > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Figure 2
def make_figure2(out: Path, lambda_grid: np.ndarray) -> pd.DataFrame:
    summary = deterministic_summary(lambda_grid)

    lam_low = 1.0
    x_star_low = equilibrium(lam_low)
    x0 = np.array([0.5, 0.8])
    t_eval = np.linspace(0, 100, 2001)
    sol = solve_ivp(lambda t, x: drift(t, x, lam_low), (0, 100), x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    fig, axes = plt.subplots(3, 1, figsize=(7, 10))

    axes[0].plot(sol.t, sol.y[0], label="M (mitochondrial reserve)")
    axes[0].plot(sol.t, sol.y[1], label="S (stress burden)")
    axes[0].axhline(x_star_low[0], linestyle="--", linewidth=1, color=INK2)
    axes[0].axhline(x_star_low[1], linestyle="--", linewidth=1, color=INK2)
    axes[0].set_xlabel("Time (a.u.)")
    axes[0].set_ylabel("State variables")
    axes[0].set_title("A) Convergence to the survival equilibrium (λ = 1.00)")
    axes[0].legend(loc="best")

    axes[1].plot(summary["lambda"], summary["M_star"], marker="o", ms=4, label=r"$M^*$")
    axes[1].plot(summary["lambda"], summary["S_star"], marker="s", ms=4, label=r"$S^*$")
    axes[1].set_xlabel(LOAD_LABEL)
    axes[1].set_ylabel("Equilibrium values")
    axes[1].set_title("B) Equilibrium shift under sustained load")
    axes[1].legend(loc="best")

    axes[2].plot(summary["lambda"], summary["mu_max"], marker="o", ms=4)
    axes[2].axhline(0, linestyle="--", linewidth=1, color=INK2)
    axes[2].set_xlabel(LOAD_LABEL)
    axes[2].set_ylabel(r"$\mu_{\max}(\lambda)$")
    axes[2].set_title("C) Dominant Jacobian eigenvalue")

    fig.tight_layout()
    save(fig, out, "Figure2_deterministic")
    summary.to_csv(out / "deterministic_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------- Figure 3
def make_figure3(out: Path, summary: pd.DataFrame, seed: int) -> None:
    lams = summary["lambda"].to_numpy()
    fl = simulated_fluctuations(lams, seed=seed)
    fl["var_S_lna"] = [lna_fluctuations(l)["var_S"] for l in lams]
    fl["acf_S_lna"] = [lna_fluctuations(l)["acf_S"] for l in lams]
    fl["var_S_lna_additive_0.10"] = [lna_fluctuations(l, sigma_S_const=0.10)["var_S"] for l in lams]
    fl["acf_S_lna_additive_0.10"] = [lna_fluctuations(l, sigma_S_const=0.10)["acf_S"] for l in lams]
    fl.to_csv(out / "Figure3_fluctuation_statistics.csv", index=False)
    lam_fine = np.linspace(lams.min(), lams.max(), 200)
    lna = pd.DataFrame([lna_fluctuations(l) for l in lam_fine])
    lna_add = pd.DataFrame([lna_fluctuations(l, sigma_S_const=0.10) for l in lam_fine])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.plot(lams, summary["relaxation_time"], marker="o", ms=5, color=BLUE,
            label=r"$\tau_{relax}$ (finite perturbation, $e^{-1}$ criterion)")
    ax.plot(lams, summary["linear_relaxation_scale"], linestyle="--", linewidth=1.5, color=INK2,
            label=r"$-1/\mu_{\max}$ (linearization)")
    ax.set_xlabel(LOAD_LABEL)
    ax.set_ylabel("Relaxation time (a.u.)")
    ax.set_title("A) Recovery from a small perturbation")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.semilogy(lams, fl["var_S_sim"], "o", ms=5, color=BLUE, label="simulation (state-dependent noise)")
    ax.semilogy(lam_fine, lna["var_S"], color=INK, linewidth=1.5, label="linear noise approximation")
    ax.semilogy(lam_fine, lna_add["var_S"], color=ORANGE, linewidth=1.5, linestyle="--",
                label=r"linear noise approx., additive $\sigma_S$ = 0.10")
    ax.set_xlabel(LOAD_LABEL)
    ax.set_ylabel(r"Stationary variance of $S$")
    ax.set_title("B) Variance of stress fluctuations")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[2]
    ax.plot(lams, fl["acf_S_sim"], "o", ms=5, color=BLUE, label="simulation (state-dependent noise)")
    ax.plot(lam_fine, lna["acf_S"], color=INK, linewidth=1.5,
            label="linear noise approximation\n(identical for additive noise)")
    ax.set_xlabel(LOAD_LABEL)
    ax.set_ylabel(r"Lag-1 autocorrelation of $S$")
    ax.set_title("C) Autocorrelation of stress fluctuations")
    ax.set_ylim(0.6, 1.0)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    save(fig, out, "Figure3_slowing_down")


# ---------------------------------------------------------------- Figure 4
def make_figure4(out: Path, S_c: float, M_c: float) -> None:
    lambda_low, lambda_high = 1.0, 1.40
    x0_low, x0_high = equilibrium(lambda_low), equilibrium(lambda_high)
    t_low, M_low, S_low = simulate_stochastic(lambda_low, x0_low, T=200, dt=0.02, seed=1)
    t_high, M_high, S_high = simulate_stochastic(lambda_high, x0_high, T=200, dt=0.02, seed=2)

    entry_idx = np.where((S_high > S_c) & (M_high < M_c))[0]
    t_entry = t_high[entry_idx[0]] if len(entry_idx) else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_low, M_low, label="M (mitochondrial reserve)", linewidth=1.2)
    axes[0].plot(t_low, S_low, label="S (stress burden)", linewidth=1.2)
    axes[0].axhline(x0_low[0], linestyle="--", linewidth=1, color=INK2)
    axes[0].axhline(x0_low[1], linestyle="--", linewidth=1, color=INK2)
    axes[0].set_xlabel("Time (a.u.)")
    axes[0].set_ylabel("State variables")
    axes[0].set_title("A) Confined fluctuations (λ = 1.00)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)

    axes[1].plot(t_high, M_high, label="M (mitochondrial reserve)", linewidth=1.2)
    axes[1].plot(t_high, S_high, label="S (stress burden)", linewidth=1.2)
    axes[1].axhline(S_c, linestyle="--", linewidth=1, color=INK2, label=r"$S=S_c$")
    axes[1].axhline(M_c, linestyle=":", linewidth=1, color=INK2, label=r"$M=M_c$")
    if t_entry is not None:
        axes[1].axvline(t_entry, linestyle="-.", linewidth=1, color=INK, label=r"First entry into $\mathcal{F}$")
    axes[1].set_xlabel("Time (a.u.)")
    axes[1].set_title("B) Large excursions at reduced resilience (λ = 1.40)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    fig.tight_layout()
    save(fig, out, "Figure4_stochastic_trajectories")


# ---------------------------------------------------------------- Figure 5
def make_figure5(out: Path, n_reps: int, T: float, dt: float, S_c: float, M_c: float, seed: int) -> None:
    lams = np.round(np.arange(1.25, 1.4501, 0.01), 2)
    tau = monte_carlo_first_entry_times(lams, n_reps=n_reps, T=T, dt=dt, S_c=S_c, M_c=M_c, seed=seed)
    summary = first_entry_summary(lams, tau, T, S_c=S_c, M_c=M_c, noise="state-dependent (baseline)")
    summary.to_csv(out / "monte_carlo_first_entry_summary.csv", index=False)

    selected = [1.37, 1.39, 1.41, 1.43, 1.45]
    t_grid = np.linspace(0, T, 800)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for color, lam_sel in zip(SEQ_BLUE, selected):
        i = int(np.argmin(np.abs(lams - lam_sel)))
        t_obs = np.where(np.isnan(tau[i]), np.inf, tau[i])
        surv = np.array([np.mean(t_obs > t) for t in t_grid])
        axes[0].plot(t_grid, surv, color=color, label=fr"$\lambda = {lams[i]:.2f}$")
    axes[0].set_xlabel("Time (a.u.)")
    axes[0].set_ylabel(r"$P(\tau > t)$")
    axes[0].set_title(r"A) Fraction of cells not yet in $\mathcal{F}$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="best")

    axes[1].errorbar(summary["lambda"], summary["p_entry"], yerr=summary["p_entry_ci95_halfwidth"],
                     marker="o", ms=5, capsize=2, color=BLUE)
    axes[1].set_xlabel(LOAD_LABEL)
    axes[1].set_ylabel(fr"$P(\tau < T)$, $T = {T:.0f}$")
    axes[1].set_title(r"B) Probability of entering $\mathcal{F}$ within the window")
    axes[1].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    save(fig, out, "Figure5_monte_carlo_first_entry")


# ---------------------------------------------------------------- Supplementary Figure S1
def make_figure_s1(out: Path, n_reps: int, dt: float, seed: int) -> None:
    lams = np.round(np.arange(1.20, 1.5001, 0.01), 2)
    base = ModelParams()
    configs = {
        "A": [
            ("T = 50", dict(T=50.0, S_c=3.5, M_c=0.15, p=base)),
            ("T = 500 (baseline)", dict(T=500.0, S_c=3.5, M_c=0.15, p=base)),
            ("T = 2000", dict(T=2000.0, S_c=3.5, M_c=0.15, p=base)),
        ],
        "B": [
            (r"$S_c$ = 2.5, $M_c$ = 0.25", dict(T=500.0, S_c=2.5, M_c=0.25, p=base)),
            (r"$S_c$ = 3.5, $M_c$ = 0.15 (baseline)", dict(T=500.0, S_c=3.5, M_c=0.15, p=base)),
            (r"$S_c$ = 4.5, $M_c$ = 0.12", dict(T=500.0, S_c=4.5, M_c=0.12, p=base)),
        ],
        "C": [
            (r"state-dependent $\sigma_S$ (baseline)", dict(T=500.0, S_c=3.5, M_c=0.15, p=base)),
            (r"additive $\sigma_S$ = 0.10", dict(T=500.0, S_c=3.5, M_c=0.15,
                                                p=ModelParams(sigma_S0=0.10, sigma_S1=0.0))),
            (r"additive $\sigma_S$ = 0.15", dict(T=500.0, S_c=3.5, M_c=0.15,
                                                p=ModelParams(sigma_S0=0.15, sigma_S1=0.0))),
        ],
    }
    titles = {
        "A": "A) Observation window",
        "B": r"B) Boundaries of $\mathcal{F}$",
        "C": "C) Noise structure",
    }
    styles = [(BLUE, "o", "-"), (ORANGE, "s", "--"), (AQUA, "^", ":")]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, (key, cfgs) in zip(axes, configs.items()):
        for (label, cfg), (color, marker, ls) in zip(cfgs, styles):
            tau = monte_carlo_first_entry_times(lams, n_reps=n_reps, dt=dt, seed=seed, **cfg)
            df = first_entry_summary(lams, tau, cfg["T"], S_c=cfg["S_c"], M_c=cfg["M_c"],
                                     sigma_S0=cfg["p"].sigma_S0, sigma_S1=cfg["p"].sigma_S1, panel=key)
            rows.append(df)
            ax.plot(df["lambda"], df["p_entry"], color=color, marker=marker, ms=5, linestyle=ls, label=label)
            if key == "B":
                # load at which the deterministic equilibrium itself enters F
                p = cfg["p"]
                # Both conditions defining F must hold: M* < M_c and S* > S_c.
                k = p.b_M * p.a_S / (p.a_M * p.b_S)
                lam_M = np.sqrt((p.M_rest - cfg["M_c"]) / k)
                S_c_cfg = cfg["S_c"]
                # S*(lam) = S_c  <=>  k S_c lam^2 + (a_S/b_S) lam - M_rest S_c = 0
                a2, a1, a0 = k * S_c_cfg, p.a_S / p.b_S, -p.M_rest * S_c_cfg
                lam_S = (-a1 + np.sqrt(a1**2 - 4 * a2 * a0)) / (2 * a2)
                lam_in = max(lam_M, lam_S)
                ax.axvline(lam_in, color=color, linewidth=1, linestyle=ls, alpha=0.8)
        ax.set_title(titles[key])
        ax.set_xlabel(LOAD_LABEL)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel(r"$P(\tau < T)$")
    axes[1].plot([], [], color=INK2, linewidth=1, label=r"load at which $x^*(\lambda)$ enters $\mathcal{F}$")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, out, "FigureS1_sensitivity")
    pd.concat(rows).to_csv(out / "FigureS1_sensitivity_summary.csv", index=False)


# ---------------------------------------------------------------- Figure 6 (panels A-B)
def make_figure6ab(out: Path, S_c: float, M_c: float, n_cells: int, seed: int) -> None:
    p = ModelParams()
    lam_c = np.sqrt(p.a_M * p.b_S * p.M_rest / (p.b_M * p.a_S))
    lam_path = np.linspace(0.6, 1.50, 400)
    eq_path = np.array([equilibrium(l) for l in lam_path])
    marks = [1.00, 1.37, 1.43]

    for with_c in (False, True):
        _figure6(out, p, lam_c, eq_path, marks, S_c, M_c, n_cells, seed, with_c)


def _figure6(out, p, lam_c, eq_path, marks, S_c, M_c, n_cells, seed, with_c):
    if with_c:
        fig = plt.figure(figsize=(15, 10.6))
        outer = fig.add_gridspec(2, 1, height_ratios=[5.2, 5.0], hspace=0.28)
        gs = outer[0].subgridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.28)
    else:
        fig = plt.figure(figsize=(15, 5.2))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.28)
    axA = fig.add_subplot(gs[0, 0])
    axB = [fig.add_subplot(gs[0, i]) for i in (1, 2, 3)]

    xlim, ylim = (0.0, 0.9), (0.0, 7.0)

    def draw_F(ax):
        ax.add_patch(Rectangle((xlim[0], S_c), M_c - xlim[0], ylim[1] - S_c, facecolor=F_FILL,
                               edgecolor=INK2, hatch="///", linewidth=1, zorder=0))
        ax.text(0.012, ylim[1] - 0.25, r"$\mathcal{F}$", fontsize=13, va="top", color=INK)

    # Panel A: equilibrium path and slow direction
    draw_F(axA)
    axA.plot(eq_path[:, 0], eq_path[:, 1], color=INK, linewidth=1.8, zorder=2)
    for lam in [0.6, 1.0, 1.2, 1.37, 1.43]:
        M, S = equilibrium(lam)
        J = jacobian(M, S, lam)
        w, V = np.linalg.eig(J)
        v = np.real(V[:, np.argmax(np.real(w))])
        v = v / np.hypot(v[0] / (xlim[1] - xlim[0]), v[1] / (ylim[1] - ylim[0])) * 0.07
        axA.annotate("", xy=(M + v[0], S + v[1]), xytext=(M - v[0], S - v[1]),
                     arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6), zorder=3)
        axA.plot(M, S, "o", color=INK, ms=6, zorder=4, markeredgecolor="white", markeredgewidth=1.2)
        offset = (-12, -16) if lam == 0.6 else (12, 2)
        axA.annotate(fr"$\lambda$ = {lam:.2f}", (M, S), textcoords="offset points", xytext=offset,
                     fontsize=9, color=INK2, ha="left" if lam != 0.6 else "right")
    axA.text(0.88, 6.75, fr"$x^*(\lambda)\to(0,\infty)$ as $\lambda\to\lambda_c\approx{lam_c:.2f}$",
             fontsize=9, ha="right", va="top", color=INK2)
    axA.plot([], [], color=INK, label=r"survival equilibrium $x^*(\lambda)$")
    axA.plot([], [], color=BLUE, label="slow direction (dominant eigenvector)")
    axA.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=9)
    axA.set_xlim(*xlim)
    axA.set_ylim(*ylim)
    axA.set_xlabel("Mitochondrial reserve M")
    axA.set_ylabel("Composite stress S")
    axA.set_title("A) Equilibrium path and viability region")

    # Panels B1-B3: population of independent cells (no absorption) at t = 200
    rng_seed = seed
    for ax, lam, tag in zip(axB, marks, ["B1", "B2", "B3"]):
        draw_F(ax)
        rng = np.random.default_rng(rng_seed)
        rng_seed += 1
        M0, S0 = equilibrium(lam)
        M = np.full(n_cells, M0)
        S = np.full(n_cells, S0)
        ever = np.zeros(n_cells, bool)
        dt, T = 0.02, 200.0
        sq = np.sqrt(dt)
        traj = [(M[0], S[0])]
        for k in range(int(T / dt)):
            dM = p.a_M * (p.M_rest - M) - p.b_M * lam * S * M
            dS = p.a_S * lam - p.b_S * M * S
            sS = p.sigma_S0 + p.sigma_S1 * lam * np.maximum(S, 0)
            M = np.abs(M + dM * dt + p.sigma_M_const * sq * rng.standard_normal(n_cells))
            S = np.abs(S + dS * dt + sS * sq * rng.standard_normal(n_cells))
            ever |= (S > S_c) & (M < M_c)
            if k % 10 == 0:
                traj.append((M[0], S[0]))
        traj = np.array(traj)
        ax.scatter(M, S, s=6, color=INK2, alpha=0.35, linewidths=0, zorder=1)
        ax.plot(traj[:, 0], traj[:, 1], color=BLUE, linewidth=0.8, alpha=0.9, zorder=2)
        ax.plot(M0, S0, "o", color=INK, ms=7, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Mitochondrial reserve M")
        ax.set_title(fr"{tag}) $\lambda$ = {lam:.2f}")
        ax.text(0.97, 0.97, fr"entered $\mathcal{{F}}$ by $t$=200: {100 * ever.mean():.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
        if tag != "B1":
            ax.set_yticklabels([])
    axB[0].scatter([], [], s=12, color=INK2, alpha=0.6, label=f"{n_cells} cells at t = 200")
    axB[0].plot([], [], color=BLUE, linewidth=1, label="one sample path")
    axB[0].legend(loc="center right", fontsize=8.5)
    if with_c:
        draw_panel_c(fig.add_subplot(outer[1]))
        save(fig, out, "Figure6_full")
    else:
        save(fig, out, "Figure6AB_state_space")


def draw_panel_c(ax) -> None:
    """Schematic mapping of PD-associated processes onto the model (Figure 6C)."""
    from matplotlib.patches import FancyBboxPatch, Polygon, Circle

    ax.set_xlim(0, 21.6)
    ax.set_ylim(0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("C) PD-associated processes mapped onto model parameters", loc="left")
    LOAD_FILL, LOAD_EDGE = "#dce8f8", "#2a78d6"
    REST_FILL, REST_EDGE = "#d9f0e6", "#148a5f"
    BOX_FILL, BOX_EDGE = "#f1f0ec", "#8f8e89"

    def box(x, y, w, h, text, fc=BOX_FILL, ec=BOX_EDGE, fs=10.5, weight="normal"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight)

    def arrow(x0, y0, x1, y1, color=INK2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, shrinkA=0, shrinkB=0,
                                    mutation_scale=12))

    processes = [
        ("α-synuclein misfolding", "L"),
        ("Ca²⁺-dependent\npacemaking", "L"),
        ("Dopamine oxidation", "LR"),
        ("Neuromelanin-associated\nchanges", "LR"),
        ("Iron dysregulation", "LR"),
        ("Glutathione limitation", "R"),
        ("Glial/inflammatory\nstress", "R"),
    ]
    bx, bw, bh = 2.4, 4.4, 0.8
    ys = np.linspace(6.6, 0.55, len(processes))
    yL, yR = 4.85, 2.15
    tri_x0, tri_x1, tri_h = 7.9, 11.7, 2.0
    for (label, tgt), y in zip(processes, ys):
        box(bx, y, bw, bh, label, fs=10)
        for t in tgt:
            ty = yL if t == "L" else yR
            arrow(bx + bw / 2, y, tri_x0, ty + float(np.clip((y - ty) * 0.12, -0.55, 0.55)),
                  color=LOAD_EDGE if t == "L" else REST_EDGE)
    for yc, fc, ec, txt in [(yL, LOAD_FILL, LOAD_EDGE, "↑ Effective\nsustained load"),
                            (yR, REST_FILL, REST_EDGE, "↓ Restorative\ncapacity")]:
        ax.add_patch(Polygon([[tri_x0, yc + tri_h / 2], [tri_x0, yc - tri_h / 2], [tri_x1, yc]],
                             closed=True, fc=fc, ec=ec, lw=1.4))
        ax.text(tri_x0 + 0.15, yc, txt, ha="left", va="center", fontsize=10, weight="bold", color=INK)
    cx, r = 13.9, 1.2
    for yc, fc, ec, txt in [(yL, LOAD_FILL, LOAD_EDGE, "λ: effective\nsustained\nload"),
                            (yR, REST_FILL, REST_EDGE, "Recovery rate\n(|μ$_{\\max}$|)")]:
        ax.add_patch(Circle((cx, yc), r, fc=fc, ec=ec, lw=1.4))
        ax.text(cx, yc, txt, ha="center", va="center", fontsize=10.5, color=INK)
        arrow(tri_x1, yc, cx - r, yc, color=ec)
    for yc, fc, ec, txt in [(yL, LOAD_FILL, LOAD_EDGE, "Prolonged\ncompensation"),
                            (yR, REST_FILL, REST_EDGE, "Probability of\nentering $\\mathcal{F}$")]:
        box(18.4, yc, 3.6, 1.1, txt, fc=fc, ec=ec, fs=10.5)
        arrow(cx + r, yc, 18.4 - 1.8, yc, color=ec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--n-reps", type=int, default=1000, help="Monte Carlo trajectories per lambda")
    parser.add_argument("--T-survival", type=float, default=500.0, help="Monte Carlo time horizon")
    parser.add_argument("--dt", type=float, default=0.02, help="Euler-Maruyama time step")
    parser.add_argument("--S-c", type=float, default=3.5, help="Stress threshold")
    parser.add_argument("--M-c", type=float, default=0.15, help="Mitochondrial reserve threshold")
    parser.add_argument("--seed", type=int, default=20240529, help="Base random seed")
    parser.add_argument("--skip-supplementary", action="store_true")
    args = parser.parse_args()

    configure_plots()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lambda_grid = np.linspace(0.6, 1.45, 22)
    print("Figure 2 and deterministic summary...")
    summary = make_figure2(out, lambda_grid)
    print("Figure 3...")
    make_figure3(out, summary, args.seed)
    print("Figure 4...")
    make_figure4(out, args.S_c, args.M_c)
    print("Figure 5 and Monte Carlo summary...")
    make_figure5(out, args.n_reps, args.T_survival, args.dt, args.S_c, args.M_c, args.seed)
    print("Figure 6 panels A-B...")
    make_figure6ab(out, args.S_c, args.M_c, n_cells=args.n_reps, seed=args.seed)
    if not args.skip_supplementary:
        print("Supplementary Figure S1 (sensitivity)...")
        make_figure_s1(out, args.n_reps, args.dt, args.seed)
    print(f"Done. Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
