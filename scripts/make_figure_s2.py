#!/usr/bin/env python3
"""Matched-relaxation-rate first-entry controls for Supplementary Figure S2.

Manuscript/Appendix: two sustained-load configurations (lambda=1.37, 1.40),
matched slow linear times by uniformly rescaling a_M, b_M, a_S and b_S;
T=500, dt=0.02, n=1000; common Gaussian increments across conditions;
state-dependent stress noise and additive sigma_S=0.15; Wilson 95% CIs.

Unlike the baseline main-figure Monte Carlo, the S2 controls use common random
numbers *at each integration step* across all four rate configurations and the
two noise conditions. The script provides a complete documented implementation;
its seed assignment is explicit and might differ from earlier unpublished code.

Run from an installed repository: pip install -e .
python scripts/make_figure_s2.py --out outputs
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pd_resilience_model.model import ModelParams, equilibrium, jacobian


CONF = [
    ("far", "fast", 1.37, 1.0),
    ("far", "slow", 1.37, 0.7601),
    ("near", "fast", 1.40, 1.3156),
    ("near", "slow", 1.40, 1.0),
]


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if not (0 <= k <= n and n > 0):
        raise ValueError("Expected 0 <= k <= n and n > 0")
    z = NormalDist().inv_cdf(0.5 + level / 2)
    phat = k / n
    d = 1 + z*z/n
    center = (phat + z*z/(2*n)) / d
    half = z * np.sqrt(phat * (1-phat)/n + z*z/(4*n*n)) / d
    return max(0.0, center-half), min(1.0, center+half)


def rescaled_params(r: float) -> ModelParams:
    p = ModelParams()
    return replace(p, a_M=p.a_M*r, b_M=p.b_M*r,
                   a_S=p.a_S*r, b_S=p.b_S*r)


def run_experiment(n_reps: int, T: float, dt: float, seed: int,
                   M_c: float = 0.15, S_c: float = 3.5) -> pd.DataFrame:
    if n_reps < 2 or T <= 0 or dt <= 0 or abs(round(T/dt)*dt-T) > 1e-8:
        raise ValueError("Require n_reps >= 2 and positive T, dt with integer T/dt")

    p0 = ModelParams()
    conditions = []
    for position, speed, lam, r in CONF:
        p = rescaled_params(r)
        ms = equilibrium(lam, p)
        J = jacobian(*ms, lam, p)
        tau_linear = -1 / np.max(np.linalg.eigvals(J).real)
        conditions.append(dict(position=position, speed=speed, lam=lam, r=r,
                               p=p, ms=ms, tau_linear=tau_linear))

    # shape (2 noise prescriptions, 4 conditions, N trajectories)
    M = np.broadcast_to(np.asarray([c["ms"][0] for c in conditions])[None, :, None],
                        (2, 4, n_reps)).copy()
    S = np.broadcast_to(np.asarray([c["ms"][1] for c in conditions])[None, :, None],
                        (2, 4, n_reps)).copy()
    lam = np.asarray([c["lam"] for c in conditions])[None, :, None]
    aM = np.asarray([c["p"].a_M for c in conditions])[None, :, None]
    bM = np.asarray([c["p"].b_M for c in conditions])[None, :, None]
    aS = np.asarray([c["p"].a_S for c in conditions])[None, :, None]
    bS = np.asarray([c["p"].b_S for c in conditions])[None, :, None]
    entered = (M < M_c) & (S > S_c)
    first_time = np.full(M.shape, np.nan)
    first_time[entered] = 0.0
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    for step in range(1, round(T/dt)+1):
        # Exact same independent standard normals for matched replicates
        # in every rate and noise configuration. Still draw if entered.
        zM = rng.standard_normal(n_reps)[None, None, :]
        zS = rng.standard_normal(n_reps)[None, None, :]
        sigma_S = np.empty_like(S)
        sigma_S[0] = p0.sigma_S0 + p0.sigma_S1*lam*np.maximum(S[0], 0)
        sigma_S[1] = 0.15
        dM = aM*(p0.M_rest-M) - bM*lam*S*M
        dS = aS*lam - bS*M*S
        M = np.abs(M + dM*dt + p0.sigma_M_const*sqrt_dt*zM)
        S = np.abs(S + dS*dt + sigma_S*sqrt_dt*zS)
        new = (~entered) & (M < M_c) & (S > S_c)
        first_time[new] = step*dt
        entered |= new

    rows = []
    for noise_index, name in enumerate(("state-dependent", "additive-0.15")):
        for i, c in enumerate(conditions):
            k = int(np.isfinite(first_time[noise_index, i]).sum())
            lower, upper = wilson_interval(k, n_reps)
            rows.append({"noise": name, "position": c["position"], "speed": c["speed"],
                         "lambda": c["lam"], "rate_multiplier": c["r"],
                         "M_star": c["ms"][0], "S_star": c["ms"][1],
                         "tau_linear": c["tau_linear"], "n": n_reps, "entered": k,
                         "p_entry": k/n_reps, "wilson_lower": lower, "wilson_upper": upper,
                         "T": T, "dt": dt, "seed": seed,
                         "first_entry_includes_T": True})
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    palette = {"fast": ("#2a78d6", "o"), "slow": ("#eb6834", "s")}
    for ax, noise in zip(axes, ("state-dependent", "additive-0.15")):
        subset = df[df.noise == noise]
        for speed in ("fast", "slow"):
            d = subset[subset.speed == speed].set_index("position").loc[["far", "near"]]
            color, marker = palette[speed]
            y = d.p_entry.to_numpy()*100
            lo = (d.p_entry-d.wilson_lower).to_numpy()*100
            hi = (d.wilson_upper-d.p_entry).to_numpy()*100
            xs = np.arange(2)+(0.08 if speed == "slow" else -0.08)
            ax.errorbar(xs,y,yerr=np.vstack([lo,hi]),linestyle="-",linewidth=1.4,
                        marker=marker,markersize=7,capsize=4,color=color,label=f"{speed.capitalize()} recovery")
        ax.set_xticks([0,1],["Farther from $\\mathcal{F}$\n$\\lambda=1.37$",
                               "Closer to $\\mathcal{F}$\n$\\lambda=1.40$"])
        ax.set_xlim(-0.22,1.22)
        ax.set_ylim(0,100)
        ax.grid(axis="y",alpha=.22)
        ax.set_title("A) State-dependent noise" if noise == "state-dependent" else
                     "B) Additive stress noise ($\\sigma_S=0.15$)")
        ax.legend(frameon=False)
    axes[0].set_ylabel("First-entry probability by $T=500$ (%)" )
    fig.suptitle("Equilibrium position and restoring timescale: matched controls")
    fig.tight_layout(rect=(0,0.02,1,0.94))
    for ext in ("png","pdf"):
        fig.savefig(out/f"FigureS2_matched_recovery.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",default="outputs")
    a.add_argument("--n-reps",type=int,default=1000)
    a.add_argument("--T",type=float,default=500.0)
    a.add_argument("--dt",type=float,default=0.02)
    a.add_argument("--seed",type=int,default=20240529)
    args=a.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=run_experiment(args.n_reps,args.T,args.dt,args.seed)
    df.to_csv(out/"FigureS2_matched_recovery.csv",index=False)
    make_figure(df,out)
    print(df[["noise","position","speed","rate_multiplier","tau_linear","entered","p_entry"]].to_string(index=False))
    print(f"Saved Figure S2 data and figure to {out}")

if __name__=="__main__":main()
