#!/usr/bin/env python3
"""Reproduce the manuscript's 3-step-size Monte Carlo comparison.

Important: the manuscript timestep-control counts use exactly THREE loads
[1.37, 1.40, 1.43] in the same vectorized seeded simulation, not the full
21-load baseline Figure 5 grid. The sample counts depend on that choice.

Paper check: python scripts/check_timestep_convergence.py --out outputs
Quick smoke: python scripts/check_timestep_convergence.py --n-reps 20 --T 20 --dt 0.02 0.01 --out outputs_smoke
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pd_resilience_model.analysis import monte_carlo_first_entry_times


def run(n_reps: int, T: float, dt_values: list[float], seed: int, full_grid: bool = False) -> pd.DataFrame:
    lams = (np.round(np.arange(1.25, 1.4501, 0.01), 2) if full_grid
            else np.array([1.37, 1.40, 1.43]))
    rows=[]
    for dt in dt_values:
        tau=monte_carlo_first_entry_times(lams,n_reps=n_reps,T=T,dt=dt,seed=seed)
        for lam, times in zip(lams,tau):
            k=int(np.isfinite(times).sum())
            rows.append({"lambda":lam,"dt":dt,"n_reps":n_reps,"T":T,"seed":seed,
                         "entered":k,"p_entry":k/n_reps})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="outputs")
    ap.add_argument("--n-reps",type=int,default=1000)
    ap.add_argument("--T",type=float,default=500)
    ap.add_argument("--dt",nargs="+",type=float,default=[0.02,0.01,0.005])
    ap.add_argument("--seed",type=int,default=20240529)
    ap.add_argument("--full-grid",action="store_true",help="Optional extended check: use Figure 5 load grid; counts will differ")
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=run(args.n_reps,args.T,args.dt,args.seed,full_grid=args.full_grid)
    df.to_csv(out/("timestep_convergence_full_grid.csv" if args.full_grid else "timestep_convergence_selected.csv"),index=False)
    chosen=df[df["lambda"].isin([1.37,1.40,1.43])]
    print(chosen.to_string(index=False))
    if args.n_reps==1000 and args.T==500 and args.seed==20240529 and not args.full_grid and args.dt==[0.02,0.01,0.005]:
        expected = {0.02:[49,526,968],0.01:[55,523,973],0.005:[56,539,963]}
        for dt, counts in expected.items():
            got = chosen[chosen.dt == dt].entered.tolist()
            assert got == counts, f"Different results at dt={dt}: {got} vs {counts}"
        print("PASS: All nine published timestep-control counts reproduced exactly.")
    print("NOTE: Monte Carlo probabilities will vary across timesteps even with the same seed.")

if __name__=="__main__":main()
