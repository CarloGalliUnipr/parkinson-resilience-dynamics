#!/usr/bin/env python3
"""Generate the manuscript's Supplementary Table S1 from the Figure 5 first-entry run.

The same complete seeded lambda grid must be simulated, rather than selected
loads only, to retain the released vectorized RNG allocation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pd_resilience_model.analysis import monte_carlo_first_entry_times

def run(n_reps=1000,T=500.,dt=.02,seed=20240529):
    lams=np.round(np.arange(1.25,1.4501,.01),2)
    tau=monte_carlo_first_entry_times(lams,n_reps=n_reps,T=T,dt=dt,seed=seed)
    rows=[]
    for lam,t in zip(lams,tau):
        hit=np.isfinite(t); k=int(hit.sum());p=k/n_reps
        total_exposure=np.where(hit,t,T).sum()
        hazard=k/total_exposure if total_exposure>0 else np.nan
        haz_se=np.sqrt(k)/total_exposure if total_exposure>0 else np.nan
        mean=1/hazard if hazard>0 else np.inf
        times=t[hit]
        rows.append(dict(lambda_value=lam,entered=k,n=n_reps,T=T,dt=dt,seed=seed,
                         p_entry=p,p_wald_low=max(0.,p-1.96*np.sqrt(p*(1-p)/n_reps)),
                         p_wald_high=min(1.,p+1.96*np.sqrt(p*(1-p)/n_reps)),
                         exposure_time=total_exposure,hazard=hazard,
                         hazard_ci95_low=max(0.,hazard-1.96*haz_se),
                         hazard_ci95_high=hazard+1.96*haz_se,
                         reciprocal_hazard=mean,
                         reciprocal_hazard_in_unloaded_recovery_times=mean/2,
                         median_entry_time_given_entry=np.median(times) if k else np.nan,
                         cv_entry_time_given_entry=(np.std(times,ddof=1)/np.mean(times)) if k>1 else np.nan))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='outputs')
    ap.add_argument('--n-reps',type=int,default=1000)
    ap.add_argument('--T',type=float,default=500.)
    ap.add_argument('--dt',type=float,default=.02)
    ap.add_argument('--seed',type=int,default=20240529)
    x=ap.parse_args();out=Path(x.out);out.mkdir(exist_ok=True,parents=True)
    df=run(x.n_reps,x.T,x.dt,x.seed)
    df.to_csv(out/'Supplementary_Table_S1.csv',index=False)
    print(df.loc[df.lambda_value.isin([1.35,1.36,1.37,1.40,1.43]),
                 ['lambda_value','entered','p_entry','hazard','median_entry_time_given_entry','cv_entry_time_given_entry']].to_string(index=False))

if __name__=='__main__':main()
