# Parkinson resilience dynamics — reproducible reserve–stress model

This repository contains Python code for the phenomenological, two-variable stochastic model described in the manuscript **Resilience Loss and Noise-Driven Threshold Crossing in a Minimal Stochastic Model of Dopamine Neuron Vulnerability**. The model is **not calibrated** to measured molecular rates, neuronal lifetimes, or clinical progression.

**Version v1.1.0** adds the balanced rate-rescaling controls (Supplementary Figure S2), Monte Carlo timestep verification, the Supplementary Table S1 summary and numerical tests. The equations and primary Figure 2–6 generators are unchanged from release v1.0.

## Model

The state is mitochondrial functional reserve `M >= 0` and composite stress `S >= 0` at fixed sustained load `λ`. The deterministic drift is

```text
dM/dt = a_M (M_rest - M) - b_M λ S M
dS/dt = a_S λ - b_S M S
```

Parameters: `a_M=0.5`, `b_M=0.4`, `M_rest=1.0`, `a_S=0.3`, `b_S=0.6`. For `λ < sqrt(a_M*b_S*M_rest/(b_M*a_S)) ≈ 1.5811`, the positive survival equilibrium is unique and stable. The stochastic dynamics use independent Itô Wiener drivers, `σ_M=0.003`, `σ_S=0.015+0.03 λ max(S,0)` and numerical reflection `abs` at each boundary. The operational first-entry criterion is **simultaneously** `M < 0.15` and `S > 3.5`. The underlying drift has no absorbing death state; irreversible commitment is an imposed survival-analysis convention.

## Install

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e .
```

Requirements: Python >= 3.10, NumPy, SciPy, pandas and Matplotlib. `numba` is optional and is **not** used for the manuscript's primary simulations. Use a fixed environment and the release tag to reproduce exact pseudo-random results across machines.

## Reproduce the figures and controls

From the repository root after installing:

```bash
# Original Figures 2, 3, 4, 5, 6 and Supplementary Figure S1
python scripts/make_figures.py --out outputs --n-reps 1000

# New balanced numerical controls: Supplementary Figure S2
python scripts/make_figure_s2.py --out outputs --n-reps 1000

# Manuscript timestep checks (the three selected loads in one seeded run)
python scripts/check_timestep_convergence.py --out outputs --n-reps 1000

# Supplementary Table S1 (same full grid as Figure 5)
python scripts/make_table_s1.py --out outputs --n-reps 1000

# Basic reproducibility and mathematical assertions
python -m unittest discover -s tests -v
```

For short smoke tests replace `--n-reps 1000` by `--n-reps 20` and add `--T 20` to the two new control scripts as appropriate; the original `make_figures.py` has its own `--skip-supplementary` quick mode. The full three-timestep convergence run adds substantial computation relative to a smoke test.

**Outputs** include the existing figure PNG/PDF pairs and tabular CSVs, plus `FigureS2_matched_recovery.png/pdf`, `FigureS2_matched_recovery.csv`, `timestep_convergence_selected.csv` (optional `timestep_convergence_full_grid.csv` with `--full-grid`), and `Supplementary_Table_S1.csv`.

### Figure S2 protocol

Two equilibrium configurations (`λ=1.37` and `1.40`) are each studied at matched linear relaxation scales of approximately **28.36** and **37.31** time units. At `λ=1.37`, the slow-recovery case rescales all four deterministic rate coefficients by `r=0.7601`; at `λ=1.40`, the fast-recovery case uses `r=1.3156`. The two other cases retain `r=1.0`. Common normal increments are applied across all four configurations under (A) the original state-dependent stress noise and (B) additive `σ_S=0.15`. Reserve noise stays at `0.003`, `T=500`, `dt=0.02`, `n=1000`, with strict threshold inequalities. Error bars are Wilson binomial 95% confidence intervals. The script uses base seed `20240529`.

The rate rescaling **does not move the equilibrium** at a fixed load but changes the linear timescale. Since stochastic noise amplitudes are held fixed, this is a mathematical control rather than a pure biological intervention. Changing λ also changes nonlinear drift structure: the comparison is not a unique causal decomposition.

### Timestep comparison

`check_timestep_convergence.py` repeats **only the three selected loads** λ = 1.37, 1.40 and 1.43 together for `dt = 0.02, 0.01, 0.005`, with the same seed, `T=500`, `n=1000`. This exactly reproduces the nine published counts: 49/526/968, 55/523/973 and 56/539/963 out of 1000. Add `--full-grid` for an optional 21-load check, but it uses a different assignment of the same seeded random stream and therefore gives different realized counts. The main Figure 5 and Table S1 reproduce the full-grid run; the independent timestep controls reproduce the three-load run. Changing `dt` changes random-number consumption: a fixed seed does *not* couple Wiener sample paths across resolutions. Statistical comparison, not equality of sample paths, is appropriate.

### Scope of figure reproducibility

Figures 2–5, manuscript Figure 6 (the `Figure6AB_state_space` file), and S1 are generated by `scripts/make_figures.py`; S2 is generated separately. The generated `Figure6_full` is a legacy extra composite containing a panel C and must not be mistaken for the revised manuscript Figure 6. The manuscript's conceptual Figures 1 and 7 contain illustrations/schematics, and are not generated by these numerical scripts. Figure 6C in the original repository is a **separate** illustration from manuscript Figure 7. The existing source files for Figures 2–6 and S1 are retained unchanged in v1.1.0.

## Release information and citation

Use the immutable GitHub tag `v1.1.0` for the computations accompanying the current manuscript. A new version-specific Zenodo DOI must be checked **after** the new release is archived. Do not automatically reuse the earlier version-specific DOI as if it contained the v1.1.0 scripts. The article should cite the archived version that actually contains all reported numerical analyses.

The present model is an abstract first-entry model. Its commitment criterion does not demonstrate ferroptosis, cuproptosis, apoptosis or any other molecular death pathway.
