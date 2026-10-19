# Minimal stochastic resilience–stress model of dopaminergic neuron vulnerability

This repository contains the Python code used to generate the simulations and figures for the manuscript:

**Resilience Loss and Noise-Driven Escape in a Minimal Stochastic Model of Dopaminergic Neuron Vulnerability** (title under revision)

The model is a two-variable stochastic dynamical system describing the interaction between mitochondrial reserve (`M`) and composite cellular stress (`S`) under an effective sustained load parameter (`lambda`). The code reproduces the deterministic equilibrium analysis, local stability analysis, relaxation-time calculations, stochastic trajectories, Monte Carlo first-entry analysis, the state-space panels of Figure 6, and the supplementary sensitivity analysis reported in the manuscript.

## Repository structure

```text
pd_resilience_model/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── scripts/
│   └── make_figures.py
└── src/
    └── pd_resilience_model/
        ├── __init__.py
        ├── model.py
        ├── analysis.py
        └── numba_mc.py
```

## Installation

Create a clean environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -e .
```

Alternatively:

```bash
pip install -r requirements.txt
```

## Reproducing the figures

To generate all main computational figures and CSV summaries:

```bash
python scripts/make_figures.py --out outputs --n-reps 1000
```

This produces:

- `Figure2_deterministic.png/pdf`
- `Figure3_relaxation_time.png/pdf`
- `Figure4_stochastic_trajectories.png/pdf`
- `Figure5_monte_carlo_first_entry.png/pdf`
- `Figure6AB_state_space.png/pdf` (panels A–B of Figure 6; panel C is a schematic)
- `FigureS1_sensitivity.png/pdf`
- `deterministic_summary.csv`, `monte_carlo_first_entry_summary.csv`, `FigureS1_sensitivity_summary.csv`

All Monte Carlo results use the seeded, vectorized Euler–Maruyama integrator in `analysis.monte_carlo_first_entry_times` (Itô interpretation), so they are reproducible for a given `--seed`. The optional `numba_mc.py` module is retained for speed experiments but is not used for the manuscript figures. A full run takes a few minutes on a laptop.

For a quick test run:

```bash
python scripts/make_figures.py --out outputs_test --n-reps 100 --skip-supplementary
```

## Main parameter values

The default deterministic parameters are:

```text
a_M = 0.5
b_M = 0.4
M_rest = 1.0
a_S = 0.3
b_S = 0.6
```

The stochastic noise amplitudes are:

```text
sigma_M = 0.003
sigma_S(S, lambda) = 0.015 + 0.03 * lambda * max(S, 0)
```

The operational loss-of-viability region F used for first-entry simulations is:

```text
S > 3.5 and M < 0.15
```

The deterministic drift does not trap trajectories in F: started inside F, they return toward the survival equilibrium. In the simulations F is absorbing by assumption (it stands for a commitment point not represented by the two variables), and each trajectory is stopped at its first entry. Supplementary Figure S1 shows how the probability of entry depends on the boundaries of F, the observation window and the noise structure.

Monte Carlo simulations in the manuscript used 1000 trajectories per lambda, time horizon `T = 500`, and timestep `dt = 0.02`.

## Notes on interpretation

For every `lambda` below `lambda_c = sqrt(a_M b_S M_rest / (b_M a_S)) ≈ 1.58` the model has a single equilibrium, which attracts every trajectory in the positive quadrant. As `lambda` approaches `lambda_c`, `M*` tends to 0 and `S*` diverges; above `lambda_c` there is no equilibrium and `S` grows without bound.

The model is intentionally minimal and phenomenological. The code is provided to make the numerical analyses reproducible; it is not intended as a calibrated physiological model of Parkinson's disease progression or dopaminergic neuron lifetime.

## Citation

If you use or modify this code, please cite the associated manuscript once available.
