# Minimal stochastic resilience–stress model of dopaminergic neuron vulnerability

This repository contains the Python code used to generate the simulations and figures for the manuscript:

**Resilience Loss and Noise-Driven Escape in a Minimal Stochastic Model of Dopaminergic Neuron Vulnerability**

The model is a two-variable stochastic dynamical system describing the interaction between mitochondrial reserve (`M`) and composite cellular stress (`S`) under an effective chronic load parameter (`lambda`). The code reproduces the deterministic equilibrium analysis, local stability analysis, relaxation-time calculations, stochastic trajectories, and Monte Carlo first-passage survival analysis reported in the manuscript.

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
python scripts/make_figures.py --out outputs --n-reps 1000 --use-numba
```

This produces:

- `Figure2_deterministic.png/pdf`
- `Figure3_relaxation_separation.png/pdf`
- `Figure4_stochastic_trajectories.png/pdf`
- `Figure5_monte_carlo_survival_analysis.png/pdf`
- `deterministic_summary.csv`
- `monte_carlo_escape_summary.csv`

The first Numba run may take longer because the functions are compiled on first use. Subsequent runs should be faster.

For a quick test run:

```bash
python scripts/make_figures.py --out outputs_test --n-reps 100 --no-numba
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

The operational failure region used for first-passage simulations is:

```text
S > 3.5 and M < 0.15
```

Monte Carlo simulations in the manuscript used 1000 trajectories per lambda, time horizon `T = 500`, and timestep `dt = 0.02`.

## Notes on interpretation

The model is intentionally minimal and phenomenological. The code is provided to make the numerical analyses reproducible; it is not intended as a calibrated physiological model of Parkinson's disease progression or dopaminergic neuron lifetime.

## Citation

If you use or modify this code, please cite the associated manuscript once available.
