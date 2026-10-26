# v1.1.0 — manuscript reproducibility update

Compared with published GitHub release v1.0 (2026-10-20):

- Preserve the same deterministic equations, baseline rates, Itô noise, threshold inequalities and original Figures 2–6 / S1 simulation logic.
- Add `scripts/make_figure_s2.py`: reproduces the four balanced recovery / equilibrium configurations for Supplementary Figure S2, under both stress-noise prescriptions, with matched Gaussian increments, Wilson 95% intervals and CSV output.
- Add `scripts/check_timestep_convergence.py`: reproduces the manuscript's 0.02 / 0.01 / 0.005 integration timestep Monte Carlo control, preserving the three selected-lambda seeded allocation used for the reported numerical checks (with optional full-grid analysis).
- Add `scripts/make_table_s1.py`: exports the first-entry, hazard and conditional entry-time statistics behind Supplementary Table S1.
- Add `tests/test_matched_recovery.py`: analytical and numerical smoke tests.
- Align `pyproject.toml` package version with the GitHub release (`1.1.0`) and revise `README.md` to separate computational figures from conceptual schematics.

All new controls use the same baseline dynamical model: no new cell-death mechanism or extra dynamical state is introduced. Unabsorbed trajectories and first-entry accounting are kept conceptually separate.

**Important version note:** The earlier repository release tag was `v1.0`, although the Python package reported `0.2.0`; the current manuscript and appendix contain references to both names. Use the precise new GitHub tag and the new version-specific Zenodo DOI after archiving. No new Zenodo DOI is claimed here.
