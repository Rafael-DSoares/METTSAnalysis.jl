# METTSAnalysis

[![Build Status](https://github.com/awietek/METTSAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/awietek/METTSAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amain)

Analysis of METTS (minimally entangled typical thermal states) simulations: pruning and
thinning of METTS chains, plain METTS averages with autocorrelation-aware errors, and
MBAR reweighting of re-evolved product states to obtain observables, their β-derivatives
and free energies on a continuous β grid.

## Workflow

1. **Thermalize.** Run a METTS chain per seed at each `beta_collapse`, storing a 1D
   `energy` series and the `product_state` of every step.
2. **Prune and thin.** `prune_analysis` plots the chains and records the start/end cut of
   each file in `<name>_pruning.toml`. `get_uncorrelated_product_states` then keeps product
   states about one autocorrelation time apart. See `examples/prune_and_thin.jl`.
3. **Re-evolve.** Evolve each kept product state σ from β = 0 over a β window, storing
   `betas`, the observables and `log_norm = log‖e^{-βH/2}σ‖` (one file per state).
4. **Reweight.** Pool the files of each `beta_collapse` and run MBAR
   (`bootstrap_mbar_reweight`, `bootstrap_mbar_reweight_dev`, ...). See
   `examples/mbar_analysis.jl`.

Plain METTS averages from the thermalization chains are in `examples/metts_analysis.jl`.

## Requirements for correct results

- **Same settings in every stage.** The product states must be re-evolved with the
  Hamiltonian (lattice, couplings) and truncation (`maxm`, `cutoff`) that sampled them.
  Store the settings in a `params/` group of each file and the thermalization settings in
  `sampling_params/`; `check_run_parameters(filenames)` then refuses to mix files.
- **Every β window must contain every `beta_collapse`.** MBAR needs the weight of each
  sample at each sampled temperature; `MBARState` raises an error otherwise.
- **Neighbouring states must overlap.** The bootstrap routines warn when the overlap
  between neighbouring `beta_collapse`s drops below 0.03 (`check_overlap`); add an
  intermediate `beta_collapse` when that happens.
- **Collapse basis.** Collapsing in the X basis and restarting from the Z-basis state with
  the same labels samples the right distribution only for SU(2)-symmetric Hamiltonians.

## Errors

- Plain METTS averages (`average_observable_single`) use `std/√ESS` by default, where ESS
  is the effective sample size of the correlated chain. Pass the seeds as separate chains
  (`group_chains_single`) to combine them correctly and to be warned when they disagree.
  `error_method=:naive` and `:bootstrap` treat samples as independent and underestimate
  the error of a Markov chain.
- MBAR errors come from bootstrapping the (independent) re-evolved product states.
- Free energies are reduced, `f(β) = ln Z(β₁) − ln Z(β)`: only differences are physical.

## Plotting

The pruning plots need CairoMakie, which is an optional dependency: run
`using CairoMakie` before calling `prune_analysis` or `prune_analysis_interval`.
Everything else works without it.
