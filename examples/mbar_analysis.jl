# MBAR reweighting of re-evolved product states.
#
# Usage:
#   julia mbar_analysis.jl outfile_job_1.hdf5 outfile_job_2.hdf5 ...
#
# Each file holds ONE product state (sampled at its beta_collapse) evolved over a beta
# window, with `beta_collapse`, `betas`, `energy` and `log_norm` (= log‖e^{-βH/2}σ‖).
# Every window must contain all beta_collapses.
using METTSAnalysis
using Dumper

filenames = length(ARGS) > 0 ? ARGS : error("usage: julia mbar_analysis.jl file1.hdf5 file2.hdf5 ...")

# Refuse to mix files run with different settings (lattice, maxm, cutoff, ...)
check_run_parameters(filenames)

data = [load_evolved_metts_file(f, ["energy"]) for f in filenames]
sort!(data; by=first)
beta_collapses   = getindex.(data, 1)
all_betas        = getindex.(data, 2)
all_measurements = getindex.(data, 3)

# One MBAR state per beta_collapse
all_measurements, all_betas, beta_collapses = pool_states(all_measurements, all_betas, beta_collapses)

n_bootstrap = 200
betas, energy, energy_err, energy_neff, f, f_err =
    bootstrap_mbar_reweight(all_measurements, all_betas, beta_collapses, "energy"; n_bootstrap=n_bootstrap)

# dE/dβ = -(⟨H²⟩ - ⟨H⟩²), i.e. the specific heat C = β² ⟨ΔH²⟩ up to sign
_, dE, dE_err, _, _, _ =
    bootstrap_mbar_reweight_dev(all_measurements, all_betas, beta_collapses, "energy"; n_bootstrap=n_bootstrap)

out = DumpFile("mbar_result.h5")
out["betas"]           = betas
out["energy_mean"]     = energy
out["energy_err"]      = energy_err
out["energy_neff"]     = energy_neff
out["dE_dbeta"]        = dE
out["dE_dbeta_err"]    = dE_err
out["free_energies"]   = f        # reduced, f(β_k) - f(β_1); not F
out["free_energy_err"] = f_err
out["beta_collapses"]  = beta_collapses
out["overlap_matrix"]  = compute_overlap_matrix(all_measurements, all_betas, beta_collapses)
println("Results written to mbar_result.h5")
