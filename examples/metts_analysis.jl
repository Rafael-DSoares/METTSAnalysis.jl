# Plain METTS averages from the thermalization chains (no reweighting).
#
# Usage:
#   julia metts_analysis.jl outfile_therm.seed.1.hdf5 outfile_therm.seed.2.hdf5 ...
#
# Files may mix several beta_collapses and seeds. Each file is pruned with its
# `_pruning.toml` (see prune_and_thin.jl).
using METTSAnalysis
using Dumper

filenames = length(ARGS) > 0 ? ARGS : error("usage: julia metts_analysis.jl file1.hdf5 file2.hdf5 ...")
check_run_parameters(filenames)

data = [load_metts_file(f, ["energy"]) for f in filenames]
sort!(data; by=first)
beta_collapses   = getindex.(data, 1)
all_measurements = getindex.(data, 2)

# Keep the seeds of each beta_collapse as separate chains: the error of each chain
# accounts for its autocorrelation (error_method=:ess) and the seeds are checked
# for consistency before being combined.
chains_per_beta, beta_collapses = group_chains_single(all_measurements, beta_collapses)
results = average_observable_single.(chains_per_beta)

energy_mean = [res.energy[1] for res in results]
energy_err  = [res.energy[2] for res in results]

out = DumpFile("metts_result.h5")
out["energy_mean"]    = energy_mean
out["energy_err"]     = energy_err
out["beta_collapses"] = beta_collapses
println("Results written to metts_result.h5")
