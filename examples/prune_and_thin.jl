# Step 2 of the workflow: choose the thermalization cut of every seed, then pick
# decorrelated product states to re-evolve.
#
# Usage:
#   julia prune_and_thin.jl outfile_therm.seed.1.hdf5 outfile_therm.seed.2.hdf5 ...
#
# Each file is a thermalization chain at one beta_collapse with a 1D `energy` series
# and the `product_state` of every step.
using METTSAnalysis
using CairoMakie   # needed for the interactive pruning plots
using HDF5

filenames = length(ARGS) > 0 ? ARGS : error("usage: julia prune_and_thin.jl file1.hdf5 file2.hdf5 ...")

# 1. Plot the energy of all seeds and type the start/end cut of each file.
#    The cuts are saved next to each file as `<name>_pruning.toml`.
prune_analysis(filenames, ["energy"]; save_figs=true)

# 2. Keep product states separated by about one autocorrelation time, at most 250 per seed.
h5open("product_states.hdf5", "w") do h5out
    job_id = 1
    for file in filenames
        states, beta_collapse = get_uncorrelated_product_states(file, "energy"; max_kept_states=250)
        params = read_run_parameters(file)       # thermalization settings, if stored
        for state in eachrow(states)
            grp = create_group(h5out, "job_$job_id")
            write(grp, "product_state", collect(state))
            write(grp, "beta_collapse", beta_collapse)
            for (k, v) in params
                write(grp, "sampling_params/$k", v)
            end
            job_id += 1
        end
    end
    write(h5out, "total_jobs", job_id - 1)
end
