using Statistics
using Random
using Printf


let

    filenames = length(ARGS) > 0 ? ARGS : error("usage: julia chained_bar_analysis.jl file1.h5 file2.h5 ...")

    n_bootstrap = 20
    tags_to_analyze = ["energy", "magnetization"]

    data = [load_metts_file(f, tags_to_analyze) for f in filenames]

    sort!(data; by=first)

    beta_collapses = getindex.(data, 1)
    all_measurements = getindex.(data, 2)


    results = average_observable_single(all_measurements; n_bootstrap=n_bootstrap)

    energy_mean, energy_err = results.energy
    mag_mean, mag_err = results.magnetization

    out = DumpFile("single_metts_result.h5")
    out["energy_mean"] = energy_mean
    out["energy_err"] = energy_err
    out["mag_mean"] = mag_mean
    out["mag_err"] = mag_err
    out["beta_collapses"] = beta_collapses
    println("\nResults written to single_metts_result.h5")
end