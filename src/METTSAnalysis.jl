module METTSAnalysis
using Dumper
using Statistics
using StatsBase
using Dierckx

export prune_analysis, prune, prune_interval, prune_analysis_interval
include("pruning.jl")

export average_observable_single
include("analysis.jl")

export load_metts_file_interval, load_metts_file, write_product_states_uncorrelated_interval, write_product_states_uncorrelated
include("io.jl")

export MBARState, reweight_observable, bootstrap_mbar_reweight, bootstrap_mbar_reweight_dev, compute_overlap_matrix
include("mbar.jl")


export pool_states
include("aux.jl")
end
