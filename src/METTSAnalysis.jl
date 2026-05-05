module METTSAnalysis
using Dumper
using Statistics
using StatsBase

export prune_analysis, prune, prune_interval, prune_analysis_interval
include("pruning.jl")

export average_observable_single
include("analysis.jl")

export load_metts_file_interval, load_metts_file
include("io.jl")

export MBARState, reweight_observable, bootstrap_mbar_reweight
include("mbar.jl")


export pool_states
include("aux.jl")
end
