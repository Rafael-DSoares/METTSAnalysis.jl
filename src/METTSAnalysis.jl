module METTSAnalysis
using Dumper
using Statistics
using StatsBase
using Dierckx

export pool_states, pool_states_single, group_chains_single, logsumexp
include("aux.jl")

export prune_analysis, prune, prune_interval, prune_analysis_interval
include("pruning.jl")

export average_observable_single, effective_sample_size
include("analysis.jl")

export load_metts_file_interval, load_metts_file, load_evolved_metts_file, write_product_states_uncorrelated_interval, write_product_states_uncorrelated, get_uncorrelated_product_states, read_run_parameters, check_run_parameters
include("io.jl")

export MBARState, reweight_observable, bootstrap_mbar_reweight, bootstrap_mbar_reweight_dev, compute_overlap_matrix, bootstrap_all_free_energies, bootstrap_mbar_magneto_caloric_effect, check_overlap, add_beta_derivatives
include("mbar.jl")

function __init__()
    # Point users to CairoMakie when they call the plotting functions without it
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f === prune_analysis || exc.f === prune_analysis_interval
            if isdefined(Main, :CairoMakie)
                # CairoMakie is loaded but the extension is not: the Manifest predates the extension
                print(io, "\n$(nameof(exc.f)) needs the CairoMakie extension, which did not load. " *
                          "Run `import Pkg; Pkg.resolve()` in this environment and restart Julia.")
            else
                print(io, "\n$(nameof(exc.f)) needs CairoMakie: run `using CairoMakie` first.")
            end
        end
    end
end

end