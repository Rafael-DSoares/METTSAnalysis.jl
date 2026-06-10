using MCMCDiagnosticTools
using HDF5


function _get_pruning_indices(filename::AbstractString, 
                              toml_name::Union{String, Nothing}, 
                              n_samples_total::Int, 
                              not_use_prune::Bool)
    start_idx, end_idx = 1, n_samples_total
    
    not_use_prune && return start_idx, end_idx

    toml_file = toml_name === nothing ? splitext(filename)[1] * "_pruning.toml" : joinpath(dirname(filename), toml_name)

    if isfile(toml_file)
        try
            prune_dict = TOML.parsefile(toml_file)
            start_idx = get(prune_dict, "start", 1)
            end_idx   = get(prune_dict, "end", n_samples_total)
        catch e
            @warn "Failed to parse pruning file at $toml_file. Using full range.\nError: $e"
        end
    else
        @warn "Pruning file not found at $toml_file. Loading full range."
    end
    
    # Guarantee that the parsed indices are valid for the array sizes
    start_idx = clamp(start_idx, 1, n_samples_total)
    end_idx   = clamp(end_idx, start_idx, n_samples_total)

    return start_idx, end_idx
end


function load_metts_file_interval(filename::AbstractString, observables::Vector{String}; 
                                 toml_name::Union{String, Nothing}=nothing, 
                                 not_use_prune::Bool=false)

    dfile = DumpFile(filename)

    # 1. Read Metadata
    beta_collapse = read_data(dfile, "beta_collapse")
    betas         = vec(read_data(dfile, "betas"))
    log_norm_mat  = read_data(dfile, "log_norm")
    n_samples_total = size(log_norm_mat, 1)

    # 2. Determine Global Pruning via Helper
    start_idx, end_idx = _get_pruning_indices(filename, toml_name, n_samples_total, not_use_prune)
    
    # 3. Read requested data
    temp_storage = Dict{Symbol, Matrix{Float64}}()
    for obs in observables
        mat = read_data(dfile, obs)
        if ndims(mat) != 2 || size(mat, 2) == 1
             error("Dataset '$obs' is not a multi-beta 2D matrix.")
        end
        temp_storage[Symbol(obs)] = Float64.(mat)
    end
    temp_storage[:log_norm] = Float64.(log_norm_mat)

    # 4. Zip into NamedTuples
    keys_tuple = Tuple(keys(temp_storage))
    range = start_idx:end_idx
    
    measurements = map(range) do i
        vals = Tuple(temp_storage[k][i, :] for k in keys_tuple)
        return NamedTuple{keys_tuple}(vals)
    end

    return beta_collapse, betas, measurements
end

function load_metts_file(filename::AbstractString, observables::Vector{String}; 
                         toml_name::Union{String, Nothing}=nothing, 
                         not_use_prune::Bool=false)

    dfile = DumpFile(filename)

    # 1. Read Metadata
    beta_collapse = read_data(dfile, "beta_collapse")
    first_obs_data = read_data(dfile, observables[1])
    n_samples_total = length(first_obs_data)

    # 2. Determine Global Pruning via Helper
    start_idx, end_idx = _get_pruning_indices(filename, toml_name, n_samples_total, not_use_prune)
    
    # 3. Read requested data
    temp_storage = Dict{Symbol, Vector{Float64}}()
    for obs in observables
        data = read_data(dfile, obs)
        temp_storage[Symbol(obs)] = Float64.(vec(data))
    end

    # 4. Zip into NamedTuples
    keys_tuple = Tuple(keys(temp_storage))
    range = start_idx:end_idx
    
    measurements = map(range) do i
        vals = Tuple(temp_storage[k][i] for k in keys_tuple)
        return NamedTuple{keys_tuple}(vals)
    end

    return beta_collapse, measurements
end



"""
    write_product_states_uncorrelated_interval(...)

Reads a multi-temperature METTS file, calculates the true autocorrelation time (tau) 
on a primary observable at `beta_collapse`, and writes an uncorrelated subset of the 
product states to a new `_pd.hdf5` file. Anchors the subset at the most thermalized state.
"""
function write_product_states_uncorrelated_interval(filename::AbstractString, 
                                                    primary_obs::String; 
                                                    toml_name::Union{String, Nothing}=nothing, 
                                                    not_use_prune::Bool=false,
                                                    max_kept_states::Union{Int, Nothing}=nothing)

    dfile = DumpFile(filename)

    # 1. Read Metadata
    beta_collapse = read_data(dfile, "beta_collapse")
    betas         = vec(read_data(dfile, "betas"))
    
    # Read the primary observable strictly to calculate tau
    mat_primary = read_data(dfile, primary_obs)
    if ndims(mat_primary) != 2 || size(mat_primary, 2) == 1
         error("Dataset '$primary_obs' is not a multi-beta 2D matrix.")
    end
    n_samples_total = size(mat_primary, 1)
    beta_index = argmin(abs.(betas .- beta_collapse))

    # 2. Determine Global Pruning via Helper
    start_idx, end_idx = _get_pruning_indices(filename, toml_name, n_samples_total, not_use_prune)
    
    # 3. Extract the pruned primary observable
    series = Float64.(mat_primary[start_idx:end_idx, beta_index])
    n_pruned = length(series)
    
    # 4. Calculate Mathematical Tau & Apply Safety Guards
    if n_pruned < 10
        @warn "Chain too short ($n_pruned samples). Defaulting to tau = 1."
        ess = NaN
        stat_tau = 1
    else
        ess = MCMCDiagnosticTools.ess(series)
        
        if isnan(ess) || ess <= 1.0
            @warn "ESS is <= 1.0 or NaN for $(basename(filename)). Chain is highly correlated or flat. Defaulting to tau = length."
            stat_tau = n_pruned
        else
            stat_tau = max(1, round(Int, n_pruned / ess))
        end
    end
    
    # 5. Apply User Compute Budget (max_kept_states override)
    if max_kept_states !== nothing && max_kept_states > 0
        forced_tau = ceil(Int, n_pruned / max_kept_states)
        tau = max(stat_tau, forced_tau)
    else
        tau = stat_tau
    end
    
    # 6. Create the reverse-anchored range
    thinned_indices = end_idx:-tau:start_idx
    n_kept = length(thinned_indices)
    
    # Terminal Report
    ess_display = isnan(ess) ? "NaN" : string(round(ess, digits=1))
    println("\n--- Thinning Report for $(basename(filename)) ---")
    println("  Using Observable : $primary_obs @ beta=$beta_collapse")
    println("  Pruned Range     : $start_idx to $end_idx ($n_pruned samples)")
    println("  Effective Samples: $ess_display")
    println("  Stat Tau         : $stat_tau")
    if tau > stat_tau
        println("  Forced Tau       : $tau (Budget limit: $max_kept_states)")
    end
    println("  Kept States      : $n_kept (Reversed, anchoring at $end_idx)")
    println("-------------------------------------------------")

    # 7. Prepare Output Filename
    base, ext = splitext(filename)
    out_filename = base * "_pd" * ext
    
    # 8. Read and write the product states
    states = read_data(dfile, "product_state")
    
    h5open(out_filename, "w") do h5out
        write(h5out, "beta_collapse", beta_collapse)
        write(h5out, "betas", betas)
        
        if ndims(states) == 1
            write(h5out, "product_state", states[thinned_indices])
        else
            write(h5out, "product_state", states[thinned_indices, :])
        end
    end
    
    println("  ✓ Wrote $n_kept uncorrelated product states to: $(basename(out_filename))\n")
    return out_filename
end


"""
    write_product_states_uncorrelated(...)
Reads a single-temperature METTS file, calculates the true autocorrelation time (tau), 
and writes an uncorrelated subset of the product states to a new `_pd.hdf5` file. 
Anchors the subset at the most thermalized state.
"""
function write_product_states_uncorrelated(filename::AbstractString, 
                                           primary_obs::String; 
                                           toml_name::Union{String, Nothing}=nothing, 
                                           not_use_prune::Bool=false,
                                           max_kept_states::Union{Int, Nothing}=nothing)

    dfile = DumpFile(filename)

    # 1. Read Metadata
    beta_collapse = read_data(dfile, "beta_collapse")
    
    # Read the primary observable strictly to calculate tau
    data_primary = read_data(dfile, primary_obs)
    if !(ndims(data_primary) == 1 || (ndims(data_primary) == 2 && size(data_primary, 2) == 1))
         error("Dataset '$primary_obs' is not a 1D array.")
    end
    
    vec_primary = vec(data_primary)
    n_samples_total = length(vec_primary)

    # 2. Determine Global Pruning via Helper
    start_idx, end_idx = _get_pruning_indices(filename, toml_name, n_samples_total, not_use_prune)
    
    # 3. Extract the pruned primary observable
    series = Float64.(vec_primary[start_idx:end_idx])
    n_pruned = length(series)
    
    # 4. Calculate Mathematical Tau & Apply Safety Guards
    if n_pruned < 10
        @warn "Chain too short ($n_pruned samples). Defaulting to tau = 1."
        ess = NaN
        stat_tau = 1
    else
        ess = MCMCDiagnosticTools.ess(series)
        
        if isnan(ess) || ess <= 1.0
            @warn "ESS is <= 1.0 or NaN for $(basename(filename)). Chain is highly correlated or flat. Defaulting to tau = length."
            stat_tau = n_pruned
        else
            stat_tau = max(1, round(Int, n_pruned / ess))
        end
    end
    
    # 5. Apply User Compute Budget (max_kept_states override)
    if max_kept_states !== nothing && max_kept_states > 0
        forced_tau = ceil(Int, n_pruned / max_kept_states)
        tau = max(stat_tau, forced_tau)
    else
        tau = stat_tau
    end
    
    # 6. Create the reverse-anchored range
    thinned_indices = end_idx:-tau:start_idx
    n_kept = length(thinned_indices)
    
    # Terminal Report
    ess_display = isnan(ess) ? "NaN" : string(round(ess, digits=1))
    println("\n--- Thinning Report for $(basename(filename)) ---")
    println("  Using Observable : $primary_obs @ beta=$beta_collapse")
    println("  Pruned Range     : $start_idx to $end_idx ($n_pruned samples)")
    println("  Effective Samples: $ess_display")
    println("  Stat Tau         : $stat_tau")
    if tau > stat_tau
        println("  Forced Tau       : $tau (Budget limit: $max_kept_states)")
    end
    println("  Kept States      : $n_kept (Reversed, anchoring at $end_idx)")
    println("-------------------------------------------------")

    # 7. Prepare Output Filename
    base, ext = splitext(filename)
    out_filename = base * "_pd" * ext
    
    # 8. Read and write the product states
    states = read_data(dfile, "product_state")
    
    h5open(out_filename, "w") do h5out
        write(h5out, "beta_collapse", beta_collapse)
        
        if ndims(states) == 1
            write(h5out, "product_state", states[thinned_indices])
        else
            write(h5out, "product_state", states[thinned_indices, :])
        end
    end
    
    println("  ✓ Wrote $n_kept uncorrelated product states to: $(basename(out_filename))\n")
    return out_filename
end



"""
    get_uncorrelated_product_states(...)

Reads a single-temperature METTS file, calculates the true autocorrelation time (tau),
and returns an uncorrelated subset of the product states along with beta_collapse.
Anchors the subset at the most thermalized state.
"""
function get_uncorrelated_product_states(filename::AbstractString,
                                         primary_obs::String;
                                         toml_name::Union{String, Nothing}=nothing,
                                         not_use_prune::Bool=false,
                                         max_kept_states::Union{Int, Nothing}=nothing)

    dfile = DumpFile(filename)

    # 1. Read Metadata
    beta_collapse = read_data(dfile, "beta_collapse")

    # Read the primary observable strictly to calculate tau
    data_primary = read_data(dfile, primary_obs)
    if !(ndims(data_primary) == 1 || (ndims(data_primary) == 2 && size(data_primary, 2) == 1))
         error("Dataset '$primary_obs' is not a 1D array.")
    end

    vec_primary = vec(data_primary)
    n_samples_total = length(vec_primary)

    # 2. Determine Global Pruning via Helper
    start_idx, end_idx = _get_pruning_indices(filename, toml_name, n_samples_total, not_use_prune)

    # 3. Extract the pruned primary observable
    series = Float64.(vec_primary[start_idx:end_idx])
    n_pruned = length(series)

    # 4. Calculate Mathematical Tau & Apply Safety Guards
    if n_pruned < 10
        @warn "Chain too short ($n_pruned samples). Defaulting to tau = 1."
        ess = NaN
        stat_tau = 1
    else
        ess = MCMCDiagnosticTools.ess(series)

        if isnan(ess) || ess <= 1.0
            @warn "ESS is <= 1.0 or NaN for $(basename(filename)). Chain is highly correlated or flat. Defaulting to tau = length."
            stat_tau = n_pruned
        else
            stat_tau = max(1, round(Int, n_pruned / ess))
        end
    end

    # 5. Apply User Compute Budget (max_kept_states override)
    if max_kept_states !== nothing && max_kept_states > 0
        forced_tau = ceil(Int, n_pruned / max_kept_states)
        tau = max(stat_tau, forced_tau)
    else
        tau = stat_tau
    end

    # 6. Create the reverse-anchored range
    thinned_indices = end_idx:-tau:start_idx
    n_kept = length(thinned_indices)

    # Terminal Report
    ess_display = isnan(ess) ? "NaN" : string(round(ess, digits=1))
    println("\n--- Thinning Report for $(basename(filename)) ---")
    println("  Using Observable : $primary_obs @ beta=$beta_collapse")
    println("  Pruned Range     : $start_idx to $end_idx ($n_pruned samples)")
    println("  Effective Samples: $ess_display")
    println("  Stat Tau         : $stat_tau")
    if tau > stat_tau
        println("  Forced Tau       : $tau (Budget limit: $max_kept_states)")
    end
    println("  Kept States      : $n_kept (Reversed, anchoring at $end_idx)")
    println("-------------------------------------------------")

    # 7. Read and extract the specific product states
    states = read_data(dfile, "product_state")

    if ndims(states) == 1
        thinned_states = states[thinned_indices]
    else
        thinned_states = states[thinned_indices, :]
    end

    println("  ✓ Extracted $n_kept uncorrelated product states\n")

    # 8. Return the objects instead of writing to disk
    return thinned_states, beta_collapse
end
