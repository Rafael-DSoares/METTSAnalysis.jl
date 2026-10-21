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
    _check_sampling_parameters(filename)

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
    load_evolved_metts_file(filename, observables)

Loads a re-evolved METTS file (one product state evolved over a beta window), which must
contain exactly one sample. A file with more rows was usually written by a job that ran
twice (e.g. after a requeue) and would count the same product state more than once.
Pruning does not apply to these files. Returns `(beta_collapse, betas, measurements)`.
"""
function load_evolved_metts_file(filename::AbstractString, observables::Vector{String})
    beta_collapse, betas, measurements = load_metts_file_interval(filename, observables; not_use_prune=true)
    if length(measurements) != 1
        error("$(basename(filename)) contains $(length(measurements)) samples instead of 1; " *
              "the job was probably run more than once. Remove the duplicate rows or the file.")
    end
    return beta_collapse, betas, measurements
end



"""
    _plan_thinning(filename, primary_obs; interval, toml_name, not_use_prune, max_kept_states)

Shared by the product-state thinning routines. Reads `primary_obs` (a multi-beta matrix
taken at beta_collapse when `interval`, a 1D series otherwise), applies the pruning,
estimates the autocorrelation time tau = n / ESS, optionally enlarges tau to respect
`max_kept_states`, and prints a report. Returns
`(dfile, beta_collapse, betas, thinned_indices, n_samples_total)`, with `betas === nothing`
when `!interval`. The thinned indices run backwards from the pruned end, anchoring the
subset at the most thermalized state.
"""
function _plan_thinning(filename::AbstractString, primary_obs::String;
                        interval::Bool,
                        toml_name::Union{String, Nothing},
                        not_use_prune::Bool,
                        max_kept_states::Union{Int, Nothing})
    dfile = DumpFile(filename)
    beta_collapse = read_data(dfile, "beta_collapse")
    data_primary  = read_data(dfile, primary_obs)

    if interval
        betas = vec(read_data(dfile, "betas"))
        if ndims(data_primary) != 2 || size(data_primary, 2) == 1
            error("Dataset '$primary_obs' is not a multi-beta 2D matrix.")
        end
        vec_primary = data_primary[:, argmin(abs.(betas .- beta_collapse))]
    else
        betas = nothing
        if !(ndims(data_primary) == 1 || (ndims(data_primary) == 2 && size(data_primary, 2) == 1))
            error("Dataset '$primary_obs' is not a 1D array.")
        end
        vec_primary = vec(data_primary)
    end
    n_samples_total = length(vec_primary)

    # 1. Global pruning
    start_idx, end_idx = _get_pruning_indices(filename, toml_name, n_samples_total, not_use_prune)
    series = Float64.(vec_primary[start_idx:end_idx])
    n_pruned = length(series)

    # 2. Statistical tau, with safety guards
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

    # 3. User compute budget (max_kept_states override)
    if max_kept_states !== nothing && max_kept_states > 0
        tau = max(stat_tau, ceil(Int, n_pruned / max_kept_states))
    else
        tau = stat_tau
    end

    # 4. Reverse-anchored range
    thinned_indices = end_idx:-tau:start_idx
    n_kept = length(thinned_indices)

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

    return dfile, beta_collapse, betas, thinned_indices, n_samples_total
end

"""
    _read_thinned_states(dfile, filename, thinned_indices, n_samples_total)

Reads `product_state` and returns the rows `thinned_indices`. Errors if the number of
product states differs from the number of samples of the primary observable: the two
are then misaligned (typically a job killed between writing the observable and the
product state, then resumed), so the indices would pick the wrong states.
"""
function _read_thinned_states(dfile, filename::AbstractString, thinned_indices, n_samples_total::Int)
    states = read_data(dfile, "product_state")
    n_states = size(states, 1)
    if n_states != n_samples_total
        error("$(basename(filename)) has $n_states product states but $n_samples_total samples of the primary observable. " *
              "The chain was probably interrupted between the two writes and resumed, so they are misaligned.")
    end
    return ndims(states) == 1 ? states[thinned_indices] : states[thinned_indices, :]
end

function _write_thinned_states(filename::AbstractString, beta_collapse, betas, states)
    base, ext = splitext(filename)
    out_filename = base * "_pd" * ext

    h5open(out_filename, "w") do h5out
        write(h5out, "beta_collapse", beta_collapse)
        betas === nothing || write(h5out, "betas", betas)
        write(h5out, "product_state", states)
    end

    println("  ✓ Wrote $(size(states, 1)) uncorrelated product states to: $(basename(out_filename))\n")
    return out_filename
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
    dfile, beta_collapse, betas, idxs, n_total = _plan_thinning(filename, primary_obs; interval=true,
        toml_name=toml_name, not_use_prune=not_use_prune, max_kept_states=max_kept_states)
    states = _read_thinned_states(dfile, filename, idxs, n_total)
    return _write_thinned_states(filename, beta_collapse, betas, states)
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
    dfile, beta_collapse, _, idxs, n_total = _plan_thinning(filename, primary_obs; interval=false,
        toml_name=toml_name, not_use_prune=not_use_prune, max_kept_states=max_kept_states)
    states = _read_thinned_states(dfile, filename, idxs, n_total)
    return _write_thinned_states(filename, beta_collapse, nothing, states)
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
    dfile, beta_collapse, _, idxs, n_total = _plan_thinning(filename, primary_obs; interval=false,
        toml_name=toml_name, not_use_prune=not_use_prune, max_kept_states=max_kept_states)
    thinned_states = _read_thinned_states(dfile, filename, idxs, n_total)
    println("  ✓ Extracted $(length(idxs)) uncorrelated product states\n")
    return thinned_states, beta_collapse
end



# ==============================================================================
# RUN-PARAMETER CONSISTENCY CHECKS
# ==============================================================================

"""
Parameters that must be identical between the thermalization run that sampled a
product state and the run that re-evolved it; otherwise the samples are not drawn
from the distribution MBAR reweights them with.
"""
const STRICT_SAMPLING_PARAMS = ["L", "W", "J2", "lattice", "maxm", "cutoff"]

"""
    read_run_parameters(filename; group="params")

Reads the simulation parameters stored as datasets in the HDF5 group `group` of
`filename` (e.g. `params/maxm`, `params/cutoff`, `params/lattice`).
Returns an empty Dict if the group is absent.

Store there only settings that must be identical across all files of one analysis;
per-file values such as `beta_collapse`, the seed or the job id belong at the root.
"""
function read_run_parameters(filename::AbstractString; group::AbstractString="params")
    return h5open(filename, "r") do fid
        haskey(fid, group) || return Dict{String, Any}()
        g = fid[group]
        Dict{String, Any}(k => read(g, k) for k in keys(g))
    end
end

_param_equal(a, b) = (a isa Real && b isa Real) ? isapprox(a, b) : isequal(a, b)

function _param_diff(p1::AbstractDict, p2::AbstractDict)
    return sort([k for k in union(keys(p1), keys(p2))
                 if !(haskey(p1, k) && haskey(p2, k) && _param_equal(p1[k], p2[k]))])
end

"""
    _check_sampling_parameters(filename; group, sampling_group, strict_keys)

Within one file, compares the parameters of the run that produced it (`group`) with
those of the thermalization run that sampled its product state (`sampling_group`).
A mismatch in `strict_keys` is an error; other differing shared keys (e.g. `tau`)
only warn. Does nothing unless both groups are present.
"""
function _check_sampling_parameters(filename::AbstractString;
                                    group::AbstractString="params",
                                    sampling_group::AbstractString="sampling_params",
                                    strict_keys::Vector{String}=STRICT_SAMPLING_PARAMS)
    run_params      = read_run_parameters(filename; group=group)
    sampling_params = read_run_parameters(filename; group=sampling_group)
    (isempty(run_params) || isempty(sampling_params)) && return nothing

    for k in intersect(keys(run_params), keys(sampling_params))
        _param_equal(run_params[k], sampling_params[k]) && continue
        msg = "$(basename(filename)): '$k' = $(run_params[k]) in '$group' but $(sampling_params[k]) in '$sampling_group'"
        k in strict_keys ? error(msg * ". The product states were sampled with different settings.") : @warn msg
    end
    for k in strict_keys
        if haskey(run_params, k) != haskey(sampling_params, k)
            error("$(basename(filename)): '$k' is stored in only one of '$group' and '$sampling_group'.")
        end
    end
    return nothing
end

"""
    check_run_parameters(filenames; group="params", sampling_group="sampling_params",
                         strict_keys=STRICT_SAMPLING_PARAMS)

Guards against mixing files produced with different simulation settings before they
are pooled or fed to MBAR:

1. The `group` parameters of all files must agree (an error names the first file that
   differs and the keys involved). Files with and without parameters cannot be mixed.
2. Within each file, `sampling_group` must agree with `group` on `strict_keys`
   (see `_check_sampling_parameters`).

Returns the common parameter Dict (empty, with a warning, if no file stores any).
"""
function check_run_parameters(filenames::AbstractVector{<:AbstractString};
                              group::AbstractString="params",
                              sampling_group::AbstractString="sampling_params",
                              strict_keys::Vector{String}=STRICT_SAMPLING_PARAMS)
    isempty(filenames) && return Dict{String, Any}()

    ref = read_run_parameters(filenames[1]; group=group)
    for f in filenames
        params = read_run_parameters(f; group=group)
        diff = _param_diff(ref, params)
        if !isempty(diff)
            details = join(["'$k': $(get(ref, k, "missing")) vs $(get(params, k, "missing"))" for k in diff], ", ")
            error("Run parameters of $(basename(f)) differ from $(basename(filenames[1])): $details")
        end
        _check_sampling_parameters(f; group=group, sampling_group=sampling_group, strict_keys=strict_keys)
    end

    if isempty(ref)
        @warn "None of the $(length(filenames)) files stores a '$group' group; run parameters cannot be verified."
    end
    return ref
end
