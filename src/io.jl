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
    
    # Guarantee that the parsed indices are valid for our array sizes
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