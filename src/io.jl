


function load_metts_file_interval(filename::AbstractString, observables::Vector{String}; 
                                 toml_name::Union{String, Nothing}=nothing, 
                                 not_use_prune::Bool=false)

    dfile = DumpFile(filename)

    # 1. Read Metadata
    beta_collapse = read_data(dfile, "beta_collapse")
    betas         = vec(read_data(dfile, "betas"))   
    log_norm_mat  = read_data(dfile, "log_norm")
    n_samples_total = size(log_norm_mat, 1)

    # 2. Determine Global Pruning (Check TOML once)
    toml_file = toml_name === nothing ? splitext(filename)[1] * "_pruning.toml" : joinpath(dirname(filename), toml_name)
    start_idx, end_idx = 1, n_samples_total
    
    if !not_use_prune && isfile(toml_file)
        prune_dict = TOML.parsefile(toml_file)
        start_idx = get(prune_dict, "start", 1)
        end_idx   = get(prune_dict, "end", n_samples_total)
    end

    # 3. Read all observables into memory (One loop over tags)
    # We store them in a temp dict to zip them later
    temp_storage = Dict{Symbol, Matrix{Float64}}()
    for obs in observables
        mat = read_data(dfile, obs)
        if ndims(mat) != 2 || size(mat, 2) == 1
             error("Dataset '$obs' is not a multi-beta 2D matrix.")
        end
        temp_storage[Symbol(obs)] = Float64.(mat)
    end
    temp_storage[:log_norm] = Float64.(log_norm_mat)

    # 4. Prepare the schema for the NamedTuple
    keys_tuple = Tuple(keys(temp_storage))

    # 5. Build the measurements vector (Slicing happens here)
    range = start_idx:end_idx
    measurements = map(range) do i
        # For each sample index i, grab the i-th row of every matrix in storage
        vals = Tuple(temp_storage[k][i, :] for k in keys_tuple)
        return NamedTuple{keys_tuple}(vals)
    end

    return beta_collapse, betas, measurements
end



function load_metts_file(filename::AbstractString, observables::Vector{String}; 
                         toml_name::Union{String, Nothing}=nothing, 
                         not_use_prune::Bool=false)

    dfile = DumpFile(filename)

    beta_collapse = read_data(dfile, "beta_collapse")
    
    first_obs_data = read_data(dfile, observables[1])
    n_samples_total = length(first_obs_data)

    toml_file = toml_name === nothing ? splitext(filename)[1] * "_pruning.toml" : joinpath(dirname(filename), toml_name)
    start_idx, end_idx = 1, n_samples_total

    if !not_use_prune && isfile(toml_file)
        prune_dict = TOML.parsefile(toml_file)
        start_idx = get(prune_dict, "start", 1)
        end_idx   = get(prune_dict, "end", n_samples_total)
    end

    # 2. Read all requested tags once
    temp_storage = Dict{Symbol, Vector{Float64}}()
    for obs in observables
        data = read_data(dfile, obs)
        temp_storage[Symbol(obs)] = Float64.(vec(data))
    end

    # 3. Zip into NamedTuples
    keys_tuple = Tuple(keys(temp_storage))
    range = start_idx:end_idx
    
    measurements = map(range) do i
        vals = Tuple(temp_storage[k][i] for k in keys_tuple)
        return NamedTuple{keys_tuple}(vals)
    end

    return beta_collapse, measurements
end