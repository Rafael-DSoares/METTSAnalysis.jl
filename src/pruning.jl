using Printf
using TOML
using HDF5
using Dumper

"""
    prune_analysis(data_files, observable_tags; toml_name=nothing, axis_kwargs=Dict(), save_figs=false, fig_path=nothing)

Plots each observable for all seeds (1D chains) and asks for the start/end cut of every
file, which is saved as a `_pruning.toml` next to it.

Needs CairoMakie: run `using CairoMakie` first. The plotting lives in a package
extension so that analysis scripts load without it.
"""
function prune_analysis end

"""
    prune_analysis_interval(data_files, observable_tags; toml_name=nothing, axis_kwargs=Dict(), save_figs=false, fig_path=nothing)

Like `prune_analysis` for multi-beta files, plotting each observable at beta_collapse.
Needs CairoMakie: run `using CairoMakie` first.
"""
function prune_analysis_interval end

function prune(data_files::Vector{String}, observable_tags::Vector{String}; 
    toml_name::Union{String,Nothing}=nothing,
    global_start::Union{Int,Nothing}=nothing,
    global_end::Union{Int,Nothing}=nothing)

    data_pruned = Dict{String,Vector{Vector{Float64}}}()
    for tag in observable_tags
        data_pruned[tag] = Vector{Float64}[]
    end

    use_global = (global_start !== nothing) || (global_end !== nothing)

    for data_file in data_files
        dfile = DumpFile(data_file)

        # 1. Determine base start/end logic for this file
        local base_s::Int
        local base_e::Union{Int, Nothing}

        if use_global
            base_s = global_start !== nothing ? global_start : 1
            base_e = global_end
        else
            toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
            if !isfile(toml_file)
                error("Pruning file not found for: " * data_file * "\nProvide TOML files or use global_start/global_end keyword arguments.")
            end
            prune_dict = TOML.parsefile(toml_file)
            base_s = prune_dict["start"]
            base_e = prune_dict["end"]
        end

        # 2. Extract and safely clamp data for each tag
        for tag in observable_tags
            data = read_data(dfile, tag)

            if !(ndims(data) == 1 || (ndims(data) == 2 && size(data, 2) == 1))
                error("Dataset '$tag' must be 1D or Nx1")
            end

            vec_data = vec(data)
            local_len = length(vec_data)

            actual_start = clamp(base_s, 1, local_len)
            actual_end = base_e !== nothing ? clamp(base_e, actual_start, local_len) : local_len

            push!(data_pruned[tag], Float64.(vec_data[actual_start:actual_end]))
        end
    end
    
    return data_pruned
end

function prune_interval(data_files::Vector{String}, observable_tags::Vector{String}; 
    toml_name::Union{String,Nothing}=nothing,
    global_start::Union{Int,Nothing}=nothing,
    global_end::Union{Int,Nothing}=nothing)

    data_pruned = Dict{String,Vector{Matrix{Float64}}}()
    for tag in observable_tags
        data_pruned[tag] = Matrix{Float64}[]
    end

    use_global = (global_start !== nothing) || (global_end !== nothing)

    for data_file in data_files
        dfile = DumpFile(data_file)

        # 1. Determine base start/end logic for this file
        local base_s::Int
        local base_e::Union{Int, Nothing}

        if use_global
            base_s = global_start !== nothing ? global_start : 1
            base_e = global_end
        else
            toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
            if !isfile(toml_file)
                error("Pruning file not found for: " * data_file * "\nProvide TOML files or use global_start/global_end keyword arguments.")
            end
            prune_dict = TOML.parsefile(toml_file)
            base_s = prune_dict["start"]
            base_e = prune_dict["end"]
        end

        # 2. Extract and safely clamp data for each tag
        for tag in observable_tags
            data = read_data(dfile, tag)

            if ndims(data) != 2 || size(data, 2) == 1
                error("Dataset '$tag' must be 2D with multiple columns")
            end

            local_len = size(data, 1)

            actual_start = clamp(base_s, 1, local_len)
            actual_end = base_e !== nothing ? clamp(base_e, actual_start, local_len) : local_len

            push!(data_pruned[tag], Float64.(data[actual_start:actual_end, :]))
        end
    end
    
    return data_pruned
end