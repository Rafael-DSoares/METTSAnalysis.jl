using Printf
using TOML
using HDF5
using Dumper
using GLMakie

@doc raw"""
    prune_analysis(data_files::Vector{String}, observable_tags::Vector{String}; toml_name=nothing, axis_kwargs=Dict())

Interactively prune MC data stored in HDF5 files. Plots up to 4 observables at a time.

# Keyword Arguments
- `toml_name`: TOML filename (defaults to `<data_file_basename>_pruning.toml`).
- `axis_kwargs`: A `Dict` mapping tags to `NamedTuple`s for Makie Axis customization.
  Example: `Dict("energy" => (; yscale=log10), "mag" => (; yreversed=true))`
"""
function prune_analysis(data_files::Vector{String}, observable_tags::Vector{String}; 
                        toml_name::Union{String, Nothing}=nothing,
                        axis_kwargs::Dict{String, <:NamedTuple}=Dict{String, NamedTuple}())
    
    for data_file in data_files
        dfile = DumpFile(data_file)
        prune_data = Dict("start" => Dict{String, Int}(), "end" => Dict{String, Int}())

        println("@ File: " * data_file)

        for tag in observable_tags
            data = read_data(dfile, tag)

            # Check if it is strictly 1D or an Nx1 matrix
            if !(ndims(data) == 1 || (ndims(data) == 2 && size(data, 2) == 1))
                error(@sprintf("Dataset \"%s\" must be 1D or Nx1 (got shape %s)", tag, size(data)))
            end
            
            # Flatten Nx1 down to a 1D vector
            data = vec(data)
            
            # Create and display a single plot for each observable
            f = Figure()
            kwargs = get(axis_kwargs, tag, (;)) 
            ax = Axis(f[1, 1]; title=tag, xlabel="MC Step", ylabel=tag, kwargs...)
            
            scatter!(ax, 1:length(data), data)
            lines!(ax, 1:length(data), data)
            display(f)

            # Prompt immediately after displaying
            len = length(data)
            println("  @ Observable: " * tag)

            print("    Start (default 1): ")
            rd = strip(readline())
            start_idx = rd != "" ? parse(Int, rd) : 1

            print("    End   (default $len): ")
            rd = strip(readline())
            end_idx = rd != "" ? parse(Int, rd) : len

            prune_data["start"][tag] = clamp(start_idx, 1, len)
            prune_data["end"][tag]   = clamp(end_idx, 1, len)
        end

        toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
        open(toml_file, "w") do fl
            TOML.print(fl, prune_data)
        end
        println("Pruning written to: ", toml_file)
    end
end

function prune(data_files::Vector{String}, observable_tags::Vector{String}; toml_name::Union{String, Nothing}=nothing)
    # Type stability: Explicitly defining the structure. 
    # Using Float64 ensures math operations downstream are predictable.
    data_pruned = Dict{String, Vector{Vector{Float64}}}()
    for tag in observable_tags
        data_pruned[tag] = Vector{Float64}[]
    end

    for data_file in data_files
        toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
        if !isfile(toml_file)
            error("Pruning file not found for: " * data_file)
        end
        prune_dict = TOML.parsefile(toml_file)

        dfile = DumpFile(data_file)
        for tag in observable_tags
            data = read_data(dfile, tag)
            
            # Check if it is strictly 1D or an Nx1 matrix
            if !(ndims(data) == 1 || (ndims(data) == 2 && size(data, 2) == 1))
                error(@sprintf("Dataset \"%s\" must be 1D or Nx1 (got shape %s)", tag, size(data)))
            end
            
            # Flatten Nx1 down to a 1D vector
            data = vec(data)
            
            # TOML.parsefile natively reads these as Ints
            start_idx = prune_dict["start"][tag]
            end_idx   = prune_dict["end"][tag]
            
            # Explicit broadcast to Float64 guarantees type stability
            push!(data_pruned[tag], Float64.(data[start_idx:end_idx]))
        end
    end
    return data_pruned
end


function prune_analysis_interval(data_files::Vector{String}, observable_tags::Vector{String}; 
                                 toml_name::Union{String, Nothing}=nothing,
                                 axis_kwargs::Dict{String, <:NamedTuple}=Dict{String, NamedTuple}())
    
    for data_file in data_files
        dfile = DumpFile(data_file)
        beta_collapse = read_data(dfile, "beta_collapse")
        betas = vec(read_data(dfile, "betas"))
        beta_index = argmin(abs.(betas .- beta_collapse))
        
        prune_data = Dict("start" => Dict{String, Int}(), "end" => Dict{String, Int}())
        println("@ File: " * data_file)

        for tag in observable_tags
            data = read_data(dfile, tag)
            
            # Reject if not 2D, OR if it's an Nx1 matrix
            if ndims(data) != 2 || size(data, 2) == 1
                error(@sprintf("Dataset \"%s\" must be 2D with multiple columns (got shape %s)", tag, size(data)))
            end
            
            vals = data[:, beta_index]

            # Create and display a single plot for this observable
            f = Figure()
            kwargs = get(axis_kwargs, tag, (;))
            ax = Axis(f[1, 1]; title=@sprintf("%s (beta=%.4f)", tag, beta_collapse), xlabel="MC Step", ylabel=tag, kwargs...)
            
            scatter!(ax, 1:length(vals), vals)
            lines!(ax, 1:length(vals), vals)
            display(f)

            # Prompt immediately after displaying
            nmetts = size(data, 1)
            println("  @ Observable: " * tag)

            print("    Start (default 1): ")
            rd = strip(readline())
            start_idx = rd != "" ? parse(Int, rd) : 1

            print("    End   (default $nmetts): ")
            rd = strip(readline())
            end_idx = rd != "" ? parse(Int, rd) : nmetts

            prune_data["start"][tag] = clamp(start_idx, 1, nmetts)
            prune_data["end"][tag]   = clamp(end_idx, 1, nmetts)
        end

        toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
        open(toml_file, "w") do fl
            TOML.print(fl, prune_data)
        end
        println("Pruning written to: ", toml_file)
    end
end


function prune_interval(data_files::Vector{String}, observable_tags::Vector{String}; toml_name::Union{String, Nothing}=nothing)
    # Type stability for 2D arrays: Vector of Matrices
    data_pruned = Dict{String, Vector{Matrix{Float64}}}()
    for tag in observable_tags
        data_pruned[tag] = Matrix{Float64}[]
    end

    for data_file in data_files
        toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
        if !isfile(toml_file)
            error("Pruning file not found for: " * data_file)
        end
        prune_dict = TOML.parsefile(toml_file)

        dfile = DumpFile(data_file)
        for tag in observable_tags
            data = read_data(dfile, tag)
            
            # Reject if not 2D, OR if it's an Nx1 matrix
            if ndims(data) != 2 || size(data, 2) == 1
                error(@sprintf("Dataset \"%s\" must be 2D with multiple columns (got shape %s)", tag, size(data)))
            end
            
            start_idx = prune_dict["start"][tag]
            end_idx   = prune_dict["end"][tag]
            
            push!(data_pruned[tag], Float64.(data[start_idx:end_idx, :]))
        end
    end
    return data_pruned
end