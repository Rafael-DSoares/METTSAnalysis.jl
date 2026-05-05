using Printf
using TOML
using HDF5
using Dumper
using CairoMakie

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
        println("@ File: " * data_file)

        chain_length = 0
        open_screens = []

        for tag in observable_tags
            data = read_data(dfile, tag)

            if !(ndims(data) == 1 || (ndims(data) == 2 && size(data, 2) == 1))
                error("Dataset '$tag' must be 1D or Nx1 (got shape $(size(data)))")
            end

            data = vec(data)
            chain_length = length(data)

            # Create and display plot
            f = Figure()
            kwargs = get(axis_kwargs, tag, (;))
            ax = Axis(f[1, 1]; title="Reviewing: $tag", xlabel="MC Step", ylabel=tag, kwargs...)

            scatter!(ax, 1:chain_length, data)
            lines!(ax, 1:chain_length, data)
            sc = display(f)
            push!(open_screens, sc)
        end

        println("--- Global Cut for $data_file ---")
        print("    Start Index (default 1): ")
        rd = strip(readline())
        start_idx = rd != "" ? parse(Int, rd) : 1

        print("    End Index   (default $chain_length): ")
        rd = strip(readline())
        end_idx = rd != "" ? parse(Int, rd) : chain_length

        prune_data = Dict(
            "start" => clamp(start_idx, 1, chain_length),
            "end"   => clamp(end_idx, 1, chain_length)
        )

        toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
        open(toml_file, "w") do fl
            TOML.print(fl, prune_data)
        end
        println("Pruning written to: ", toml_file)
        # 3. Close the windows once the TOML file is written
        for sc in open_screens
            try
                close(sc)
            catch
                # Silently catch the error in case the user
                # already manually clicked the 'X' to close the window
            end
        end
    end
end

function prune(data_files::Vector{String}, observable_tags::Vector{String}; toml_name::Union{String, Nothing}=nothing)
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
        
        # Global indices for this file
        s, e = prune_dict["start"], prune_dict["end"]

        dfile = DumpFile(data_file)
        for tag in observable_tags

            data = read_data(dfile, tag)
            
            if !(ndims(data) == 1 || (ndims(data) == 2 && size(data, 2) == 1))
                error("Dataset '$tag' must be 1D or Nx1")
            end
            
            push!(data_pruned[tag], Float64.(vec(data)[s:e]))
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
        
        println("@ File: " * data_file)
        nmetts = 0

        open_screens = []

        for tag in observable_tags
            data = read_data(dfile, tag)
            if ndims(data) != 2 || size(data, 2) == 1
                error("Dataset '$tag' must be 2D with multiple columns")
            end
            nmetts = size(data, 1)
            vals = data[:, beta_index]

            f = Figure()
            kwargs = get(axis_kwargs, tag, (;))
            
            ax = Axis(f[1, 1]; title=@sprintf("%s (beta=%.4f)", tag, beta_collapse), 
                      xlabel="MC Step", ylabel=tag, kwargs...)
            
            scatter!(ax, 1:nmetts, vals)
            lines!(ax, 1:nmetts, vals)
            sc = display(f)
            push!(open_screens, sc)
        end

        println("--- Global Cut for $data_file ---")
        print("    Start Index (default 1): ")
        rd = strip(readline())
        start_idx = rd != "" ? parse(Int, rd) : 1

        print("    End Index   (default $nmetts): ")
        rd = strip(readline())
        end_idx = rd != "" ? parse(Int, rd) : nmetts

        prune_data = Dict(
            "start" => clamp(start_idx, 1, nmetts),
            "end"   => clamp(end_idx, 1, nmetts)
        )

        toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
        open(toml_file, "w") do fl
            TOML.print(fl, prune_data)
        end
        println("Pruning written to: ", toml_file)
        # 3. Close the windows once the TOML file is written
        for sc in open_screens
            try
                close(sc)
            catch
                # Silently catch the error in case the user
                # already manually clicked the 'X' to close the window
            end
        end
    end
end

function prune_interval(data_files::Vector{String}, observable_tags::Vector{String}; toml_name::Union{String, Nothing}=nothing)
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
        s, e = prune_dict["start"], prune_dict["end"]

        dfile = DumpFile(data_file)
        for tag in observable_tags
            data = read_data(dfile, tag)
            
            if ndims(data) != 2 || size(data, 2) == 1
                error("Dataset '$tag' must be 2D with multiple columns")
            end
            
            push!(data_pruned[tag], Float64.(data[s:e, :]))
        end
    end
    return data_pruned
end
