using Printf
using TOML
using HDF5
using Dumper
using CairoMakie

@doc raw"""
    prune_analysis(data_files::Vector{String}, observable_tags::Vector{String}; toml_name=nothing, axis_kwargs=Dict())

Plots all seeds on the same axis and prompts the user to enter a manual burn-in interval per seed.
"""
function prune_analysis(data_files::Vector{String}, observable_tags::Vector{String};
    toml_name::Union{String,Nothing}=nothing,
    axis_kwargs::Dict{String,<:NamedTuple}=Dict{String,NamedTuple}())

    open_screens = []

    # 1. Loop over observables to create the overlaid plots
    for tag in observable_tags
        f = Figure(size=(1000, 500))
        kwargs = get(axis_kwargs, tag, (;))
        ax = Axis(f[1, 1]; title="Reviewing All Seeds: $tag", xlabel="MC Step", ylabel=tag, kwargs...)

        for (i, data_file) in enumerate(data_files)
            dfile = DumpFile(data_file)
            data = read_data(dfile, tag)

            if !(ndims(data) == 1 || (ndims(data) == 2 && size(data, 2) == 1))
                error("Dataset '$tag' in $data_file must be 1D or Nx1")
            end

            data = vec(data)
            chain_length = length(data)

            lines!(ax, 1:chain_length, data, label="Seed $i", alpha=0.7, linewidth=1.5)
        end

        axislegend(ax, position=:rt)
        sc = display(f)
        push!(open_screens, sc)
    end

    # 2. Prompt for individual cuts per file
    println("\n--- Enter Cuts for Each Seed ---")
    for (i, data_file) in enumerate(data_files)
        dfile = DumpFile(data_file)
        local_len = length(vec(read_data(dfile, observable_tags[1])))

        println("\n▶ File $i: $(basename(data_file))")

        print("    Start Index (default 1): ")
        rd = strip(readline())
        # FIX: Added '1' as the default fallback
        start_idx = rd != "" ? parse(Int, rd) : 1

        print("    End Index (default $local_len): ")
        rd = strip(readline())
        end_idx = rd != "" ? parse(Int, rd) : local_len

        prune_data = Dict(
            "start" => clamp(start_idx, 1, local_len),
            "end" => clamp(end_idx, 1, local_len)
        )

        toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
        open(toml_file, "w") do fl
            TOML.print(fl, prune_data)
        end
        println("    ✓ Saved to: ", basename(toml_file))
    end

    for sc in open_screens
        try
            close(sc)
        catch
        end
    end
end

function prune(data_files::Vector{String}, observable_tags::Vector{String}; toml_name::Union{String,Nothing}=nothing)
    data_pruned = Dict{String,Vector{Vector{Float64}}}()
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

@doc raw"""
    prune_analysis_interval(data_files::Vector{String}, observable_tags::Vector{String}; ...)
"""
function prune_analysis_interval(data_files::Vector{String}, observable_tags::Vector{String};
    toml_name::Union{String,Nothing}=nothing,
    axis_kwargs::Dict{String,<:NamedTuple}=Dict{String,NamedTuple}())

    open_screens = []
    bold_colors = [:firebrick, :dodgerblue, :forestgreen, :darkorange, :purple, :saddlebrown, :magenta, :teal]

    dfile_first = DumpFile(data_files[1])
    beta_collapse = read_data(dfile_first, "beta_collapse")
    betas = vec(read_data(dfile_first, "betas"))
    beta_index = argmin(abs.(betas .- beta_collapse))

    # 1. Loop over observables to create the overlaid plots
    for tag in observable_tags
        f = Figure(size=(1000, 500))
        kwargs = get(axis_kwargs, tag, (;))
        ax = Axis(f[1, 1]; title=@sprintf("Reviewing All Seeds: %s (beta=%.4f)", tag, beta_collapse),
            xlabel="MC Step", ylabel=tag, kwargs...)

        for (i, data_file) in enumerate(data_files)
            dfile = DumpFile(data_file)
            data = read_data(dfile, tag)

            if ndims(data) != 2 || size(data, 2) == 1
                error("Dataset '$tag' in $data_file must be 2D with multiple columns")
            end

            vals = data[:, beta_index]
            nmetts = length(vals)

            seed_color = bold_colors[mod1(i, length(bold_colors))]

            lines!(ax, 1:nmetts, vals, label="Seed $i", color=seed_color, alpha=0.85, linewidth=1.5)
        end

        axislegend(ax, position=:rt)
        sc = display(f)
        push!(open_screens, sc)
    end

    # 2. Prompt for individual cuts per file
    println("\n--- Enter Cuts for Each Seed ---")
    for (i, data_file) in enumerate(data_files)
        dfile = DumpFile(data_file)
        data = read_data(dfile, observable_tags[1])
        nmetts = size(data, 1)

        println("\n▶ File $i: $(basename(data_file))")

        print("    Start Index (default 1): ")
        rd = strip(readline())
        # FIX: Added '1' as the default fallback
        start_idx = rd != "" ? parse(Int, rd) : 1

        print("    End Index (default $nmetts): ")
        rd = strip(readline())
        end_idx = rd != "" ? parse(Int, rd) : nmetts

        prune_data = Dict(
            "start" => clamp(start_idx, 1, nmetts),
            "end" => clamp(end_idx, 1, nmetts)
        )

        toml_file = toml_name === nothing ? splitext(data_file)[1] * "_pruning.toml" : joinpath(dirname(data_file), toml_name)
        open(toml_file, "w") do fl
            TOML.print(fl, prune_data)
        end
        println("✓ Saved to: ", basename(toml_file))
    end

    for sc in open_screens
        try
            close(sc)
        catch
        end
    end
end

function prune_interval(data_files::Vector{String}, observable_tags::Vector{String}; toml_name::Union{String,Nothing}=nothing)
    data_pruned = Dict{String,Vector{Matrix{Float64}}}()
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