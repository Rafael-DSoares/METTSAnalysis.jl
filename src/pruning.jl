using Printf
using TOML
using HDF5
using Dumper
using CairoMakie

function prune_analysis(data_files::Vector{String}, observable_tags::Vector{String};
    toml_name::Union{String,Nothing}=nothing,
    axis_kwargs::Dict{String,<:NamedTuple}=Dict{String,NamedTuple}(),
    save_figs::Bool=false, 
    fig_path::Union{String,Nothing}=nothing)

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
        
        if save_figs
            # Use user-provided path or fallback to the data file's directory
            base_dir = fig_path !== nothing ? fig_path : dirname(data_files[1])
            
            if !isdir(base_dir)
                mkpath(base_dir)
            end

            out_file = joinpath(base_dir, "tmp_prune_$tag.png")
            save(out_file, f)
            println("▶ Saved review plot for $tag to: $out_file")
        else
            sc = display(f)
            push!(open_screens, sc)
        end
    end

    # 2. Prompt for individual cuts per file
    println("\n--- Enter Cuts for Each Seed ---")
    for (i, data_file) in enumerate(data_files)
        dfile = DumpFile(data_file)
        local_len = length(vec(read_data(dfile, observable_tags[1])))

        println("\n▶ File $i: $(basename(data_file))")

        print("    Start Index (default 1): ")
        rd = strip(readline())
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

function prune_analysis_interval(data_files::Vector{String}, observable_tags::Vector{String};
    toml_name::Union{String,Nothing}=nothing,
    axis_kwargs::Dict{String,<:NamedTuple}=Dict{String,NamedTuple}(),
    save_figs::Bool=false,
    fig_path::Union{String,Nothing}=nothing)

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
        
        if save_figs
            # Use user-provided path or fallback to the data file's directory
            base_dir = fig_path !== nothing ? fig_path : dirname(data_files[1])
            
            if !isdir(base_dir)
                mkpath(base_dir)
            end

            out_file = joinpath(base_dir, "tmp_prune_interval_$tag.png")
            save(out_file, f)
            println("▶ Saved review plot for $tag to: $out_file")
        else
            sc = display(f)
            push!(open_screens, sc)
        end
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