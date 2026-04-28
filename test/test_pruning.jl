using Test
using TOML
using Dumper


# Simulates keyboard input by using a temporary file stream
function simulate_input(f::Function, input_str::String)
    tmp = tempname()
    write(tmp, input_str)
    open(tmp, "r") do io
        redirect_stdin(f, io)
    end
    rm(tmp, force=true)
end

function make_1d_file(filename, tags, data_dict)
    dfile = DumpFile(filename)
    for tag in tags
        for val in data_dict[tag]
            dump!(dfile, tag, val)
        end
    end
end

function make_2d_file(filename, tags, data_dict, beta_collapse, betas)
    dfile = DumpFile(filename)
    write_data!(dfile, "beta_collapse", beta_collapse)
    write_data!(dfile, "betas", betas)
    for tag in tags
        for row in eachrow(data_dict[tag])
            dump!(dfile, tag, collect(row))
        end
    end
end


@testset "Global Pruning (1D and Nx1)" begin

    @testset "prune_analysis writes correct Global TOML" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = splitext(tmp_file)[1] * "_pruning.toml"

        tags = ["energy", "magnetization"]
        data_dict = Dict(
            "energy" => Float64.(1:20),
            "magnetization" => Float64.(1:20)
        )
        make_1d_file(tmp_file, tags, data_dict)

        # Global logic: only 2 inputs ("start\nend\n") even with 2 tags
        simulate_input("5\n15\n") do
            prune_analysis([tmp_file], tags; toml_name=nothing)
        end

        @test isfile(toml_file)
        prune_dict = TOML.parsefile(toml_file)

        # Checking global structure: start = 5, end = 15
        @test prune_dict["start"] == 5
        @test prune_dict["end"] == 15

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end

    @testset "prune returns correct slices" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = splitext(tmp_file)[1] * "_pruning.toml"

        tags = ["energy"]
        data_dict = Dict("energy" => Float64.(1:20))
        make_1d_file(tmp_file, tags, data_dict)

        # Write Global TOML manually
        open(toml_file, "w") do fl
            write(fl, "start = 5\nend = 15\n")
        end

        result = prune([tmp_file], tags)
        @test length(result["energy"][1]) == 11      # indices 5:15
        @test result["energy"][1][1] ≈ 5.0
        @test result["energy"][1][end] ≈ 15.0

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end

    @testset "prune multiple seeds (Global)" begin
        tmp_files = [tempname() * ".hdf5" for _ in 1:3]
        toml_files = [splitext(f)[1] * "_pruning.toml" for f in tmp_files]

        for (i, tmp_file) in enumerate(tmp_files)
            make_1d_file(tmp_file, ["energy"], Dict("energy" => Float64.(1:20) .+ i))
            open(toml_files[i], "w") do fl
                write(fl, "start = 3\nend = 10\n")
            end
        end

        result = prune(tmp_files, ["energy"])
        @test length(result["energy"]) == 3
        @test length(result["energy"][1]) == 8 # 3 to 10

        foreach(f -> rm(f, force=true), tmp_files)
        foreach(f -> rm(f, force=true), toml_files)
    end
end

@testset "Global Pruning (2D Intervals)" begin

    @testset "prune_analysis_interval writes correct Global TOML" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = splitext(tmp_file)[1] * "_pruning.toml"

        betas = [0.5, 1.0, 2.0]
        make_2d_file(tmp_file, ["energy"], Dict("energy" => rand(20, 3)), 1.0, betas)

        # Global input: start 5, end 15
        simulate_input("5\n15\n") do
            prune_analysis_interval([tmp_file], ["energy"])
        end

        @test isfile(toml_file)
        prune_dict = TOML.parsefile(toml_file)
        @test prune_dict["start"] == 5
        @test prune_dict["end"] == 15

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end

    @testset "prune_interval returns correct slices" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = splitext(tmp_file)[1] * "_pruning.toml"

        betas = [0.5, 1.0, 2.0]
        energy_data = rand(20, 3)
        make_2d_file(tmp_file, ["energy"], Dict("energy" => energy_data), 1.0, betas)

        open(toml_file, "w") do fl
            write(fl, "start = 5\nend = 15\n")
        end

        result = prune_interval([tmp_file], ["energy"])
        @test size(result["energy"][1]) == (11, 3)     # rows 5:15
        @test result["energy"][1] ≈ energy_data[5:15, :]

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end

    @testset "prune_interval errors on Nx1 data" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = splitext(tmp_file)[1] * "_pruning.toml"

        # Nx1 matrix triggers error in prune_interval (expects multi-beta)
        dfile = DumpFile(tmp_file)
        write_data!(dfile, "energy", rand(20, 1))

        open(toml_file, "w") do fl
            write(fl, "start = 1\nend = 5\n")
        end

        @test_throws ErrorException prune_interval([tmp_file], ["energy"])

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end
end