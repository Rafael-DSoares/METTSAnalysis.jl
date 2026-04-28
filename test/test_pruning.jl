using Test
using TOML
using Dumper

# ---- Helpers ----------------------------------------------------------------

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

# ---- Tests ------------------------------------------------------------------

@testset "prune and prune_analysis (1D)" begin

    @testset "prune_analysis writes correct TOML" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = splitext(tmp_file)[1] * "_pruning.toml"

        tags = ["energy", "magnetization"]
        data_dict = Dict(
            "energy" => Float64.(1:20),
            "magnetization" => Float64.(1:20)
        )
        make_1d_file(tmp_file, tags, data_dict)

        # Simulate user input: start=5, end=15 for each observable
        simulate_input("5\n15\n5\n15\n") do
            prune_analysis([tmp_file], tags; toml_name=nothing)
        end

        @test isfile(toml_file)
        prune_dict = TOML.parsefile(toml_file)

        # Native TOML ints, no parse() needed!
        @test prune_dict["start"]["energy"] == 5
        @test prune_dict["end"]["energy"] == 15
        @test prune_dict["start"]["magnetization"] == 5
        @test prune_dict["end"]["magnetization"] == 15

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end

    @testset "prune_analysis default indices (empty input)" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = splitext(tmp_file)[1] * "_pruning.toml"

        tags = ["energy"]
        data_dict = Dict("energy" => Float64.(1:20))
        make_1d_file(tmp_file, tags, data_dict)

        # Empty input -> defaults to 1 and len
        simulate_input("\n\n") do
            prune_analysis([tmp_file], tags)
        end

        prune_dict = TOML.parsefile(toml_file)
        @test prune_dict["start"]["energy"] == 1
        @test prune_dict["end"]["energy"] == 20

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end

    @testset "prune returns correct slices" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = splitext(tmp_file)[1] * "_pruning.toml"

        tags = ["energy"]
        data_dict = Dict("energy" => Float64.(1:20))
        make_1d_file(tmp_file, tags, data_dict)

        # Write TOML manually - Use native integers (no quotes around 5 and 15)
        open(toml_file, "w") do fl
            write(fl, "[start]\nenergy = 5\n\n[end]\nenergy = 15\n")
        end

        result = prune([tmp_file], tags)
        @test haskey(result, "energy")
        @test length(result["energy"]) == 1          # one seed
        @test length(result["energy"][1]) == 11      # indices 5:15
        @test result["energy"][1][1] ≈ 5.0
        @test result["energy"][1][end] ≈ 15.0

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end

    @testset "prune errors on missing TOML" begin
        tmp_file = tempname() * ".hdf5"
        make_1d_file(tmp_file, ["energy"], Dict("energy" => [1.0]))

        @test_throws ErrorException prune([tmp_file], ["energy"])
        rm(tmp_file, force=true)
    end

    @testset "prune errors on non-1D data" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = splitext(tmp_file)[1] * "_pruning.toml"

        # Manually write 2D data to trigger the error
        dfile = DumpFile(tmp_file)
        for row in eachrow(rand(10, 3))
            dump!(dfile, "energy", collect(row))
        end

        open(toml_file, "w") do fl
            write(fl, "[start]\nenergy = 1\n\n[end]\nenergy = 5\n")
        end

        @test_throws ErrorException prune([tmp_file], ["energy"])

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end

    @testset "prune multiple seeds" begin
        tmp_files = [tempname() * ".hdf5" for _ in 1:3]
        toml_files = [splitext(f)[1] * "_pruning.toml" for f in tmp_files]

        for (i, tmp_file) in enumerate(tmp_files)
            data_dict = Dict("energy" => Float64.(1:20) .+ i)
            make_1d_file(tmp_file, ["energy"], data_dict)

            open(splitext(tmp_file)[1] * "_pruning.toml", "w") do fl
                write(fl, "[start]\nenergy = 3\n\n[end]\nenergy = 10\n")
            end
        end

        result = prune(tmp_files, ["energy"])
        @test length(result["energy"]) == 3
        for i in 1:3
            @test length(result["energy"][i]) == 8   # indices 3:10
        end

        foreach(f -> rm(f, force=true), tmp_files)
        foreach(f -> rm(f, force=true), toml_files)
    end

    @testset "prune_analysis custom toml_name" begin
        tmp_file = tempname() * ".hdf5"
        toml_file = joinpath(dirname(tmp_file), "custom_pruning.toml")

        make_1d_file(tmp_file, ["energy"], Dict("energy" => Float64.(1:20)))

        simulate_input("2\n18\n") do
            prune_analysis([tmp_file], ["energy"]; toml_name="custom_pruning.toml")
        end

        @test isfile(toml_file)
        prune_dict = TOML.parsefile(toml_file)
        @test prune_dict["start"]["energy"] == 2
        @test prune_dict["end"]["energy"] == 18

        rm(tmp_file, force=true)
        rm(toml_file, force=true)
    end


    @testset "prune_interval and prune_analysis_interval" begin

        @testset "prune_analysis_interval writes correct TOML" begin
            tmp_file = tempname() * ".hdf5"
            toml_file = splitext(tmp_file)[1] * "_pruning.toml"

            betas = [0.5, 1.0, 2.0]
            beta_collapse = 1.0
            nmetts = 20
            energy_data = rand(nmetts, 3)

            make_2d_file(tmp_file, ["energy"], Dict("energy" => energy_data), beta_collapse, betas)

            simulate_input("5\n15\n") do
                prune_analysis_interval([tmp_file], ["energy"]) 
            end

            @test isfile(toml_file)
            prune_dict = TOML.parsefile(toml_file)
            @test prune_dict["start"]["energy"] == 5
            @test prune_dict["end"]["energy"]   == 15

            rm(tmp_file, force=true)
            rm(toml_file, force=true)
        end

        @testset "prune_analysis_interval default indices" begin
            tmp_file = tempname() * ".hdf5"
            toml_file = splitext(tmp_file)[1] * "_pruning.toml"

            betas = [0.5, 1.0, 2.0]
            beta_collapse = 1.0
            nmetts = 20

            make_2d_file(tmp_file, ["energy"], Dict("energy" => rand(nmetts, 3)), beta_collapse, betas)

            simulate_input("\n\n") do
                prune_analysis_interval([tmp_file], ["energy"])
            end

            prune_dict = TOML.parsefile(toml_file)
            @test prune_dict["start"]["energy"] == 1
            @test prune_dict["end"]["energy"]   == nmetts

            rm(tmp_file, force=true)
            rm(toml_file, force=true)
        end

        @testset "prune_interval returns correct slices" begin
            tmp_file = tempname() * ".hdf5"
            toml_file = splitext(tmp_file)[1] * "_pruning.toml"

            betas = [0.5, 1.0, 2.0]
            beta_collapse = 1.0
            nmetts = 20
            energy_data = rand(nmetts, 3)

            make_2d_file(tmp_file, ["energy"], Dict("energy" => energy_data), beta_collapse, betas)

            open(toml_file, "w") do fl
                write(fl, "[start]\nenergy = 5\n\n[end]\nenergy = 15\n")
            end

            result = prune_interval([tmp_file], ["energy"])
            @test haskey(result, "energy")
            @test length(result["energy"]) == 1            # one seed
            @test size(result["energy"][1]) == (11, 3)     # rows 5:15, all betas
            @test result["energy"][1] ≈ energy_data[5:15, :]

            rm(tmp_file, force=true)
            rm(toml_file, force=true)
        end

        @testset "prune_interval errors on non-2D data" begin
            tmp_file = tempname() * ".hdf5"
            toml_file = splitext(tmp_file)[1] * "_pruning.toml"

            # Write 1D data to trigger the error
            make_1d_file(tmp_file, ["energy"], Dict("energy" => Float64.(1:20)))

            # Need to append betas to trick the function into reading it
            dfile = DumpFile(tmp_file)
            write_data!(dfile, "beta_collapse", 1.0)
            write_data!(dfile, "betas", [0.5, 1.0, 2.0])

            open(toml_file, "w") do fl
                write(fl, "[start]\nenergy = 1\n\n[end]\nenergy = 5\n")
            end

            @test_throws ErrorException prune_interval([tmp_file], ["energy"])

            rm(tmp_file, force=true)
            rm(toml_file, force=true)
        end

        @testset "prune_interval multiple seeds" begin
            tmp_files = [tempname() * ".hdf5" for _ in 1:3]
            toml_files = [splitext(f)[1] * "_pruning.toml" for f in tmp_files]

            betas = [0.5, 1.0, 2.0]
            beta_collapse = 1.0
            nmetts = 20

            for tmp_file in tmp_files
                make_2d_file(tmp_file, ["energy"], Dict("energy" => rand(nmetts, 3)), beta_collapse, betas)

                open(splitext(tmp_file)[1] * "_pruning.toml", "w") do fl
                    write(fl, "[start]\nenergy = 3\n\n[end]\nenergy = 10\n")
                end
            end

            result = prune_interval(tmp_files, ["energy"])
            @test length(result["energy"]) == 3
            for i in 1:3
                @test size(result["energy"][i]) == (8, 3)  # rows 3:10, all betas
            end

            foreach(f -> rm(f, force=true), tmp_files)
            foreach(f -> rm(f, force=true), toml_files)
        end

        @testset "prune_interval custom toml_name" begin
            tmp_file = tempname() * ".hdf5"
            toml_file = joinpath(dirname(tmp_file), "custom_pruning.toml")

            make_2d_file(tmp_file, ["energy"], Dict("energy" => rand(20, 3)), 1.0, [0.5, 1.0, 2.0])

            open(toml_file, "w") do fl
                write(fl, "[start]\nenergy = 2\n\n[end]\nenergy = 18\n")
            end

            result = prune_interval([tmp_file], ["energy"]; toml_name="custom_pruning.toml")
            @test size(result["energy"][1]) == (17, 3)

            rm(tmp_file, force=true)
            rm(toml_file, force=true)
        end

    end
end