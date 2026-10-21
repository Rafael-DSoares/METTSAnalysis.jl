using Test
using TOML
using Statistics
using StatsBase
using Random

function make_tmp_metts_file(filename, nmetts, nbetas; add_mag=false)
    dfile = DumpFile(filename)
    
    write_data!(dfile, "beta_collapse", 1.0)
    write_data!(dfile, "betas", collect(range(0.1, 2.0, length=nbetas)))
    
    energy_data = rand(nmetts, nbetas)
    log_norm_data = rand(nmetts, nbetas)
    
    for i in 1:nmetts
        dump!(dfile, "energy", energy_data[i, :])
        dump!(dfile, "log_norm", log_norm_data[i, :])
        if add_mag
            dump!(dfile, "magnetization", rand(nbetas))
        end
    end
    return energy_data, log_norm_data
end

@testset "I/O: load_metts_file" begin
    tmp_file = tempname() * ".hdf5"
    toml_file = splitext(tmp_file)[1] * "_pruning.toml"
    
    nmetts = 20
    nbetas = 5
    energy_full, log_norm_full = make_tmp_metts_file(tmp_file, nmetts, nbetas; add_mag=true)

    @testset "No TOML file -> Warns and loads full data" begin
        bc, betas, meas = @test_logs (:warn, r"Pruning file not found at .* Loading full range.") load_metts_file_interval(tmp_file, ["energy"])

        @test bc == 1.0
        @test length(betas) == nbetas
        @test length(meas) == nmetts

        @test meas isa Vector{<:NamedTuple}
        @test meas[1] isa NamedTuple
        
        @test hasproperty(meas[1], :energy)
        @test hasproperty(meas[1], :log_norm)
        @test keys(meas[1]) == (:energy, :log_norm)

        @test meas[1].energy isa Vector{Float64}
        @test length(meas[1].energy) == nbetas
        @test meas[1].energy ≈ energy_full[1, :]
        
        mid = nmetts ÷ 2
        @test meas[mid].log_norm ≈ log_norm_full[mid, :]
    end

    @testset "TOML exists -> Loads pruned data without warnings" begin
        open(toml_file, "w") do fl
            write(fl, "start = 5\n end = 15\n")
        end

        bc, betas, meas = @test_logs load_metts_file_interval(tmp_file, ["energy"])
        
        @test length(meas) == 11 # Indices 5 through 15 inclusive
        @test meas[1].energy ≈ energy_full[5, :]
        @test meas[end].log_norm ≈ log_norm_full[15, :]
    end
    
    @testset "Malformed TOML -> Warns and loads full data" begin
        open(toml_file, "w") do fl
            write(fl, "start = 5\n end : 15\n") 
        end

        bc, betas, meas = @test_logs (:warn, r"Failed to parse pruning file at .* Using full range.") load_metts_file_interval(tmp_file, ["energy"])
        
        @test length(meas) == nmetts 
    end

    @testset "not_use_prune = true -> Ignores TOML, no warnings" begin
        bc, betas, meas = @test_logs load_metts_file_interval(tmp_file, ["energy"]; not_use_prune=true)
        
        @test length(meas) == nmetts
        @test meas[5].energy ≈ energy_full[5, :]
    end
    
    # Cleanup
    rm(tmp_file, force=true)
    rm(toml_file, force=true)
end
@testset "I/O: file integrity checks" begin
    @testset "Evolved files must hold exactly one sample" begin
        f = tempname() * ".hdf5"
        d = DumpFile(f)
        write_data!(d, "beta_collapse", 1.0)
        dump!(d, "betas", [0.5, 1.0])
        dump!(d, "energy", [1.0, 2.0])
        dump!(d, "log_norm", [0.1, 0.2])
        bc, betas, meas = @test_logs load_evolved_metts_file(f, ["energy"])
        @test length(meas) == 1 && betas == [0.5, 1.0]

        dump!(d, "energy", [1.0, 2.0])     # the job ran a second time
        dump!(d, "log_norm", [0.1, 0.2])
        @test_throws ErrorException load_evolved_metts_file(f, ["energy"])
        rm(f, force=true)
    end

    @testset "Misaligned energy / product_state lengths are rejected" begin
        f = tempname() * ".hdf5"
        d = DumpFile(f)
        write_data!(d, "beta_collapse", 1.0)
        for i in 1:30
            dump!(d, "energy", randn())
            dump!(d, "product_state", rand(1:2, 4))
        end
        states, _ = redirect_stdout(devnull) do
            get_uncorrelated_product_states(f, "energy"; not_use_prune=true)
        end
        @test size(states, 2) == 4

        dump!(d, "energy", randn())        # killed before product_state was written
        err = try
            redirect_stdout(devnull) do
                get_uncorrelated_product_states(f, "energy"; not_use_prune=true)
            end
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("misaligned", err.msg)
        rm(f, force=true)
    end
end
