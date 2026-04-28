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

# ---- Tests ------------------------------------------------------------------

@testset "I/O: load_metts_file" begin
    tmp_file = tempname() * ".hdf5"
    toml_file = splitext(tmp_file)[1] * "_pruning.toml"
    
    nmetts = 20
    nbetas = 5
    energy_full, log_norm_full = make_tmp_metts_file(tmp_file, nmetts, nbetas; add_mag=true)

    @testset "No TOML file -> Warns and loads full data" begin
        # @test_logs catches the specific warning message to ensure it fired
        bc, betas, meas = @test_logs (:warn, r"Pruning file not found") load_metts_file_interval(tmp_file, "energy")
        
        @test bc == 1.0
        @test length(betas) == nbetas
        @test length(meas) == nmetts
        @test meas[1] isa NamedTuple
        @test hasproperty(meas[1], :energy)
        @test hasproperty(meas[1], :log_norm)
        @test meas[1].energy ≈ energy_full[1, :]
    end

    @testset "TOML exists -> Loads pruned data without warnings" begin
        open(toml_file, "w") do fl
            write(fl, "[start]\nenergy = 5\n\n[end]\nenergy = 15\n")
        end

        bc, betas, meas = @test_logs load_metts_file_interval(tmp_file, "energy")
        
        @test length(meas) == 11 # Indices 5 through 15 inclusive
        @test meas[1].energy ≈ energy_full[5, :]
        @test meas[end].log_norm ≈ log_norm_full[15, :]
    end

    @testset "not_use_prune = true -> Ignores TOML, no warnings" begin
        # TOML still exists from previous test
        bc, betas, meas = @test_logs load_metts_file_interval(tmp_file, "energy"; not_use_prune=true)
        
        @test length(meas) == nmetts # Back to full size
        @test meas[5].energy ≈ energy_full[5, :]
    end

    @testset "TOML exists but missing requested observable -> Warns and loads full" begin
        # TOML only has "energy", but we ask for "magnetization"
        bc, betas, meas = @test_logs (:warn, r"Observable 'magnetization' not found") load_metts_file_interval(tmp_file, "magnetization")
        
        @test length(meas) == nmetts
        @test hasproperty(meas[1], :magnetization)
    end

    # Cleanup
    rm(tmp_file, force=true)
    rm(toml_file, force=true)
end