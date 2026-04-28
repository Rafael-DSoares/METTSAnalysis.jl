using Test
using TOML
using Random
using Statistics
using Test
using Random

@testset "Analysis: average_observable_single and _boot" begin

    # --- Setup Data ---
    Random.seed!(42)
    n_samples = 100
    n_betas = 5
    betas = collect(range(0.1, 1.0, length=n_betas))
    beta_collapse = 0.55

    mock_measurements = map(1:n_samples) do i
        (
            energy = randn(n_betas) .+ 10.0,  # Mean 10
            magz   = randn(n_betas) .- 1.0    # Mean -1
        )
    end

    @testset "Standard Analytical (No Bootstrap)" begin
        # 1. Test the "at beta_collapse" version
        results = average_observable_single(mock_measurements, beta_collapse, betas)
        
        @test results isa NamedTuple
        @test isapprox(results.energy[1], 10.0, atol=0.3)
        # Check that error is roughly sigma/sqrt(N) -> 1.0/sqrt(100) = 0.1
        @test 0.05 < results.energy[2] < 0.15

        # 2. Test the "all data" version (assuming 1D obs per sample)
        # Let's create a simpler 1D set for this
        simple_measurements = [(energy=randn()+5.0,) for _ in 1:n_samples]
        res_simple = average_observable_single(simple_measurements)
        @test isapprox(res_simple.energy[1], 5.0, atol=0.2)
    end

    @testset "Bootstrap Version" begin
        tags = ["energy"]
        results_boot = average_observable_single_boot(
            mock_measurements, beta_collapse, betas, tags; n_bootstrap=100
        )
        
        @test hasproperty(results_boot, :energy)
        @test !hasproperty(results_boot, :magz) # Should only have requested tags
        @test isapprox(results_boot.energy[1], 10.0, atol=0.3)
        @test results_boot.energy[2] > 0.0 # Error should be positive
    end

    @testset "Edge Case: Empty Data" begin
        empty_data = NamedTuple[]
        @test average_observable_single(empty_data) == NamedTuple()
        @test average_observable_single_boot(empty_data, 0.5, [0.5]) == NamedTuple()
    end
end