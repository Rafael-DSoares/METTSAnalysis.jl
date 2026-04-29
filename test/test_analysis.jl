using Test
using TOML
using Random
using Statistics

@testset "Analysis: average_observable_single (Standard & Bootstrap)" begin

    # --- Setup Data ---
    Random.seed!(42)
    n_samples = 100
    n_betas = 5
    betas = collect(range(0.1, 1.0, length=n_betas))
    beta_collapse = 0.55

    mock_measurements = map(1:n_samples) do i
        (
            energy=randn(n_betas) .+ 10.0,
            magz=randn(n_betas) .- 1.0,
            log_norm=zeros(n_betas)
        )
    end

    @testset "Standard Deviation (No Bootstrap)" begin
        results = average_observable_single(mock_measurements, beta_collapse, betas)

        @test results isa NamedTuple
        @test isapprox(results.energy[1], 10.0, atol=0.3)

        @test hasproperty(results, :magz)
        @test hasproperty(results, :energy)
        @test !hasproperty(results, :lognorm)


        @test 0.05 < results.energy[2] < 0.15


        simple_measurements = [(energy=randn() + 5.0,) for _ in 1:n_samples]
        res_simple = average_observable_single(simple_measurements)
        @test isapprox(res_simple.energy[1], 5.0, atol=0.2)
    end


    @testset "Average values only for specific tags" begin

        results = average_observable_single(mock_measurements, beta_collapse, betas, ["magz"])

        @test results isa NamedTuple


        @test hasproperty(results, :magz)
        @test !hasproperty(results, :energy)
        @test !hasproperty(results, :log_norm)


        @test isapprox(results.magz[1], -1.0, atol=0.3)
        @test 0.05 < results.magz[2] < 0.15


        simple_measurements = [(energy=randn() + 5.0, magz=randn() - 2.0) for _ in 1:n_samples]
        res_simple = average_observable_single(simple_measurements, ["magz"])


        @test hasproperty(res_simple, :magz)
        @test !hasproperty(res_simple, :energy)

        # Verify the math for the new mock magz data
        @test isapprox(res_simple.magz[1], -2.0, atol=0.3)
    end


    @testset "Bootstrap Version" begin
        tags = ["energy"]
        results_boot = average_observable_single(
            mock_measurements, beta_collapse, betas, tags; bootstrap=true, n_bootstrap=100
        )

        @test results_boot isa NamedTuple


        @test !hasproperty(results_boot, :magz)
        @test hasproperty(results_boot, :energy)
        @test !hasproperty(results_boot, :lognorm)

        @test isapprox(results_boot.energy[1], 10.0, atol=0.3)
        @test results_boot.energy[2] > 0.0 # Error should be positive
    end

    @testset "Edge Case: Empty Data" begin
        empty_data = NamedTuple[]
        @test average_observable_single(empty_data) == NamedTuple()
        @test average_observable_single(empty_data, 0.5, [0.5]; bootstrap=true) == NamedTuple()
    end
end