using Test
using TOML
using Random
using Statistics

@testset "Math: bootstrap_average_observable" begin

    @testset "1D Vector Bootstrap" begin
        Random.seed!(42)
        N = 1000
        mock_data = randn(N) .+ 10.0 
        
        obs_mean, obs_err = bootstrap_average_observable(mock_data; n_bootstrap=500)

        @test isapprox(obs_mean, 10.0, atol=0.1)
        @test isapprox(obs_err, 0.0316, atol=0.005) 
    end

    @testset "NamedTuple Bootstrap" begin
        Random.seed!(42)
        nt_data = (
            energy = randn(500) .- 2.5,
            magz = randn(500) .+ 1.0
        )
        
        results = bootstrap_average_observable(nt_data; n_bootstrap=200)
        
        @test results isa NamedTuple
        @test hasproperty(results, :energy)
        @test hasproperty(results, :magz)
        
        @test results.energy isa Tuple{Float64, Float64}
        @test isapprox(results.energy[1], -2.5, atol=0.1)
        @test isapprox(results.magz[1], 1.0, atol=0.1)
    end
end