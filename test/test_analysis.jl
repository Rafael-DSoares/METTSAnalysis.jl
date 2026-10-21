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
@testset "Analysis: autocorrelation-aware errors" begin
    rng = MersenneTwister(11)
    ar1(n, φ) = (x = zeros(n); x[1] = randn(rng) / sqrt(1 - φ^2);
                 for i in 2:n; x[i] = φ * x[i-1] + randn(rng); end; x)

    @testset "ESS error matches the exact AR(1) error" begin
        φ, n = 0.9, 5000
        exact = sqrt((1 + φ) / (1 - φ) / n) / sqrt(1 - φ^2)   # σ sqrt(τ_int / n)
        series = [ar1(n, φ) for _ in 1:20]
        ess_err   = mean(last(METTSAnalysis.ess_average_observable(x)) for x in series)
        naive_err = mean(last(METTSAnalysis.standard_average_observable(x)) for x in series)
        @test isapprox(ess_err, exact; rtol=0.15)
        @test naive_err < exact / 3
    end

    @testset "Independent samples: ESS error is never below the naive error" begin
        x = randn(rng, 2000)
        @test last(METTSAnalysis.ess_average_observable(x)) >= last(METTSAnalysis.standard_average_observable(x))
        @test effective_sample_size(x) <= length(x)
    end

    @testset "error_method selection" begin
        data = [(energy=v,) for v in ar1(2000, 0.9)]
        e_ess   = average_observable_single(data).energy[2]
        e_naive = average_observable_single(data; error_method=:naive).energy[2]
        e_boot  = average_observable_single(data; bootstrap=true, n_bootstrap=50).energy[2]
        @test e_ess > 2 * e_naive
        @test e_boot < e_ess
        @test_throws ErrorException average_observable_single(data; error_method=:foo)
    end

    @testset "Short chains fall back to independent samples" begin
        x = randn(rng, 10)
        @test (@test_logs (:warn, r"too few") effective_sample_size(x)) == 10.0
    end

    @testset "Several chains are combined with length weights" begin
        c1 = [(energy=v,) for v in ar1(3000, 0.5)]
        c2 = [(energy=v,) for v in ar1(1000, 0.5)]
        m1, e1 = average_observable_single(c1).energy
        m2, e2 = average_observable_single(c2).energy
        m, e = average_observable_single([c1, c2]).energy
        @test m ≈ 0.75 * m1 + 0.25 * m2
        @test e ≈ sqrt((0.75 * e1)^2 + (0.25 * e2)^2)

        # Interval data: values at beta_collapse
        betas = [0.5, 1.0]
        ic = [[(energy=[v, -v], log_norm=[0.0, 0.0]) for v in ar1(500, 0.5)] for _ in 1:3]
        res = average_observable_single(ic, 1.0, betas)
        @test keys(res) == (:energy,)
        @test res.energy[1] ≈ mean(vcat([[-m.energy[1] for m in c] for c in ic]...))
    end

    @testset "Disagreeing seeds are flagged" begin
        good = [[(energy=v,) for v in ar1(2000, 0.5)] for _ in 1:3]
        bad  = [(energy=v + 1.0,) for v in ar1(2000, 0.5)]
        @test_logs (:warn, r"disagree") average_observable_single([good; [bad]])
    end

    @testset "group_chains_single keeps seeds separate" begin
        meas = [[(energy=1.0,)], [(energy=2.0,), (energy=3.0,)], [(energy=4.0,)]]
        chains, bcs = group_chains_single(meas, [1.0, 1.0 + 1e-12, 2.0])
        @test bcs == [1.0, 2.0]
        @test length.(chains) == [2, 1]
        @test length.(chains[1]) == [1, 2]
    end
end
