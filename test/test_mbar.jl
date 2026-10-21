using Test
using Random
using Statistics
using LinearAlgebra
using HDF5

# Exactly solvable toy: 6-site Heisenberg ring plus a small field (to break degeneracies).
# Product states σ are sampled exactly from P_β(σ) ∝ ⟨σ|e^{-βH}|σ⟩ and their trajectories
# log‖e^{-βH/2}σ‖ and ⟨H⟩ are computed by exact diagonalization, mimicking METTS reruns.
module ToyED
using LinearAlgebra, Random

const N  = 6
const sz = [0.5 0.0; 0.0 -0.5]
const sp = [0.0 1.0; 0.0 0.0]

site_op(o, i) = kron([j == i ? o : Matrix(1.0I, 2, 2) for j in 1:N]...)

function hamiltonian()
    H = zeros(2^N, 2^N)
    for i in 1:N
        j = mod1(i + 1, N)
        H += site_op(sz, i) * site_op(sz, j) +
             0.5 * (site_op(sp, i) * site_op(sp', j) + site_op(sp', i) * site_op(sp, j))
    end
    H += 0.3 * sum(site_op(sz, i) for i in 1:N)
    return H
end

const H = hamiltonian()
const ev, V = let F = eigen(Symmetric(H)); (F.values, F.vectors) end

expH(b) = V * Diagonal(exp.(-b .* ev)) * V'
lnZ(b) = log(sum(exp.(-b .* ev)))
function boltzmann(b)
    w = exp.(-b .* (ev .- ev[1]))
    return w ./ sum(w)
end
energy(b)  = sum(boltzmann(b) .* ev)
denergy(b) = (w = boltzmann(b); -(sum(w .* ev .^ 2) - sum(w .* ev)^2))

function trajectory(s, betas, cache)
    log_norm = Float64[]
    en = Float64[]
    for b in betas
        v = get!(() -> expH(b / 2), cache, b)[:, s]
        nv = norm(v)
        push!(log_norm, log(nv))
        φ = v / nv
        push!(en, dot(φ, H * φ))
    end
    return (log_norm=log_norm, energy=en)
end

function sample_states(rng, b, n)
    c = cumsum(diag(expH(b)))
    c ./= c[end]
    return [min(searchsortedfirst(c, rand(rng)), length(c)) for _ in 1:n]
end

function dataset(rng, beta_collapses, grids, nsamp)
    cache = Dict{Float64, Matrix{Float64}}()
    meas = [[trajectory(s, grids[k], cache) for s in sample_states(rng, beta_collapses[k], nsamp)]
            for k in eachindex(beta_collapses)]
    return meas, [collect(Float64, g) for g in grids]
end
end # module ToyED

@testset "MBAR" begin
    rng = MersenneTwister(1234)
    bcs = [0.6, 1.6, 3.0]

    # Same nominal grid built two ways, as METTS does for different collapse points:
    # the betas differ by float noise (e.g. 0.6 vs 0.6000000000000001).
    grid_a = [2 * (i * 0.1) for i in 2:20]
    grid_b = [2 * (0.3 + i * 0.1) for i in -1:17]
    @assert length(unique(vcat(grid_a, grid_b))) > length(grid_a)  # the noise is really there

    meas, all_betas = ToyED.dataset(rng, bcs, [grid_a, grid_b, grid_a], 400)
    mbar = MBARState(meas, all_betas, bcs; max_iter=5000, tol=1e-10)

    @testset "Free energies match exact ln Z ratios" begin
        @test mbar.converged
        f_exact = [ToyED.lnZ(bcs[1]) - ToyED.lnZ(b) for b in bcs]
        @test maximum(abs.(mbar.free_energies .- f_exact)) < 0.1
    end

    @testset "Reweighted energy and dE/dβ match exact values" begin
        betas_out, e_mean, _ = reweight_observable(mbar, "energy")
        @test maximum(abs.(e_mean .- ToyED.energy.(betas_out))) < 0.05

        _, de_mean, _ = METTSAnalysis.reweight_observable_dev(mbar, "energy")
        @test maximum(abs.(de_mean .- ToyED.denergy.(betas_out))) < 0.1
    end

    @testset "Output betas have no float-noise duplicates" begin
        betas_out, _, _ = reweight_observable(mbar, "energy")
        @test length(betas_out) == length(grid_a)
        @test all(diff(betas_out) .> 1e-6)
    end

    @testset "Free-energy profile is gauged at the first beta_collapse" begin
        betas_out, f = METTSAnalysis.compute_mbar_free_energies(mbar)
        idx = [findfirst(b -> isapprox(b, bc; atol=1e-8), betas_out) for bc in bcs]
        @test f[idx] ≈ mbar.free_energies atol=1e-6
    end

    @testset "Windows missing a beta_collapse are rejected" begin
        short = [all_betas[1], all_betas[2], filter(b -> b > 1.0, all_betas[3])]
        meas_short = [meas[1], meas[2], [(log_norm=m.log_norm[end-length(short[3])+1:end],
                                          energy=m.energy[end-length(short[3])+1:end]) for m in meas[3]]]
        @test_throws ErrorException MBARState(meas_short, short, bcs)
    end

    @testset "Non-convergence is reported and keeps the last iterate" begin
        m = @test_logs (:warn, r"did not converge") match_mode=:any MBARState(meas, all_betas, bcs; max_iter=2)
        @test !m.converged
        @test all(isfinite, m.free_energies)
    end
end

@testset "pool_states groups beta_collapses within tolerance" begin
    meas = [[(log_norm=[0.0],)], [(log_norm=[1.0],)], [(log_norm=[2.0],)]]
    betas = [[1.0], [1.0], [1.0]]
    pooled, _, bcs = pool_states(meas, betas, [1.0, 1.0 + 1e-12, 2.0])
    @test length(bcs) == 2
    @test length.(pooled) == [2, 1]
end

@testset "Run-parameter checks" begin
    function make_file(params; sampling=nothing)
        f = tempname() * ".hdf5"
        h5open(f, "w") do fid
            for (k, v) in params
                fid["params/$k"] = v
            end
            if sampling !== nothing
                for (k, v) in sampling
                    fid["sampling_params/$k"] = v
                end
            end
        end
        return f
    end
    base = Dict("L" => 12, "W" => 6, "J2" => 0.0, "lattice" => "YC", "maxm" => 1024, "cutoff" => 1e-6, "tau" => 0.1)

    f1 = make_file(base)
    f2 = make_file(base)
    @test read_run_parameters(f1)["lattice"] == "YC"
    @test check_run_parameters([f1, f2]) == base

    f3 = make_file(merge(base, Dict("maxm" => 2048)))
    @test_throws ErrorException check_run_parameters([f1, f3])

    f4 = make_file(Dict{String, Any}())
    @test_throws ErrorException check_run_parameters([f1, f4])
    @test_logs (:warn, r"cannot be verified") check_run_parameters([f4])

    f5 = make_file(base; sampling=merge(base, Dict("lattice" => "XC")))
    @test_throws ErrorException check_run_parameters([f5])

    f6 = make_file(base; sampling=merge(base, Dict("tau" => 0.5)))
    @test_logs (:warn, r"'tau'") check_run_parameters([f6])

    foreach(f -> rm(f, force=true), [f1, f2, f3, f4, f5, f6])
end
