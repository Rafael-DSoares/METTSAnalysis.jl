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
const Sz_total = sum(site_op(sz, i) for i in 1:N)
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
    mz = Float64[]
    for b in betas
        v = get!(() -> expH(b / 2), cache, b)[:, s]
        nv = norm(v)
        push!(log_norm, log(nv))
        φ = v / nv
        push!(en, dot(φ, H * φ))
        push!(mz, dot(φ, Sz_total * φ))
    end
    return (log_norm=log_norm, energy=en, mz=mz)
end

function entropy(Hλ, b)
    e = eigvals(Symmetric(Hλ))
    p = exp.(-b .* (e .- e[1]))
    p ./= sum(p)
    return -sum(x * log(x) for x in p if x > 0)
end

"""Exact (∂T/∂λ)_S for H + λ Sz_total at λ = 0, by following the isentrope."""
function adiabatic_dT_dλ(b; δ=1e-4)
    S0 = entropy(H, b)
    function isentropic_T(λ)
        lo, hi = b / 2, 2b                    # entropy decreases with β
        for _ in 1:200
            mid = (lo + hi) / 2
            entropy(H + λ * Sz_total, mid) > S0 ? (lo = mid) : (hi = mid)
        end
        return 2 / (lo + hi)
    end
    return (isentropic_T(δ) - isentropic_T(-δ)) / (2δ)
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

    @testset "Energy key, precomputed derivatives and elastocaloric formula" begin
        renamed = [[(log_norm=m.log_norm, en=m.energy, mz=m.mz) for m in c] for c in meas]
        mb2 = MBARState(renamed, all_betas, bcs; max_iter=5000, tol=1e-10)
        _, d_ref, _ = METTSAnalysis.reweight_observable_dev(mbar, "mz")
        _, d_ren, _ = METTSAnalysis.reweight_observable_dev(mb2, "mz"; energy=:en)
        @test d_ren ≈ d_ref

        pre = add_beta_derivatives(meas, all_betas, ["mz", "energy"])
        mb3 = MBARState(pre, all_betas, bcs; max_iter=5000, tol=1e-10)
        @test METTSAnalysis.reweight_observable_dev(mb3, "mz")[2] ≈ d_ref

        @test_throws ErrorException METTSAnalysis.compute_elastocaloric_effect(mbar, "energy", ["mz"], [1.0, 2.0])
    end

    @testset "Exhaustive enumeration at beta_collapse = 0 is exact" begin
        # At β_c = 0 every product state has q = 1, so one copy of each is an exact
        # "sample": the MBAR weights are exactly ⟨σ|e^{-βH}|σ⟩ and only spline errors remain.
        grid = collect(0.0:0.05:3.0)
        cache = Dict{Float64, Matrix{Float64}}()
        all_states = [[ToyED.trajectory(s, grid, cache) for s in 1:2^ToyED.N]]
        mb0 = MBARState(all_states, [grid], [0.0])

        betas_out, e_mean, _ = reweight_observable(mb0, "energy")
        @test e_mean ≈ ToyED.energy.(betas_out) rtol=1e-10

        _, de_mean, _ = METTSAnalysis.reweight_observable_dev(mb0, "energy")
        @test maximum(abs.(de_mean .- ToyED.denergy.(betas_out))) < 1e-3

        _, dTdλ, _ = METTSAnalysis.compute_elastocaloric_effect(mb0, "energy", ["mz"], [1.0])
        @test isnan(dTdλ[1])                                   # β = 0
        for b in (1.0, 2.0)
            i = findfirst(x -> isapprox(x, b; atol=1e-8), betas_out)
            @test isapprox(dTdλ[i], ToyED.adiabatic_dT_dλ(b); rtol=1e-3)
        end
    end

    @testset "Overlap matrix and check" begin
        O = compute_overlap_matrix(mbar)
        @test all(sum(O, dims=2) .≈ 1)
        @test O ≈ compute_overlap_matrix(meas, all_betas, bcs; max_inter_mbar=5000, mbar_tol=1e-10) atol=1e-6
        @test_logs check_overlap(mbar; threshold=1e-6)
        @test_logs (:warn, r"Low overlap") match_mode=:any check_overlap(mbar; threshold=0.999)
    end

    @testset "Bootstrap routines run quietly with verbose=false" begin
        out = @test_logs bootstrap_mbar_reweight(meas, all_betas, bcs, "energy"; n_bootstrap=3, verbose=false)
        @test length(out) == 6
        @test all(out[3] .> 0)
    end

    @testset "Non-convergence is reported and keeps the last iterate" begin
        m = @test_logs (:warn, r"did not converge") match_mode=:any MBARState(meas, all_betas, bcs; max_iter=2)
        @test !m.converged
        @test all(isfinite, m.free_energies)
    end

    @testset "Block bootstrap" begin
        blocks_of = METTSAnalysis._blocks
        @test blocks_of([1, 1, 1, 2, 2, 3], 2) == [1:2, 3:3, 4:5, 6:6]
        @test blocks_of([1, 1, 1, 2, 2, 3], 10) == [1:3, 4:5, 6:6]
        @test blocks_of([1, 1, 2], 1) == [1:1, 2:2, 3:3]
        @test_throws ErrorException blocks_of([1, 2, 1], 2)   # chain 1 not contiguous
        @test_throws ErrorException blocks_of([1, 1], 0)

        # A replicate is a sequence of whole blocks, the last one possibly cut
        blocks = blocks_of(repeat(1:4, inner=5), 3)
        idx = METTSAnalysis._block_resample(MersenneTwister(1), blocks, 20)
        @test length(idx) == 20
        p = 1
        while p <= 20
            b = findfirst(bl -> first(bl) == idx[p], blocks)
            @test !isnothing(b)
            len = min(length(blocks[b]), 21 - p)
            @test idx[p:p+len-1] == collect(blocks[b])[1:len]
            p += len
        end

        # One sample per chain: blocks of one sample, i.e. the ordinary bootstrap
        estimator = m -> reweight_observable(m, "energy")
        opts = (n_bootstrap=5, max_inter_mbar=1000, mbar_tol=1e-8, verbose=false)
        plain = METTSAnalysis._bootstrap_mbar(estimator, meas, all_betas, bcs; opts..., rng=MersenneTwister(7))
        singles = METTSAnalysis._bootstrap_mbar(estimator, meas, all_betas, bcs; opts..., rng=MersenneTwister(7),
                                                groups=[collect(eachindex(m)) for m in meas], block_length=3)
        @test plain.values_err == singles.values_err

        # Every state stored twice in a row (perfectly correlated pairs): the ordinary
        # bootstrap sees twice the independent samples and underestimates the error by √2;
        # blocks of the two copies restore it
        doubled = [vcat([[x, x] for x in m]...) for m in meas]
        pairs = [repeat(1:length(m), inner=2) for m in meas]
        err_plain = bootstrap_mbar_reweight(doubled, all_betas, bcs, "energy"; n_bootstrap=300, verbose=false)[3]
        err_block = bootstrap_mbar_reweight(doubled, all_betas, bcs, "energy"; n_bootstrap=300, verbose=false,
                                            groups=pairs, block_length=2)[3]
        @test 1.25 < mean(err_block ./ err_plain) < 1.6

        @test_throws ErrorException bootstrap_mbar_reweight(meas, all_betas, bcs, "energy"; n_bootstrap=2, verbose=false,
                                                            groups=[[1, 2]], block_length=2)

        # Whole chains with a single chain per state: nothing to resample
        one_chain = [fill(1, length(m)) for m in meas]
        @test_throws ErrorException bootstrap_mbar_reweight(meas, all_betas, bcs, "energy"; n_bootstrap=2, verbose=false,
                                                            groups=one_chain, block_length=typemax(Int))

        # Whole chains: four chains per state, errors include √(4/3)
        four = [repeat(1:4, inner=cld(length(m), 4))[1:length(m)] for m in meas]
        whole = METTSAnalysis._bootstrap_mbar(estimator, meas, all_betas, bcs; opts..., n_bootstrap=50,
                                              groups=four, block_length=typemax(Int), rng=MersenneTwister(3))
        @test all(isfinite, whole.values_err) && all(whole.values_err .> 0)
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
