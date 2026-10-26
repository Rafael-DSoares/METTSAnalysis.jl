using Dierckx
using Random

"""
Holds the converged state of an MBAR calculation, including the precalculated 
weight_matrix matrices.
"""
struct MBARState{T}
    measurements::T                       # The raw Vector of NamedTuples
    all_betas::Vector{Vector{Float64}}    # The beta arrays for each state
    beta_collapses::Vector{Float64}       # The target beta values
    K::Int                                # Number of states
    Nk::Vector{Int}                       # Number of samples per state
    U::Vector{Matrix{Float64}}            # Pre-calculated log weight matrix log(p_α)
    free_energies::Vector{Float64}        # The converged free energies (f)
    log_denom::Vector{Vector{Float64}}    # The converged denominators for reweighting
    converged::Bool                       # Whether the self-consistent iteration reached `tol`
end


const BETA_ATOL = 1e-8

_beta_index(betas, β) = findfirst(b -> isapprox(b, β; atol=BETA_ATOL), betas)

"""
    _common_betas(all_betas)

Betas present in every state's grid (compared within `BETA_ATOL`), taken from the first grid.

MBAR can only reweight to a β where every sample's log_norm is known; reweighting with
the subset of samples whose window contains β is biased. The tolerance also removes the
float-noise duplicates that `unique` keeps (e.g. 1.2 and 1.2000000000000002).
"""
function _common_betas(all_betas)
    common = [β for β in sort(all_betas[1]) if all(b -> !isnothing(_beta_index(b, β)), all_betas)]
    n_all = length(_unique_approx(sort(vcat(all_betas...)); atol=BETA_ATOL))
    if length(common) < n_all
        @warn "Only $(length(common)) of $n_all betas are present in every state's window; reweighting is restricted to those." maxlog=1
    end
    return common
end

"""
    MBARState(all_measurements, all_betas, beta_collapses; max_iter=500, tol=1e-10, alpha=0.6, verbose=true)

Constructs the MBARState by running the self-consistency equations until convergence.

Each measurement must carry `log_norm = log‖e^{-βH/2}σ‖` (the norm, not its square) on the
state's beta grid; the weight of a sample at β is `exp(2 log_norm)`. With `verbose=false`
a non-converged run is only recorded in `converged` instead of warning.
"""
function MBARState(all_measurements, all_betas::Vector{Vector{Float64}}, beta_collapses::Vector{Float64}; max_iter::Int64=500, tol::Float64=1e-10,alpha::Float64=0.6, verbose::Bool=true)
    
    @assert issorted(beta_collapses) "beta_collapses must be sorted ascending"

    K  = length(beta_collapses)
    Nk = [length(meas) for meas in all_measurements] ###  the number of measurments per each beta_collapse. Length only looks to the outer data. We assume that the data is organized as the function load_metts work.
    
    @assert length(all_measurements) == K && length(all_betas) == K

    # Every sample needs log_norm at every beta_collapse: a missing entry is not a zero
    # probability but an unknown one, and dropping it from the denominator biases MBAR.
    for k in 1:K, j in 1:K
        if isnothing(_beta_index(all_betas[k], beta_collapses[j]))
            error("beta_collapses[$j] = $(beta_collapses[j]) is outside the beta window of state $k " *
                  "([$(first(all_betas[k])), $(last(all_betas[k]))]). Every state's window must contain all beta_collapses.")
        end
    end

    # 1. Initialize first with chained BAR (this makes MBAR converge faster!)
    _, f_init = chained_bar(all_measurements, all_betas, beta_collapses)
    f = copy(f_init)
    f_new = zeros(K)

    ## create U matrix:
    #U = [fill(-Inf, Nk[k], K) for k in 1:K]

    U = [fill(NaN, Nk[k], K) for k in 1:K]

    for k in 1:K ## Loop over all states
        for j in 1:K ## Loop over all states
            idx = _beta_index(all_betas[k], beta_collapses[j]) ## index in state k's grid of the collapse temperature of state j (guaranteed to exist by the check above)
            for i in 1:Nk[k]
                U[k][i, j] = 2 * all_measurements[k][i].log_norm[idx]
            end
        end
    end


    log_denom = [zeros(Float64,Nk[k]) for k in 1:K] ### log of the MBAR denominator

    ### try to convege the free_energies
    converged = false
    last_error = Inf
    n_iter = 0

    for iter in 1:max_iter
        n_iter = iter
        # Step A: Update log denominators
        _update_log_denom!(log_denom, U, f, K, Nk)

        # Step B: Update free energies
        _update_free_energies!(f_new, f, U, log_denom, K, Nk)

        # Check convergence on the undamped update, so that `tol` is the actual tolerance
        last_error = maximum(abs.(f_new .- f))
        if last_error < tol
            converged = true
            f .= f_new
            break
        end

        f .= alpha .* f .+ (1.0 - alpha) .* f_new #### simple non-linear solver with relexation parameter
    end

    if converged
        @debug "MBAR successfully converged in $n_iter iterations."
    elseif verbose
        # The last iterate is a better estimate than the one-sided chained FEP initial guess
        @warn "MBAR did not converge after $max_iter iterations (last error: $last_error). Using the last iterate."
    end

    _update_log_denom!(log_denom, U, f, K, Nk)

    return MBARState(all_measurements, all_betas, beta_collapses, K, Nk, U, f, log_denom, converged)
end


"""
Estimates free energy differences between adjacent temperature states using
chained backward FEP (exponential averaging).
"""
function chained_bar(all_measurements, all_betas, beta_collapses)
    K = length(beta_collapses)
    @assert issorted(beta_collapses) "beta_collapses must be sorted ascending"
    @assert length(all_measurements) == K && length(all_betas) == K

    delta_f = zeros(K - 1)

    for k in 2:K
        β_k   = beta_collapses[k]
        β_km1 = beta_collapses[k - 1]
        betas_k = all_betas[k]
        meas_k  = all_measurements[k]
        N_k = length(meas_k)

        idx_km1 = findfirst(b -> isapprox(b, β_km1; atol=1e-8), betas_k)
        idx_k   = findfirst(b -> isapprox(b, β_k;   atol=1e-8), betas_k)
        
        isnothing(idx_km1) && error("beta_collapses[$(k-1)] = $β_km1 not found in betas of state $k")
        isnothing(idx_k)   && error("beta_collapses[$k] = $β_k not found in betas of state $k")

        log_ratios = [2 * (meas_k[i].log_norm[idx_km1] - meas_k[i].log_norm[idx_k]) for i in 1:N_k]
        delta_f[k - 1] = logsumexp(log_ratios) - log(N_k)
    end

    return delta_f, [0.0; cumsum(delta_f)]
end



"""
    _update_log_denom!(log_denom, U, f, K, Nk)
"""
function _update_log_denom!(log_denom::Vector{Vector{Float64}}, U::Vector{Matrix{Float64}}, f::Vector{Float64}, K::Int, Nk::Vector{Int})
    for k in 1:K ## Loop over all states
        for i in 1:Nk[k] ## Loop over all states
            m = -Inf
            for j in 1:K
                if !isnan(U[k][i, j])
                    val = log(Nk[j]) + f[j] + U[k][i, j]
                    if val > m
                        m = val
                    end
                end
            end

            sum_exp = 0.0
            for j in 1:K
                if !isnan(U[k][i, j])
                    sum_exp += exp(log(Nk[j]) + f[j] + U[k][i, j] - m)
                end
            end
            log_denom[k][i] = m + log(sum_exp)
        end
    end
    return log_denom
end

"""

_update_free_energies!(f_new::Vector{Float64}, 
                           f::Vector{Float64}, 
                           U::Vector{Matrix{Float64}}, 
                           log_denom::Vector{Vector{Float64}}, 
                           K::Int, 
                           Nk::Vector{Int})
"""
function _update_free_energies!(f_new::Vector{Float64}, 
                                f::Vector{Float64}, 
                                U::Vector{Matrix{Float64}}, 
                                log_denom::Vector{Vector{Float64}}, 
                                K::Int, 
                                Nk::Vector{Int})
    for j in 1:K
        m = -Inf
        # Find max for log-sum-exp trick
        for k in 1:K
            for i in 1:Nk[k]
                if !isnan(U[k][i, j])
                    val = U[k][i, j] - log_denom[k][i]
                    if val > m
                        m = val
                    end
                end
            end
        end
        
        if m == -Inf
            f_new[j] = f[j]
        else
            sum_exp = 0.0
            for k in 1:K
                for i in 1:Nk[k]
                    if !isnan(U[k][i, j])
                        sum_exp += exp(U[k][i, j] - log_denom[k][i] - m)
                    end
                end
            end
            f_new[j] = -(m + log(sum_exp))
        end
    end
    
    f_new .-= f_new[1] # gauge fixing!
    return f_new
end



# ==============================================================================
# REWEIGHTING METHODS
# ==============================================================================

"""
    _log_weights(mbar, beta)

Unnormalized MBAR log-weights `ln q_β(x_n) - ln Σ_j N_j e^{f_j} q_j(x_n)` of all samples
at `beta`, ordered state by state, and the index of `beta` in each state's grid.
"""
function _log_weights(mbar::MBARState, beta::Real)
    idxs = [_beta_index(mbar.all_betas[k], beta) for k in 1:mbar.K]
    log_w = [2 * mbar.measurements[k][i].log_norm[idxs[k]] - mbar.log_denom[k][i]
             for k in 1:mbar.K for i in 1:mbar.Nk[k]]
    return log_w, idxs
end

"""
    _weights(mbar, beta)

Normalized MBAR weights at `beta`, the grid indices (see `_log_weights`) and the Kish
effective sample size `1/Σw²`.
"""
function _weights(mbar::MBARState, beta::Real)
    log_w, idxs = _log_weights(mbar, beta)
    w = exp.(log_w .- maximum(log_w))
    w ./= sum(w)
    return w, idxs, 1.0 / sum(w .^ 2)
end

"""Values of observable `sym` of every sample at the grid indices `idxs`, in the order of the weights."""
_values(measurements, sym::Symbol, idxs) =
    [measurements[k][i][sym][idxs[k]] for k in eachindex(measurements) for i in eachindex(measurements[k])]

"""Weighted covariance `Σ w (a - ā)(b - b̄)` for normalized weights `w`."""
function _wcov(w, a, b)
    a_mean = sum(w .* a)
    b_mean = sum(w .* b)
    return sum(w .* (a .- a_mean) .* (b .- b_mean))
end

"""
    _reweight_at_betas(estimate, mbar)

Calls `estimate(w, idxs, beta)` at every beta common to all states and collects
`(betas, values, n_eff)`.
"""
function _reweight_at_betas(estimate, mbar::MBARState)
    all_beta_vals = _common_betas(mbar.all_betas)
    obs_mean = zeros(length(all_beta_vals))
    obs_neff = zeros(length(all_beta_vals))

    for (b_idx, beta) in enumerate(all_beta_vals)
        w, idxs, n_eff = _weights(mbar, beta)
        obs_mean[b_idx] = estimate(w, idxs, beta)
        obs_neff[b_idx] = n_eff
    end
    return all_beta_vals, obs_mean, obs_neff
end


"""
    reweight_observable(mbar::MBARState, observable::Union{String, Symbol})

Uses a solved MBARState to cheaply reweight an observable to every beta common to all
states. Returns `(betas, mean, n_eff)`.
"""
function reweight_observable(mbar::MBARState, observable::Union{String, Symbol})
    obs_sym = Symbol(observable)
    return _reweight_at_betas(mbar) do w, idxs, beta
        sum(w .* _values(mbar.measurements, obs_sym, idxs))
    end
end


# ==============================================================================
# β-DERIVATIVES
# ==============================================================================

_derivative_symbol(observable) = Symbol("∂β_", observable)

"""
    add_beta_derivatives(all_measurements, all_betas, observables)

Returns a copy of the measurements in which every sample also carries `∂β_<obs>`, the
β-derivative of its trajectory from a cubic interpolating spline on the state's grid.

The derivative estimators call this themselves when needed. Calling it once before
bootstrapping avoids refitting the same splines in every replicate, which the bootstrap
routines below do automatically.
"""
function add_beta_derivatives(all_measurements, all_betas, observables)
    syms = Symbol.(observables)
    dsyms = Tuple(_derivative_symbol.(syms))

    return map(eachindex(all_measurements)) do k
        betas_k = all_betas[k]
        spline_order = min(3, length(betas_k) - 1)

        map(all_measurements[k]) do m
            devs = Tuple(spline_order > 0 ?
                         derivative(Spline1D(betas_k, m[s], k=spline_order), betas_k, 1) :
                         zeros(length(betas_k))
                         for s in syms)
            merge(m, NamedTuple{dsyms}(devs))
        end
    end
end

"""Measurements with `∂β_<obs>` for every `observable`, fitting only the missing ones."""
function _with_derivatives(all_measurements, all_betas, observables)
    sample = first(first(all_measurements))
    missing_obs = filter(o -> !hasproperty(sample, _derivative_symbol(o)), unique(Symbol.(observables)))
    isempty(missing_obs) && return all_measurements
    return add_beta_derivatives(all_measurements, all_betas, missing_obs)
end

"""
    _derivative_estimate(w, measurements, idxs, observable, energy)

Reweighted `d⟨O⟩/dβ = -Cov(O, E) + ⟨dO/dβ⟩`: the first term comes from the β-dependence of
the weights (`d ln q_β/dβ = -E` per sample), the second from that of each trajectory.
"""
function _derivative_estimate(w, measurements, idxs, observable::Symbol, energy::Symbol)
    O  = _values(measurements, observable, idxs)
    E  = _values(measurements, energy, idxs)
    dO = _values(measurements, _derivative_symbol(observable), idxs)
    return -_wcov(w, O, E) + sum(w .* dO)
end

"""
    reweight_observable_dev(mbar::MBARState, observable; energy=:energy)

Reweighted β-derivative `d⟨O⟩/dβ` of an observable at every beta common to all states.
`energy` names the measured energy of the Hamiltonian that generated the evolution.
Returns `(betas, d⟨O⟩/dβ, n_eff)`.
"""
function reweight_observable_dev(mbar::MBARState, observable::Union{String, Symbol};
                                 energy::Union{String, Symbol}=:energy)
    obs_sym, e_sym = Symbol(observable), Symbol(energy)
    meas = _with_derivatives(mbar.measurements, mbar.all_betas, [obs_sym, e_sym])

    return _reweight_at_betas(mbar) do w, idxs, beta
        _derivative_estimate(w, meas, idxs, obs_sym, e_sym)
    end
end

"""
    compute_elastocaloric_effect(mbar, energy, strain_terms, linear_coefficients)

Adiabatic temperature response `(∂T/∂λ)_S = T (∂⟨A⟩/∂T) / C = (d⟨A⟩/dβ) / (β d⟨E⟩/dβ)` for a
perturbation `H = H₀ + λ A` with `A = Σ_s c_s O_s` (`strain_terms` and
`linear_coefficients`). Returns `(betas, (∂T/∂λ)_S, n_eff)`; NaN where `d⟨E⟩/dβ` or β vanish.
"""
function compute_elastocaloric_effect(mbar::MBARState, energy::Union{String, Symbol},
                                      strain_terms::AbstractVector{<:Union{String, Symbol}},
                                      linear_coefficients::Vector{Float64})
    length(strain_terms) == length(linear_coefficients) ||
        error("Got $(length(strain_terms)) strain_terms but $(length(linear_coefficients)) linear_coefficients.")

    e_sym = Symbol(energy)
    s_syms = Symbol.(strain_terms)
    meas = _with_derivatives(mbar.measurements, mbar.all_betas, [e_sym; s_syms])

    return _reweight_at_betas(mbar) do w, idxs, beta
        dE_dbeta = _derivative_estimate(w, meas, idxs, e_sym, e_sym)   # ∝ heat capacity
        if isapprox(dE_dbeta, 0.0; atol=1e-12) || isapprox(beta, 0.0; atol=1e-12)
            return NaN
        end
        dA_dbeta = sum(c * _derivative_estimate(w, meas, idxs, s, e_sym) for (s, c) in zip(s_syms, linear_coefficients))
        dA_dbeta / (dE_dbeta * beta)
    end
end


# ==============================================================================
# FREE ENERGIES AND OVERLAP
# ==============================================================================

"""
    compute_mbar_free_energies(mbar::MBARState)

Computes the reduced free energies f(β) = -ln Z(β) at every beta present in all
states' windows, using the MBAR self-consistency formula.

f is only determined up to an additive constant and is gauged so that
f(beta_collapses[1]) = 0, i.e. f(β) = ln Z(β₁) - ln Z(β). Only differences are
physical: f/β is NOT the free energy F unless an absolute reference (e.g. β = 0,
where Z = dim H) is included.

Returns `(betas, f)`.
"""
function compute_mbar_free_energies(mbar::MBARState)
    all_beta_vals = _common_betas(mbar.all_betas)
    f_interp = [-logsumexp(first(_log_weights(mbar, beta))) for beta in all_beta_vals]

    # Gauge: f = 0 at the first beta_collapse, as in mbar.free_energies
    base_idx = _beta_index(all_beta_vals, mbar.beta_collapses[1])
    f_interp .-= f_interp[base_idx]

    return all_beta_vals, f_interp
end

"""
    compute_overlap_matrix(mbar::MBARState)

MBAR overlap matrix `O[i, j]`: the average, over samples of state i, of the probability
that a sample came from state j. Rows sum to one; small off-diagonal neighbours mean
the states barely overlap and their free-energy difference is poorly determined.
"""
function compute_overlap_matrix(mbar::MBARState)
    K = mbar.K
    O = zeros(Float64, K, K)
    for i in 1:K, j in 1:K
        log_base = log(mbar.Nk[j]) + mbar.free_energies[j]
        O[i, j] = sum(exp(log_base + mbar.U[i][n, j] - mbar.log_denom[i][n]) for n in 1:mbar.Nk[i]) / mbar.Nk[i]
    end
    return O
end

"""
    compute_overlap_matrix(all_measurements, all_betas, beta_collapses; max_inter_mbar=1000, mbar_tol=1e-8)

Solves MBAR for the data and returns its overlap matrix.
"""
function compute_overlap_matrix(all_measurements, all_betas, beta_collapses;
                                max_inter_mbar::Int=1000, mbar_tol::Float64=1e-8)
    mbar = MBARState(all_measurements, all_betas, beta_collapses; max_iter=max_inter_mbar, tol=mbar_tol)
    return compute_overlap_matrix(mbar)
end

const MIN_NEIGHBOUR_OVERLAP = 0.03

"""
    check_overlap(mbar::MBARState; threshold=0.03)

Warns for every pair of neighbouring states whose overlap `min(O[k,k+1], O[k+1,k])` is
below `threshold` (the usual rule of thumb, e.g. Klimovich, Shirts & Mobley 2015). Below
it, MBAR results between those states are unreliable even if the bootstrap errors look
small; add a beta_collapse in between. Returns the overlap matrix.
"""
function check_overlap(mbar::MBARState; threshold::Float64=MIN_NEIGHBOUR_OVERLAP)
    O = compute_overlap_matrix(mbar)
    for k in 1:mbar.K-1
        o = min(O[k, k+1], O[k+1, k])
        if o < threshold
            @warn @sprintf("Low overlap %.3g between beta_collapse %.4g and %.4g (threshold %.3g). Results between them are unreliable; add a beta_collapse in between.",
                           o, mbar.beta_collapses[k], mbar.beta_collapses[k+1], threshold)
        end
    end
    return O
end


# ==============================================================================
# BOOTSTRAP
# ==============================================================================

function _warn_failed_replicates(n_failed::Int, n_bootstrap::Int)
    if n_failed > 0
        @warn "$n_failed of $n_bootstrap bootstrap replicates did not converge (they use their last MBAR iterate). Increase max_inter_mbar or loosen mbar_tol."
    end
end

"""
    _blocks(chain, block_length)

Splits samples `1:length(chain)` into consecutive blocks of at most `block_length` samples
that never cross from one chain to the next. `chain[i]` labels the chain (e.g. the seed) of
sample `i`; the samples of each chain must be contiguous and in chain order.
"""
function _blocks(chain::AbstractVector, block_length::Int)
    block_length >= 1 || error("block_length must be at least 1, got $block_length.")
    n = length(chain)
    finished = Set{eltype(chain)}()
    blocks = UnitRange{Int}[]
    start = 1
    for i in 2:(n + 1)
        new_chain = i <= n && chain[i] != chain[start]
        if i > n || new_chain || i - start == block_length
            push!(blocks, start:(i - 1))
            if new_chain
                push!(finished, chain[start])
                chain[i] in finished &&
                    error("The samples of chain $(chain[i]) are not contiguous; sort the samples of each state by chain.")
            end
            start = i
        end
    end
    return blocks
end

"""
    _block_resample(rng, blocks, n)

Indices of a block-bootstrap replicate: whole blocks drawn with replacement until `n`
samples are reached, the last block cut so that the replicate has exactly `n` samples
(MBAR needs the number of samples per state). With blocks of length 1 this is the ordinary
bootstrap.
"""
function _block_resample(rng::AbstractRNG, blocks::Vector{UnitRange{Int}}, n::Int)
    idx = Int[]
    while length(idx) < n
        append!(idx, blocks[rand(rng, 1:length(blocks))])
    end
    return resize!(idx, n)
end

"""
    _bootstrap_mbar(estimator, all_measurements, all_betas, beta_collapses; n_bootstrap, max_inter_mbar, mbar_tol, verbose, groups=nothing, block_length=1, rng=Random.default_rng())

Runs `estimator(mbar) -> (betas, values, n_eff)` on the full data and on `n_bootstrap`
data sets resampled with replacement within each state, re-solving MBAR each time.
Checks the overlap of the full solution. Returns a NamedTuple with `mbar`, `betas`,
`values`, `values_err`, `n_eff` and `free_energy_err` (of the beta_collapse free energies).

By default single samples are resampled, which assumes that the product states of a state
are independent. States thinned from a Markov chain can still be correlated, and their
errors then come out too small. For them pass `groups`, one vector per state with the chain
(e.g. the seed) of every sample, the samples of a chain contiguous and in chain order, and
`block_length > 1`: runs of up to `block_length` consecutive samples of one chain are
resampled together (block bootstrap). A `block_length` of at least the chain length
resamples whole chains, which is valid whatever the correlation within a chain but is itself
uncertain when there are few chains.

With G resampled units (samples or blocks) the bootstrap variance of a mean is too small by
(G-1)/G, which matters when whole chains are resampled (G = 5 seeds: errors 11% too small).
The errors are multiplied by √(G/(G-1)), with G the smallest number of units of any state.
"""
function _bootstrap_mbar(estimator, all_measurements, all_betas, beta_collapses;
                         n_bootstrap::Int, max_inter_mbar::Int, mbar_tol::Float64, verbose::Bool,
                         groups=nothing, block_length::Int=1, rng::AbstractRNG=Random.default_rng())
    # Blocks of every state; without groups the samples of a state form one chain
    isnothing(groups) || length(groups) == length(all_measurements) ||
        error("groups has $(length(groups)) entries but there are $(length(all_measurements)) states.")
    blocks = map(eachindex(all_measurements)) do k
        chain = isnothing(groups) ? fill(1, length(all_measurements[k])) : groups[k]
        length(chain) == length(all_measurements[k]) ||
            error("groups[$k] has $(length(chain)) labels but state $k has $(length(all_measurements[k])) samples.")
        _blocks(chain, block_length)
    end
    n_units = minimum(length, blocks)
    n_units >= 2 || error("A state has only $n_units bootstrap unit(s); the bootstrap needs at least 2 (more chains, or a shorter block_length).")
    small_sample_factor = sqrt(n_units / (n_units - 1))

    if verbose
        for (k, (bc, betas, meas)) in enumerate(zip(beta_collapses, all_betas, all_measurements))
            println("  state $k: beta_collapse = $bc, window = [$(betas[1]), $(betas[end])], N = $(length(meas))" *
                    (block_length > 1 ? ", $(length(blocks[k])) blocks" : ""))
        end
        block_length > 1 && println("  block bootstrap with block_length = $block_length")
    end

    mbar_full = MBARState(all_measurements, all_betas, beta_collapses; max_iter=max_inter_mbar, tol=mbar_tol)
    check_overlap(mbar_full)
    betas_out, values, n_eff = estimator(mbar_full)

    boot_vals = zeros(n_bootstrap, length(betas_out))
    boot_f    = zeros(n_bootstrap, mbar_full.K)
    n_failed  = 0

    for b in 1:n_bootstrap
        meas_boot = [meas[_block_resample(rng, blocks[k], length(meas))] for (k, meas) in enumerate(all_measurements)]
        mbar_boot = MBARState(meas_boot, all_betas, beta_collapses; max_iter=max_inter_mbar, tol=mbar_tol, verbose=false)

        boot_vals[b, :] = estimator(mbar_boot)[2]
        boot_f[b, :]    = mbar_boot.free_energies
        n_failed += !mbar_boot.converged
    end
    _warn_failed_replicates(n_failed, n_bootstrap)

    return (mbar=mbar_full, betas=betas_out, values=values, values_err=small_sample_factor .* vec(std(boot_vals, dims=1)),
            n_eff=n_eff, free_energy_err=small_sample_factor .* vec(std(boot_f, dims=1)))
end

function _print_bootstrap_summary(res, beta_collapses, label, n_bootstrap)
    println("\nMBAR free energies (bootstrap n=$n_bootstrap):")
    println("  beta_collapse    f (reduced)  stderr")
    for (k, bc) in enumerate(beta_collapses)
        @printf("  %10.4f  %12.6f  %12.6f\n", bc, res.mbar.free_energies[k], res.free_energy_err[k])
    end

    println("\nMBAR $label estimates (bootstrap n=$n_bootstrap):")
    println("  beta       value         stderr        N_eff")
    for (beta, v, err, neff) in zip(res.betas, res.values, res.values_err, res.n_eff)
        @printf("  %6.3f  %12.6f  %12.6f  %8.1f\n", beta, v, err, neff)
    end
end

"""
    bootstrap_mbar_reweight(all_measurements, all_betas, beta_collapses, observable; n_bootstrap=500, max_inter_mbar=1000, mbar_tol=1e-8, verbose=true, groups=nothing, block_length=1)

Reweights `observable` with MBAR and estimates its errors by bootstrapping.
Returns `(betas, mean, err, n_eff, free_energies, free_energy_err)`.

The bootstrap resamples single product states, which assumes they are independent. If the
states of a beta_collapse were thinned from Markov chains and are still correlated, pass
`groups` (one vector per state with the chain, e.g. the seed, of every sample; the samples of
each chain contiguous and in chain order) and `block_length > 1`: runs of up to
`block_length` consecutive states of one chain are then resampled together (block
bootstrap). `block_length` at least as long as the chains resamples whole chains; tested on
exact METTS samples, this is the choice that stays correct for strongly correlated states
(blocks of 2–4 states recover only part of the error). The errors include the factor
√(G/(G-1)) for G resampled units. The other `bootstrap_*` functions take the same keywords.
"""
function bootstrap_mbar_reweight(all_measurements, all_betas, beta_collapses,
                                 observable::Union{String, Symbol}; n_bootstrap::Int=500, max_inter_mbar::Int=1000,
                                 mbar_tol::Float64=1e-8, verbose::Bool=true, groups=nothing, block_length::Int=1)
    res = _bootstrap_mbar(mbar -> reweight_observable(mbar, observable),
                          all_measurements, all_betas, beta_collapses;
                          n_bootstrap=n_bootstrap, max_inter_mbar=max_inter_mbar, mbar_tol=mbar_tol, verbose=verbose,
                          groups=groups, block_length=block_length)
    verbose && _print_bootstrap_summary(res, beta_collapses, string(observable), n_bootstrap)
    return res.betas, res.values, res.values_err, res.n_eff, res.mbar.free_energies, res.free_energy_err
end

"""
    bootstrap_mbar_reweight_dev(all_measurements, all_betas, beta_collapses, observable; energy=:energy, n_bootstrap=500, max_inter_mbar=1000, mbar_tol=1e-8, verbose=true, groups=nothing, block_length=1)

Bootstrapped `d⟨O⟩/dβ` (see `reweight_observable_dev`). The trajectory splines are fitted
once, before resampling. Returns `(betas, d⟨O⟩/dβ, err, n_eff, free_energies, free_energy_err)`.
"""
function bootstrap_mbar_reweight_dev(all_measurements, all_betas, beta_collapses,
                                     observable::Union{String, Symbol}; energy::Union{String, Symbol}=:energy,
                                     n_bootstrap::Int=500, max_inter_mbar::Int=1000,
                                     mbar_tol::Float64=1e-8, verbose::Bool=true, groups=nothing, block_length::Int=1)
    meas = _with_derivatives(all_measurements, all_betas, [Symbol(observable), Symbol(energy)])
    res = _bootstrap_mbar(mbar -> reweight_observable_dev(mbar, observable; energy=energy),
                          meas, all_betas, beta_collapses;
                          n_bootstrap=n_bootstrap, max_inter_mbar=max_inter_mbar, mbar_tol=mbar_tol, verbose=verbose,
                          groups=groups, block_length=block_length)
    verbose && _print_bootstrap_summary(res, beta_collapses, "d$(observable)/dβ", n_bootstrap)
    return res.betas, res.values, res.values_err, res.n_eff, res.mbar.free_energies, res.free_energy_err
end

"""
    bootstrap_all_free_energies(all_measurements, all_betas, beta_collapses; n_bootstrap=500, max_inter_mbar=1000, mbar_tol=1e-8, verbose=true, groups=nothing, block_length=1)

Bootstrapped reduced free energies f(β) = ln Z(β₁) - ln Z(β) at every common beta
(see `compute_mbar_free_energies`). Returns `(betas, f, f_err)`.
"""
function bootstrap_all_free_energies(all_measurements, all_betas, beta_collapses;
                                     n_bootstrap::Int=500, max_inter_mbar::Int=1000,
                                     mbar_tol::Float64=1e-8, verbose::Bool=true, groups=nothing, block_length::Int=1)
    estimator = mbar -> begin
        betas, f = compute_mbar_free_energies(mbar)
        (betas, f, fill(NaN, length(betas)))
    end
    res = _bootstrap_mbar(estimator, all_measurements, all_betas, beta_collapses;
                          n_bootstrap=n_bootstrap, max_inter_mbar=max_inter_mbar, mbar_tol=mbar_tol, verbose=verbose,
                          groups=groups, block_length=block_length)

    if verbose
        println("\nMBAR Intermediate Free Energies (bootstrap n=$n_bootstrap):")
        println("  beta        f (reduced)   f_stderr")
        for (beta, f, err) in zip(res.betas, res.values, res.values_err)
            @printf("  %8.4f  %12.6f  %12.6f\n", beta, f, err)
        end
    end
    return res.betas, res.values, res.values_err
end

"""
    bootstrap_mbar_magneto_caloric_effect(all_measurements, all_betas, beta_collapses, energy, terms, linear_coefficients; n_bootstrap=500, max_inter_mbar=1000, mbar_tol=1e-8, verbose=true, groups=nothing, block_length=1)

Bootstrapped `(∂T/∂λ)_S` (see `compute_elastocaloric_effect`). The trajectory splines
are fitted once, before resampling.
Returns `(betas, (∂T/∂λ)_S, err, n_eff, free_energies, free_energy_err)`.
"""
function bootstrap_mbar_magneto_caloric_effect(all_measurements, all_betas, beta_collapses, energy::Union{String, Symbol},
                                               plaquete_terms::AbstractVector{<:Union{String, Symbol}}, linear_coefficients::Vector{Float64};
                                               n_bootstrap::Int=500, max_inter_mbar::Int=1000,
                                               mbar_tol::Float64=1e-8, verbose::Bool=true, groups=nothing, block_length::Int=1)
    meas = _with_derivatives(all_measurements, all_betas, [Symbol(energy); Symbol.(plaquete_terms)])
    res = _bootstrap_mbar(mbar -> compute_elastocaloric_effect(mbar, energy, plaquete_terms, linear_coefficients),
                          meas, all_betas, beta_collapses;
                          n_bootstrap=n_bootstrap, max_inter_mbar=max_inter_mbar, mbar_tol=mbar_tol, verbose=verbose,
                          groups=groups, block_length=block_length)
    verbose && _print_bootstrap_summary(res, beta_collapses, "(∂T/∂λ)_S", n_bootstrap)
    return res.betas, res.values, res.values_err, res.n_eff, res.mbar.free_energies, res.free_energy_err
end
