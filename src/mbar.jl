function logsumexp(x::AbstractVector{<:Real})
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
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

@doc raw"""
    _update_log_denom!(log_denom, U, f, K, Nk)

Computes the log-denominator for the MBAR self-consistency equations. 
For each sample $i$ from state $k$, we calculate:
$$\text{log\_denom}_{k,i} = \ln \sum_{j=1}^K N_j \exp(f_j + U_{k,i,j})$$

where:
- $N_j$ is the number of samples in state $j$[cite: 7].
- $f_j$ is the current estimate of the reduced free energy for state $j$[cite: 7].
- $U_{k,i,j}$ is the reduced potential (2 * log_norm) of sample $i$ from state $k$ evaluated at state $j$[cite: 7].
"""
function _update_log_denom!(log_denom::Vector{Vector{Float64}}, U::Vector{Matrix{Float64}}, f::Vector{Float64}, K::Int, Nk::Vector{Int})
    for k in 1:K ## Loop over all states
        for i in 1:Nk[k] ## Loop over all pruned samples
            # logsumexp trick to prevent overflow
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

@doc raw"""
    _update_free_energies!(f_new::Vector{Float64}, 
                           f::Vector{Float64}, 
                           U::Vector{Matrix{Float64}}, 
                           log_denom::Vector{Vector{Float64}}, 
                           K::Int, 
                           Nk::Vector{Int})

Updates the reduced free energy estimates:
$$f_j = -\ln \sum_{k=1}^K \sum_{i=1}^{N_k} \exp(U_{k,i,j} - \text{log\_denom}_{k,i})$$
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
    
    f_new .-= f_new[1] ## the first f is zero otherwise the system is impossible to solve
    return f_new
end

"""
Holds the converged state of an MBAR calculation, including the precalculated 
energy matrices and weights.
"""
struct MBARState{T}
    measurements::T                       # The raw Vector of NamedTuples
    all_betas::Vector{Vector{Float64}}    # The beta arrays for each state
    beta_collapses::Vector{Float64}       # The target beta values
    K::Int                                # Number of states
    Nk::Vector{Int}                       # Number of samples per state
    U::Vector{Matrix{Float64}}            # Pre-calculated energy matrix
    free_energies::Vector{Float64}        # The converged free energies (f)
    log_denom::Vector{Vector{Float64}}    # The converged denominators for reweighting
end

"""
    MBARState(all_measurements, all_betas, beta_collapses; max_iter=500, tol=1e-10)

Constructs the MBARState by running the self-consistency equations until convergence.
"""
function MBARState(all_measurements, all_betas, beta_collapses; max_iter::Int64=500, tol::Float64=1e-10)

    @warn("Beware that the MBAR routines assume that log_norm is passed and not log_norm_square...")

    K  = length(beta_collapses)
    Nk = [length(meas) for meas in all_measurements] ###  the number of measurments per each beta_collapse. Length only looks to the outer data. We assume that the data is organized as the function load_metts work.
    
    @assert issorted(beta_collapses) "beta_collapses must be sorted ascending"
    @assert length(all_measurements) == K && length(all_betas) == K
    
    # 1. Initialize first with chained BAR (this makes MBAR converge faster!)
    _, f_init = chained_bar(all_measurements, all_betas, beta_collapses)
    f = copy(f_init)
    f_new = zeros(K)

    # 2. Pre-calculate the probability matrix U 
    U = [fill(NaN, Nk[k], K) for k in 1:K]
    for k in 1:K
        for j in 1:K
            idx = findfirst(b -> isapprox(b, beta_collapses[j]; atol=1e-8), all_betas[k])
            if !isnothing(idx)
                for i in 1:Nk[k]
                    U[k][i, j] = 2 * all_measurements[k][i].log_norm[idx] ## the factor of 2 comes from the fact that we pass log_norm and not log_norm_square!! (this could be a big source of error if people have this wrong)
                end
            end
        end
    end

    log_denom = [zeros(Nk[k]) for k in 1:K]

    ### try to convege the free_energies

    converged = false
    for iter in 1:max_iter
        # Step A: Update log denominators
        _update_log_denom!(log_denom, U, f, K, Nk)

        # Step B: Update free energies
        _update_free_energies!(f_new, f, U, log_denom, K, Nk)
        
        # Check convergence
        if maximum(abs.(f_new .- f)) < tol
            converged = true
            break
        end
        f .= f_new
    end
    
    !converged && @warn "MBAR did not converge within $max_iter iterations."

    f .= f_new
    _update_log_denom!(log_denom, U, f, K, Nk)

    return MBARState(all_measurements, all_betas, beta_collapses, K, Nk, U, f, log_denom)
end

# ==============================================================================
# REWEIGHTING METHODS
# ==============================================================================

"""
    reweight_observable(mbar::MBARState, observable::Union{String, Symbol})

Uses a solved MBARState to cheaply reweight an observable across all unique betas.
"""
function reweight_observable(mbar::MBARState, observable::Union{String, Symbol})
    obs_sym = Symbol(observable)
    
    all_beta_vals = sort(unique(vcat(mbar.all_betas...)))
    n_betas = length(all_beta_vals)
    
    obs_mean = zeros(n_betas)
    obs_neff = zeros(n_betas)

    for (b_idx, beta) in enumerate(all_beta_vals)
        log_w_tmp = Float64[]
        obs_tmp   = Float64[]

        for k in 1:mbar.K
            idx = findfirst(b -> isapprox(b, beta; atol=1e-8), mbar.all_betas[k])
            isnothing(idx) && continue
            
            for i in 1:mbar.Nk[k]
                ln_num = 2 * mbar.measurements[k][i].log_norm[idx]
                
                # Instantly retrieve the precalculated denominator
                push!(log_w_tmp, ln_num - mbar.log_denom[k][i])
                push!(obs_tmp, mbar.measurements[k][i][obs_sym][idx])
            end
        end

        if !isempty(log_w_tmp)
            log_w_tmp .-= maximum(log_w_tmp)
            w = exp.(log_w_tmp)
            w ./= sum(w)
            
            obs_mean[b_idx] = sum(w .* obs_tmp)
            obs_neff[b_idx] = 1.0 / sum(w .^ 2)
        end
    end

    return all_beta_vals, obs_mean, obs_neff
end

"""
    bootstrap_mbar_reweight(all_measurements, all_betas, beta_collapses, observable; n_bootstrap=500)

Performs full MBAR bootstrap error analysis by generating bootstrapped MBARStates.
"""
function bootstrap_mbar_reweight(all_measurements, all_betas, beta_collapses, 
                                 observable::Union{String, Symbol}; n_bootstrap::Int=500,max_inter_mbar::Int=1000 )
    

    mbar_full = MBARState(all_measurements, all_betas, beta_collapses)
    betas_out, obs_mean, obs_neff = reweight_observable(mbar_full, observable)

    n_betas  = length(betas_out)
    K        = mbar_full.K
    
    boot_obs_mat = zeros(n_bootstrap, n_betas)
    boot_f_mat   = zeros(n_bootstrap, K)

    # 2. Resample and create new MBAR states
    for b in 1:n_bootstrap
        meas_boot = [all_measurements[k][rand(1:length(all_measurements[k]), length(all_measurements[k]))] for k in 1:K]
        
        mbar_boot = MBARState(meas_boot, all_betas, beta_collapses; max_iter=max_inter_mbar)
        _, boot_obs, _ = reweight_observable(mbar_boot, observable)
        
        boot_obs_mat[b, :] = boot_obs
        boot_f_mat[b, :]   = mbar_boot.free_energies
    end

    obs_err         = vec(std(boot_obs_mat, dims=1))
    free_energy_err = vec(std(boot_f_mat, dims=1))
    
    return betas_out, obs_mean, obs_err, obs_neff, mbar_full.free_energies, free_energy_err
end