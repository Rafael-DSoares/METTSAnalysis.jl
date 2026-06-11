using Dierckx

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
end

"""
    MBARState(all_measurements, all_betas, beta_collapses; max_iter=500, tol=1e-8)

Constructs the MBARState by running the self-consistency equations until convergence.
"""
function MBARState(all_measurements, all_betas::Vector{Vector{Float64}}, beta_collapses::Vector{Float64}; max_iter::Int64=500, tol::Float64=1e-10,alpha::Float64=0.6)
    
    @assert issorted(beta_collapses) "beta_collapses must be sorted ascending"

    K  = length(beta_collapses)
    Nk = [length(meas) for meas in all_measurements] ###  the number of measurments per each beta_collapse. Length only looks to the outer data. We assume that the data is organized as the function load_metts work.
    
    @assert length(all_measurements) == K && length(all_betas) == K
    
    # 1. Initialize first with chained BAR (this makes MBAR converge faster!)
    _, f_init = chained_bar(all_measurements, all_betas, beta_collapses)
    f = copy(f_init)
    f_new = zeros(K)

    ## create U matrix:
    U = [fill(-Inf, Nk[k], K) for k in 1:K]

    for k in 1:K ## Loop over all states
        for j in 1:K ## Loop over all states
            idx = findfirst(b -> isapprox(b, beta_collapses[j]; atol=1e-8), all_betas[k]) ## find for a;; states in k which measurments were done at the collapse temperature. If there is none the probabity is zero.
            if !isnothing(idx)
                for i in 1:Nk[k]
                    U[k][i, j] = 2 * all_measurements[k][i].log_norm[idx] 
                end
            end
        end
    end


    log_denom = [zeros(Float64,Nk[k]) for k in 1:K] ### log of the MBAR denominator

    ### try to convege the free_energies
    converged = false
    last_error = 0.0

    for iter in 1:max_iter
        # Step A: Update log denominators
        _update_log_denom!(log_denom, U, f, K, Nk)

        # Step B: Update free energies
        _update_free_energies!(f_new, f, U, log_denom, K, Nk)

        f_new .= alpha .* f .+ (1.0 - alpha) .* f_new #### simple non-linear solver with relexation parameter

        last_error = maximum(abs.(f_new .- f))
        # Check convergence
        if last_error < tol
            converged = true
            break
        end
        f .= f_new
    end
    
    if converged
        @info "MBAR successfully converged in $max_iter iterations."
        f .= f_new
    else
        @warn "MBAR did not converge (Last error: $last_error). Falling back to exact Chained BAR (FEP) free energies."
        f .= f_init
    end

    _update_log_denom!(log_denom, U, f, K, Nk)

    return MBARState(all_measurements, all_betas, beta_collapses, K, Nk, U, f, log_denom)
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

Computes the log-denominator for the MBAR self-consistency equations. 
For each sample $i$ from state $k$, we calculate:
$$\text{log\_denom}_{k,i} = \ln \sum_{j=1}^K N_j \exp(f_j + U_{k,i,j})$$

where:
- $N_j$ is the number of samples in state $j$.
- $f_j$ is the current estimate of the reduced free energy for state $j$.
- $U_{k,i,j}$ is the reduced potential (2 * log_norm) of sample $i$ from state $k$ evaluated at state $j$.
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
    
    f_new .-= f_new[1] # gauge fixing!
    return f_new
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
                                 observable::Union{String, Symbol}; n_bootstrap::Int=500,max_inter_mbar::Int=1000, mbar_tol::Float64=1e-8 )


    for (k, (bc, betas, meas)) in enumerate(zip(beta_collapses, all_betas, all_measurements))
        println("  state $k: beta_collapse = $bc, window = [$(betas[1]), $(betas[end])], N = $(length(meas))")
    end
    
    @warn("Beware that the MBAR routines assume that log_norm is passed and not log_norm_square...")


    mbar_full = MBARState(all_measurements, all_betas, beta_collapses,max_iter=max_inter_mbar, tol=mbar_tol)
    betas_out, obs_mean, obs_neff = reweight_observable(mbar_full, observable)

    n_betas  = length(betas_out)
    K        = mbar_full.K
    
    boot_obs_mat = zeros(n_bootstrap, n_betas)
    boot_f_mat   = zeros(n_bootstrap, K)

    # 2. Resample and create new MBAR states
    for b in 1:n_bootstrap
        meas_boot = [all_measurements[k][rand(1:length(all_measurements[k]), length(all_measurements[k]))] for k in 1:K]
        
        mbar_boot = MBARState(meas_boot, all_betas, beta_collapses; max_iter=max_inter_mbar, tol=mbar_tol)
        _, boot_obs, _ = reweight_observable(mbar_boot, observable)
        
        boot_obs_mat[b, :] = boot_obs
        boot_f_mat[b, :]   = mbar_boot.free_energies
    end

    obs_err         = vec(std(boot_obs_mat, dims=1))
    free_energy_err = vec(std(boot_f_mat, dims=1))


    println("\nMBAR free energies (bootstrap n=$n_bootstrap):")
    println("  beta_collapse    F            stderr")
    for (k, bc) in enumerate(beta_collapses)
        @printf("  %10.4f  %12.6f  %12.6f\n", bc,  mbar_full.free_energies[k], free_energy_err[k])
    end

    println("\nMBAR energy estimates (bootstrap n=$n_bootstrap):")
    println("  beta       energy        stderr        N_eff")
    for (beta, e, err, neff) in zip(betas_out, obs_mean, obs_err, obs_neff)
        @printf("  %6.3f  %12.6f  %12.6f  %8.1f\n", beta, e, err, neff)
    end
    
    return betas_out, obs_mean, obs_err, obs_neff, mbar_full.free_energies, free_energy_err
end


"""
    precompute_observable_derivatives(mbar::MBARState, observable::Union{String, Symbol})

Fits cubic splines to individual sample trajectories using their strictly native original beta arrays, and precalculates the 1st derivatives.
"""
function precompute_observable_derivatives(mbar::MBARState, observable::Union{String, Symbol})
    obs_sym = Symbol(observable)
    precomputed_devs = Vector{Vector{Float64}}[]
    
    for k in 1:mbar.K
        betas_k = mbar.all_betas[k]
        n_points = length(betas_k)
        
        spline_order = min(3, n_points - 1) 
        
        devs_k = Vector{Float64}[]
        for i in 1:mbar.Nk[k]
            y_vals = mbar.measurements[k][i][obs_sym]
            
            if spline_order > 0
                spline = Spline1D(betas_k, y_vals, k=spline_order)

                push!(devs_k, derivative(spline, betas_k, 1))
            else
                push!(devs_k, zeros(n_points))
            end
        end
        push!(precomputed_devs, devs_k)
    end
    
    return precomputed_devs
end

"""
    reweight_observable_dev(mbar::MBARState, observable::Union{String, Symbol})

Uses a solved MBARState to cheaply reweight an observable and its derivative across all unique betas.
"""
function reweight_observable_dev(mbar::MBARState, observable::Union{String, Symbol})

    obs_sym = Symbol(observable)
    
    all_beta_vals = sort(unique(vcat(mbar.all_betas...)))
    n_betas = length(all_beta_vals)
    
    obs_mean = zeros(n_betas)
    obs_neff = zeros(n_betas)

    precomputed_devs = precompute_observable_derivatives(mbar, observable)

    for (b_idx, beta) in enumerate(all_beta_vals)
        log_w_tmp  = Float64[]
        obs_tmp    = Float64[]
        obs_dev    = Float64[]
        energy_tmp = Float64[]

        for k in 1:mbar.K
            idx = findfirst(b -> isapprox(b, beta; atol=1e-8), mbar.all_betas[k])
            isnothing(idx) && continue
            
            for i in 1:mbar.Nk[k]
                ln_num = 2 * mbar.measurements[k][i].log_norm[idx]
                
                push!(log_w_tmp, ln_num - mbar.log_denom[k][i])
                push!(obs_tmp, mbar.measurements[k][i][obs_sym][idx])
                push!(energy_tmp, mbar.measurements[k][i][:energy][idx])
                push!(obs_dev, precomputed_devs[k][i][idx])
            end
        end

        if !isempty(log_w_tmp)
            log_w_tmp .-= maximum(log_w_tmp)
            w = exp.(log_w_tmp)
            w ./= sum(w)
            
            observable_mean = sum(w .* obs_tmp)
            energy_mean     = sum(w .* energy_tmp)
            dev_mean        = sum(w .* obs_dev)

            # Equation: Fluctuation Term + Mean Derivative
            obs_mean[b_idx] = -sum(w .* (obs_tmp .- observable_mean) .* (energy_tmp .- energy_mean)) + dev_mean

            obs_neff[b_idx] = 1.0 / sum(w .^ 2)
        end
    end

    return all_beta_vals, obs_mean, obs_neff
end



"""
    bootstrap_mbar_reweight(all_measurements, all_betas, beta_collapses, observable; n_bootstrap=500)
Performs full MBAR bootstrap error analysis by generating bootstrapped MBARStates.
"""
function bootstrap_mbar_reweight_dev(all_measurements, all_betas, beta_collapses, 
                                 observable::Union{String, Symbol}; n_bootstrap::Int=500,max_inter_mbar::Int=1000, mbar_tol::Float64=1e-8 )


    for (k, (bc, betas, meas)) in enumerate(zip(beta_collapses, all_betas, all_measurements))
        println("  state $k: beta_collapse = $bc, window = [$(betas[1]), $(betas[end])], N = $(length(meas))")
    end
    
    @warn("Beware that the MBAR routines assume that log_norm is passed and not log_norm_square...")


    mbar_full = MBARState(all_measurements, all_betas, beta_collapses,max_iter=max_inter_mbar, tol=mbar_tol)
    betas_out, obs_mean, obs_neff = reweight_observable_dev(mbar_full, observable)

    n_betas  = length(betas_out)
    K        = mbar_full.K
    
    boot_obs_mat = zeros(n_bootstrap, n_betas)
    boot_f_mat   = zeros(n_bootstrap, K)

    # 2. Resample and create new MBAR states
    for b in 1:n_bootstrap
        meas_boot = [all_measurements[k][rand(1:length(all_measurements[k]), length(all_measurements[k]))] for k in 1:K]
        
        mbar_boot = MBARState(meas_boot, all_betas, beta_collapses; max_iter=max_inter_mbar, tol=mbar_tol)
        _, boot_obs, _ = reweight_observable_dev(mbar_boot, observable)
        
        boot_obs_mat[b, :] = boot_obs
        boot_f_mat[b, :]   = mbar_boot.free_energies
    end

    obs_err         = vec(std(boot_obs_mat, dims=1))
    free_energy_err = vec(std(boot_f_mat, dims=1))


    println("\nMBAR free energies (bootstrap n=$n_bootstrap):")
    println("  beta_collapse    F            stderr")
    for (k, bc) in enumerate(beta_collapses)
        @printf("  %10.4f  %12.6f  %12.6f\n", bc,  mbar_full.free_energies[k], free_energy_err[k])
    end

    println("\nMBAR energy estimates (bootstrap n=$n_bootstrap):")
    println("  beta       energy        stderr        N_eff")
    for (beta, e, err, neff) in zip(betas_out, obs_mean, obs_err, obs_neff)
        @printf("  %6.3f  %12.6f  %12.6f  %8.1f\n", beta, e, err, neff)
    end
    
    return betas_out, obs_mean, obs_err, obs_neff, mbar_full.free_energies, free_energy_err
end



"""
    compute_overlap_matrix(mbar::MBARState)

Computes the $K \times K$ MBAR overlap matrix $\mathbb{O}$ using the converged 
free energies and precomputed log-denominators from an `MBARState`.

The element $\mathbb{O}_{i,j}$ represents the average weight of the samples 
from state $i$ when evaluated at state $j$. A well-connected MBAR dataset 
will have significant off-diagonal elements.
"""
function compute_overlap_matrix(all_measurements, all_betas, beta_collapses; n_bootstrap::Int=500,max_inter_mbar::Int=1000, mbar_tol::Float64=1e-8)

    mbar = MBARState(all_measurements, all_betas, beta_collapses,max_iter=max_inter_mbar, tol=mbar_tol)

    K = mbar.K
    O = zeros(Float64, K, K)
    
    for i in 1:K
        N_i = mbar.Nk[i]
        if N_i == 0
            continue # Skip states with no samples to avoid division by zero
        end
        
        for j in 1:K
            # If the target state j has no samples, its weight contribution is 0
            if mbar.Nk[j] == 0
                O[i, j] = 0.0
                continue
            end
            
            # Base log-weight term for state j: ln(N_j) + f_j
            base_val = log(mbar.Nk[j]) + mbar.free_energies[j]
            sum_weights = 0.0
            
            for n in 1:N_i
                u_val = mbar.U[i][n, j]
                
                # Check for NaN values just as handled in the MBAR iterations
                if !isnan(u_val)
                    # W_{i,n,j} = exp(ln(N_j) + f_j + U_{i,n,j} - log_denom_{i,n})
                    log_w = base_val + u_val - mbar.log_denom[i][n]
                    sum_weights += exp(log_w)
                end
            end
            
            # Average the weights over all samples originally drawn from state i
            O[i, j] = sum_weights / N_i
        end
    end
    
    return O
end


"""
    compute_mbar_free_energies(mbar::MBARState)

Computes the reduced (f) and actual (F) free energies across all unique 
intermediate temperatures (betas) present in the MBAR measurements using 
the exact MBAR self-consistency formula.
"""
function compute_mbar_free_energies(mbar::MBARState)
    # Collect and sort all unique betas available in the dataset
    all_beta_vals = sort(unique(vcat(mbar.all_betas...)))
    n_betas = length(all_beta_vals)
    
    f_interp = zeros(n_betas)
    F_interp = zeros(n_betas)

    for (b_idx, beta) in enumerate(all_beta_vals)
        log_w_tmp = Float64[]

        for k in 1:mbar.K
            idx = findfirst(b -> isapprox(b, beta; atol=1e-8), mbar.all_betas[k])
            isnothing(idx) && continue
            
            for i in 1:mbar.Nk[k]
                # Extract the reduced potential for the sample at the target beta
                ln_num = 2 * mbar.measurements[k][i].log_norm[idx]
                
                # Accumulate the weight term: U_{k,i,beta} - log_denom_{k,i}
                push!(log_w_tmp, ln_num - mbar.log_denom[k][i])
            end
        end

        if !isempty(log_w_tmp)
            # f_beta = -ln( \sum exp(U - log_denom) )
            f_interp[b_idx] = -logsumexp(log_w_tmp)
        else
            f_interp[b_idx] = NaN
        end
    end

    # Align the reference state to 0.0 to exactly match mbar.free_energies.
    # We find the index of the first beta collapse and shift the entire array.
    base_idx = findfirst(b -> isapprox(b, mbar.beta_collapses[1]; atol=1e-8), all_beta_vals)
    if !isnothing(base_idx)
        f_interp .-= f_interp[base_idx]
    else
        f_interp .-= f_interp[1] # Fallback shift
    end

    # Compute actual free energies F = f / beta
    for i in 1:n_betas
        beta = all_beta_vals[i]
        if isapprox(beta, 0.0; atol=1e-12)
            F_interp[i] = NaN # Handle edge case to avoid DivisionByZero
        else
            F_interp[i] = f_interp[i] / beta
        end
    end

    return all_beta_vals, f_interp, F_interp
end




"""
    bootstrap_intermediate_free_energies(all_measurements, all_betas, beta_collapses; kwargs...)

Performs full MBAR bootstrap error analysis on the intermediate free energies.
Generates `n_bootstrap` resampled datasets, reconverges MBAR for each, and 
calculates the standard deviation of both the reduced (f) and actual (F) free energies.
"""
function bootstrap_all_free_energies(all_measurements, all_betas, beta_collapses; 
                                              n_bootstrap::Int=500, max_inter_mbar::Int=1000, mbar_tol::Float64=1e-8)

    # 1. Compute the reference state using the full, original dataset
    mbar_full = MBARState(all_measurements, all_betas, beta_collapses; max_iter=max_inter_mbar, tol=mbar_tol)
    all_beta_vals, f_mean, F_mean = compute_mbar_free_energies(mbar_full)
    
    n_betas = length(all_beta_vals)
    K       = mbar_full.K
    
    # 2. Initialize matrices to hold the bootstrap results
    boot_f_mat = zeros(n_bootstrap, n_betas)
    boot_F_mat = zeros(n_bootstrap, n_betas)

    println("Starting bootstrap for intermediate free energies (n=$n_bootstrap)...")

    # 3. Resample and create new MBAR states
    for b in 1:n_bootstrap
        # Resample measurements with replacement
        meas_boot = [all_measurements[k][rand(1:length(all_measurements[k]), length(all_measurements[k]))] for k in 1:K]
        
        # Converge a new MBAR state for this resampled data
        mbar_boot = MBARState(meas_boot, all_betas, beta_collapses; max_iter=max_inter_mbar, tol=mbar_tol)
        
        # Compute intermediate free energies for the boot state
        _, f_boot, F_boot = compute_mbar_free_energies(mbar_boot)
        
        boot_f_mat[b, :] = f_boot
        boot_F_mat[b, :] = F_boot
    end

    # 4. Compute standard errors across the bootstrap dimension (dims=1)
    f_err = vec(std(boot_f_mat, dims=1))
    F_err = vec(std(boot_F_mat, dims=1))

    # 5. Print results gracefully
    println("\nMBAR Intermediate Free Energies (bootstrap n=$n_bootstrap):")
    println("  beta        F (actual)    F_stderr      f (reduced)   f_stderr")
    for i in 1:n_betas
        # Handle formatting for cases where beta=0 might result in NaN for F
        if isnan(F_mean[i])
            @printf("  %8.4f           NaN           NaN  %12.6f  %12.6f\n", 
                    all_beta_vals[i], f_mean[i], f_err[i])
        else
            @printf("  %8.4f  %12.6f  %12.6f  %12.6f  %12.6f\n", 
                    all_beta_vals[i], F_mean[i], F_err[i], f_mean[i], f_err[i])
        end
    end
    
    return all_beta_vals, F_mean, F_err
end



function compute_magneto_caloric_effect(mbar::MBARState, energy::Union{String, Symbol}, plaquete_terms::Vector{Union{String, Symbol}}, linear_coefficients::Vector{Float64})
    energy_sym = Symbol(energy)
    
    all_beta_vals = sort(unique(vcat(mbar.all_betas...)))
    n_betas = length(all_beta_vals)
    
    obs_mean = zeros(n_betas)

    obs_neff = zeros(n_betas)

    precomputed_devs = precompute_observable_derivatives(mbar, energy_sym)

    ## first sum over the plaquete terms:

    for (p_idx, plaquete) in enumerate(plaquete_terms)
        
        plaquete_sym = Symbol(plaquete)
        precomputed_devs = precompute_observable_derivatives(mbar, plaquete_sym)

        for (b_idx, beta) in enumerate(all_beta_vals)
            log_w_tmp  = Float64[]
            obs_tmp    = Float64[]
            obs_dev    = Float64[]
            energy_tmp = Float64[]

            for k in 1:mbar.K
                idx = findfirst(b -> isapprox(b, beta; atol=1e-8), mbar.all_betas[k])
                isnothing(idx) && continue
                
                for i in 1:mbar.Nk[k]
                    ln_num = 2 * mbar.measurements[k][i].log_norm[idx]
                    
                    push!(log_w_tmp, ln_num - mbar.log_denom[k][i])
                    push!(obs_tmp, mbar.measurements[k][i][plaquete_sym][idx])
                    push!(energy_tmp, mbar.measurements[k][i][:energy][idx])
                    push!(obs_dev, precomputed_devs[k][i][idx])
                end
            end

            if !isempty(log_w_tmp)
                log_w_tmp .-= maximum(log_w_tmp)
                w = exp.(log_w_tmp)
                w ./= sum(w)
                
                observable_mean = sum(w .* obs_tmp)
                energy_mean     = sum(w .* energy_tmp)
                dev_mean        = sum(w .* obs_dev)

                # Equation: Fluctuation Term + Mean Derivative
                dP_dbeta = -sum(w .* (obs_tmp .- observable_mean) .* (energy_tmp .- energy_mean)) + dev_mean

                obs_mean[b_idx] += dP_dbeta * linear_coefficients[p_idx] 
        end
    end

    for (b_idx, beta) in enumerate(all_beta_vals)
        log_w_tmp  = Float64[]
        obs_tmp    = Float64[]
        obs_dev    = Float64[]
        energy_tmp = Float64[]

        for k in 1:mbar.K
            idx = findfirst(b -> isapprox(b, beta; atol=1e-8), mbar.all_betas[k])
            isnothing(idx) && continue
            
            for i in 1:mbar.Nk[k]
                ln_num = 2 * mbar.measurements[k][i].log_norm[idx]
                
                push!(log_w_tmp, ln_num - mbar.log_denom[k][i])
                push!(obs_tmp, mbar.measurements[k][i][energy_sym][idx])
                push!(energy_tmp, mbar.measurements[k][i][:energy][idx])
                push!(obs_dev, precomputed_devs[k][i][idx])
            end
        end

        if !isempty(log_w_tmp)
            log_w_tmp .-= maximum(log_w_tmp)
            w = exp.(log_w_tmp)
            w ./= sum(w)
            
            observable_mean = sum(w .* obs_tmp)
            energy_mean     = sum(w .* energy_tmp)
            dev_mean        = sum(w .* obs_dev)

            # Equation: Fluctuation Term + Mean Derivative
            dE_dbeta = -sum(w .* (obs_tmp .- observable_mean) .* (energy_tmp .- energy_mean)) + dev_mean

            obs_mean[b_idx] /= dE_dbeta * beta 

            obs_neff[b_idx] = 1.0 / sum(w .^ 2)
        end
    end

    return all_beta_vals, obs_mean, obs_neff
end


function bootstrap_mbar_magneto_caloric_effect(all_measurements, all_betas, beta_collapses, energy::Union{String, Symbol}, plaquete_terms::Vector{Union{String, Symbol}}, linear_coefficients::Vector{Float64}; 
    n_bootstrap::Int=500,max_inter_mbar::Int=1000, mbar_tol::Float64=1e-8 )


    for (k, (bc, betas, meas)) in enumerate(zip(beta_collapses, all_betas, all_measurements))
        println("  state $k: beta_collapse = $bc, window = [$(betas[1]), $(betas[end])], N = $(length(meas))")
    end
    
    @warn("Beware that the MBAR routines assume that log_norm is passed and not log_norm_square...")


    mbar_full = MBARState(all_measurements, all_betas, beta_collapses,max_iter=max_inter_mbar, tol=mbar_tol)
    betas_out, obs_mean, obs_neff = compute_magneto_caloric_effect(mbar_full, energy, plaquete_terms, linear_coefficients)

    n_betas  = length(betas_out)
    K        = mbar_full.K
    
    boot_obs_mat = zeros(n_bootstrap, n_betas)
    boot_f_mat   = zeros(n_bootstrap, K)

    # 2. Resample and create new MBAR states
    for b in 1:n_bootstrap
        meas_boot = [all_measurements[k][rand(1:length(all_measurements[k]), length(all_measurements[k]))] for k in 1:K]
        
        mbar_boot = MBARState(meas_boot, all_betas, beta_collapses; max_iter=max_inter_mbar, tol=mbar_tol)
        _, boot_obs, _ = compute_magneto_caloric_effect(mbar_full, energy, plaquete_terms, linear_coefficients)
        
        boot_obs_mat[b, :] = boot_obs
        boot_f_mat[b, :]   = mbar_boot.free_energies
    end

    obs_err         = vec(std(boot_obs_mat, dims=1))
    free_energy_err = vec(std(boot_f_mat, dims=1))


    println("\nMBAR free energies (bootstrap n=$n_bootstrap):")
    println("  beta_collapse    F            stderr")
    for (k, bc) in enumerate(beta_collapses)
        @printf("  %10.4f  %12.6f  %12.6f\n", bc,  mbar_full.free_energies[k], free_energy_err[k])
    end

    println("\nMBAR energy estimates (bootstrap n=$n_bootstrap):")
    println("  beta       energy        stderr        N_eff")
    for (beta, e, err, neff) in zip(betas_out, obs_mean, obs_err, obs_neff)
        @printf("  %6.3f  %12.6f  %12.6f  %8.1f\n", beta, e, err, neff)
    end
      
    return betas_out, obs_mean, obs_err, obs_neff, mbar_full.free_energies, free_energy_err
end