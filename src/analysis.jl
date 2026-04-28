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
        isnothing(idx_km1) && error("beta_collapses[$(k-1)] = $β_km1 not found in " *
                                    "betas of state $k — extend the measurement window")
        isnothing(idx_k)   && error("beta_collapses[$k] = $β_k not found in betas of state $k")

        # CHANGED: Use .log_norm instead of ["log_norm"]
        log_ratios = [2 * (meas_k[i].log_norm[idx_km1] - meas_k[i].log_norm[idx_k])
                      for i in 1:N_k]
        delta_f[k - 1] = logsumexp(log_ratios) - log(N_k)
    end

    return delta_f, [0.0; cumsum(delta_f)]
end

"""
Refine free energy estimates by iterating the MBAR self-consistency equations.
"""
function mbar_free_energies(all_measurements, all_betas, beta_collapses, f_init;
                            max_iter=500, tol=1e-10)
    K  = length(beta_collapses)
    Nk = [length(all_measurements[k]) for k in 1:K]
    f  = copy(f_init)

    avail_states = [Int[] for _ in 1:K]
    avail_idx    = [Int[] for _ in 1:K]
    for k in 1:K, j in 1:K
        idx = findfirst(b -> isapprox(b, beta_collapses[j]; atol=1e-8), all_betas[k])
        if !isnothing(idx)
            push!(avail_states[k], j)
            push!(avail_idx[k], idx)
        end
    end

    for _ in 1:max_iter
        # CHANGED: Use .log_norm instead of ["log_norm"]
        log_denom = [[logsumexp([log(Nk[j]) + f[j] +
                                 2 * all_measurements[k][i].log_norm[avail_idx[k][jj]]
                                 for (jj, j) in enumerate(avail_states[k])])
                      for i in 1:Nk[k]]
                     for k in 1:K]

        f_new = zeros(K)
        for j in 1:K
            terms = Float64[]
            for k in 1:K
                jj = findfirst(==(j), avail_states[k])
                isnothing(jj) && continue
                for i in 1:Nk[k]
                    # CHANGED: Use .log_norm instead of ["log_norm"]
                    push!(terms, 2 * all_measurements[k][i].log_norm[avail_idx[k][jj]] -
                                 log_denom[k][i])
                end
            end
            f_new[j] = isempty(terms) ? f[j] : -logsumexp(terms)
        end
        f_new .-= f_new[1]

        maximum(abs.(f_new .- f)) < tol && return f_new
        f .= f_new
    end
    return f
end

"""
MBAR reweighting of an observable using chained BAR free energies.
"""
function mbar_reweight_observable(all_measurements, all_betas, beta_collapses,
                                  free_energies, observable::Union{String, Symbol})
    # CHANGED: Cast observable to Symbol once outside the loop for speed
    obs_sym = Symbol(observable)
    
    K  = length(beta_collapses)
    Nk = [length(all_measurements[k]) for k in 1:K]

    avail_states = [Int[] for _ in 1:K]
    avail_idx    = [Int[] for _ in 1:K]
    for k in 1:K
        for j in 1:K
            idx = findfirst(b -> isapprox(b, beta_collapses[j]; atol=1e-8), all_betas[k])
            if !isnothing(idx)
                push!(avail_states[k], j)
                push!(avail_idx[k],    idx)
            end
        end
    end

    # CHANGED: Use .log_norm instead of ["log_norm"]
    log_denom = [[logsumexp([log(Nk[j]) + free_energies[j] +
                             2 * all_measurements[k][i].log_norm[avail_idx[k][jj]]
                             for (jj, j) in enumerate(avail_states[k])])
                  for i in 1:Nk[k]]
                 for k in 1:K]

    all_beta_vals = sort(unique(vcat(all_betas...)))

    betas_out = Float64[]
    obs_mean  = Float64[]
    obs_neff  = Float64[]

    for beta in all_beta_vals
        log_w = Float64[]
        obs   = Float64[]

        for k in 1:K
            beta_idx = findfirst(b -> isapprox(b, beta; atol=1e-8), all_betas[k])
            isnothing(beta_idx) && continue
            for i in 1:Nk[k]
                # CHANGED: Use .log_norm and [obs_sym]
                ln_num = 2 * all_measurements[k][i].log_norm[beta_idx]
                push!(log_w, ln_num - log_denom[k][i])
                push!(obs,   all_measurements[k][i][obs_sym][beta_idx])
            end
        end

        isempty(log_w) && continue
        log_w .-= maximum(log_w)
        w = exp.(log_w)
        w ./= sum(w)

        push!(betas_out, beta)
        push!(obs_mean,  sum(w .* obs))
        push!(obs_neff,  1 / sum(w .^ 2))
    end

    return betas_out, obs_mean, obs_neff
end

"""
Same as mbar_reweight_observable but also returns bootstrap standard errors.
"""
function bootstrap_mbar_reweight_observable(all_measurements, all_betas, beta_collapses,
                                            observable::Union{String, Symbol}; n_bootstrap::Int=500)
    # Full-data estimate
    _, f_bar       = chained_bar(all_measurements, all_betas, beta_collapses)
    free_energies  = mbar_free_energies(all_measurements, all_betas, beta_collapses, f_bar)
    betas_out, obs_mean, obs_neff =
        mbar_reweight_observable(all_measurements, all_betas, beta_collapses,
                                 free_energies, observable)

    n_betas  = length(betas_out)
    K        = length(beta_collapses)
    boot_obs_mat = zeros(n_bootstrap, n_betas)
    boot_f_mat   = zeros(n_bootstrap, K)

    for b in 1:n_bootstrap
        meas_boot = [all_measurements[k][rand(1:length(all_measurements[k]),
                                              length(all_measurements[k]))]
                     for k in 1:K]
        _, f_boot      = chained_bar(meas_boot, all_betas, beta_collapses)
        f_boot_mbar    = mbar_free_energies(meas_boot, all_betas, beta_collapses, f_boot)
        _, boot_obs, _ = mbar_reweight_observable(meas_boot, all_betas, beta_collapses,
                                                   f_boot_mbar, observable)
        boot_obs_mat[b, :] = boot_obs
        boot_f_mat[b, :]   = f_boot_mbar
    end

    obs_err         = [std(boot_obs_mat[:, j]) for j in 1:n_betas]
    free_energy_err = [std(boot_f_mat[:, j])   for j in 1:K]
    return betas_out, obs_mean, obs_err, obs_neff, free_energies, free_energy_err
end