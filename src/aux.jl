"""
    pool_states(all_measurements, all_betas, beta_collapses)

Groups entries that share the same beta_collapse into a single pooled state.
Multiple files simulated at the same collapse temperature are concatenated
into one measurement vector. Their `all_betas` arrays must be the same..

Returns `(pooled_measurements, pooled_betas, unique_beta_collapses)`.
"""
function pool_states(all_measurements, all_betas, beta_collapses)
    @assert issorted(beta_collapses) "beta_collapses must be sorted ascending"

    unique_betas = unique(beta_collapses)  # preserves order since input is sorted

    pooled_measurements = Vector{eltype(all_measurements)}(undef, length(unique_betas))
    pooled_betas        = Vector{eltype(all_betas)}(undef, length(unique_betas))

    for (j, β) in enumerate(unique_betas)
        idxs = findall(b -> isapprox(b, β; atol=1e-8), beta_collapses)

        # All files for this β must have been run at the same temperatures
        ref_betas = all_betas[idxs[1]]
        for i in idxs[2:end]
            @assert all_betas[i] ≈ ref_betas "Files for β=$β have mismatched all_betas arrays at index $i"
        end

        pooled_betas[j]        = ref_betas
        pooled_measurements[j] = vcat([all_measurements[i] for i in idxs]...)
    end

    return pooled_measurements, pooled_betas, unique_betas
end




"""
    pool_states_single(all_measurements, beta_collapses)

Groups entries sharing the same beta_collapse into a single pooled state,
for the case where each file only contains measurements at its own beta_collapse
(no log_norm, no cross-beta reweighting).

Returns `(pooled_measurements, unique_beta_collapses)`.
"""
function pool_states_single(all_measurements, beta_collapses)
    @assert issorted(beta_collapses) "beta_collapses must be sorted ascending"

    unique_betas = unique(beta_collapses)

    pooled_measurements = Vector{eltype(all_measurements)}(undef, length(unique_betas))

    for (j, β) in enumerate(unique_betas)
        idxs = findall(b -> isapprox(b, β; atol=1e-8), beta_collapses)
        pooled_measurements[j] = vcat([all_measurements[i] for i in idxs]...)
    end

    return pooled_measurements, unique_betas
end
