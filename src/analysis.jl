using Statistics
using StatsBase
using MCMCDiagnosticTools

const MIN_ESS_SAMPLES = 20

"""
    bootstrap_average_observable(measurements::AbstractVector; n_bootstrap=500)

Calculates the mean and bootstrap standard error of a 1D array of measurements.
"""
function bootstrap_average_observable(measurements::AbstractVector; n_bootstrap::Int=500)
    n_samples = length(measurements)
    obs_mean = mean(measurements)
    
    # Resample with replacement and calculate the mean for each bootstrap sample
    boot_means = [mean(sample(measurements, n_samples; replace=true)) for _ in 1:n_bootstrap]
    obs_err = std(boot_means)

    return obs_mean, obs_err
end

"""
    standard_average_observable(measurements::AbstractVector)

Calculates the mean and standard error of a 1D array of measurements analytically.
"""
function standard_average_observable(measurements::AbstractVector)
    n_samples = length(measurements)
    obs_mean = mean(measurements)
    
    # Prevent division by zero if there is only 1 sample
    if n_samples <= 1
        return obs_mean, 0.0
    end
    
    obs_err = std(measurements) / sqrt(n_samples)
    
    return obs_mean, obs_err
end


"""
    effective_sample_size(series::AbstractVector)

Effective sample size of the mean of a Markov-chain time series (split-chain basic ESS
from MCMCDiagnosticTools, with the autocorrelation summed up to lag n/2 instead of the
default 250). Capped at `length(series)`, so errors are never smaller than the naive ones.
Chains shorter than `MIN_ESS_SAMPLES` return their length, with a warning.
"""
function effective_sample_size(series::AbstractVector)
    n = length(series)
    if n < MIN_ESS_SAMPLES
        @warn "Only $n samples: too few to estimate the autocorrelation. Treating them as independent." maxlog=1
        return float(n)
    end
    n_eff = MCMCDiagnosticTools.ess(Float64.(series); kind=:basic, maxlag=n ÷ 2 - 1)
    isnan(n_eff) && return float(n)  # constant series: the error is zero anyway
    return clamp(n_eff, 1.0, float(n))
end

"""
    ess_average_observable(measurements::AbstractVector)

Calculates the mean and standard error of a correlated (Markov-chain) time series:
`std / sqrt(ESS)`, see `effective_sample_size`.
"""
function ess_average_observable(measurements::AbstractVector)
    n_samples = length(measurements)
    obs_mean = mean(measurements)
    n_samples <= 1 && return obs_mean, 0.0
    return obs_mean, std(measurements) / sqrt(effective_sample_size(measurements))
end

function _average_series(series::AbstractVector; error_method::Symbol, n_bootstrap::Int)
    error_method === :ess       && return ess_average_observable(series)
    error_method === :naive     && return standard_average_observable(series)
    error_method === :bootstrap && return bootstrap_average_observable(series; n_bootstrap=n_bootstrap)
    error("Unknown error_method :$error_method. Use :ess, :naive or :bootstrap.")
end

"""
    _average_chains(chains; error_method, n_bootstrap, label)

Combines independent chains (e.g. seeds) of one observable: the mean is weighted by chain
length and the errors of the chain means are added in quadrature. Warns when the chain
means scatter more than their errors allow (reduced χ² > 3), which usually means a seed
is not thermalized or was pruned too late.
"""
function _average_chains(chains::AbstractVector{<:AbstractVector}; error_method::Symbol, n_bootstrap::Int, label="")
    chains = filter(!isempty, chains)
    isempty(chains) && return (NaN, NaN)
    length(chains) == 1 && return _average_series(chains[1]; error_method=error_method, n_bootstrap=n_bootstrap)

    stats = [_average_series(c; error_method=error_method, n_bootstrap=n_bootstrap) for c in chains]
    weights = length.(chains) ./ sum(length.(chains))
    means, errs = first.(stats), last.(stats)
    obs_mean = sum(weights .* means)
    obs_err = sqrt(sum((weights .* errs) .^ 2))

    if all(errs .> 0)
        red_chi2 = sum(((means .- obs_mean) ./ errs) .^ 2) / (length(chains) - 1)
        if red_chi2 > 3
            @warn "Chain means of $label disagree beyond their errors (reduced χ² = $(round(red_chi2, digits=2)) over $(length(chains)) chains). Check the thermalization/pruning of each seed." means=means errors=errs
        end
    end
    return obs_mean, obs_err
end


"""
    average_observable_single(data::AbstractVector{<:NamedTuple}, tags::Vector{String}; bootstrap=false, n_bootstrap=500)

Averages the specified observables of one Markov chain.

`error_method` selects the error bar:
- `:ess` (default): `std / sqrt(ESS)`, accounting for the autocorrelation of the chain.
- `:naive`: `std / sqrt(N)`, only valid for independent samples.
- `:bootstrap`: resampling of individual samples; like `:naive`, it assumes independence.

`bootstrap=true` is kept for compatibility and selects `:bootstrap`.

Several independent chains (seeds) can be passed as a vector of chains instead; see
`group_chains_single`.
"""
function average_observable_single(data::AbstractVector{<:NamedTuple}, 
                                   tags::Vector{String}; 
                                   bootstrap::Bool=false, 
                                   n_bootstrap::Int=500,
                                   error_method::Symbol=bootstrap ? :bootstrap : :ess)
    isempty(data) && return NamedTuple()
    
    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)
        series = [m[sym_tag] for m in data]
        
        obs_mean, obs_err = _average_series(series; error_method=error_method, n_bootstrap=n_bootstrap)
            
        return sym_tag => (obs_mean, obs_err)
    end
    return NamedTuple(results_pairs)
end

"""
    average_observable_single(data::AbstractVector{<:NamedTuple}; kwargs...)

Averages all observables present in the data.
"""
function average_observable_single(data::AbstractVector{<:NamedTuple}; kwargs...)
    isempty(data) && return NamedTuple()
    
    all_tags = collect(string.(keys(first(data))))
    
    return average_observable_single(data, all_tags; kwargs...)
end

"""
    average_observable_single(data::AbstractVector{<:NamedTuple}, beta_collapse::Real, betas::AbstractVector{<:Real}, tags::Vector{String}; kwargs...)

Extracts the physical value at beta_collapse from the imaginary time trajectory 
and computes the average and standard error.
"""
function average_observable_single(data::AbstractVector{<:NamedTuple},
                                   beta_collapse::Real,
                                   betas::AbstractVector{<:Real},
                                   tags::Vector{String};
                                   bootstrap::Bool=false,
                                   n_bootstrap::Int=500,
                                   error_method::Symbol=bootstrap ? :bootstrap : :ess)
    isempty(data) && return NamedTuple()
    beta_index = argmin(abs.(betas .- beta_collapse))

    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)

        series = [m[sym_tag][beta_index] for m in data]

        obs_mean, obs_err = _average_series(series; error_method=error_method, n_bootstrap=n_bootstrap)

        return sym_tag => (obs_mean, obs_err)
    end
    return NamedTuple(results_pairs)
end

"""
    average_observable_single(data::AbstractVector{<:NamedTuple}, beta_collapse::Real, betas::AbstractVector{<:Real}; kwargs...)

Averages all observables present in the data at `beta_collapse`, explicitly filtering out `log_norm`.
"""
function average_observable_single(data::AbstractVector{<:NamedTuple}, 
                                   beta_collapse::Real, 
                                   betas::AbstractVector{<:Real}; 
                                   kwargs...)
    isempty(data) && return NamedTuple()
    all_tags = filter(t -> t != "log_norm", collect(string.(keys(first(data)))))
    return average_observable_single(data, beta_collapse, betas, all_tags; kwargs...)
end



# ==============================================================================
# SEVERAL INDEPENDENT CHAINS (SEEDS)
# ==============================================================================

const ChainList = AbstractVector{<:AbstractVector{<:NamedTuple}}

"""
    average_observable_single(chains::AbstractVector{<:AbstractVector{<:NamedTuple}}, tags::Vector{String}; kwargs...)

Averages observables over several independent chains (e.g. the seeds of one beta_collapse,
as returned by `group_chains_single`). The error of each chain is computed with
`error_method` and the chains are combined as in `_average_chains`, which also warns
when the seeds disagree. Prefer this over pooling the seeds into one vector.
"""
function average_observable_single(chains::ChainList,
                                   tags::Vector{String};
                                   bootstrap::Bool=false,
                                   n_bootstrap::Int=500,
                                   error_method::Symbol=bootstrap ? :bootstrap : :ess)
    all(isempty, chains) && return NamedTuple()

    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)
        series = [[m[sym_tag] for m in chain] for chain in chains]
        return sym_tag => _average_chains(series; error_method=error_method, n_bootstrap=n_bootstrap, label=tag)
    end
    return NamedTuple(results_pairs)
end

function average_observable_single(chains::ChainList; kwargs...)
    all(isempty, chains) && return NamedTuple()
    all_tags = collect(string.(keys(first(first(filter(!isempty, chains))))))
    return average_observable_single(chains, all_tags; kwargs...)
end

"""
    average_observable_single(chains::AbstractVector{<:AbstractVector{<:NamedTuple}}, beta_collapse, betas, tags; kwargs...)

Multi-chain version of the interval method: takes the value at `beta_collapse` from each
trajectory and combines the chains as above.
"""
function average_observable_single(chains::ChainList,
                                   beta_collapse::Real,
                                   betas::AbstractVector{<:Real},
                                   tags::Vector{String};
                                   bootstrap::Bool=false,
                                   n_bootstrap::Int=500,
                                   error_method::Symbol=bootstrap ? :bootstrap : :ess)
    all(isempty, chains) && return NamedTuple()
    beta_index = argmin(abs.(betas .- beta_collapse))

    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)
        series = [[m[sym_tag][beta_index] for m in chain] for chain in chains]
        label = "$tag at beta=$beta_collapse"
        return sym_tag => _average_chains(series; error_method=error_method, n_bootstrap=n_bootstrap, label=label)
    end
    return NamedTuple(results_pairs)
end

function average_observable_single(chains::ChainList,
                                   beta_collapse::Real,
                                   betas::AbstractVector{<:Real};
                                   kwargs...)
    all(isempty, chains) && return NamedTuple()
    all_tags = filter(t -> t != "log_norm", collect(string.(keys(first(first(filter(!isempty, chains)))))))
    return average_observable_single(chains, beta_collapse, betas, all_tags; kwargs...)
end
