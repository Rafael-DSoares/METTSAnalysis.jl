using Statistics
using StatsBase

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
    bootstrap_average_observable(data::AbstractVector{<:NamedTuple}, tags::Vector{String}; n_bootstrap=500)

Loops over a list of observables (tags), extracts their 1D series from the NamedTuple data, 
and returns a NamedTuple containing the (mean, error) for each.
"""
function average_observable_single(data::AbstractVector{<:NamedTuple}, tags::Vector{String}; n_bootstrap::Int=500)
    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)
        
        series = [m[sym_tag] for m in data]
    
        obs_mean, obs_err = bootstrap_average_observable(series; n_bootstrap=n_bootstrap)
        
        return sym_tag => (obs_mean, obs_err)
    end
    return NamedTuple(results_pairs)
end

"""
    bootstrap_average_observable(data::AbstractVector{<:NamedTuple}; n_bootstrap=500)

Loops over ALL observables present in the NamedTuple data, extracts their 1D series, 
and returns a NamedTuple containing the (mean, error) for each.
"""
function average_observable_single(data::AbstractVector{<:NamedTuple}; n_bootstrap::Int=500)
    isempty(data) && return NamedTuple()

    all_tags = keys(first(data))
    
    results_pairs = map(all_tags) do sym_tag
        series = [m[sym_tag] for m in data]
        
        obs_mean, obs_err = bootstrap_average_observable(series; n_bootstrap=n_bootstrap)
        
    
        return sym_tag => (obs_mean, obs_err)
    end
    
    return NamedTuple(results_pairs)
end