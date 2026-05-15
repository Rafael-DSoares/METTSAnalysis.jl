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
    average_observable_single(data::AbstractVector{<:NamedTuple}, tags::Vector{String}; bootstrap=false, n_bootstrap=500)

Averages the specified observables. Uses standard analytical error by default. 
Set `bootstrap=true` to use bootstrap resampling.
"""
function average_observable_single(data::AbstractVector{<:NamedTuple}, 
                                   tags::Vector{String}; 
                                   bootstrap::Bool=false, 
                                   n_bootstrap::Int=500)
    isempty(data) && return NamedTuple()
    
    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)
        series = [m[sym_tag] for m in data]
        
        obs_mean, obs_err = bootstrap ? 
            bootstrap_average_observable(series; n_bootstrap=n_bootstrap) : 
            standard_average_observable(series)
            
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
                                   n_bootstrap::Int=500)
    isempty(data) && return NamedTuple()
    beta_index = argmin(abs.(betas .- beta_collapse))

    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)

        series = [m[sym_tag][beta_index] for m in data]

        obs_mean, obs_err = bootstrap ?
            bootstrap_average_observable(series; n_bootstrap=n_bootstrap) :
            standard_average_observable(series)

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