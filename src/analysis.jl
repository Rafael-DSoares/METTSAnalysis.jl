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


# ==============================================================================
# 1. BOOTSTRAP FAMILY (average_observable_single_boot)
# ==============================================================================

"""
    average_observable_single_boot(data::AbstractVector{<:NamedTuple}, tags::Vector{String}; n_bootstrap=500)
"""
function average_observable_single_boot(data::AbstractVector{<:NamedTuple}, tags::Vector{String}; n_bootstrap::Int=500)
    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)
        series = [m[sym_tag] for m in data]
        obs_mean, obs_err = bootstrap_average_observable(series; n_bootstrap=n_bootstrap)
        return sym_tag => (obs_mean, obs_err)
    end
    return NamedTuple(results_pairs)
end

"""
    average_observable_single_boot(data::AbstractVector{<:NamedTuple}; n_bootstrap=500)
"""
function average_observable_single_boot(data::AbstractVector{<:NamedTuple}; n_bootstrap::Int=500)
    isempty(data) && return NamedTuple()
    all_tags = keys(first(data))
    
    results_pairs = map(all_tags) do sym_tag
        series = [m[sym_tag] for m in data]
        obs_mean, obs_err = bootstrap_average_observable(series; n_bootstrap=n_bootstrap)
        return sym_tag => (obs_mean, obs_err)
    end
    return NamedTuple(results_pairs)
end



function average_observable_single_boot(data::AbstractVector{<:NamedTuple}, 
                                        beta_collapse::Real, 
                                        betas::AbstractVector{<:Real}, 
                                        tags::Vector{String}; 
                                        n_bootstrap::Int=500)
    beta_index = argmin(abs.(betas .- beta_collapse))
    
    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)
        series = [m[sym_tag][beta_index] for m in data]
        obs_mean, obs_err = bootstrap_average_observable(series; n_bootstrap=n_bootstrap)
        return sym_tag => (obs_mean, obs_err)
    end
    return NamedTuple(results_pairs)
end



function average_observable_single_boot(data::AbstractVector{<:NamedTuple}, 
                                   beta_collapse::Real, 
                                   betas::AbstractVector{<:Real})
    isempty(data) && return NamedTuple()
    all_tags = filter(t -> t != "log_norm", collect(string.(keys(first(data)))))
    return average_observable_single_boot(data, beta_collapse, betas, all_tags)
end

# ==============================================================================
# 2. STANDARD ANALYTICAL FAMILY (average_observable_single)
# ==============================================================================

"""
    average_observable_single(data::AbstractVector{<:NamedTuple}, tags::Vector{String})
"""
function average_observable_single(data::AbstractVector{<:NamedTuple}, tags::Vector{String})
    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)
        series = [m[sym_tag] for m in data]
        obs_mean, obs_err = standard_average_observable(series)
        return sym_tag => (obs_mean, obs_err)
    end
    return NamedTuple(results_pairs)
end

"""
    average_observable_single(data::AbstractVector{<:NamedTuple})
"""
function average_observable_single(data::AbstractVector{<:NamedTuple})
    isempty(data) && return NamedTuple()
    all_tags = keys(first(data))
    
    results_pairs = map(all_tags) do sym_tag
        series = [m[sym_tag] for m in data]
        obs_mean, obs_err = standard_average_observable(series)
        return sym_tag => (obs_mean, obs_err)
    end
    return NamedTuple(results_pairs)
end


"""
    average_observable_single(data::AbstractVector{<:NamedTuple}, 
                              beta_collapse::Real, 
                              betas::AbstractVector{<:Real}, 
                              tags::Vector{String})

Extracts the physical value at beta_collapse from the imaginary time trajectory 
of every METTS sample in the file and computes the average and standard error.
"""
function average_observable_single(data::AbstractVector{<:NamedTuple}, 
                                   beta_collapse::Real, 
                                   betas::AbstractVector{<:Real}, 
                                   tags::Vector{String})
    isempty(data) && return NamedTuple()

    beta_index = argmin(abs.(betas .- beta_collapse))
    
    results_pairs = map(tags) do tag
        sym_tag = Symbol(tag)
        
        series = [m[sym_tag][beta_index] for m in data]
    
        # 3. Calculate mean and standard error (sigma / sqrt(N-1))
        obs_mean, obs_err = standard_average_observable(series)
        
        return sym_tag => (obs_mean, obs_err)
    end
    
    return NamedTuple(results_pairs)
end

function average_observable_single(data::AbstractVector{<:NamedTuple}, 
                                   beta_collapse::Real, 
                                   betas::AbstractVector{<:Real})
    isempty(data) && return NamedTuple()
    all_tags = filter(t -> t != "log_norm", collect(string.(keys(first(data)))))
    return average_observable_single(data, beta_collapse, betas, all_tags)
end