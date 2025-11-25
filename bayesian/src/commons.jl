module Commons

export center, standardize, flip_likert,
    logistic, logit, ai_xp_to_indexes,
    dev1_skill_to_indexes,
    plot_ordered_logistic,
    credibility_interval,
    remap_scale,
    remap_selected_columns


using Plots: vline!
using Statistics
using Bijectors: bijector

using Turing, ArviZ, Random

using Plots: plot, plot!

function plotzero!(p)
    vline!(p, [0.0], style=:dash, color=:black, label=missing)
    return p
end

function center(v)
    return v .- mean(v)
end

function standardize(v)
    return center(v) ./ std(v)
end

function flip_likert(x)
    return (x - 3) * -1 + 3
end

function logistic(x)
    return 1 / (1 + exp(-x))
end

function logit(p)
        return log(p/(1-p))
end

function sample_model(model)
    return sample(model, NUTS(0.65), MCMCThreads(), 1000, 4)
end

# Aggregation and categorical variables

function ai_experience(has_used_ai, has_used_ai_likert)
    if !has_used_ai
        return 1
    else
        return convert(Integer, has_used_ai_likert)
    end
end

function dev1_skill_to_indexes(dev1_skills)
        indexin(dev1_skills, ["Beginner", "Intermediate", "Advanced"])
end

# Bayesian utilities
function posterior_summary(posterior_dist)
    ci = credibility_interval(posterior_dist, 0.95)
    m = round(mean(posterior_dist), digits=2)

    return (mean=m, interval_95=ci)
end

function credibility_interval(posterior_dist, credibility)
    # if 90% credibility, we want to remove 5% on each side
    # Or keep 45% around the 50% mark.
    qmin = 0.5 - credibility / 2
    qmax = 0.5 + credibility / 2

    return round.(quantile(posterior_dist, [qmin, qmax]),
                  digits=2)
end

function plot_ordered_logistic(discrimination, cutoffs)
    # For each cutoff, we build a function
    functions = map(c -> (x -> logistic(discrimination * (x-c))),
                    cutoffs)

    xrange = (minimum(cutoffs) - 1, maximum(cutoffs) + 1)

    p = plot(xrange=xrange, yrange=(0,1))
    for f in functions
        plot!(p, x -> f(x))
    end
    return p
end

# Not needed, but keeping it for reference.
function item_response_curves(cutoffs, level, x)
    functions = map(c -> (x -> logistic((x-c))), cutoffs)

    if level == 1
        y = 1 - functions[1](x)
        return y
    end

    if level == length(cutoffs) + 1
        y = functions[level-1](x)
        return y
    end

    y = functions[level-1](x) - functions[level](x)

    return y
end

function plot_item_response_curves(cutoffs)
    levels = 1:length(cutoffs)+1

    xrange = (minimum(cutoffs) - 2, maximum(cutoffs) + 2)
    yrange = (0, 1)
    p = plot(;xrange, yrange)

    for l in levels
        f = x -> pdf(OrderedLogistic(x, cutoffs), l)
        plot!(p, f, label="p(level=$l)", lw=3)

    end
    vline!(p, cutoffs,
           style=:dash,
           color=:black,
           label=missing)

    # functions = [x -> item_response_curves(cutoffs, l, x)
    #              for l in 1:length(cutoffs)+1]

    # xrange = (minimum(cutoffs) - 1, maximum(cutoffs) + 1)

    # p = plot(xrange=xrange, yrange=(0,1))
    # for i in eachindex(functions)
    #     f = functions[i]
    #     plot!(p, x -> f(x), label="p(level=$i)")
    # end

    # vline!(p, cutoffs,
    #        style=:dash,
    #        color=:black,
    #        label=missing)
    return p
end

function response_ordered_logistic(discrimination, cutoffs)
    functions = map(c -> (x -> logistic(discrimination * (x-c))),
                    cutoffs)

    accumulate(x) = 1 + sum([f(x) for f in functions])
    return accumulate
end


function plot_ordered_logistic_cumulative(discrimination, cutoffs)
    xrange = (minimum(cutoffs) - 1, maximum(cutoffs) + 1)

    response = response_ordered_logistic(discrimination,
                                         cutoffs)
    p = plot(response, xrange=xrange)
    return p
end



"""
A function which remaps from a range into another

(Useful for mapping from one type of likert scale to another)
"""
function remap_scale(xs, input_scale, output_scale)
    if length(input_scale) != length(output_scale)
       throw(DomainError((input_scale, output_scale), "The ranges don't have the same size"))
    end

    for x in xs
        if x ∉ input_scale
            throw(DomainError(x, "This value is not part of the input range"))
        end
    end

    i = indexin(xs, input_scale)

    ys = getindex(output_scale, i)

    return ys
end

"""
Remaps the selected columns from an input scale
to an output scale

inverted is a list of columns for which we use an inverted scale
"""
function remap_selected_columns(dataframe,
        selected_columns,
        input_scale, output_scale;
        inverted=[])

    for q in 1:length(selected_columns)
        selected_column = selected_columns[q]

        if selected_column ∉ inverted
            remapped = remap_scale(dataframe[!,selected_column],
                input_scale,
                output_scale)
        else
            remapped = remap_scale(dataframe[!,selected_column],
                input_scale,
                reverse(output_scale))
        end
        dataframe[!,selected_column] = remapped
    end
    return dataframe
end

"""
Returns a list of variables (that have posteriors) in a model
"""
function model_params(model)
    _, sym2range = bijector(model, Val(true));
    return sym2range

    # symbols = []
    # for k in keys(sym2range)
    #     range = sym2range[k][1]
    #     if length(range) == 1
    #         append!(symbols, [k])
    #     else
    #         # If the param p is an array,
    #         # we generate params
    #         # p[1], p[2], p[3]...
    #         for i in 1:length(range)
    #             s = Symbol("$k[$i]")
    #             append!(symbols, [s])
    #         end
    #     end
    # end


end

function index(name, values)
    coords = NamedTuple([name => values])
    return coords
end

function single_var(name)
    # They're empty
    # because only the arrays are
    # used in the loo_model function
    return (coords=[],
            dims=[])
end

function var_array(name, indexes...)
    # We merge all the indexes
    # (built with the index function above)
    coords = merge(indexes...)
    dims = NamedTuple([name => keys(coords)])

    return (coords=coords, dims=dims)
end


function model_mapping(outcome, parameters...)
    coords = merge(outcome[:coords],
                   merge(map(p -> p[:coords], parameters)...)
                   )

    dims = merge(outcome[:dims],
                 merge(map(p -> p[:dims], parameters)...))

    return (coords=coords, dims=dims)
end


"""
Takes two models, one for training, one for prediction (with outputs set to missing)
and a dictionary representing the way we map the different dimensions

- model_train: The model you want to check
- model_predict: A model to make predictions (based on posteriors)
- coords: a named tuple of coordinate name to possible values
- dims: a named tuple of all arrays in the model, and how they're indexed with coordinates
- data: The data you use for the training.
"""
function inference_data(model_train, model_predict;
                        coords::NamedTuple,
                        dims::NamedTuple,
                        data::NamedTuple,
                        n_samples::Integer=1000,
                        rng::Random.AbstractRNG=Random.default_rng())

    @assert all(typeof(v) <: AbstractArray for v in values(coords))
    @assert all(typeof(v) <: Tuple for v in values(dims))
    @assert all(typeof(v) <: AbstractArray for v in values(data))

    n_samples_warmup = n_samples
    sampler = NUTS(n_samples_warmup, 0.8)

    chain_prior = sample(rng, model_train, Prior(), n_samples)
    chain_posterior = sample(rng, model_train, sampler, MCMCSerial(), n_samples, 4)

    prior_predictive = predict(rng, model_predict, chain_prior)
    posterior_predictive = predict(rng, model_predict, chain_posterior)

    log_likelihoods = let
        # second argument is to avoid a lot of error messages...
        log_likelihood = Turing.pointwise_loglikelihoods(
            model_train, MCMCChains.get_sections(chain_posterior, :parameters)
        )
        # Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
        # Convert all symbols to strings...
        ynames = string.(keys(posterior_predictive))
        # Index the log_likelihood dict with ynames (re-orders it)
        log_likelihood_y = getindex.(Ref(log_likelihood), ynames)
        # Create a NamedTuple, with y as the log_likelihoods.
        # (; y=cat(log_likelihood_y...; dims=3))
        NamedTuple(k => reshape(log_likelihood_y, size(data[k]))
                   for k in keys(data))
    end

    return from_mcmcchains(chain_posterior;
                           posterior_predictive,
                           log_likelihood=log_likelihoods,
                           prior=chain_prior,
                           prior_predictive,
                           observed_data=data,
                           coords=coords,
                           dims=dims,
                           library=Turing)
end

end
