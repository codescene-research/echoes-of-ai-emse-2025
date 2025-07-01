# This is a library of functions that we will use to make simulations
# For each variable listed in the causal graph, we will make a function that generates a new value
# based on latent parameters, and values from other variables.

module Simulation

import ..Commons

using Distributions

using DataFrames

using Statistics, StatsBase

# Make a tiny simulation to show how logistic regression works.
function ordered_logistic_simulation(N)
    xs = rand(Normal(), N)
    ys = rand(Normal(), N)
    noise = rand(Normal(0, 0.05), N)
    zs = 1.0 .+ xs .+ ys .+ noise

    cutoffs = [(0, 1), (1, 2), (2, 3)]

    observed = map(z -> to_ordinal(z, cutoffs), zs)

    return DataFrame(:x => xs,
                     :y => ys,
                     :z => zs,
                     :observed => observed)

end

function gen_dev1()
    # We don't know that variable, so we assume it's normally distributed with mean 0
    return rand(Normal(0, 1))
end

# Next variable is ordinal, so we'll use a quantization function
function to_ordinal(quantitative, thresholds_labels)
    for (threshold, label) in reverse(thresholds_labels)
        if quantitative >= threshold
            return label
        end
    end
    return first(thresholds_labels)[2]
end

function ai_xp_function(dev1, noise)
    # 1 for low, 2 for medium, 3 for high
    #
    # Here we try something, what it devs with less experience
    # Considered themselves more knowledgeable about AI?
    v = -dev1 + noise
    return to_ordinal(v, [(-1, 1), (-0.0, 2), (1.0, 3)])
end

function gen_ai_xp(dev1)
    # It seems (Gelman) that we model the noise with a logistic distribution
    return ai_xp_function(dev1, rand(Logistic(0, 0.5)))
end

# Returns the log-odds of the ai_preference
function ai_pref_function(dev1, ai_xp, noise)
    # Developers with low experience in AI are less likely to use it
    table = [-2, 0, 2]

    # Here, we encode that mediocre developers want to use AI more
    return table[ai_xp] - 3 * dev1 + noise
end

function gen_ai_pref(dev1, ai_xp)
    # This is a yes/no variable.
    # We use the latent variable definition of logistic regression
    # Modeling the noise as a logistic.
    # See: https://en.wikipedia.org/wiki/Logistic_regression#As_a_generalized_linear_model

    v = ai_pref_function(dev1, ai_xp, rand(Logistic(0, 0.5)))

    return Commons.logistic(v) > 0.5
end

function gen_ai_use(ai_pref)
    # This is not very easy to model this as
    # A direct function, because we want to stratify
    # (To make sure there's as many devs using AI than those who don't?)

    # Approximating it here...
    if ai_pref
         return rand(Bernoulli(0.80))
    else
         return rand(Bernoulli(0.20))
    end
end

function dev_exp_function(dev, noise)
    # Dev is on range -3 to 3 (approx)
    # Target range is 1 to 6
    return to_ordinal(dev + noise, [(-3, 1), (-2, 2), (-1, 3),
        (0, 4), (1, 5), (2, 6)])
end

function gen_dev_exp(dev)
    return dev_exp_function(dev, rand(Logistic(0.1)))
end

function java_exp_function(dev, noise)
    return to_ordinal(dev + noise, [(-1, 1), (-0.5, 2), (0.5, 3)])
end

function gen_java_exp(dev)
    return java_exp_function(dev, rand(Logistic(0.1)))
end

function dev_skill_function(dev_exp, java_exp)
    if dev_exp ∈ [1, 2]
        return 1
    elseif java_exp == 1
        return 1
    elseif (dev_exp ∈ [4, 5, 6]) && java_exp == 3
        return 3
    else
        return 2
    end
end

function gen_dev_skill(dev)
    dev_exp = gen_dev_exp(dev)
    java_exp = gen_java_exp(dev)
    return dev_skill_function(dev_exp, java_exp)
end

function code1q_function(dev1_skill, ai_use, ai_xp, noise)
    # Less experienced developer produce worse code.
    # But devs more experienced with AI produce better code!
    ai_xp_effect = [0, 1, 2]

    dev1_skill_effect = [0.5, 1.0, 1.5]

    if ai_use
        v = dev1_skill_effect[dev1_skill] + 0.5 + 0.15 * ai_xp_effect[ai_xp]
    else
        v = dev1_skill_effect[dev1_skill]
    end
    return v + noise
end

function gen_code1q(dev1_skill, ai_use, ai_xp)
    return code1q_function(dev1_skill, ai_use, ai_xp, rand(Normal(0, 0.25)))
end

function gen_dev2()
    return gen_dev1()
end

function code2q_function(code1q, dev2, noise)
    return code1q + 0.5 * dev2 + noise
end

function gen_code2q(code1q, dev2)
    return code2q_function(code1q, dev2, rand(Normal(0.0, 1.0)))
end

function time_function(dev2, code1q, noise)
    base = log(3*60) # We imagine the mean is about 3 hours
    return exp(base - 0.5 * code1q - 0.25 * dev2 + noise)
end

function gen_time(dev2, code1)
    return time_function(dev2, code1, rand(Normal(0.0, 0.5)))
end

function codehealth_function(code2q, noise)
    labels = 1:10

    # Cutoffs from -2 to 2, evenly spaced
    cutoffs = cumsum(ones(10) * 0.45) .- 2.5

    return to_ordinal(code2q + noise, collect(zip(cutoffs, labels)))
end

function gen_codehealth(code2)
    return codehealth_function(code2, rand(Logistic(0, 0.2)))
end

# One other option to model this type of variable
# Is to use a collection of logistic functions, with an intercept
# and a common slope

function productivity_function(question, dev2, code1, noise)
    productivity = 0.5 * dev2 + 0.5 * code1 + noise

    return productivity_answer(question, productivity)
end

function productivity_answer(question, productivity)
    # Now we encode how the questions reflect real productity
    #
    # We set some questions as inverted scale
    # Skill is -1, +1
    # slopes_skill = [1, 0, -1, 1,   0, -1, 1, 0, -1, 1, -1]
    inverted = [1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1]

    intercepts = [0, -1, 1, 0, -1, 1, 0, 1, 0, -1, 0]
    slopes = [1, 2, 3, 1, 0.5, 0.5, 1, 1, 1, 1, 1] .* inverted

    score = intercepts[question] + slopes[question] * productivity

    if score < -1.5
        return -2
    elseif score < -0.5
        return -1
    elseif score < 0.5
        return 0
    elseif score < 1.5
        return 1
    else
        return 2
    end
end

function gen_productivity(question, dev2, code1)
    return productivity_function(question, dev2, code1, rand(Logistic(0.5)))
end

# This returns the test coverage.
# Higher quality code has better test coverage
# Google study reported they got approximately between 70% and 90%
# See: https://research.google/pubs/code-coverage-at-google/

# TODO: Model this with a Beta distribution
# And check the validity of data-transformations
function test_coverage_function(code2q, noise)
    return Commons.logistic(0.25 * code2q + 1.45 + noise)
end

function gen_test_coverage(code2)
    return test_coverage_function(code2, rand(Normal(0, 0.25)))
end

# We have a function that determines if we have missing data!
# For a start we treat it as an ordinary column.
# function data_missingness_function(time, noise)
#     # time is logNormal, maybe be careful with the noise here
#     t1 = exp(log(time) + noise)
#     if (t1 > 8*60) true else false end
# end

function data_missingness_function(code1q, dev2, noise)
    # Higher quality code and more experiences dev2s
    # Are less likely to fail to do assignment 2...
    # Base rate is already quite low
    log_odds_missing = -0.25 + -1 * code1q + -0.5 * dev2 + noise

    return Commons.logistic(log_odds_missing)
end

function gen_data_missingness(code1q, dev2)
    return data_missingness_function(code1q, dev2,
                                     rand(Logistic(0.0, 0.20))) > 0.5
end



# Group all functions together to generate a new data-point
function gen_data_point()
    dev1 = gen_dev1()
    dev2 = gen_dev2()

    ai_xp = gen_ai_xp(dev1)
    ai_pref = gen_ai_pref(dev1, ai_xp)
    dev1_skill = gen_dev_skill(dev1)

    ai_use = gen_ai_use(ai_pref)

    code1 = gen_code1q(dev1_skill, ai_use, ai_xp)

    # Does dev2 succeed??
    data_missingness = gen_data_missingness(code1, dev2)

    time = gen_time(dev2, code1)
    code2 = gen_code2q(code1, dev2)

    code_health = gen_codehealth(code2)
    test_coverage = gen_test_coverage(code2)

    # productivity
    productivities = [gen_productivity(q, dev2, code1) for q in 1:10]
    symbols = [Symbol("productivity_q$i") for i in 1:10]
    prod_median = median(productivities)

    # Data
    data = Dict(:ai_xp => ai_xp,
            :ai_pref => ai_pref,
            :ai_use => ai_use,
            :dev1_skill => dev1_skill,
            :dev1 => dev1,
            :dev2 => dev2,
            :code1q => code1,
            :code2q => code2,
            :code_health => code_health,
            :test_coverage=>test_coverage,
            :productivity_median => prod_median,
            :time=>time,
            :data_missingness => data_missingness)

    for (symbol, prod) in zip(symbols, productivities)
        data[symbol] = prod
    end
    return NamedTuple(collect(data))
end

function gen_perfect_dataset()
    dev1s = -3:0.5:3
    dev2s = -3:0.5:3

    records = []
    for dev1 in dev1s
        for dev2 in dev2s
            for ai_use in [false, true]
                ai_xp = ai_xp_function(dev1, 0)
                ai_pref = ai_pref_function(dev1, ai_xp, 0)
                dev1_java_exp = java_exp_function(dev1, 0)
                dev1_dev_exp = dev_exp_function(dev1, 0)
                dev1_skill = dev_skill_function(dev1_dev_exp, dev1_java_exp)

                # Consequences of ai_use are more interesting
                code1q = code1q_function(dev1_skill, ai_use, ai_xp, 0)
                code2q = code2q_function(code1q, dev2, 0)

                time = time_function(dev2, code1q, 0)
                code_health = codehealth_function(code2q, 0)
                test_coverage = test_coverage_function(code2q, 0)

                record = Dict(
                    :dev1 => dev1,
                    :dev2 => dev2,
                    :ai_use => ai_use,
                    :ai_xp => ai_xp,
                    :ai_pref => ai_pref,
                    :dev1_skill => dev1_skill,
                    :code1q => code1q,
                    :code2q => code2q,
                    :time => time,
                    :code_health => code_health,
                    :test_coverage => test_coverage)

                scores = [productivity_function(question, dev2, code1q,0 )
                          for question in 1:10]

                record[:productivity_median] = median(scores)

                for productivity_question in 1:10
                    score = scores[productivity_question]
                    symbol = Symbol("productivity_q$productivity_question")
                    record[symbol] = score
                end

                append!(records, [NamedTuple(collect(record))])
            end
        end
    end
    return DataFrame(records)
end

# Make a data frame with many data points.
function gen_data_set(N)
    points = [gen_data_point() for r in 1:N]
    return DataFrame(points)
end

# There's a function that decides if the data is missing or not!
# DEV1 can fail if he's not good enough, or his AI doesn't work well.
# But that's compensated by the researchers.
# Otherwise, the dev2 can fail to make the code...

# We should simulate a dropout for the people who spend more than 8 hours on the task...
# (Uses the data_missingness to decide to remove some values.)
function dropout(simu::DataFrame, targets::Vector{Symbol})
#=     productivity_outcomes = vcat([:productivity_median],
        [Symbol("productivity_q$i") for i in 1:10])

    outcomes = vcat([:time, :code_health, :test_coverage],
        productivity_outcomes) =#

    return set_missing_values!(copy(simu), :data_missingness, targets)
end

function dropout(simu::DataFrame, target::Symbol)
    return dropout(simu, [target])
end

function set_missing_values!(df::DataFrame, missingness::Symbol, targets::Vector{Symbol})
    # We need to make sure the type allows "Missing"
    for col in targets
        col_type = eltype(df[!,col])
        df[!, col] = convert(Vector{Union{Missing, col_type}}, df[!, col])
    end
    for row in eachrow(df)
        if row[missingness] == 1
            for col in targets
                row[col] = missing
            end
        end
    end
    return df
end

end
