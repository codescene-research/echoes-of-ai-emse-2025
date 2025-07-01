module Models

# Models for the various data
using Statistics
using Turing
using Distributions
using LinearAlgebra

using StatsFuns: logistic

# BIAS: Does the AI group differ from the human-only group?
# TODO: We did find a significant effect of skill on AI_use
# Which indicates that our experimental setup has a bit of bias:
# The group with AI users has slightly less experienced people.
@model function bias_analysis(gender, age_group, background, dev1exp, ai_xp, ai_use)
    intercept ~ Normal()

    # Gender: Unordered, three levels (man, woman, undisclosed)
    sigma_gender ~ Exponential()
    effect_gender_z ~ filldist(Normal(0, 1), 3)
    effect_gender = effect_gender_z .* sigma_gender

    # Age is ordered (in groups)
    # We use a Dirichlet allocator
    effect_age ~ Normal()
    effect_age_group_factor ~ Dirichlet(5, 1.0)
    effect_age_group = cumsum(effect_age_group_factor) .* effect_age
    # effect_age_group ~ filldist(Normal(), 5)

    # Background, undordered, three levels
    sigma_background ~ Exponential()
    effect_background_z ~ filldist(Normal(0, 1), 4)
    effect_background = effect_background_z .* sigma_background

    # Experience in Java
    effect_skill ~ Normal()
    effect_skill_factor ~ Dirichlet(3, 1.0)
    effect_skill = cumsum(effect_skill_factor) .* effect_skill

    # AI experience
    effect_ai ~ Normal()
    effect_ai_xp_factor ~ Dirichlet(5, 1.0)
    effect_ai_xp = cumsum(effect_ai_xp_factor) .* effect_ai

    effect_gender_ = getindex(effect_gender, gender)
    # Effect of age for each sample
    effect_age_ = getindex(effect_age_group, age_group)
    # Effect background
    effect_background_ = getindex(effect_background, background)
    # Effect of skill for each sample
    effect_dev1exp_ = getindex(effect_skill, dev1exp)
    # Effect of AI for each sample
    effect_ai_xp_ = getindex(effect_ai_xp, ai_xp)

    log_odds_ai = intercept .+ effect_gender_ .+ effect_age_ .+ effect_background_ .+ effect_ai_xp_ .+ effect_dev1exp_

    proba_ai = logistic.(log_odds_ai)

    for i in 1:length(ai_use)
        ai_use[i] ~ Bernoulli(proba_ai[i])
    end
end


# COMPLETION TIME
# We assume that the samples are centered (mean of 0)

# Effect of just ai on completion time
@model function completion_time_analysis(ai_uses, log_times)
    # We assume the samples are centered...
    intercept ~ Normal(0, 1)

    ai_use_effect ~ Normal()

    predicted_log_times = intercept .+ ai_use_effect .* ai_uses

    sigma ~ Exponential()

    log_times ~ MvNormal(predicted_log_times, sigma)
end

# ai_use: Vector of Booleans
# dev2_uninterrupted: Vector of Indexes (1: Yes, 2: Yes, but breaks, 3: No)
# log_time: Vector of Floats
@model function completion_time_with_interruptions(ai_use, dev2_uninterrupted, log_time)
    intercept ~ Normal()

    ai_use_effect ~ Normal()

    # One effect for each group...
    interruption_effect ~ filldist(Normal(), 3)

    # One standard deviation is expected for the noise of each
    noise ~ filldist(Exponential(), 3)

    interruption_effect_ = getindex(interruption_effect, dev2_uninterrupted)

    log_time_predicted = intercept .+ ai_use_effect .* ai_use .+ interruption_effect_

    # A vector of standard deviations...
    noise_ = getindex(noise, dev2_uninterrupted)

    for i in 1:length(log_time)
        log_time[i] ~ Normal(log_time_predicted[i], noise_[i])
    end
end


@model function completion_time_analysis_with_controls(ai_use, ai_xp, dev1_skill, log_time, prior_effect_ai=Normal())
    # We assume the samples are centered.
    intercept ~ Normal(0, 1)
    # effect_ai ~ Normal(-0.55, 0.3) # Based on the study by Google.

    effect_ai ~ prior_effect_ai
    # Removed this, because the effects are centered at zero
    # For skill
    # effect_skill ~ Normal()

    # Effect of AI_xp
    sigma_ai_xp ~ Exponential(0.1)
    effect_ai_xp_z ~ filldist(Normal(0, 1), 5)
    effect_ai_xp = effect_ai_xp_z * sigma_ai_xp .+ effect_ai

    sigma_skill ~ Exponential(0.1)
    effect_dev1_skill_z ~ filldist(Normal(0, 1), 3)
    effect_dev1_skill = effect_dev1_skill_z * sigma_skill # .+ effect_skill

    # We need to fetch the relevant effects to add, depending on the categorical variables.
    effect_ai_xp_ = getindex(effect_ai_xp, ai_xp)
    effect_dev1_skill_ = getindex(effect_dev1_skill, dev1_skill)

    log_time_predicted = intercept .+ effect_dev1_skill_ .+ (effect_ai_xp_ .* ai_use)

    sigma ~ Exponential()

    # log_time ~ MvNormal(log_time_predicted, sigma)
    # We use a loop, because MvNormal doesn't support missing data in the vector
    for i in 1:length(log_time)
        log_time[i] ~ Normal(log_time_predicted[i], sigma)
    end
end

@model function completion_time_analysis_with_controls_2(ai_use, ai_xp, dev1_skill, log_time, prior_effect_ai=Normal())
    # We assume the samples are centered.
    intercept ~ Normal(0, 1)
    # effect_ai ~ Normal(-0.55, 0.3) # Based on the study by Google.


    # Effect of AI xp
    effect_ai ~ prior_effect_ai
    ladder_ai_xp ~ Dirichlet(5, 1.0)
    effect_ai_xp = cumsum(ladder_ai_xp) .* effect_ai

    # Effect of Skill
    effect_skill ~ Normal()
    ladder_skill ~ Dirichlet(3, 1.0)
    effect_dev1_skill = cumsum(ladder_skill) .* effect_skill

    # We need to fetch the relevant effects to add, depending on the categorical variables.
    effect_ai_xp_ = getindex(effect_ai_xp, ai_xp)
    effect_dev1_skill_ = getindex(effect_dev1_skill, dev1_skill)

    log_time_predicted = intercept .+ effect_dev1_skill_ .+ (effect_ai_xp_ .* ai_use)

    sigma ~ Exponential()

    # log_time ~ MvNormal(log_time_predicted, sigma)
    # We use a loop, because MvNormal doesn't support missing data in the vector
    for i in 1:length(log_time)
        log_time[i] ~ Normal(log_time_predicted[i], sigma)
    end
end

@model function completion_time_analysis_with_controls_and_interruptions(ai_use, ai_xp, dev1_skill, dev2_uninterrupted, log_time, prior_effect_ai=Normal())
    # We assume the samples are centered.
    intercept ~ Normal(0, 1)
    # effect_ai ~ Normal(-0.55, 0.3) # Based on the study by Google.

    effect_ai ~ prior_effect_ai
    # Removed this, because the effects are centered at zero
    # For skill
    # effect_skill ~ Normal()

    # Effect of AI_xp
    sigma_ai_xp ~ Exponential(0.1)
    effect_ai_xp_z ~ filldist(Normal(0, 1), 5)
    effect_ai_xp = effect_ai_xp_z * sigma_ai_xp .+ effect_ai

    # Effect of skill
    sigma_skill ~ Exponential(0.1)
    effect_dev1_skill_z ~ filldist(Normal(0, 1), 3)
    effect_dev1_skill = effect_dev1_skill_z * sigma_skill # .+ effect_skill

    # Effect of interuptions
    sigma_interruptions ~ Exponential(0.1)
    effect_interruptions_z ~ filldist(Normal(), 3)
    effect_interruptions = effect_interruptions_z * sigma_interruptions

    # We need to fetch the relevant effects to add, depending on the categorical variables.
    effect_ai_xp_ = getindex(effect_ai_xp, ai_xp)
    effect_dev1_skill_ = getindex(effect_dev1_skill, dev1_skill)
    effect_interruptions_ = getindex(effect_interruptions, dev2_uninterrupted)

    log_time_predicted = intercept .+ effect_dev1_skill_ .+ (effect_ai_xp_ .* ai_use) .+ effect_interruptions_

    # One standard deviation for each group (interrupted, took breaks, not interrupted)
    residuals ~ filldist(Exponential(), 3)

    residuals_ = getindex(residuals, dev2_uninterrupted)

    # log_time ~ MvNormal(log_time_predicted, sigma)
    # We use a loop, because MvNormal doesn't support missing data in the vector
    for i in 1:length(log_time)
        log_time[i] ~ Normal(log_time_predicted[i], residuals_[i])
    end
end

@model function completion_time_analysis_with_controls_and_interruptions_2(ai_use, ai_xp, dev1_skill, dev2_uninterrupted, log_time, prior_effect_ai=Normal())
    # We assume the samples are centered.
    intercept ~ Normal(0, 1)
    # effect_ai ~ Normal(-0.55, 0.3) # Based on the study by Google.


    # Effect of AI xp
    effect_ai ~ prior_effect_ai
    ladder_ai_xp ~ Dirichlet(5, 1.0)
    effect_ai_xp = cumsum(ladder_ai_xp) .* effect_ai

    # Effect of Skill
    effect_skill ~ Normal()
    ladder_skill ~ Dirichlet(3, 1.0)
    effect_dev1_skill = cumsum(ladder_skill) .* effect_skill

    effects_ai = getindex(effect_ai_xp, ai_xp)
    effects_skill = getindex(effect_dev1_skill, dev1_skill)

    # Effect of interuptions
    effect_interruptions ~ Normal()
    ladder_interruptions ~ Dirichlet(3, 1.0)
    effect_interruptions = cumsum(ladder_interruptions) .* effect_interruptions

    # We need to fetch the relevant effects to add, depending on the categorical variables.
    effect_ai_xp_ = getindex(effect_ai_xp, ai_xp)
    effect_dev1_skill_ = getindex(effect_dev1_skill, dev1_skill)
    effect_interruptions_ = getindex(effect_interruptions, dev2_uninterrupted)

    log_time_predicted = intercept .+ effect_dev1_skill_ .+ (effect_ai_xp_ .* ai_use) .+ effect_interruptions_

    # One standard deviation for each group (interrupted, took breaks, not interrupted)
    residuals ~ filldist(Exponential(), 3)

    residuals_ = getindex(residuals, dev2_uninterrupted)

    # log_time ~ MvNormal(log_time_predicted, sigma)
    # We use a loop, because MvNormal doesn't support missing data in the vector
    for i in 1:length(log_time)
        log_time[i] ~ Normal(log_time_predicted[i], residuals_[i])
    end
end


# CODE HEALTH

# INVALID: The CH is averaged over many files...
@model function code_health_analysis(ai_use, code_health)
    # Expected average code health
    intercept ~ Normal(7, 1)

    # One effect of AI
    effect_ai ~ Normal()

    # Cutoffs at 0, 1, 2...
    # Cutoffs are fixed, evenly spaced..
    # cutoffs = collect(0:9)
    # Alternative method
    # (Not constrained to be evenly spaced)
    offsets ~ filldist(Exponential(1), 10-2)
    cutoffs = vcat([0], cumsum(offsets))

    effect_ai_ = effect_ai .* ai_use

    for i in 1:length(code_health)
        code_health[i] ~ OrderedLogistic(intercept .+ effect_ai_[i], cutoffs)
    end

    return cutoffs
end

@model function code_health_analysis_gaussian(ai_use, code_health)
    # Expected average code health
    intercept ~ Normal(7, 1)

    # One effect of AI
    effect_ai ~ Normal()

    effect_ai_ = effect_ai .* ai_use

    sigma ~ Exponential()

    # code_health ~ MvNormal(intercept .+ effect_ai_, sigma)
    for i in 1:length(code_health)
        code_health[i] ~ Normal(intercept .+ effect_ai_[i], sigma)
    end

end


@model function code_health_analysis_with_controls(ai_use, ai_xp, dev1_skill, code_health)
    intercept ~ Normal(7, 1)

    # If we don't use a hyper-prior here, we see more shrinkage
    # The prior of mean = 0 pulls the effects noticeably.
    effect_ai ~ Normal()
    # effect_skill ~ Normal()

    # Effect of AI xp
    sigma_ai_xp ~ Exponential(0.1)
    effect_ai_xp_z ~ filldist(Normal(0, 1), 5)
    effect_ai_xp = effect_ai_xp_z * sigma_ai_xp .+ effect_ai

    # Effect of Skill
    sigma_skill ~ Exponential(0.1)
    effect_dev1_skill_z ~ filldist(Normal(0, 1), 3)
    effect_dev1_skill = effect_dev1_skill_z * sigma_skill # .+ effect_skill

    effects_ai = getindex(effect_ai_xp, ai_xp)
    effects_skill = getindex(effect_dev1_skill, dev1_skill)

    predicted_code_health = intercept .+ effects_ai .* ai_use .+ effects_skill

    # cutoffs = collect(0:10)
    # Cutoffs aren't constrained to be evenly spaced...
    offsets ~ filldist(Exponential(), 10-2)
    cutoffs = vcat([0], cumsum(offsets))

    # code_health .~ OrderedLogistic(predicted_code_health, cutoffs)
    for i in 1:length(code_health)
        code_health[i] ~ OrderedLogistic(predicted_code_health[i], cutoffs)
    end
end

@model function code_health_analysis_with_controls_gaussian(ai_use, ai_xp, dev1_skill, code_health, prior_effect_ai=Normal())
    intercept ~ Normal(0, 1)

    # If we don't use a hyper-prior here, we see more shrinkage
    # The prior of mean = 0 pulls the effects noticeably.
    effect_ai ~ prior_effect_ai

    # effect_skill ~ Normal()

    # Effect of AI xp
    sigma_ai_xp ~ Exponential(0.1)
    effect_ai_xp_z ~ filldist(Normal(0, 1), 5)
    effect_ai_xp = effect_ai_xp_z * sigma_ai_xp .+ effect_ai

    # Effect of Skill
    sigma_skill ~ Exponential(0.1)
    effect_dev1_skill_z ~ filldist(Normal(0, 1), 3)
    effect_dev1_skill = effect_dev1_skill_z * sigma_skill # .+ effect_skill

    effects_ai = getindex(effect_ai_xp, ai_xp)
    effects_skill = getindex(effect_dev1_skill, dev1_skill)

    predicted_code_health = intercept .+ effects_ai .* ai_use .+ effects_skill

    sigma ~ Exponential()

    # code_health ~ MvNormal(predicted_code_health, sigma)
    for i in 1:length(code_health)
        code_health[i] ~ Normal(predicted_code_health[i], sigma)
    end
end

@model function code_health_analysis_with_controls_gaussian_2(ai_use, ai_xp, dev1_skill, code_health, prior_effect_ai=Normal())
    intercept ~ Normal(0, 1)

    # If we don't use a hyper-prior here, we see more shrinkage
    # The prior of mean = 0 pulls the effects noticeably.
    effect_ai ~ prior_effect_ai
    # effect_skill ~ Normal()

    # Effect of AI xp
    ladder_ai_xp ~ Dirichlet(5, 1.0)
    effect_ai_xp = cumsum(ladder_ai_xp) .* effect_ai

    # Effect of Skill
    effect_skill ~ Normal()
    ladder_skill ~ Dirichlet(3, 1.0)
    effect_dev1_skill = cumsum(ladder_skill) .* effect_skill

    effects_ai = getindex(effect_ai_xp, ai_xp)
    effects_skill = getindex(effect_dev1_skill, dev1_skill)

    predicted_code_health = intercept .+ effects_ai .* ai_use .+ effects_skill

    sigma ~ Exponential()

    # code_health ~ MvNormal(predicted_code_health, sigma)
    for i in 1:length(code_health)
        code_health[i] ~ Normal(predicted_code_health[i], sigma)
    end

    return (;effect_ai_xp, effect_dev1_skill)
end

# regression on the logit of test coverage.
@model function test_coverage_analysis(ai_use, logit_test_coverage)
    intercept ~ Normal()
    effect_ai ~ Normal()

    predicted = intercept .+ effect_ai .* ai_use

    sigma ~ Exponential()

    # logit_test_coverage ~ MvNormal(predicted, sigma)
    for i in eachindex(logit_test_coverage)
        logit_test_coverage[i] ~ Normal(predicted[i], sigma)
    end
end

@model function test_coverage_analysis_with_controls(ai_use, ai_xp, dev1_skill, logit_test_coverage, prior_effect_ai=Normal())
    intercept ~ Normal()

    # There's 3 levels for AI XP, and DEV 1 XP
    # NOTE: Adding the hyper-priors (esp. sigma_interactions)
    # Allows for a lot of shrinking.
    effect_ai ~ prior_effect_ai
    sigma_interactions ~ Exponential(0.1)
    effect_ai_xp_z ~ filldist(Normal(), 5)
    # effect_ai_xp ~ filldist(Normal(effect_ai, sigma_interactions), 3)
    effect_ai_xp = effect_ai_xp_z .* sigma_interactions .+ effect_ai

    sigma_effect_skill ~ Exponential(0.1)
    effect_dev1_skill_z ~ filldist(Normal(), 3)
    effect_dev1_skill = effect_dev1_skill_z .* sigma_effect_skill
    # effect_dev1_skill ~ filldist(Normal(0, sigma_effect_skill), 3)

    effects_ai_xp = getindex(effect_ai_xp, ai_xp)
    effects_dev_xp = getindex(effect_dev1_skill, dev1_skill)

    effects_ai = ai_use .* effects_ai_xp

    predicted_logit = intercept .+ effects_ai .+ effects_dev_xp

    logit_sigma ~ Exponential()


    # logit_test_coverage ~ MvNormal(predicted_logit, logit_sigma)
    for i in eachindex(logit_test_coverage)
        logit_test_coverage[i] ~ Normal(predicted_logit[i], logit_sigma)
    end
end

@model function test_coverage_analysis_with_controls_2(ai_use, ai_xp, dev1_skill, logit_test_coverage, prior_effect_ai=Normal())
    intercept ~ Normal()

    # There's 3 levels for AI XP, and DEV 1 XP
    # NOTE: Adding the hyper-priors (esp. sigma_interactions)
    # Allows for a lot of shrinking.
    effect_ai ~ prior_effect_ai
    effect_ai_xp_factor ~ Dirichlet(5, 1.0)
    effect_ai_xp = cumsum(effect_ai_xp_factor) .* effect_ai

    effect_skill ~ Normal()
    effect_skill_factor ~ Dirichlet(3, 1.0)
    effect_skill = cumsum(effect_skill_factor) .* effect_skill

    effects_ai_xp = getindex(effect_ai_xp, ai_xp)
    effects_dev_xp = getindex(effect_skill, dev1_skill)

    effects_ai = ai_use .* effects_ai_xp

    predicted_logit = intercept .+ effects_ai .+ effects_dev_xp

    logit_sigma ~ Exponential()

    # logit_test_coverage ~ MvNormal(predicted_logit, logit_sigma)
    for i in eachindex(logit_test_coverage)
        logit_test_coverage[i] ~ Normal(predicted_logit[i], logit_sigma)
    end

    # We return this to help calculate parameter uncertainty.
    # That is, uncertainty without considering the residuals.
    return logistic.(predicted_logit)
end

# PRODUCTIVITY
# We have 11 answers, with some inverted scales
# Question 4 and 10 are inverted!
# We model the "productivity" as a latent variable
# and use a model to estimate how the answers are correlated with the latent

# n_levels is the number of levels for the questions (we assume they have all the same)
# ai_use is a vector of booleans
# productivities is a matrix
# (each row is a sample)
# (each column is a question)
# inverted is an array of 1 and -1 which says which questions are inverted.
@model function productivity_analysis(n_levels, ai_use, answers)
    intercept ~ Normal()

    effect_ai ~ Normal()

    # We use a linear predictor of the "overall productivity"
    # One per person
    productivities_ = intercept .+ effect_ai .* ai_use

    # But how does each productivity map to each answers to a question?
    # We assume the productivity is passed through a number of cutoff points
    # and if prod is higher that cutoff 1, then the answer is level 1 of the scale
    # lowest cutoff is constrained to be 0.
    # Number of questions is the number of columns of productivities
    n_questions = size(answers)[2]

    # Instead of cutoffs, we work with offsets
    # We make a matrix with one row per question,
    # and n_level - 2 columns (first cutoff is at zero, and last doesn't have an offset)
    offsets ~ filldist(truncated(Normal(1, 1), 0, Inf),
                       n_questions, n_levels-2)
    # Cutoff for first level is zero for all questions
    # and then the cutoffs are a sum of the offsets.
    cutoffs = hcat(zeros(n_questions),
                   cumsum(offsets, dims=2))

    # For each person
    for p in 1:length(ai_use)
        # For each question
        for q in 1:n_questions
            # Productivity answer of person p on question q
            # is modeled with the overall prod, and the cutoffs
            # of that specific question.
            answers[p, q] ~ OrderedLogistic(productivities_[p], cutoffs[q,:])
        end
    end
    return cutoffs
end

@model function productivity_analysis_gaussian(ai_use, answers)
    intercept ~ Normal()
    effect_ai ~ Normal()

    productivities_ = intercept .+ effect_ai .* ai_use

    # For each question, one intercept + one factor
    n_samples = size(answers)[1]
    n_questions = size(answers)[2]

    intercept_questions ~ filldist(Normal(3,1), n_questions)
    load_questions ~ filldist(Normal(1,1), n_questions)

    predicted_ = hcat(ones(n_samples), productivities_) * vcat(intercept_questions', load_questions')

    sigma ~ filldist(Exponential(), n_questions)

    for p in 1:length(ai_use)
        for q in 1:n_questions
            answers[p,q] ~ Normal(predicted_[p,q], sigma[q])
        end
    end
end

@model function productivity_analysis_basic(n_levels, answers)
    n_samples = size(answers)[1]
    n_questions = size(answers)[2]

    productivities ~ filldist(Normal(0,1), n_samples)

    # Instead of cutoffs, we work with offsets
    # We make a matrix with one row per question,
    # and n_level - 2 columns (first cutoff is at zero, and last doesn't have an offset)
    offsets ~ filldist(truncated(Normal(1, 1), 0, Inf),
                       n_questions, n_levels-2)
    # Cutoff for first level is zero for all questions
    # and then the cutoffs are a sum of the offsets.
    cutoffs = hcat(zeros(n_questions),
                   cumsum(offsets, dims=2))

    # Discriminations
    # One per question
    # (Determines how accurately, each question reflects the productivity)
    discriminations ~ filldist(Exponential(), n_questions)

    # We use a latent value matrix
    # One row per person, one column per question
    # And the value is the latent score of person p on question q
    # (We then use cutoffs to determine the answers)
    latents_ = productivities * discriminations'

    # For each person
    for p in 1:n_samples
        # For each question
        for q in 1:n_questions
            # Productivity answer of person p on question q
            # is modeled with the overall prod, and the cutoffs
            # of that specific question.
            answers[p,q] ~ OrderedLogistic(latents_[p,q], cutoffs[q,:])
        end
    end

    return cutoffs
end

@model function productivity_analysis_with_controls(n_levels, ai_use, ai_xp, dev1exp, answers)
    intercept ~ Normal()

    effect_ai ~ Normal()

    # z-score of effect of AI for different levels
    # like other models
    sigma_effect_ai_xp ~ Exponential()
    ai_xp_z ~ filldist(Normal(0, 1), 5)
    effect_ai_xp = ai_xp_z .* sigma_effect_ai_xp .+ effect_ai
    # effect_ai_xp ~ filldist(Normal(effect_ai, 0.5), 3)

    # This vector has a mean centered at 0
    # (regularization)
    sigma_dev1_skill ~ Exponential()
    dev1exp_z ~ filldist(Normal(0, 1), 3)
    effect_dev1exp = dev1exp_z .* sigma_dev1_skill
    # effect_dev1exp ~ filldist(Normal(0, 1), 3)

    # Effect of AI for each sample
    effect_ai_xp_ = getindex(effect_ai_xp, ai_xp)
    # Effect of skill for each sample
    effect_dev1exp_ = getindex(effect_dev1exp, dev1exp)

    # We use a linear predictor of the "overall productivity"
    # One productivity per person
    productivities_ = intercept .+ effect_dev1exp_ .+ effect_ai_xp_ .* ai_use

    # But how does each productivity map to each answers to a question?
    # We assume the productivity is passed through a number of cutoff points
    # and if prod is higher that cutoff 1, then the answer is level 1 of the scale
    # lowest cutoff is constrained to be 0.
    # Number of questions is the number of columns of productivities
    n_questions = size(answers)[2]

    # Instead of cutoffs, we work with offsets
    # We make a matrix with one row per question,
    # and n_level - 2 columns (first cutoff is at zero, and last doesn't have an offset)
    offsets ~ filldist(truncated(Normal(1,1), 0, Inf),
                       n_questions, n_levels-2)
    # Cutoff for first level is zero for all questions
    # and then the cutoffs are a sum of the offsets.
    cutoffs = hcat(zeros(n_questions),
                   cumsum(offsets, dims=2))

    # For each person
    for p in 1:length(ai_use)
        # For each question
        for q in 1:n_questions
            # Productivity answer of person p on question q
            # is modeled with the overall prod, and the cutoffs
            # of that specific question.
            answers[p, q] ~ OrderedLogistic(productivities_[p], cutoffs[q,:])
        end
    end
    return cutoffs
end



# This model (using a Dirichlet distribution for the effects)
# Samples FAR better than the other.
@model function productivity_analysis_with_controls_2(n_levels, ai_use, ai_xp, dev1exp, answers, prior_effect_ai=Normal())
    intercept ~ Normal()

    # We use a Dirichlet allocator
    effect_ai ~ prior_effect_ai
    effect_ai_xp_factor ~ Dirichlet(5, 1.0)
    effect_ai_xp = cumsum(effect_ai_xp_factor) .* effect_ai

    effect_skill ~ Normal()
    effect_skill_factor ~ Dirichlet(3, 1.0)
    effect_skill = cumsum(effect_skill_factor) .* effect_skill

    # Effect of AI for each sample
    effect_ai_xp_ = getindex(effect_ai_xp, ai_xp)
    # Effect of skill for each sample
    effect_dev1exp_ = getindex(effect_skill, dev1exp)

    # We use a linear predictor of the "overall productivity"
    # One productivity per person
    productivities_ = intercept .+ effect_dev1exp_ .+ effect_ai_xp_ .* ai_use

    # But how does each productivity map to each answers to a question?
    # We assume the productivity is passed through a number of cutoff points
    # and if prod is higher that cutoff 1, then the answer is level 1 of the scale
    # lowest cutoff is constrained to be 0.
    # Number of questions is the number of columns of productivities
    n_questions = size(answers)[2]

    # Instead of cutoffs, we work with offsets
    # We make a matrix with one row per question,
    # and n_level - 2 columns (first cutoff is at zero, and last doesn't have an offset)
    # offsets ~ filldist(truncated(Normal(1,1), 0, Inf),
    #                    n_questions, n_levels-2)
    offsets ~ filldist(Exponential(),
                       n_questions, n_levels-2)
    # Cutoff for first level is zero for all questions
    # and then the cutoffs are a sum of the offsets.
    cutoffs = hcat(zeros(n_questions),
                   cumsum(offsets, dims=2))

    # For each person
    for p in 1:length(ai_use)
        # For each question
        for q in 1:n_questions
            # Productivity answer of person p on question q
            # is modeled with the overall prod, and the cutoffs
            # of that specific question.
            answers[p, q] ~ OrderedLogistic(productivities_[p], cutoffs[q,:])
        end
    end

    standardized_effect_ai = effect_ai_xp ./ std(productivities_)

    return (;cutoffs,effect_ai_xp, effect_skill, productivities_, standardized_effect_ai)
    # We return a tuple with names = to the variables
    # return (;cutoffs,effect_ai_xp, effect_skill, productivities_)
end

# MODELS FOR ANALYSING EACH QUESTION
# We make models for analysing answers to one single question.
@model function productivity_analysis_question(answers, ai_use)
    intercept ~ Normal()

    effect_ai ~ Normal()

    # We model the "productivity" of the user
    # As a continuous variable
    scores = intercept .+ effect_ai .* ai_use

    # The problem is that we have levels,
    # But we don't know if the levels have the same "width"
    # So we have cutoffs that represent the cutoff after which we pass
    # to the next level
    # We assume a default size of 1 per level
    offsets ~ filldist(truncated(Normal(1,1), 0, Inf),
                       5-2)

    cutoffs = vcat([0], cumsum(offsets))

    for i in 1:length(answers)
        answers[i] ~ OrderedLogistic(scores[i], cutoffs)
    end

    return (cutoffs=cutoffs)
end

# We make a second model, that has access to ai_xp and dev1exp
@model function productivity_analysis_question_with_controls(answers, ai_use, ai_xp, dev1exp)

    intercept ~ Normal()

    # For these factors, we need some
    # constraining priors.
    # If the scale is inverted, the model can flip-flop any of these
    # To negative scale to make it work, so the inference is unstable.
    effect_ai ~ Normal()
    sigma_effect_ai_xp ~ Exponential()

    slopes_ai_xp_z ~ filldist(Normal(0, 1), 5)
    slopes_ai_xp = slopes_ai_xp_z .* sigma_effect_ai_xp .+ effect_ai

    # This vector has a mean centered at 0
    # (regularization)
    sigma_dev1_skill ~ Exponential()
    slopes_dev1exp_z ~ filldist(Normal(0, 1), 3)
    slopes_dev1exp = slopes_dev1exp_z .* sigma_dev1_skill

    # We set cutoffs evenly spaced, because the model can
    # Decide the slope of each factor.
    # cutoffs = [0, 1, 2, 3]

    offsets ~ filldist(truncated(Normal(1, 1), 0, Inf),
        5-2)
    cutoffs = vcat([0], cumsum(offsets))

    slopes_ai_xp_ = getindex(slopes_ai_xp, ai_xp)
    slopes_dev1exp_ = getindex(slopes_dev1exp, dev1exp)

    # Remember that ai_xp should have no effect unless AI is used
    scores_ = intercept .+ (slopes_ai_xp_ .* ai_use .+ slopes_dev1exp_)

    for i in eachindex(answers)
        answers[i] ~ OrderedLogistic(scores_[i], cutoffs)
    end
end

@model function gh_edits_analysis(gh_edits, ai_use, N=length(gh_edits))
    intercept ~ Normal()

    effect_ai ~ Normal()

    score_ = intercept .+ effect_ai .* ai_use

    # p_ = logistic.(score_)
    sigma ~ Exponential()

    for i in 1:N
        # gh_edits[i] ~ NegativeBinomial(4, p_[i])
        gh_edits[i] ~ Normal(score_[i], sigma)
    end
end


@model function gh_edits_analysis_with_controls(gh_edits, ai_use, ai_xp, dev1exp, N=length(gh_edits))
    # intercept ~ Normal()
    intercept = 0

    effect_ai ~ Normal(0.0, 0.25)

    # z-score of effect of AI for different levels
    # like other models
    sigma_effect_ai_xp ~ Exponential(0.5)
    ai_xp_z ~ filldist(Normal(0, 1), 5)
    effect_ai_xp = ai_xp_z .* sigma_effect_ai_xp .+ effect_ai
    # effect_ai_xp ~ filldist(Normal(effect_ai, 0.5), 3)

    # This vector has a mean centered at 0
    # (regularization)
    sigma_dev1_skill ~ Exponential(0.5)
    dev1exp_z ~ filldist(Normal(0, 1), 3)
    effect_dev1exp = dev1exp_z .* sigma_dev1_skill
    # effect_dev1exp ~ filldist(Normal(0, 1), 3)

    # Effect of AI for each sample
    effect_ai_xp_ = getindex(effect_ai_xp, ai_xp)
    # Effect of skill for each sample
    effect_dev1exp_ = getindex(effect_dev1exp, dev1exp)

    # We use a linear predictor of the "overall productivity"
    # One productivity per person
    productivities_ = intercept .+ effect_dev1exp_ .+ effect_ai_xp_ .* ai_use

    # Log odds of a successful line, basically
    # prob_success_ = logistic.(productivities_)

    sigma ~ Exponential()
    for i in 1:N
        # gh_edits[i] ~ Poisson(exp(productivities_[i]))
        # gh_edits[i] ~ NegativeBinomial(2, prob_success_[i])
        gh_edits[i] ~ Normal(productivities_[i], sigma)
    end
end

end
