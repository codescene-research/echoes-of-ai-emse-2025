"""
A script that makes tests to check our implementation of LOO is valid.
"""

using Test

using ai_codev_study.Commons

using Random, Turing, Distributions, ArviZ


@testset "loo-tests" begin

    @testset failfast = true "loo-simulation" begin
        xs = collect(1:20)
        a = 0.2
        b = 0.3
        sigma = 1
        ys :: Vector{Float64} = b .+ a .* xs + rand(Normal(), 20).*sigma

        # Only mean and variance
        @model function model_avg(xs, ys)
            mu ~ Normal(0.0, 5.0)
            sig ~ Exponential()
            # We MUST use a loop, otherwise it doesn't work...
            for i in eachindex(ys)
                ys[i] ~ Normal(mu, sig)
            end
        end

        # Linear regression
        @model function linreg(xs, ys)
            a ~ Normal()
            b ~ Normal()
            sig ~ Exponential()

            line = b .+ a .* xs

            for i in eachindex(ys)
                ys[i] ~ Normal(line[i], sig)
            end
        end

        m0_train = model_avg(xs, ys)
        m0_predict = model_avg(xs, similar(ys, Missing))

        m1_train = linreg(xs, ys)
        m1_predict = linreg(xs, similar(ys, Missing))

        coords = (x=xs,)
        dims = (ys=(:x,),)
        data = (ys=ys,)
        m0_data = Commons.inference_data(m0_train, m0_predict;
                                         coords, dims, data)
        m1_data = Commons.inference_data(m1_train, m1_predict;
                                         coords, dims, data)

        comparison = compare((average=m0_data, linear=m1_data))

        ps = comparison.rank |> pairs
        @test ps[:linear] == 1
        @test ps[:average] == 2

        ps = comparison.elpd_diff |> pairs
        @test ps[:linear] == 0.0
        @test ps[:average] >= 0.0

    end


    @testset failfast = true "loo-simplified" begin
        # Dataset
        J = 8
        y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
        σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
        schools = [
            "Choate",
            "Deerfield",
            "Phillips Andover",
            "Phillips Exeter",
            "Hotchkiss",
            "Lawrenceville",
            "St. Paul's",
            "Mt. Hermon",
        ]
        ndraws = 1_000
        ndraws_warmup = 1_000
        nchains = 4

        # Turing Model
        @model function model_turing(y, σ, J=length(y))
            μ ~ Normal(0, 5)
            τ ~ truncated(Cauchy(0, 5), 0, Inf)
            θ ~ filldist(Normal(μ, τ), J)
            for i in 1:J
                y[i] ~ Normal(θ[i], σ[i])
            end
        end

        model_train = model_turing(y, σ)
        # similar builds an array with the same shape, here
        # initialized to missing values...
        model_predict = model_turing(similar(y, Missing), σ)

        # produce inference data
        rng = Random.MersenneTwister(16653)
        idata_turing = Commons.inference_data(model_train,
                                              model_predict;
                                              coords=(;school=schools),
                                              dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
                                              data=(;y),
                                              rng=rng)

        loo_result = loo(idata_turing)

        # These should be the same as the reference impl below.
        estimates = elpd_estimates(loo_result)
        println(estimates)
        @test round(estimates[:elpd]) == -31
        @test round(estimates[:elpd_mcse], digits=1) == 1.4
        @test round(estimates[:p], digits=1) == 0.9
        @test round(estimates[:p_mcse], digits=2) == 0.32
    end

    """
    We use the example from: https://julia.arviz.org/ArviZ/stable/quickstart/
    """
    @testset "loo-example" begin
        # Dataset
        J = 8
        y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
        σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
        schools = [
            "Choate",
            "Deerfield",
            "Phillips Andover",
            "Phillips Exeter",
            "Hotchkiss",
            "Lawrenceville",
            "St. Paul's",
            "Mt. Hermon",
        ]
        ndraws = 1_000
        ndraws_warmup = 1_000
        nchains = 4

        # Turing Model
        @model function model_turing(y, σ, J=length(y))
            μ ~ Normal(0, 5)
            τ ~ truncated(Cauchy(0, 5), 0, Inf)
            θ ~ filldist(Normal(μ, τ), J)
            for i in 1:J
                y[i] ~ Normal(θ[i], σ[i])
            end
        end

        # RNG, to make sure we get same results.
        rng2 = Random.MersenneTwister(16653);

        # Sampling the posterior
        param_mod_turing = model_turing(y, σ)
        sampler = NUTS(ndraws_warmup, 0.8)

        turing_chns = Turing.sample(
            rng2, model_turing(y, σ), sampler, MCMCThreads(), ndraws, nchains
        )

        # Building Inference data...
        # coords allows to create an index for an array (e.g. θ)
        # dims says that y, σ, θ are indexed by school (one value for each school)
        # This data works for some plots but we'll need more for LOO
        idata_turing_post = from_mcmcchains(
            turing_chns;
            coords=(; school=schools),
            dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
            library="Turing",
        )

        # We need the prior too
        prior = Turing.sample(rng2, param_mod_turing, Prior(), ndraws);

        # We need the pointwise log likelihoods to estimate LOO
        # Instantiate the predictive model
        param_mod_predict = model_turing(similar(y, Missing), σ)
        # and then sample!
        prior_predictive = Turing.predict(rng2, param_mod_predict, prior)
        posterior_predictive = Turing.predict(rng2, param_mod_predict, turing_chns)

        # Calculating the log-likelihoods...
        log_likelihood = let
            # second argument is to avoid a lot of error messages...
            log_likelihood = Turing.pointwise_loglikelihoods(
                param_mod_turing, MCMCChains.get_sections(turing_chns, :parameters)
            )
            # Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
            # Convert all symbols to strings...
            ynames = string.(keys(posterior_predictive))
            # Index the log_likelihood dict with ynames (re-orders it)
            log_likelihood_y = getindex.(Ref(log_likelihood), ynames)
            # Create a NamedTuple, with y as the log_likelihoods.
            (; y=cat(log_likelihood_y...; dims=3))
        end

        # We make a more complete "InferenceData"...
        idata_turing = from_mcmcchains(
            turing_chns; # Chains
            posterior_predictive, # Predictions with posterior
            log_likelihood,
            prior, # Chains with priors
            prior_predictive, # Predictions with prior
            observed_data=(; y), # The observed values...
            coords=(; school=schools),
            dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
            library=Turing,
        )

        loo_result = loo(idata_turing)

        estimates = elpd_estimates(loo_result)
        println(estimates)
        @test round(estimates[:elpd]) == -31
        @test round(estimates[:elpd_mcse], digits=1) == 1.4
        @test round(estimates[:p], digits=1) == 0.9
        @test round(estimates[:p_mcse], digits=2) == 0.32
    end


end
