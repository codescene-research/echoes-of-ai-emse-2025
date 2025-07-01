using Test
using ai_codev_study.Simulation

using DataFrames: DataFrame, nrow, filter

include("./testloo.jl")

@testset "test_gen_data" begin
    @testset "test_gen_dev_skill" begin
        dev = -4:1:4
        for skill in dev
            @test Simulation.dev_exp_function(skill, 0) ∈ collect(1:6)
            @test Simulation.java_exp_function(skill, 0) ∈ collect(1:3)
        end

        @test Simulation.dev_skill_function(6, 3) == 3
    end

    @testset "test_gen_productivity" begin
        # Productivity is described in 11 questions
        # Encoded as likert scales
        productivities = []
        for q in 1:11
            dev2skill = 0.0 # Average dev
            code1q = 0.0 # Average quality
            # We set noise to zero
            for dev2skill in -2:0.5:2
                for code1q in -2:0.5:2
                    prod = Simulation.productivity_function(q, dev2skill, code1q, 0)
                    append!(productivities, [prod])
                    @test prod in [-2, -1, 0, 1, 2]
                end
            end
        end

        # We cover the whole set of options
        @test Set(productivities) == Set([-2, -1, 0, 1, 2])
    end

    @testset "test_generators" begin
        dev1 = Simulation.gen_dev1()
        @test dev1 isa Real

        ai_xp = Simulation.gen_ai_xp(dev1)
        @test ai_xp isa Int

        ai_pref = Simulation.gen_ai_pref(dev1, ai_xp)
        @test ai_pref isa Bool

        ai_use = Simulation.gen_ai_use(ai_pref)
        @test ai_use isa Bool

        dev1_skill = Simulation.gen_dev_skill(dev1)
        @test dev1_skill isa Int

        code1 = Simulation.gen_code1q(dev1_skill, ai_use, ai_xp)
        @test code1 isa Real

        dev2 = Simulation.gen_dev2()
        @test dev2 isa Real

        time = Simulation.gen_time(dev2, code1)
        @test time isa Real
        @test time > 0

        code2 = Simulation.gen_code2q(code1, dev2)
        @test code2 isa Real

        code_health = Simulation.gen_codehealth(code2)
        @test code_health isa Int

        test_coverage = Simulation.gen_test_coverage(code2)
        @test test_coverage isa Real
        @test test_coverage > 0
        @test test_coverage < 1
    end

    @testset "test_gen_data_set" begin
        set = Simulation.gen_data_set(5)
        # Check the data set contains the productivity columns
        @test "productivity_median" ∈ names(set)

        productivitiy_columns = ["productivity_q$i" for i in 1:10]
        for column in productivitiy_columns
            @test column ∈ names(set)
        end
        @test nrow(set) == 5
    end

    @testset "test_gen_perfect_experiment" begin
        set = Simulation.gen_perfect_dataset()
        # Half the data should have AI_use as true
        @test sum(set[!,:ai_use]) == nrow(set) / 2

        with_ai = filter(r -> r[:ai_use], set)
        without_ai = filter(r -> !r[:ai_use], set)

        @test nrow(with_ai) == nrow(without_ai)

        # Causes of ai_use should have the same value regardless
        causes = [:ai_xp, :ai_pref, :dev1, :dev2, :dev1_skill]
        for col in causes
            @test with_ai[!,col] == without_ai[!,col]
        end
        # But the outcomes (consequences) should be different
        outcomes = [:code1q, :code2q, :time, :test_coverage, :productivity_q1]
        for col in outcomes
            @test with_ai[!,col] != without_ai[!,col]
        end
    end

    @testset "test_dropout" begin
        # Time is set in minutes
        set = DataFrame(:time => [1*60, 2*60, 6*60, 10*60],
            :code_health => [0.4, 0.2, 0.7, 0.75],
            :data_missingness => [false, true, false, true])

        @test "data_missingness" in names(set)

        dropped = Simulation.dropout(set, :time)

        for r in eachrow(dropped)
            @test if r[:data_missingness] ismissing(r[:time]) else !ismissing(r[:time]) end
        end
    end
end
