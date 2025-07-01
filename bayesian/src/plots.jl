module ChainPlots

using Plots
using StatsPlots

"""
Plots the posterior of a parameter, and overlays a reference value
"""
function plot(chain, parameter :: Symbol, reference :: Real; label="estimate")
    samples = chain[parameter] |> vec
    p = density(samples, label=label)
    vline!(p, [reference], 
           lw=3, style=:dash, color=:black,
           label="reference")

    return p
end

"""
Plots two posteriors from different chains for the same parameter
"""
function plot(chain_1, chain_2, parameter :: Symbol; label1="chain 1", label2="chain 2")
    ss1 = chain_1[parameter] |> vec
    ss2 = chain_2[parameter] |> vec
    p = density(ss1, label=label1)
    density!(p, ss2, label=label2)
    return p
end

end
