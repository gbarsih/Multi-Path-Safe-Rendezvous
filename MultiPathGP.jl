#=
 █████╗  ██████╗██████╗ ██╗
██╔══██╗██╔════╝██╔══██╗██║
███████║██║     ██████╔╝██║
██╔══██║██║     ██╔══██╗██║
██║  ██║╚██████╗██║  ██║███████╗
╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝
File:       rendezvous_review.jl
Author:     Gabriel Barsi Haberfeld, 2020. gbh2@illinois.edu
Function:   This program simulates all results in the paper "Geometry-Informed
            Minimum Risk Rendezvous Algorithm for Heterogeneous Agents in Urban
            Environments"
Instructions:   Run this file in juno with Julia 1.2.0 or later.
Requirements:   JuMP, Ipopt, Plots, LinearAlgebra, BenchmarkTools.
=#

using Plots, LinearAlgebra, Random, Statistics, ColorSchemes, LazySets
using JuMP, Ipopt, Measures, Printf, LaTeXStrings
using BenchmarkTools

include("Map.jl")
include("GPTools.jl")
include("OptimizationTools.jl")
include("PlotTools.jl")
include("DriverFunctions.jl")

function rankMultiPath(
    TimeSamples,
    UASPos,
    n,
    p = [1, 2],
    gp = nothing,
    ti = 0.0,
    DriverPos = 0.0,
    method = "BestFirst",
    Er = Inf,
    sol = nothing,
)
    np = size(TimeSamples, 2)
    if np != length(p)
        error("Invalid #map v #smpl")
    end
    lt = size(TimeSamples, 1)
    x = hcat(UASPos[1] * ones(lt), UASPos[2] * ones(lt)) #x contains a copy-vector of x-y coordinates of the UAS
    E = zeros(lt, np)
    Threads.@threads for i = 1:np #for each path
        PosSamples = DriverPosition(ti, TimeSamples, gp) .+ DriverPos
        v = (pathSample2Array(path.(PosSamples[:, i], i)) .- x) ./ TimeSamples
        E[:, i] =
            m[1] .* diag(v * v') .* TimeSamples[:, i] .+
            alpha * TimeSamples[:, i]
    end
    # E now contains energies for each sample group. Next choose a method
    replace!(E, NaN => Inf)
    if n == 1
        return [argmin(E[:, 1]) argmin(E[:, 2])]
    end
    if method == "BestFirst"
        elites = zeros(Int16, n, np)
        Threads.@threads for i = 1:np
            elites[:, i] = partialsortperm(E[:, i], 1:min(n, lt))
        end
        #to select the path using this policy
        #get the best from each, and output the best
        #TODO: add previous solution functionality to sway towards the Depot.
        ptgt = argmin(E[elites][1,:])
        return elites, ptgt
    else
        error("Invalid Method Argument")
    end
    Es = E[:, 1] .+ E[:, 2]
    SumElites = partialsortperm(Es, 1:min(n, lt))
end

function pathSample2Array(pathSample)
    return hcat([y[1] for y in pathSample], [y[2] for y in pathSample])
end

function SampleTime(μ, Σ, N = 1, t0 = 0.0, tf = 75.0)
    if size(μ, 1) != size(Σ, 1) || size(μ, 2) != 1 || size(Σ, 2) != 1
        error("Wrong distribution dimensions!")
    end
    ndist = size(μ, 1)
    tf = tb
    TimeSamples = μ' .+ randn(N, ndist) .* Σ'
    TimeSamples[TimeSamples.>=tf] .= tf
    TimeSamples[TimeSamples.<=t0] .= t0
    return TimeSamples
end

function TestSampling(μ, Σ, N)
    TimeSamples = SampleTime(μ, Σ, N)
    plotTimeSamples(TimeSamples, 1)
end

function CEM(μ, Σ, np, elites, OptTimeSample, TimeSamples)
    elite = elites[1, :]
    Threads.@threads for j = 1:length(np)
        μn = sum(TimeSamples[elites[:, j], j]) / Ns
        μ[j] = γ * μn + (1 - γ) * μ[j]
        Σ[j] =
            β * (sum((TimeSamples[elites[:, j], j] .- μ[j]) .^ 2) ./ Ns + 0.5) +
            (1 - β) * Σ[j]
        Σ[j] = min(Σ[j], 100)
        OptTimeSample[j] = TimeSamples[elite[j], j]
    end
    return μ, Σ, OptTimeSample
end


np = [1, 2]
μ = [50.0, 50.0] #initial means
Σ = [40.0, 40.0] #initial variances
N = 100 #number of time samples
Ns = 5
UASPos = [900, 450]
LPos = [400, 550]
ts = 0
PNRStat = false
p = [1, 2]
ptgt = 2
dt = 1
PrevPNR = [0.0, 0.0]
β = 1.0
γ = 1.0
UASPosVec = zeros(N, 2)

function iterateCEM(μ, Σ, np)
    OptTimeSample = zeros(length(p))
    for i = 1:1000
        TimeSamples = SampleTime(μ, Σ, N)
        elites = rankMultiPath(TimeSamples, UASPos, Ns, np)
        CEM(μ, Σ, np, elites, OptTimeSample, TimeSamples)
        @show μ, Σ
    end
    return μ, Σ, OptTimeSample
end

#iterateCEM(μ, Σ, np)
#=
OptTimeSample = zeros(length(p))
TimeSamples = SampleTime(μ, Σ, N)
elites = rankMultiPath(TimeSamples, UASPos, Ns, np)
pp = plot()
pp = plotpath!(np)
pp = plotTimeSamples!(TimeSamples, np)
=#
function animateGpCem(tt = 20, ti = 5, dt = 1)
    #first build an initial dataset and gp
    t = range(0.0, stop = ti, step = dt)
    li = length(t)
    D = zeros(li, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(li)
    gp = LearnDeviationFunction(D, true, "full")
    #now we move the driver, collect new data, update gp and sample on the gp
    t = range(0.0, stop = tt, step = dt)
    l = length(t)
    D = [D; zeros(l, 2)]
    DriverPos = 0.0
    DriverPosVec = zeros(l)
    DriverPosEstimateVec = zeros(l)
    DriverPosErrorVec = zeros(l)
    PosSamples = zeros(N, length(p))
    OptTimeSample = zeros(length(p))
    anim = @animate for i = 1:l
        #sample driver velocity
        DriverVelSample = DriverVelocity(t[i]) + NoiseStd * randn(1)[1]
        DriverDeviationSample = DriverVelSample - VelocityPrior(t[i])
        #add to dataset
        D[li+i, 1] = VelocityPrior(t[i])
        D[li+i, 2] = DriverDeviationSample
        #re-train gp
        gp = LearnDeviationFunction(D[1:(li+i), :], true, "DTC")
        #sample rendezvous candidates
        TimeSamples = SampleTime(μ, Σ, N)
        elites =
            rankMultiPath(TimeSamples, UASPos, Ns, np, gp, t[i], DriverPos)
        CEM(μ, Σ, np, elites, OptTimeSample, TimeSamples)
        PosSamples = DriverPosition(t[i], TimeSamples, gp) .+ DriverPos
        l = @layout [a; b]
        pp = plot()
        pp = plotpath!(np)
        pp = plotTimeSamples!(TimeSamples, np, t[i], DriverPos, gp)
        pp = drawMultiConvexHull(PosSamples, p)
        po = plot(gp, xlims = (6, 10), ylim = (-2, 2))
        pp = plot(pp, po, layout = l)
        DriverPosVec[i] = DriverPos
        DriverPosEstimateVec[i] = DriverPosition(0.0, t[i])
        DriverPosErrorVec[i] = DriverPosVec[i] - DriverPosEstimateVec[i]
        DriverPos = DriverPos + DriverVelocity(t[i]) * dt
        @show μ, Σ, i
    end
    gif(anim, "GpCem.gif", fps = 15)
end

#TODO close the loop using best first method.

#=
Closed loop mission using BestFirst. Order of things:
-Collect sensor data
-Update driver model with gp
-Update distribution with CEM
-Compute trajectory using MPC
-Send control iputs to UAS
-Update driver position
-Update UAS position
=#

function mission(Er = 18000, method = "BestFirst", tt = 80, ti = 5, dt = 1)
    clearconsole()
    UASPos = [800, 450]
    LPos = [1000, 650]
    #LPos = [800, 450]
    ts = 0
    PNRStat = false
    N = 100
    p = [1, 2]
    ptgt = 2
    dt = 1
    PrevPNR = [0.0, 0.0]
    μ = 60 * ones(2)
    Σ = 20 * ones(2)
    Ns = 5
    β = 1.0
    γ = 1.0
    UASPosVec = zeros(N, 2)
    #TimeSamples = ts:120
    t = range(0.0, stop = ti, step = dt)
    li = length(t)
    D = zeros(li, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(li)
    gp = LearnDeviationFunction(D, true, "full")
    #now we move the driver, collect new data, update gp and sample on the gp
    tv = range(0.0, stop = tt, step = dt)
    l = length(tv)
    D = [D; zeros(l, 2)]
    DriverPos = 0.0
    DriverPosVec = zeros(l)
    DriverPosEstimateVec = zeros(l)
    DriverPosErrorVec = zeros(l)
    PosSamples = zeros(N, length(p))
    OptTimeSample = zeros(length(p))
    sol = nothing
    method = "BestFirst"
    anim = @animate for i = 1:l
        #sample driver velocity
        DriverVelSample = DriverVelocity(tv[i]) + NoiseStd * randn(1)[1]
        DriverDeviationSample = DriverVelSample - VelocityPrior(tv[i])
        #add to dataset
        D[li+i, 1] = VelocityPrior(tv[i])
        D[li+i, 2] = DriverDeviationSample
        #re-train gp
        gp = LearnDeviationFunction(D[1:(li+i), :], true, "DTC")
        #sample rendezvous candidates
        UASPosVec[i, :] = UASPos
        TimeSamples = SampleTime(μ, Σ, N)
        elites, ptgt =
            rankMultiPath(TimeSamples, UASPos, Ns, np, gp, tv[i], DriverPos, method, Er, sol)
        CEM(μ, Σ, np, elites, OptTimeSample, TimeSamples)
        PosSamples = DriverPosition(tv[i], TimeSamples, gp) .+ DriverPos
        #MPC goes here
        v, t = RendezvousPlanner(
            UASPos,
            LPos,
            OptTimeSample,
            Er,
            tv[i],
            PrevPNR,
            false,
            ptgt,
            gp,
            DriverPos,
        )
        sol = (v, t)
        pp = plot()
        pp = plot!(UASPosVec[1:i, 1], UASPosVec[1:i, 2], legend = false)
        pp = drawMultiConvexHull(PosSamples, p)
        pp = plotPlan!(
            UASPos,
            LPos,
            v,
            t,
            TimeSamples,
            length(p),
            tv[i],
            DriverPos,
            gp,
        )
        pp = plotTimeSamples!(TimeSamples, p, tv[i], DriverPos, gp)
        pp = scatter!(
            path(DriverPos, ptgt),
            markersize = 6.0,
            label = "",
            markercolor = :green,
        )
        PrevPNR = v[:, 1] .* t[1] .+ UASPos
        Ed =
            m[1] * v[1, 1]^2 * t[1] +
            m[1] * v[1, 2]^2 * t[2] +
            m[2] * v[1, 3]^2 * t[3] +
            m[1] * v[2, 1]^2 * t[1] +
            m[1] * v[2, 2]^2 * t[2] +
            m[2] * v[2, 3]^2 * t[3] +
            m[1] * alpha * t[1] +
            m[1] * alpha * t[2] +
            m[2] * alpha * t[3]
        x, y, Er = uav_dynamics(
            UASPos[1],
            UASPos[2],
            v[1, 1],
            v[2, 1],
            dt,
            Er,
            vmax,
            PNRStat ? m[2] : m[1],
        )
        UASPos = [x, y]
        if ptgt == 2
            pv = pv2
        else
            pv = pv1
        end
        # if (t[1] <= 1.0 ||
        #    EuclideanDistance(
        #     path(DriverPosition(tv[i], OptTimeSample[ptgt], gp) + DriverPos),
        #     pv[2, :],
        # ) <= 10) && tv[i] > 10
        #     break
        # end
        if (t[1] <= 1.0 && tv[i] > 10)
            break
        end
        #@show t, i, Σ, μ, Er, Ed
        DriverPosVec[i] = DriverPos
        DriverPosEstimateVec[i] = DriverPosition(0.0, tv[i])
        DriverPosErrorVec[i] = DriverPosVec[i] - DriverPosEstimateVec[i]
        DriverPos = DriverPos + DriverVelocity(tv[i]) * dt
        @show i, t[1], OptTimeSample, ptgt
    end
    gif(anim, "RDV_Anim_MP.gif", fps = 15)
end
