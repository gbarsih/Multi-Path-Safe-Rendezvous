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

include("GPTools.jl")
include("OptimizationTools.jl")
include("PlotTools.jl")
include("DriverFunctions.jl")

default(palette = :tol_bright)
default(dpi = 200)
default(size = (600, 600))
default(lw = 3)
default(margin = 10mm)
FontSize = 18
default(xtickfontsize = FontSize)
default(ytickfontsize = FontSize)
default(xguidefontsize = FontSize)
default(yguidefontsize = FontSize)
default(legendfontsize = FontSize)
default(titlefontsize = FontSize)

m = [2, 1]
alpha = 5
vmax = 25
vo = 50
ho = 20
pv1 = [0 800; 600 800; 600 900; 1300 900]
pv2 = [0 800; 1000 800; 1000 1200]
Mp = 1000
MaxDriverSpeed = 8
tb = Mp / MaxDriverSpeed

function path(θ, p = 1, ArrOut = false)
    if p == 1
        nv = size(pv1, 1)
        v = pv1
    elseif p == 2
        nv = size(pv2, 1)
        v = pv2
    else
        error("invalid path choice at path function!")
    end
    ns = nv - 1
    d = zeros(ns) #distances
    for i = 1:ns
        d[i] = EuclideanDistance(v[i, :], v[i+1, :])
    end
    td = sum(d)
    Dθ = d ./ td .* Mp
    s = 1
    thisSegBound = 0
    for i = 1:ns
        thisSegBound = sum(Dθ[1:i]) #how far this segment goes
        if θ <= thisSegBound
            s = i
            break
        end
    end
    if s != 1 #not first segment, adjust bounds and theta
        prevSegBound = sum(Dθ[1:(s-1)])
        θ -= prevSegBound
        thisSegBound -= prevSegBound
    end
    x = v[s, 1] + (v[s+1, 1] - v[s, 1]) * θ / thisSegBound
    y = v[s, 2] + (v[s+1, 2] - v[s, 2]) * θ / thisSegBound
    if θ > Mp
        x = v[end, 1]
        y = v[end, 2]
    end
    if ArrOut
        return [x, y]
    end
    return x, y
end

function EuclideanDistance(x1, x2)
    sqrt((x1[1] - x2[1])^2 + (x1[2] - x2[2])^2)
end

function uav_dynamics(x, y, vx, vy, dt, rem_power = Inf, vmax = Inf, m = 1.0)
    vx > vmax ? vmax : vx
    vy > vmax ? vmax : vy
    vx < -vmax ? -vmax : vx
    vy < -vmax ? -vmax : vy
    x = x + vx * dt
    y = y + vy * dt
    rem_power = rem_power - vx^2 * m * dt - vy^2 * m * dt - m * alpha * dt
    return x, y, rem_power
end

function rankSampleMean(TimeSamples, UASPos, n = false, p = 1, plotOpt = false)
    np = 1
    lt = size(TimeSamples, 1)
    if np == 1
        x = hcat(UASPos[1] * ones(lt), UASPos[2] * ones(lt))
        v =
            (pathSample2Array(path.(DriverPosFunction(TimeSamples), p)) .- x) ./
            TimeSamples
        E = diag(v * v') .* TimeSamples .+ alpha * TimeSamples
        if plotOpt
            plot(E, ylims = (0, 18000))
        end
        replace!(E, NaN => Inf)
        if n == false
            return E
        elseif n == 1
            return argmin(E)
        else
            return partialsortperm(E, 1:min(n, lt))
        end
    elseif np == 2
        #Here we rank two paths
        #Loss function is energy between paths plus energy to min
        x = hcat(UASPos[1] * ones(lt), UASPos[2] * ones(lt)) #x contains a copy-vector of x-y coordinates of the UAS
        E = zeros(lt, np)
        for i = 1:np #for each path
            v =
                (
                    pathSample2Array(
                        path.(DriverPosFunction(TimeSamples), i),
                    ) .- x
                ) ./ TimeSample
            E[:, i] = diag(v * v') .* TimeSamples .+ alpha * TimeSamples
        end
        #E now contains energy to each path
        #Calculate minmax
        Em = min.(E[:, 1], E[:, 2])
        Ex = max.(E[:, 1], E[:, 2])
    else
        error("Invalid option!")
    end

end

function rankMultiPath(
    TimeSamples,
    UASPos,
    n,
    p = [1, 2],
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
        v =
            (
                pathSample2Array(
                    path.(DriverPosFunction(TimeSamples[:, i], p), i),
                ) .- x
            ) ./ TimeSamples
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
        return elites
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

function simDeterministicUniform(Er = 8000)
    clearconsole()
    UASPos = [800, 450]
    LPos = [400, 550]
    ts = 0
    PNRStat = false
    N = 100
    p = 2
    dt = 1
    PrevPNR = [0.0, 0.0]
    μ = 80
    Σ = 40
    Ns = 5
    β = 0.1
    γ = 1.0
    UASPosVec = zeros(N, 2)
    anim = @animate for i = 1:N
        UASPosVec[i, :] = UASPos
        #TimeSamples = ts:120
        TimeSamples = SampleTime(μ, Σ, N, ts)
        TimeSamples = TimeSamples[:]
        PosSamples = DriverPosFunction(TimeSamples)
        elites = rankSampleMean(TimeSamples, UASPos, Ns, p)
        elite = rankSampleMean(TimeSamples, UASPos, 1, p)
        μn = sum(TimeSamples[elites]) / Ns
        μ = γ * μn + (1 - γ) * μ
        Σ =
            β * (sum((TimeSamples[elites] .- μ) .^ 2) ./ Ns + 1) + (1 - β) * Σ
        Σ = min(Σ, 100)
        OptTimeSample = TimeSamples[elite]
        v, t = DeterministicUniformMPC(
            UASPos,
            LPos,
            OptTimeSample,
            Er,
            ts,
            PrevPNR,
            PNRStat,
            p,
        )
        pp = plot()
        pp = plot!(UASPosVec[1:i, 1], UASPosVec[1:i, 2], legend = false)
        pp = drawConvexHull(TimeSamples, [1, 2], :green)
        pp = drawConvexHull(TimeSamples[elites], [1, 2], :red)
        pp = plotPlan!(UASPos, LPos, v, t, TimeSamples)
        pp = scatter!(
            path(DriverPosFunction(ts), p),
            markersize = 6.0,
            label = "",
            markercolor = :green,
        )
        pp = plot!(tickfont = Plots.font("serif", pointsize = round(12.0)))
        ts = ts + dt
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
        @show t, i, Σ, μ, Er, Ed
        if p == 2
            pv = pv2
        else
            pv = pv1
        end
        if t[1] <= 1.0 ||
           EuclideanDistance(path(DriverPosFunction(ts), p, true), pv[2, :]) <= 10
            break
        end
    end
    # proceed with rendezvous
    gif(anim, "RDV_Anim.gif", fps = 15)
end

function simIndependentDistributions(Er = 8000)
    clearconsole()
    UASPos = [800, 450]
    LPos = [400, 550]
    ts = 0
    PNRStat = false
    N = 100
    p = [1, 2]
    ptgt = 2
    dt = 1
    PrevPNR = [0.0, 0.0]
    μ = 80 * ones(2)
    Σ = 40 * ones(2)
    Ns = 5
    β = 1.0
    γ = 1.0
    UASPosVec = zeros(N, 2)
    OptTimeSample = zeros(length(p))
    anim = @animate for i = 1:N
        UASPosVec[i, :] = UASPos
        #TimeSamples = ts:120
        TimeSamples = SampleTime(μ, Σ, N, ts)
        PosSamples = DriverPosFunction(TimeSamples)
        elites = rankMultiPath(TimeSamples, UASPos, Ns, p)
        elite = rankMultiPath(TimeSamples, UASPos, 1, p)
        for j = 1:length(p)
            μn = sum(TimeSamples[elites[:, j], j]) / Ns
            μ[j] = γ * μn + (1 - γ) * μ[j]
            Σ[j] =
                β * (
                    sum((TimeSamples[elites[:, j], j] .- μ[j]) .^ 2) ./ Ns + 1
                ) + (1 - β) * Σ[j]
            Σ[j] = min(Σ[j], 100)
            OptTimeSample[j] = TimeSamples[elite[j], j]
        end
        ptgt = argmin(OptTimeSample)
        v, t = DeterministicUniformMPC(
            UASPos,
            LPos,
            OptTimeSample[2],
            Er,
            ts,
            PrevPNR,
            PNRStat,
            2,
        )
        #TODO: Fix plotting of convex hull
        pp = plot()
        pp = plot!(UASPosVec[1:i, 1], UASPosVec[1:i, 2], legend = false)
        pp = drawConvexHull(TimeSamples, [1, 2], :green)
        pp = drawConvexHull(TimeSamples[elites], [1, 2], :red)
        pp = plotPlan!(UASPos, LPos, v, t, TimeSamples)
        pp = scatter!(
            path(DriverPosFunction(ts), ptgt),
            markersize = 6.0,
            label = "",
            markercolor = :green,
        )
        pp = plot!(tickfont = Plots.font("serif", pointsize = round(12.0)))
        ts = ts + dt
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
        @show t, i, Σ, μ, Er, Ed
        if ptgt == 2
            pv = pv2
        else
            pv = pv1
        end
        if t[1] <= 1.0 ||
           EuclideanDistance(
            path(DriverPosFunction(ts), ptgt, true),
            pv[2, :],
        ) <= 10
            break
        end
    end
    # proceed with rendezvous
    gif(anim, "RDV_Anim.gif", fps = 15)
end
