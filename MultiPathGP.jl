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
using JuMP, Ipopt, Measures, Printf, LaTeXStrings, Distributions
using BenchmarkTools, StatsPlots

mass = [3, 1]
m = mass
alpha = 20
vmax = 20
vo = 50
ho = 20
pv1 = [0 800; 600 800; 600 900; 1300 900]
pv2 = [0 800; 1000 800; 1000 1200]
Mp = 1000
MaxDriverSpeed = 8
tb = Mp / MaxDriverSpeed

include("Map.jl")
include("GPTools.jl")
include("OptimizationTools.jl")
include("PlotTools.jl")
include("DriverFunctions.jl")

function rowNormSquared(mat)
    A = zeros(Float64, size(mat, 2))
    for j = 1:size(mat, 2)
        for i = 1:size(mat, 1)
            A[j] += mat[i, j]^2
        end
        #A[j] = sqrt(A[j])
    end
    A
end

function rowNorm(mat)
    A = zeros(Float64, size(mat, 2))
    for j = 1:size(mat, 2)
        for i = 1:size(mat, 1)
            A[j] += mat[i, j]^2
        end
        A[j] = sqrt(A[j])
    end
    A
end

function rankMultiPath(
    TimeSamples,
    UASPos,
    Ns,
    p = [1, 2],
    gp = nothing,
    ti = 0.0,
    DriverPos = 0.0,
    rmethod = "BestFirst",
    Er = Inf,
    sol = nothing,
    riska = false,
)
    np = size(TimeSamples, 2)
    if np != length(p)
        error("Invalid #map v #smpl")
    end
    lt = size(TimeSamples, 1)
    x = hcat(UASPos[1] * ones(lt), UASPos[2] * ones(lt))
    #x contains a copy-vector of x-y coordinates of the UAS
    E = zeros(lt, np)
    PosSamples = DriverPosition(ti, TimeSamples, gp) .+ DriverPos
    if riska
        USamples = DriverUncertainty(ti, TimeSamples, gp)
        pU = PosSamples .+ USamples #Position + σ
        nU = PosSamples .- USamples #Position - σ
    end
    Threads.@threads for i = 1:np #for each path
        EuclideanPositions = pathSample2Array(path.(PosSamples[:, i], i))
        EuclideanDistances = EuclideanPositions .- x #mean distance
        if riska
            pUEuclidean = pathSample2Array(path.(pU[:, i], i)) .- x
            nUEuclidean = pathSample2Array(path.(nU[:, i], i)) .- x
            mRadii =
                max.(
                    rowNorm(pUEuclidean'), #r_j^+
                    rowNorm(nUEuclidean'), #r_j^-
                    rowNorm(EuclideanDistances'), #r_j
                )
        else
            mRadii = rowNorm(EuclideanDistances')
        end

        v = mRadii ./ TimeSamples[:, i]

        if sol == nothing
            E[:, i] =
                m[1] .* v .^ 2 .* TimeSamples[:, i] .+
                m[1] * alpha * TimeSamples[:, i]
        else #we have a solution available
            vs = sol[1]
            ts = sol[2]
            LPos = sol[3]
            tSol = ts[3] #just use the time provided by the solver

            EuclideanDistances = LPos' .- EuclideanPositions #mean distance
            if riska
                pUEuclidean = LPos' .- pathSample2Array(path.(pU[:, i], i))
                nUEuclidean = LPos' .- pathSample2Array(path.(nU[:, i], i))
                mRadii =
                    max.(
                        rowNorm(pUEuclidean'), #r_j^+
                        rowNorm(nUEuclidean'), #r_j^-
                        rowNorm(EuclideanDistances'), #r_j
                    )
            else
                mRadii = rowNorm(EuclideanDistances')
            end

            LPos = hcat(LPos[1] * ones(lt), LPos[2] * ones(lt))
            vNew = mRadii ./ tSol
            E[:, i] =
                m[1] .* v .^ 2 .* TimeSamples[:, i] .+
                m[1] * alpha * TimeSamples[:, i] .+
                1 .*
                (m[2] .* rowNormSquared(vNew') .* tSol .+ m[2] * alpha * tSol)
        end
    end
    # E now contains energies for each sample group. Next choose a rmethod
    replace!(E, NaN => Inf)
    if Ns == 1
        return [argmin(E[:, 1]) argmin(E[:, 2])]
    end
    if rmethod == "BestFirst"
        Emins = zeros(np)
        elites = zeros(Int16, Ns, np)
        Threads.@threads for i = 1:np
            elites[:, i] = partialsortperm(E[:, i], 1:min(Ns, lt))
            idx = elites[1, i]
            Emins[i] = E[idx, i]
        end
        #to select the path using this policy
        #get the best from each, and output the best
        ptgt = argmin(Emins)
        return elites, ptgt
    elseif rmethod == "WorstFirst"
        Emins = zeros(np)
        elites = zeros(Int16, Ns, np)
        Threads.@threads for i = 1:np
            elites[:, i] = partialsortperm(E[:, i], 1:min(Ns, lt))
            idx = elites[1, i]
            Emins[i] = E[idx, i]
        end
        #to select the path using this policy
        #get the best from each, and output the best
        ptgt = argmax(Emins)
        return elites, ptgt
    else
        error("Invalid rmethod Argument")
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

function CEM(μ, Σ, np, elites, OptTimeSample, TimeSamples, Ns)
    β = 0.9
    γ = 1.0
    elite = elites[1, :]
    Threads.@threads for j = 1:length(np)
        μn = sum(TimeSamples[elites[:, j], j]) / Ns
        μ[j] = γ * μn + (1 - γ) * μ[j]
        Σ[j] =
            β * (sum((TimeSamples[elites[:, j], j] .- μ[j]) .^ 2) ./ Ns + 1.0) +
            (1 - β) * Σ[j]
        Σ[j] = min(Σ[j], 100)
        OptTimeSample[j] = TimeSamples[elite[j], j]
    end
    return μ, Σ, OptTimeSample
end

function iterateCEM(μ, Σ, np)
    OptTimeSample = zeros(length(p))
    for i = 1:1000
        TimeSamples = SampleTime(μ, Σ, N)
        elites = rankMultiPath(TimeSamples, UASPos, Ns, np)
        CEM(μ, Σ, np, elites, OptTimeSample, TimeSamples)
    end
    return μ, Σ, OptTimeSample
end

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

function mission(Er = 18000, rmethod = "WorstFirst", tt = 80, ti = 5, dt = 1)
    Random.seed!(125)
    UASPos = [800, 450]
    LPos = [600, 600]
    #LPos = [800, 450]
    ts = 0
    PNRStat = false
    N = 100
    p = [1, 2]
    ptgt = 2
    dt = 1
    PrevPNR = [0.0, 0.0]
    μ = 60 * ones(2)
    Σ = 10 * ones(2)
    Ns = 5
    UASPosVec = zeros(N, 2)
    #TimeSamples = ts:120
    t = range(0.0, stop = ti, step = dt)
    li = length(t)
    D = zeros(li, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(li)
    gp = LearnDeviationFunction(D, true, "DTC")
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
    solr = 0.0
    riska = true
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
        elites, ptgt = rankMultiPath(
            TimeSamples,
            UASPos,
            Ns,
            p,
            gp,
            tv[i],
            DriverPos,
            rmethod,
            Er,
            sol,
            riska,
        )
        CEM(μ, Σ, p, elites, OptTimeSample, TimeSamples, Ns)
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
        sol = (v, t, LPos, UASPos, DriverPos, tv[i], OptTimeSample, ptgt)
        solr = sol
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
        vsq = rowNorm(sol[1])
        @show i, t[1], ptgt, vsq
    end
    #risk1 = GPCVaR(gp, solr, 1)
    #risk2 = GPCVaR(gp, solr, 2)
    #if (risk1 > 200 || risk2 > 200)
    #    println("Risk too high, aborting!")
    #end
    gif(anim, "RDV_Anim_MP.gif", fps = 15)
    return solr
end

function GPCVaR(
    gp,
    sol, #v, t, LPos, UASPos, DriverPos, ti, OptTimeSample, ptgt
    ptgt = 1,
    n = 10000,
    c_α = 0.01,
)
    vs = sol[1]
    ts = sol[2]
    LPos = sol[3]
    UASPos = sol[4]
    DriverPos = sol[5]
    ti = sol[6]

    OptTimeSample = sol[7][ptgt]

    μ = DriverPosition(ti, OptTimeSample, gp) + DriverPos #mean
    Σ = DriverUncertainty(ti, OptTimeSample, gp) #variance
    RDVPos = path(μ, ptgt, true)
    gpStd = sqrt(Σ + 1.0)
    gp_distr = Normal(μ, gpStd)
    x = rand(gp_distr, n)
    UASPosv = hcat(UASPos[1] * ones(n), UASPos[2] * ones(n))
    LPosv = hcat(LPos[1] * ones(n), LPos[2] * ones(n))
    EuclideanPositions = pathSample2Array(path.(x, ptgt))
    EuclideanDistancesUAS = EuclideanPositions .- UASPosv
    EuclideanDistancesL = EuclideanPositions .- LPosv


    EDS = rowNorm(EuclideanDistancesUAS')
    EDL = rowNorm(EuclideanDistancesL')

    t1 = (OptTimeSample - ti)
    #nominal
    sEDS = norm(UASPos - RDVPos)
    sEDL = norm(LPos - RDVPos)
    sVS = sEDS ./ t1
    Em, tm, vm = minReturnEnergy(RDVPos, LPos)

    #compute new velocities
    VS = EDS ./ t1
    VL = EDL ./ tm

    #compute nominal energy
    Γ1 = sVS^2 * m[1] / 2 * t1 + m[1] * alpha * t1 + Em

    #compute new energies
    E =
        VS .^ 2 .* m[1] ./ 2 .* t1 + .+VL .^ 2 .* m[2] ./ 2 .* tm .+
        m[1] * alpha * t1 .+ m[2] * alpha * tm
    xd = Γ1 .- E
    # To find our elusive distribution, we will sample from gp and fit.
    ddistr = fit(Normal, xd)

    #now its easy to compute CVaR of a normal distribution
    x_α = VaR(ddistr, c_α)
    CVaRα = CVaR(ddistr, x_α, c_α)
    gainmain = -CVaRα #extra distance we have to deal with on the main path

    #now redo everything for the second path, but subs t3 with new one
    maintgt = ptgt
    ptgt = ptgt == 1 ? 2 : 1

    OptTimeSample = sol[7][ptgt]

    μ = DriverPosition(ti, OptTimeSample, gp) + DriverPos #mean
    Σ = DriverUncertainty(ti, OptTimeSample, gp) #variance
    RDVPos = path(μ, ptgt, true)
    gpStd = sqrt(Σ + 1.0)
    gp_distr = Normal(μ, gpStd)
    x = rand(gp_distr, n)
    UASPosv = hcat(UASPos[1] * ones(n), UASPos[2] * ones(n))
    LPosv = hcat(LPos[1] * ones(n), LPos[2] * ones(n))
    EuclideanPositions = pathSample2Array(path.(x, ptgt))
    EuclideanDistancesUAS = EuclideanPositions .- UASPosv
    EuclideanDistancesL = EuclideanPositions .- LPosv


    EDS = rowNorm(EuclideanDistancesUAS')
    EDL = rowNorm(EuclideanDistancesL')

    t1 = (OptTimeSample - ti)
    #nominal
    sEDS = norm(UASPos - RDVPos)
    sEDL = norm(LPos - RDVPos)
    sVS = sEDS ./ t1
    Em, tm, vm = minReturnEnergy(RDVPos, LPos)

    #compute new velocities
    VS = EDS ./ t1
    VL = EDL ./ tm

    #compute nominal energy
    Γ1 = sVS^2 * m[1] / 2 * t1 + m[1] * alpha * t1 + Em

    #compute new energies
    E =
        VS .^ 2 .* m[1] ./ 2 .* t1 + .+VL .^ 2 .* m[2] ./ 2 .* tm .+
        m[1] * alpha * t1 .+ m[2] * alpha * tm
    xd = Γ1 .- E
    # To find our elusive distribution, we will sample from gp and fit.
    ddistr = fit(Normal, xd)

    #now its easy to compute CVaR of a normal distribution
    x_α = VaR(ddistr, c_α)
    CVaRα = CVaR(ddistr, x_α, c_α)
    gainalt = -CVaRα

    return gainmain, gainalt, maintgt

end

function VaR(d, α::Real)
    #truncate:
    q1 = 0.001
    q2 = 1 - q1
    @assert(α > q1)
    @assert(α < q2)
    q1 = quantile(d, q1)
    q2 = quantile(d, q2)
    x = range(q1, stop = q2, length = 10^3)
    Fx = cdf(d, x)
    i = findfirst(x -> x >= α, Fx)
    return x[i]
end

function CVaR(d, x_α::Real, α::Real)
    x_i = quantile(d, 1e-6)
    CVaRα, err = quadgk(x -> x * pdf(d, x), x_i, x_α, 1e-9)
    CVaRα = 1 / α * CVaRα
end
