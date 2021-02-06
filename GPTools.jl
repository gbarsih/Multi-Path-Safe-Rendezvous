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
using BenchmarkTools, Dates, Measures
using GaussianProcesses
using Random
using Optim
using QuadGK
using Measures

NoiseVar = 0.05
NoiseStd = sqrt(NoiseVar)
NoiseLog = log10(NoiseVar)
# setup driver learning problem.
default(size = 2 .* [800, 600])
default(palette = :tol_bright)
default(dpi = 300)
default(lw = 3)
default(margin = 10mm)
FontSize = 18
default(xtickfontsize = FontSize)
default(ytickfontsize = FontSize)
default(xguidefontsize = FontSize)
default(yguidefontsize = FontSize)
default(legendfontsize = FontSize)
default(titlefontsize = FontSize)
rng = MersenneTwister(1234)

rectangle(w, h, x, y) = Shape(x .+ [0, w, w, 0], y .+ [0, 0, h, h])

# Now test with learning

function LearnedDeviation(gp, v) #learned deviation function
    predict_y(gp, [v])
end

function LearnDeviationFunction(D, useConst = false, method = "full")
    #=This function takes in the dataset D and outputs a GP.
    #D[1,:] has historical velocities
    #D[2,:] has measured velocities
    Want to learn measured velocities as a function of velocities
    =#

    x = D[:, 1]   #predictors
    y = D[:, 2]   #regressors
    #Select mean and covariance function
    if !useConst
        mFcn = MeanZero()
    else
        mFcn = MeanConst(mean(D[:, 2]))
    end
    kern = SE(0.0, 0.0)
    logObsNoise = NoiseLog
    if method == "full"
        return GPE(x, y, mFcn, kern, logObsNoise)
    elseif method == "SOR"
        Xu = Matrix(
            quantile(
                x,
                [
                    0.2,
                    0.25,
                    0.3,
                    0.35,
                    0.4,
                    0.45,
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.98,
                ],
            )',
        )
        X = Matrix(x')
        return GaussianProcesses.SoR(X, Xu, y, mFcn, kern, logObsNoise)
    elseif method == "DTC"
        Xu = Matrix(
            quantile(
                x,
                [
                    0.02,
                    0.2,
                    0.3,
                    0.35,
                    0.4,
                    0.45,
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.8,
                    0.98,
                ],
            )',
        )
        X = Matrix(x')
        return GaussianProcesses.DTC(X, Xu, y, mFcn, kern, logObsNoise)
    end
end

function TestLearning(n = 100)
    t = range(0, stop = 100, length = n)
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    gp = LearnDeviationFunction(D, true, "DTC")
    plot(gp, ylims = (-2, 10), xlims = (0, 16))
end

function BenchmarkLearning(n = 100)
    t = range(0, stop = 100, length = n)
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    @benchmark LearnDeviationFunction($D, $true, "full")
    @benchmark LearnDeviationFunction($D, $true, "SOR")
    @benchmark LearnDeviationFunction($D, $true, "DTC")
end

function AnimateLearning(n = 100)
    t = range(0, stop = 100, length = n)
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    anim = @animate for i = 1:n
        @show i
        k = i <= 5 ? 5 : i
        gp = LearnDeviationFunction(D[1:k, :], true, "full")
        plot(
            gp,
            ylims = (-2, 6),
            xlims = (0, 16),
            legend = false,
            xlabel = "Prototypical Speed [θ/s]",
            ylabel = "Deviation Function [θ/s]",
            markeralpha = 0.3,
        )
    end
    gif(anim, "Learning_Anim.gif", fps = 30)
end

function PlotPosPrediction(n = 100, t0 = 20.0, tf = 100.0)
    t = range(0, stop = t0, length = n) #just learn the first t0s
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    gp = LearnDeviationFunction(D, true, "full")
    #plot(gp, ylims = (-2, 10), xlims = (0, 16))
    t = range(0.0, stop = tf, length = 100)
    p = zeros(length(t))
    s = zeros(length(t))
    v = zeros(length(t))
    vp = VelocityPrior.(t)
    Threads.@threads for i = 1:length(t)
        p[i] = DriverPosition(t[1], t[i], gp)
        s[i] = DriverUncertainty(t[1], t[i], gp, 1e-3)
        v[i] = DriverVelocity(t[i])
    end
    p1 = plot(
        t,
        v,
        title = "True Velocity",
        ylabel = "Driver Speed [θ/s]",
        xlabel = "Time [s]",
        label = "Driver Velocity",
        xlims = (0, 100),
        ylims = (5, 16),
    )
    p1 = plot!(t, vp, label = "Velocity Prior")
    minv = minimum(D[:, 1])
    maxv = maximum(D[:, 1])
    @show minv maxv
    p1 = plot!(
        rectangle(1000, maxv - minv, -100, minv),
        opacity = 0.2,
        label = "Learned Region",
    )
    p2 = plot(
        t,
        s,
        title = "Position Uncertainty",
        ylabel = "Variance [θ]",
        xlabel = "Time [s]",
        legend = false,
        ylims = (0, 35),
    )
    p3 = plot(
        gp,
        ylims = (-2, 6),
        xlims = (5, 16),
        title = "Gaussian Process",
        xlabel = "Prototypical Speed [θ/s]",
        ylabel = "Dev. Fcn [θ/s]",
        legend = false,
    )
    l = @layout [a b; c{0.4h}]
    p = plot(p1, p2, p3, layout = l)
    display(p)
    savefig("learning_ex.png")
end
