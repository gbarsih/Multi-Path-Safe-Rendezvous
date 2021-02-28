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
using Distributions
using Random
import Statistics
using Optim
using QuadGK
using Measures

NoiseVar = 0.05
NoiseStd = sqrt(NoiseVar)
NoiseLog = log10(NoiseVar)
NoiseStdLog = log10(NoiseStd)
# setup driver learning problem.
default(size = 0.6 .* [800, 600])
default(palette = :tol_bright)
default(dpi = 300)
default(lw = 2)
default(margin = 1mm)
FontSize = 12
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
    #kern = SE(0.0, 0.0)
    kern = Matern(3 / 2, zeros(5), log10(sqrt(5.25)))
    #lik = BernLik()
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
    else
        error("Invalid GP method")
    end
end
