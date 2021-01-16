using GaussianProcesses
using Random
using Optim

NoiseVar = 0.05
NoiseStd = sqrt(NoiseVar)
NoiseLog = log10(NoiseVar)
# setup driver learning problem.

function VelocityPrior(t) #historical model
    return 10.0 .+ 4.0 .* sin(t ./ 10)
end

function VariancePrior(t) #historical model
    return 2.0
end

function Deviation(v) #deviation function
    return v >= 10 ? 2.0 : -1.0
end

function LearnedDeviation(gp, v) #learned deviation function
    predict_y(gp, [v])
end

#I want to integrate the driver position, which is prior+deviation

function DriverVelocity(t, gp = nothing)
    v = VelocityPrior(t)
    if gp == nothing
        devfcn = Deviation(v)
    else
        devfcn = LearnedDeviation(gp, v)
    end
    return v + devfcn[1][1]
end

function DriverPosition(ti, tf, gp = nothing, tol = 1e-1)
    integral, err = quadgk(x -> DriverVelocity(x, gp), ti, tf, rtol = tol)
    return integral
end

function DriverPositionVector(tv = [0.0, 1.0], gp = nothing)
    l = length(tv) - 1
    pos = zeros(l)
    for i = 1:l
        pos[i] = DriverPosition(tv[1], tv[i+1], gp)
    end
    return pos
end

# Now test with learning

function LearnDeviationFunction(D, useConst = false)
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
    kern = SE(0.0, NoiseLog)
    logObsNoise = NoiseLog
    gp = GPE(x, y, mFcn, kern, logObsNoise)
end

function TestLearning(n = 100)
    t = range(0, stop = 100, length = n)
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    gp = LearnDeviationFunction(D, true)
    plot(gp, ylims = (-2, 10), xlims = (0, 16))
end

function AnimateLearning(n = 100)
    t = range(0, stop = 100, length = n)
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    anim = @animate for i = 1:n
        @show i
        k = i <= 5 ? 5 : i
        gp = LearnDeviationFunction(D[1:k, :], true)
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
