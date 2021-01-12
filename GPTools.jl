using GaussianProcesses
using Random

using Optim

Random.seed!(20140430)
# Training data
n = 10;                          #number of training points
x = 2π * rand(n);              #predictors
y = sin.(x) + 0.05 * randn(n);   #regressors

#Select mean and covariance function
mZero = MeanZero()
kern = SE(0.0, 0.0)

logObsNoise = -1.0
gp = GP(x, y, mZero, kern, logObsNoise)

μ, σ² = predict_y(gp, range(0, stop = 2π, length = 100));

plot(gp; xlabel = "x", ylabel = "y", title = "Gaussian process", legend = false, fmt = :png)

GaussianProcesses.optimize!(gp)

plot(gp)

# setup driver learning problem.

function VelocityPrior(t) #historical model
    return 10.0 .+ sin(t ./ 10)
end

function VariancePrior(t) #historical model
    return 2.0
end

function Deviation(v) #deviation function
    return 2.0
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

function LearnDeviationFunction(D)
    #This function takes in the dataset D and outputs a GP.
    #D[1,:] has historical velocities
    #D[2,:] has measured velocities

    D[2, :] .= D[2, :] .- D[1, :]

end
