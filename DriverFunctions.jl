function VelocityPrior(t) #historical model
    return 10.0 .+ 4.0 .* sin(t ./ 10)
end

function VariancePrior(t) #historical model
    return 2.0
end

function Deviation(v) #deviation function
    return v >= 10 ? 2.0 : -1.0
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

function DriverUncertainty(t, gp)
    v = VelocityPrior(t)
    μ, Σ = predict_y(gp, [v])
    return Σ[1]
end

function DriverPosition(ti, TimeSamples, gp = nothing, tol = 1e-1)
    if minimum(size(TimeSamples)) == 1
        integral, err = quadgk(x -> DriverVelocity(x, gp), ti, TimeSamples, rtol = tol)
        return integral
    else
        nsamples, npaths = size(TimeSamples)
        PosArray = zeros(nsamples, npaths)
        for p = 1:npaths
            Threads.@threads for i = 1:nsamples
                PosArray[i, p], err =
                    quadgk(x -> DriverVelocity(x, gp), ti, TimeSamples[i,p], rtol = tol)
                end
        end
    end
    return PosArray
end

function DriverUncertainty(ti, tf, gp, tol = 1e-1)
    integral, err = quadgk(x -> DriverUncertainty(x, gp), ti, tf, rtol = tol)
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
