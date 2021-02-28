function VelocityPrior(t) #historical model
    #return 10.0 .+ 4.0 .* sin(t ./ 10)
    return MaxDriverSpeed .+ 1.0 .* sin(t ./ 10)
end

function VariancePrior(t) #historical model
    return 5.25
end

function Deviation(v) #deviation function
    return v >= MaxDriverSpeed ? 1.0 : -1.0
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

function DriverPosition(ti, TimeSamples, gp = nothing, tol = 1e-3)
    if isa(TimeSamples, Int) || isa(TimeSamples, Float64)
        integral, err =
            quadgk(x -> DriverVelocity(x, gp), ti, TimeSamples, rtol = tol)
        return integral
    else
        nsamples, npaths = size(TimeSamples)
        PosArray = zeros(nsamples, npaths)
        for p = 1:npaths
            Threads.@threads for i = 1:nsamples
                PosArray[i, p], err = quadgk(
                    x -> DriverVelocity(x, gp),
                    ti,
                    TimeSamples[i, p],
                    rtol = tol,
                )
            end
        end
    end
    return PosArray
end

function DriverUncertainty(ti, TimeSamples, gp, tol = 1e-1)
    if isa(TimeSamples, Int) || isa(TimeSamples, Float64)
        integral, err =
            quadgk(x -> DriverUncertainty(x, gp), ti, TimeSamples, rtol = tol)
        return integral
    else
        nsamples, npaths = size(TimeSamples)
        UArray = zeros(nsamples, npaths)
        for p = 1:npaths
            Threads.@threads for i = 1:nsamples
                UArray[i, p], err = quadgk(
                    x -> DriverUncertainty(x, gp),
                    ti,
                    TimeSamples[i, p],
                    rtol = tol,
                )
            end
        end
    end
    return UArray
end

function DriverPositionVector(tv = [0.0, 1.0], gp = nothing)
    l = length(tv) - 1
    pos = zeros(l)
    for i = 1:l
        pos[i] = DriverPosition(tv[1], tv[i+1], gp)
    end
    return pos
end
