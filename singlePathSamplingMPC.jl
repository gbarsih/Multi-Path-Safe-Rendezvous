

using Plots, LinearAlgebra, Random
using JuMP, Ipopt
using BenchmarkTools
rng = MersenneTwister(1234);

m = [2, 1]
alpha = 1
vmax = 25

function path(θ, p = 1)
    if p == 1
        if θ <= 500
            return θ, 800
        elseif θ <= 900
            θ = θ - 500
            return 500, 800 + θ
        else
            θ = θ - 900
            return -θ + 500, 800 + 400
        end
    elseif p == 2
        return θ, 800
    end
end

function DriverPosFunction(t)
    return 10 .* t
end

function plotpath(ns = ones(1))
    x = zeros(2, 1001)
    for i in ns
        j = 1
        for θ in range(0, size(x, 2) - 1, step = 1)
            x[:, j] .= path(θ, i)
            j = j + 1
        end
    end
    plot(x[1, :], x[2, :], xlims = (0, 1300), ylims = (0, 1300))
end

function plotpath!(ns = ones(1))
    x = zeros(2, 1001)
    for i in ns
        j = 1
        for θ in range(0, size(x, 2) - 1, step = 1)
            x[:, j] .= path(θ, i)
            j = j + 1
        end
    end
    plot!(x[1, :], x[2, :], xlims = (0, 1300), ylims = (0, 1300))
end

function plotPosSamples(samples, p = 1)
    plotpath(p)
    c = path.(samples, p)
    scatter!(c)
end

function plotPosSamples!(samples, p = 1)
    plotpath!(p)
    c = path.(samples, p)
    scatter!(c,markersize=3.0)
end

function plotTimeSamples(samples, p = 1)
    samples = DriverPosFunction(samples)
    plotPosSamples(samples, p)
end

function plotTimeSamples!(samples, p = 1)
    samples = DriverPosFunction(samples)
    plotPosSamples!(samples, p)
end

function uav_dynamics(x, y, vx, vy, dt, rem_power = Inf, vmax = Inf, m = 1.0)
    vx > vmax ? vmax : vx
    vy > vmax ? vmax : vy
    vx < -vmax ? -vmax : vx
    vy < -vmax ? -vmax : vy
    x = x + vx * dt
    y = y + vy * dt
    rem_power = rem_power - vx^2 * m * dt - vy^2 * m * dt - alpha * dt
    return x, y, rem_power
end

function rankSampleMean(TimeSample, UASPos, p = [1])
    np = size(p,1)
    lt = size(TimeSample,1)
    if np == 1
        x = hcat(
            UASPos[1] * ones(lt),
            UASPos[2] * ones(lt),
        )
        v = (pathSample2Array(path.(DriverPosFunction(TimeSample), p)) .- x) ./ TimeSample
        E = diag(v * v') .* TimeSample .+ alpha * TimeSample
        replace!(E, NaN => Inf)
        return E
    elseif np == 2
        #Here we rank two paths
        #Loss function is energy between paths plus energy to min
        x = hcat(
            UASPos[1] * ones(lt),
            UASPos[2] * ones(lt),
        ) #x contains a copy-vector of x-y coordinates of the UAS
        E = zeros(lt,np)
        for i = 1:np #for each path
            v = (pathSample2Array(path.(DriverPosFunction(TimeSample), i)) .- x) ./ TimeSample
            E[:,i] = diag(v * v') .* TimeSample .+ alpha * TimeSample
        end
        #E now contains energy to each path
        #Calculate min
        Em = min.(E[:,1],E[:,2])
        #Calculate energy between paths
        Eb = max.(E[:,1],E[:,2]) - Em
    end
end

function pathSample2Array(pathSample)
    return hcat([y[1] for y in pathSample], [y[2] for y in pathSample])
end

function simDeterministicUniform()
    clearconsole()
    UASPos = [800, 950]
    LPos = [700, 650]
    Er = 6000
    ts = 0
    PNRStat = false
    N = 100
    p = 1
    dt = 1
    PrevPNR = [0.0, 0.0]
    anim = @animate for i = 1:100
        TimeSamples = ts:sum(bitrand(rng, 5))+1:(100-ts/3)
        PosSamples = DriverPosFunction(TimeSamples)
        rank = rankSampleMean(TimeSamples, UASPos, p)
        OptTimeSample = TimeSamples[argmin(rank)]
        v, t =
            DeterministicUniformMPC(UASPos, LPos, OptTimeSample, Er, ts, PrevPNR, PNRStat)
            pp = plot(legend=false)
        pp = plotPlan!(UASPos, LPos, RDVPos, v, t, TimeSamples)
        @show TimeSamples
        pp = scatter!(path(DriverPosFunction(ts)),label="")
        pp = plot!(tickfont = Plots.font("serif", pointsize=round(12.0)))
        #display(pp)
        ts = ts + dt
        PrevPNR = v[:, 1] .* t[1] .+ UASPos
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
        @show t, i
        if t[1] <= 1.0 && ts > 10
            break
        end
    end
    # proceed with rendezvous
    gif(anim, "anim_fps15.gif", fps = 15)
end

function DeterministicUniformMPC(
    UASPos,
    LPos,
    OptTimeSample,
    Er,
    ts,
    PrevPNR = [0, 0],
    PNRStat = false,
    p = 1,
)
    RDVPos = path(DriverPosFunction(OptTimeSample), p)
    MPC = Model(optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "max_iter" => convert(Int64, 500),
    ))
    @variable(MPC, PNR[i = 1:2])
    @variable(MPC, -vmax <= v[i = 1:2, j = 1:4] <= vmax)
    @variable(MPC, t[j = 1:4] >= 0.1)
    @constraint(MPC, PNR .== v[:, 1] .* t[1] .+ UASPos)
    @constraint(MPC, RDVPos .== v[:, 2] .* t[2] .+ PNR)
    @constraint(MPC, LPos .== v[:, 3] .* t[3] .+ RDVPos)
    @constraint(MPC, LPos .== v[:, 4] .* t[4] .+ PNR)

    @NLconstraint(
        MPC,
        m[1] * v[1, 1]^2 * t[1] +
        m[1] * v[1, 2]^2 * t[2] +
        m[2] * v[1, 3]^2 * t[3] +
        m[1] * v[2, 1]^2 * t[1] +
        m[1] * v[2, 2]^2 * t[2] +
        m[2] * v[2, 3]^2 * t[3] <= Er
    )
    @NLconstraint(
        MPC,
        m[1] * v[1, 1]^2 * t[1] +
        m[1] * v[1, 4]^2 * t[4] +
        m[1] * v[2, 1]^2 * t[1] +
        m[1] * v[2, 4]^2 * t[4] <= Er
    )
    ta = OptTimeSample - ts #available time is time to RDV minus current time
    @constraint(MPC, t[1] + t[2] <= ta)
    if PrevPNR[1] != 0 || PrevPNR[2] != 0
        PNRDist = @expression(MPC, PrevPNR - PNR)
        @constraint(MPC, PNRDist' * PNRDist <= 5.0)
    end
    @objective(MPC, Min, sum(t[i] for i = 2:4) - 1 * t[1])

    optimize!(MPC)
    v = value.(v)
    t = value.(t)
    PNR = value.(PNR)
    Ed =
        m[1] * v[1, 1]^2 * t[1] +
        m[1] * v[1, 2]^2 * t[2] +
        m[2] * v[1, 3]^2 * t[3] +
        m[1] * v[2, 1]^2 * t[1] +
        m[1] * v[2, 2]^2 * t[2] +
        m[2] * v[2, 3]^2 * t[3]
    if t[1] + t[2] > ta + 1
        tsum = t[1] + t[2]
        #@show tsum ta t v OptTimeSample ts Er Ed
    else
        #@show t v
    end
    return v, t
end

function plotPlan(UASPos, LPos, RDVPos, v, t)
    plotpath(1)
    plotpath!(2)
    PNR = UASPos + v[:, 1] * t[1]
    RDV = PNR + v[:, 2] * t[2]
    L = RDV + v[:, 3] * t[3]
    LA = PNR + v[:, 4] * t[4]
    scatter!([UASPos[1]], [UASPos[2]])
    scatter!([LPos[1]], [LPos[2]])
    scatter!([PNR[1]], [PNR[2]])
    scatter!([RDV[1]], [RDV[2]])
    scatter!([L[1]], [L[2]])
    scatter!([LA[1]], [LA[2]])
    DPlan = [UASPos PNR RDV L]
    #DPlan = [UASPos RDV L]
    p = plot!(DPlan[1, :], DPlan[2, :])
end

function plotPlan!(UASPos, LPos, RDVPos, v, t)
    plotpath!(1)
    plotpath!(2)
    PNR = UASPos + v[:, 1] * t[1]
    RDV = PNR + v[:, 2] * t[2]
    L = RDV + v[:, 3] * t[3]
    LA = PNR + v[:, 4] * t[4]
    scatter!([UASPos[1]], [UASPos[2]])
    scatter!([LPos[1]], [LPos[2]])
    scatter!([PNR[1]], [PNR[2]])
    scatter!([RDV[1]], [RDV[2]])
    scatter!([L[1]], [L[2]])
    scatter!([LA[1]], [LA[2]])
    DPlan = [UASPos PNR RDV L]
    #DPlan = [UASPos RDV L]
    p = plot!(DPlan[1, :], DPlan[2, :])
end

function plotPlan!(UASPos, LPos, RDVPos, v, t, TimeSamples)
    plotpath!(1)
    plotTimeSamples!(TimeSamples,1)
    plotpath!(2)
    plotTimeSamples!(TimeSamples,2)
    PNR = UASPos + v[:, 1] * t[1]
    RDV = PNR + v[:, 2] * t[2]
    L = RDV + v[:, 3] * t[3]
    LA = PNR + v[:, 4] * t[4]
    scatter!([UASPos[1]], [UASPos[2]], markersize=5.0)
    #scatter!([LPos[1]], [LPos[2]])
    #scatter!([PNR[1]], [PNR[2]])
    #scatter!([RDV[1]], [RDV[2]])
    #scatter!([L[1]], [L[2]])
    #scatter!([LA[1]], [LA[2]])
    DPlan = [UASPos PNR RDV L]
    p = scatter!(DPlan[1, :], DPlan[2, :])
    p = plot!(DPlan[1, :], DPlan[2, :])
end
