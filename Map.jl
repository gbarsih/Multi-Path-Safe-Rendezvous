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
alpha = 0
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
