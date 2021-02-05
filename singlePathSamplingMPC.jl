

using Plots, LinearAlgebra, Random, Statistics, ColorSchemes, LazySets
using JuMP, Ipopt, Measures, Printf, LaTeXStrings
using BenchmarkTools
rng = MersenneTwister(1234);
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
pv2 = [0 800; 1000 800; 1200 1200]
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

function DriverPosFunction(t)
    return MaxDriverSpeed .* t
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
                    path.(DriverPosFunction(TimeSamples[:, i]), i),
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
    β = 0.1
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
            μn = sum(TimeSamples[elites[:,j], j]) / Ns
            μ[j] = γ * μn + (1 - γ) * μ[j]
            Σ[j] =
                β * (sum((TimeSamples[elites[:,j], j] .- μ[j]) .^ 2) ./ Ns + 1) +
                (1 - β) * Σ[j]
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
    MPC = Model(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "max_iter" => convert(Int64, 500),
        ),
    )
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
        m[2] * v[2, 3]^2 * t[3] +
        m[1] * alpha * t[1] +
        m[1] * alpha * t[2] +
        m[2] * alpha * t[3] <= Er
    )
    @NLconstraint(
        MPC,
        m[1] * v[1, 1]^2 * t[1] +
        m[1] * v[1, 4]^2 * t[4] +
        m[1] * v[2, 1]^2 * t[1] +
        m[1] * v[2, 4]^2 * t[4] +
        m[1] * alpha * t[1] +
        m[1] * alpha * t[4] <= Er
    )
    ta = OptTimeSample - ts #available time is time to RDV minus current time
    @constraint(MPC, t[1] + t[2] <= ta)
    if PrevPNR[1] != 0 || PrevPNR[2] != 0
        PNRDist = @expression(MPC, PrevPNR - PNR)
        @constraint(MPC, PNRDist' * PNRDist <= 1000.0)
    end
    @objective(MPC, Min, sum(t[i] for i = 2:4) - 1 * t[1])

    JuMP.optimize!(MPC)
    v = value.(v)
    t = value.(t)
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

function plotPlan!(UASPos, LPos, v, t, TimeSamples)
    plotpath!(1)
    plotTimeSamples!(TimeSamples, 1)
    plotpath!(2)
    plotTimeSamples!(TimeSamples, 2)
    PNR = UASPos + v[:, 1] * t[1]
    RDV = PNR + v[:, 2] * t[2]
    L = RDV + v[:, 3] * t[3]
    LA = PNR + v[:, 4] * t[4]
    scatter!([UASPos[1]], [UASPos[2]], markersize = 5.0)
    DPlan = [UASPos PNR RDV L]
    APlan = [UASPos PNR L]
    annotate!(PNR[1], PNR[2] + vo, text("PNR", 12, :center))
    annotate!(UASPos[1], UASPos[2] - vo, text("UAS", 12, :center))
    annotate!(L[1], L[2] - vo, text("Depot", 12, :center))
    annotate!(RDV[1], RDV[2] + vo, text("RDV", 12, :right))
    p1 = pv1[2, :]
    p2 = pv2[2, :]
    annotate!(p1[1] - ho, p1[2] + vo, text("Path 1", 12, :right))
    annotate!(p2[1] - ho, p2[2] - vo, text("Path 2", 12, :right))
    p = scatter!(DPlan[1, :], DPlan[2, :])
    p = plot!(DPlan[1, :], DPlan[2, :], lw = 2)
    p = plot!(APlan[1, :], APlan[2, :], linestyle = :dash)
end

function drawConvexHull(TimeSamples, p = [1, 2], ucol = :blue)
    lt = length(TimeSamples)
    lp = length(p)
    θ = DriverPosFunction(TimeSamples)
    v = zeros(lt * lp)
    v = [zeros(2) for i = 1:lt*lp]
    for j = 1:lp
        idx1 = 1 + (j - 1) * lt
        idx2 = j * lt
        v[idx1:idx2] = [path(i, j, true) for i in θ]
    end
    hull = convex_hull(v)
    plot!(VPolygon(hull), alpha = 0.2, color = ucol)
end

function maxRange(Er, mass = [2, 1], tmax = 1000)
    m = mass
    OCP = Model(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "max_iter" => convert(Int64, 50000),
        ),
    )

    @variable(OCP, r >= 0)
    @variable(OCP, v[i = 1:2, j = 1:2] >= 0)
    @variable(OCP, t[i = 1:2] >= 0)
    @NLobjective(OCP, Max, sqrt((v[1, 1] * t[1])^2 + (v[2, 1] * t[1])^2))
    @NLconstraint(
        OCP,
        sqrt((v[1, 1] * t[1])^2 + (v[2, 1] * t[1])^2) ==
        sqrt((v[1, 2] * t[2])^2 + (v[2, 2] * t[2])^2)
    ) # one-way ranges are the same
    @NLconstraint(
        OCP,
        m[1] * v[1, 1]^2 * t[1] + #vx^2 going
        m[2] * v[1, 2]^2 * t[2] +
        m[1] * v[2, 1]^2 * t[1] + #vx^2 back
        m[2] * v[2, 2]^2 * t[2] +
        m[1] * alpha * t[1] + #hovering going
        m[2] * alpha * t[2] <= Er #hovering back
    )
    @constraint(OCP, sum(t[i] for i = 1:2) <= tmax)
    JuMP.optimize!(OCP)
    v = value.(v)
    t = value.(t)
    r = sqrt((v[1, 1] * t[1])^2 + (v[2, 1] * t[1])^2)
    #@show r, v, t
    return r
end

function minTime(Er, mass = [2, 1], xmax = 500, ymax = 500)
    m = mass
    OCP = Model(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 0,
            "max_iter" => convert(Int64, 500),
        ),
    )

    @variable(OCP, r >= 0)
    @variable(OCP, v[i = 1:2, j = 1:2] >= 0)
    @variable(OCP, t[i = 1:2] >= 0)
    @NLobjective(OCP, Min, sum(t[i] for i = 1:2))
    rmax = sqrt(xmax^2 + ymax^2)
    @NLconstraint(
        OCP,
        sqrt((v[1, 1] * t[1])^2 + (v[2, 1] * t[1])^2) ==
        sqrt((v[1, 2] * t[2])^2 + (v[2, 2] * t[2])^2)
    ) # one-way ranges are the same
    @NLconstraint(OCP, sqrt((v[1, 1] * t[1])^2 + (v[2, 1] * t[1])^2) == rmax)
    @NLconstraint(
        OCP,
        m[1] * v[1, 1]^2 * t[1] + #vx^2 going
        m[2] * v[1, 2]^2 * t[2] +
        m[1] * v[2, 1]^2 * t[1] + #vx^2 back
        m[2] * v[2, 2]^2 * t[2] +
        m[1] * alpha * t[1] + #hovering going
        m[2] * alpha * t[2] <= Er #hovering back
    )
    JuMP.optimize!(OCP)
    v = value.(v)
    t = value.(t)
    r = sqrt((v[1, 1] * t[1])^2 + (v[2, 1] * t[1])^2)
    @show r, v, t
    tmax = sum(t)
    return tmax
end

function CircleShape(h, k, r)
    θ = range(0, 2 * π, length = 500)
    h .+ r * sin.(θ), k .+ r * cos.(θ)
end

function PlotRangeContours(
    Er = 50000,
    DepotCoord = [0.0, 0.0],
    t0 = 10,
    tint = 10,
    n = 20,
    mass = [2, 1],
    keep = false,
)
    t = range(t0, step = tint, length = n)
    r = zeros(n)
    if !keep
        p = plot()
    end
    p = plot!()
    for i = 1:n
        r[i] = maxRange(Er, mass, t[i])
        p = plot!(
            CircleShape(DepotCoord[1], DepotCoord[2], r[i]),
            seriestype = [:shape],
            c = :green,
            linecolor = :black,
            linealpha = 0.05,
            #linewidth = 0.0,
            legend = :false,
            fillalpha = min(1 / n, 0.1),
            aspecratio = 1,
        )
        s1 = @sprintf("\$T_{\\operator{max}}=%0.1fs\$", t[i])
        s1 = @sprintf("%ds", t[i])
        θ = -(i^(1.3) / (5 * n) * 2 * π + π)
        annotate!(
            DepotCoord[1] + r[i] * cos(θ),
            DepotCoord[2] + r[i] * sin(θ),
            Plots.text(s1, FontSize, rotation = rad2deg(θ - π / 2)),
        )
        xlabel!(L"x")
        ylabel!(L"y")
    end
    display(p)
    return p
end

function PlotRangeTiers!(
    Er = 10000,
    UASPos = [0.0, 0.0],
    tmax = 100,
    mass = [2, 1],
)
    p = plot!(
        CircleShape(
            UASPos[1],
            UASPos[2],
            maxRange(Er, [mass[2], mass[2]], tmax),
        ),
        seriestype = [:shape],
        c = :green,
        linecolor = :black,
        linealpha = 0.1,
        #linewidth = 0.0,
        legend = :false,
        fillalpha = 0.3,
        aspecratio = 1,
    )
    p = plot!(
        CircleShape(
            UASPos[1],
            UASPos[2],
            maxRange(Er, [mass[1], mass[1]], tmax),
        ),
        seriestype = [:shape],
        c = :blue,
        linecolor = :black,
        linealpha = 0.1,
        #linewidth = 0.0,
        legend = :false,
        fillalpha = 0.3,
        aspecratio = 1,
    )
    p = plot!(
        CircleShape(UASPos[1], UASPos[2], maxRange(Er, mass, tmax)),
        seriestype = [:shape],
        c = :red,
        linecolor = :black,
        linealpha = 0.1,
        #linewidth = 0.0,
        legend = :false,
        fillalpha = 0.3,
        aspecratio = 1,
    )
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
    plot(x[1, :], x[2, :], xlims = (0, 1300), ylims = (400, 1300), lw = 2)
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
    plot!(x[1, :], x[2, :], xlims = (0, 1300), ylims = (400, 1300), lw = 2)
end

function plotPosSamples(samples, p = 1)
    plotpath(p)
    c = path.(samples, p)
    scatter!(c)
end

function plotPosSamples!(samples, p = 1)
    plotpath!(p)
    c = path.(samples, p)
    scatter!(c, markersize = 3.0)
end

function plotTimeSamples(samples, p = 1)
    samples = DriverPosFunction(samples)
    plotPosSamples(samples, p)
end

function plotTimeSamples!(samples, p = 1)
    samples = DriverPosFunction(samples)
    plotPosSamples!(samples, p)
end
