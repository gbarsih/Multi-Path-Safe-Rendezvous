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
    np = size(TimeSamples, 2)
    for i = 1:np
        plotpath!(i)
        plotTimeSamples!(TimeSamples[:, i], i)
    end
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

function drawMultiConvexHull(PosSamples, p = [1, 2], ucol = :blue)
    ls, lp = size(PosSamples)
    if lp != length(p)
        error("Incorrect input dimensions")
    end
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

function CircleShape(h, k, r)
    θ = range(0, 2 * π, length = 500)
    h .+ r * sin.(θ), k .+ r * cos.(θ)
end

function PlotRangeContours(
    Er = 50000,
    DepotCoord = [0.0, 0.0],
    t0 = 30,
    tint = 30,
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
            size = (1200, 1200),
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
    p = title!("Range for DIfferent Target Times")
    display(p)
    savefig("range_contours.png")
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

function plotpath!(ns = ones(1))
    x = zeros(2, 1001)
    p = plot!(legend = :false)
    for i in ns
        j = 1
        for θ in range(0, size(x, 2) - 1, step = 1)
            x[:, j] .= path(θ, i)
            j = j + 1
        end
        p = plot!(x[1, :], x[2, :], xlims = (0, 1300), ylims = (400, 1300))
    end
    display(p)
    return p
end

function plotPosSamples!(PosSamples, np = ones(1))
    pp = plot!()
    for p in np
        c = path.(PosSamples[:,p], p)
        pp = scatter!(c, markersize = 3.0)
    end
    display(pp)
    return pp
end

function plotTimeSamples!(TimeSamples,np,ti=0.0)
    PosSamples = DriverPosition(ti,TimeSamples)
    PosSamples[PosSamples.>=Mp] .= Mp
    plotPosSamples!(PosSamples, np)
end
