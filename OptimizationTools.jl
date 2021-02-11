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

function RendezvousPlanner(
    UASPos,
    LPos,
    OptTimeSample,
    Er,
    ts,
    PrevPNR = [0, 0],
    PNRStat = false,
    p = 1,
    gp = nothing,
    DriverPos = 0.0,
)
    RDVPos = path(DriverPosition(ti, OptTimeSample[p], gp) + DriverPos)
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
    ta = OptTimeSample[p] - ts #available time is time to RDV minus current time
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
