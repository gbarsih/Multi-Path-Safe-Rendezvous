include("MultiPathGP.jl")

function TestLearning(n = 100, method = "full")
    t = range(0, stop = 100, length = n)
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    gp = LearnDeviationFunction(D, true, method)
    plot(
        legend = false,
        xlabel = "Prototypical Speed [θ/s]",
        ylabel = "Deviation Function [θ/s]",
        title = "Deviation Function Learning Performance",
    )
    t = range(0, stop = 20, length = 1000)
    plot!(t, Deviation.(t), linestyle = :dash, color = :red, linealpha = 0.5)
    p = plot!(gp, ylims = (-2, 3), xlims = (4, 12), markeralpha = 0.6)
    display(p)
    savefig("gp_perf.png")
end

function CompareMethods(n = 100)
    t = range(0, stop = 100, length = n)
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    gp = LearnDeviationFunction(D, true, "full")
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    gp2 = LearnDeviationFunction(D, true, "DTC")
    plot(
        legend = :topleft,
        xlabel = L"\textrm{Prototypical Speed }[\theta/s]",
        ylabel = L"\textrm{Deviation Function }[\theta/s]",
        title = L"\textrm{Fitting Performance, DTC vs Full GP}",
        formatter = :latex,
    )
    t = range(0, stop = 20, length = 1000)
    p = plot!(
        gp,
        ylims = (-2, 3),
        xlims = (4, 12),
        markeralpha = 0.2,
        markercolor = :black,
        linecolor = :black,
        fillcolor = :black,
        fillalpha = 0.2,
        label = [L"\textrm{Full GP}" ""],
        formatter = :latex,
    )
    p = plot!(
        gp2,
        ylims = (-2, 3),
        xlims = (4, 12),
        markeralpha = 0.2,
        markercolor = :green,
        linecolor = :green,
        fillcolor = :green,
        fillalpha = 0.2,
        label = [L"\textrm{DTC GP}" ""],
        formatter = :latex,
    )
    plot!(
        t,
        Deviation.(t),
        linestyle = :dash,
        color = :red,
        linealpha = 0.5,
        label = L"\textrm{True Deviation}",
        formatter = :latex,
    )
    display(p)
    savefig("gp_comp.pdf")
end

function BenchmarkLearning(n = 100)
    t = range(0, stop = 100, length = n)
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    @benchmark LearnDeviationFunction($D, $true, "full")
    @benchmark LearnDeviationFunction($D, $true, "SOR")
    @benchmark LearnDeviationFunction($D, $true, "DTC")
end

function AnimateLearning(n = 100)
    t = range(0, stop = 100, length = n)
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    anim = @animate for i = 1:n
        @show i
        k = i <= 5 ? 5 : i
        gp = LearnDeviationFunction(D[1:k, :], true, "full")
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

function PlotPosPrediction(n = 100, t0 = 20.0, tf = 100.0)
    rng = MersenneTwister(1234)
    t = range(0, stop = t0, length = n) #just learn the first t0s
    D = zeros(n, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(n)
    gp = LearnDeviationFunction(D, true, "DTC")
    #plot(gp, ylims = (-2, 10), xlims = (0, 16))
    t = range(0.0, stop = tf, length = 100)
    p = zeros(length(t))
    s = zeros(length(t))
    v = zeros(length(t))
    vp = VelocityPrior.(t)
    #@benchmark DriverPosition($t[1], $t[2], $gp, 1e-1)
    Threads.@threads for i = 1:length(t)
        p[i] = DriverPosition(t[1], t[i], gp)
        s[i] = DriverUncertainty(t[1], t[i], gp, 1e-3)
        v[i] = DriverVelocity(t[i])
    end
    p1 = plot(
        t,
        v,
        title = L"\textrm{True Velocity}",
        ylabel = L"\textrm{Speed }[\theta/s]",
        xlabel = L"\textrm{Time }[s]",
        label = L"\dot{\theta^d}",
        legend = :bottomright,
        xlims = (0, 100),
        ylims = (5, 12),
        formatter = :latex,
    )
    p1 = plot!(t, vp, label = L"\dot{\theta^h}")
    minv = minimum(D[:, 1])
    maxv = maximum(D[:, 1])
    p1 = plot!(
        rectangle(1000, maxv - minv, -100, minv),
        opacity = 0.2,
        label = L"\mathcal{O}",
        formatter = :latex,
    )
    p2 = plot(
        t,
        s,
        title = L"\textrm{Position Uncertainty}",
        ylabel = L"\Sigma_d [\theta]",
        xlabel = L"\textrm{Time }[s]",
        legend = false,
        ylims = (0, 35),
        formatter = :latex,
    )
    p3 = plot(
        gp,
        ylims = 1 .* (-2, 3),
        xlims = (4, 12),
        title = L"\textrm{Gaussian Process}",
        xlabel = L"\dot{\theta^h}\thinspace[\theta/s]",
        ylabel = L"d\thinspace [\theta/s]",
        legend = false,
        formatter = :latex,
    )
    p3 = plot!(
        range(0, stop = 20, length = 1000),
        Deviation.(range(0, stop = 20, length = 1000)),
        linestyle = :dash,
        color = :red,
        linealpha = 0.5,
        label = "True Deviation",
        formatter = :latex,
    )
    l = @layout [a b; c{0.5h}]
    p = plot(p1, p2, p3, layout = l)
    display(p)
    savefig("learning_ex.pdf")
end

function fstar(x::Float64)
    return abs(x - 5) * cos(2 * x)
end

function benchmarkGP()

    n = 50:50:300

    times = zeros(length(n), 2)
    stds = copy(times)
    σy = 10.0
    for i = 1:length(n)
        @show i, n[i]
        Random.seed!(1) # for reproducibility
        Xdistr = Beta(7, 7)
        ϵdistr = Normal(0, σy)
        x = rand(Xdistr, n[i]) * 10
        X = Matrix(x')
        Y = fstar.(x) .+ rand(ϵdistr, n[i])
        k = SEIso(log(0.3), log(5.0))
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
        GPE(X, Y, MeanConst(mean(Y)), k, log(σy))
        t = @benchmark gp_full = GPE($X, $Y, $MeanConst(mean(Y)), $k, $log(σy))
        @show t
        t = t.:times
        t[t.>=3*median(t)] .= median(t)
        times[i, 1] = median(t)
        stds[i, 1] = std(t)
        gp_DTC = GaussianProcesses.DTC(X, Xu, Y, MeanConst(mean(Y)), k, log(σy))
        t = @benchmark gp_DTC = GaussianProcesses.DTC(
            $X,
            $Xu,
            $Y,
            $MeanConst(mean(Y)),
            $k,
            $log(σy),
        )
        @show t
        t = t.:times
        t[t.>=5*median(t)] .= median(t)
        times[i, 2] = median(t)
        stds[i, 2] = std(t)
    end
    p = plot(
        n,
        times[:, 1],
        yerror = stds[:, 1],
        label = L"\textrm{Full GP}",
        legend = :topleft,
        formatter = :latex,
    )
    p = plot!(n, times[:, 2], yerror = stds[:, 2], label = L"\textrm{DTC GP}")
    p = xlabel!(L"N")
    p = ylabel!(L"\textrm{CPU Time }[ns]")
    p = title!(L"\textrm{GPR Computation Times}")
    display(p)
    savefig("gpr_times.pdf")
    return times, stds, n
end

function maxt1(Er = 12000, rmethod = "WorstFirst", tt = 80, ti = 5, dt = 1)
    UASPos = [800, 450]
    LPos = [600, 600]
    #LPos = [800, 450]
    ts = 0
    PNRStat = false
    N = 100
    p = [1, 2]
    ptgt = 2
    dt = 1
    PrevPNR = [0.0, 0.0]
    μ = 60 * ones(2)
    Σ = 10 * ones(2)
    Ns = 5
    UASPosVec = zeros(N, 2)
    #TimeSamples = ts:120
    t = range(0.0, stop = ti, step = dt)
    li = length(t)
    D = zeros(li, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(li)
    gp = LearnDeviationFunction(D, true, "DTC")
    #now we move the driver, collect new data, update gp and sample on the gp
    tv = range(0.0, stop = tt, step = dt)
    l = length(tv)
    D = [D; zeros(l, 2)]
    DriverPos = 0.0
    DriverPosVec = zeros(l)
    DriverPosEstimateVec = zeros(l)
    DriverPosErrorVec = zeros(l)
    PosSamples = zeros(N, length(p))
    OptTimeSample = zeros(length(p))
    sol = nothing
    riska = true

    #data streams
    fi = 0
    t1v = zeros(length(tv), 4)

    for i = 1:l
        #sample driver velocity
        DriverVelSample = DriverVelocity(tv[i]) + NoiseStd * randn(1)[1]
        DriverDeviationSample = DriverVelSample - VelocityPrior(tv[i])
        #add to dataset
        D[li+i, 1] = VelocityPrior(tv[i])
        D[li+i, 2] = DriverDeviationSample
        #re-train gp
        gp = LearnDeviationFunction(D[1:(li+i), :], true, "DTC")
        #sample rendezvous candidates
        UASPosVec[i, :] = UASPos
        TimeSamples = SampleTime(μ, Σ, N)
        elites, ptgt = rankMultiPath(
            TimeSamples,
            UASPos,
            Ns,
            p,
            gp,
            tv[i],
            DriverPos,
            rmethod,
            Er,
            sol,
            riska,
        )
        CEM(μ, Σ, p, elites, OptTimeSample, TimeSamples)
        PosSamples = DriverPosition(tv[i], TimeSamples, gp) .+ DriverPos
        #MPC goes here
        v, t = RendezvousPlanner(
            UASPos,
            LPos,
            OptTimeSample,
            Er,
            tv[i],
            PrevPNR,
            false,
            ptgt,
            gp,
            DriverPos,
        )
        sol = (v, t, LPos)
        pp = plot()
        pp = plot!(UASPosVec[1:i, 1], UASPosVec[1:i, 2], legend = false)
        pp = drawMultiConvexHull(PosSamples, p)
        pp = plotPlan!(
            UASPos,
            LPos,
            v,
            t,
            TimeSamples,
            length(p),
            tv[i],
            DriverPos,
            gp,
        )
        pp = plotTimeSamples!(TimeSamples, p, tv[i], DriverPos, gp)
        pp = scatter!(
            path(DriverPos, ptgt),
            markersize = 6.0,
            label = "",
            markercolor = :green,
        )
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
        if ptgt == 2
            pv = pv2
        else
            pv = pv1
        end
        # if (t[1] <= 1.0 ||
        #    EuclideanDistance(
        #     path(DriverPosition(tv[i], OptTimeSample[ptgt], gp) + DriverPos),
        #     pv[2, :],
        # ) <= 10) && tv[i] > 10
        #     break
        # end
        if (t[1] <= 1.0 && tv[i] > 10)
            break
        end
        #@show t, i, Σ, μ, Er, Ed
        DriverPosVec[i] = DriverPos
        DriverPosEstimateVec[i] = DriverPosition(0.0, tv[i])
        DriverPosErrorVec[i] = DriverPosVec[i] - DriverPosEstimateVec[i]
        DriverPos = DriverPos + DriverVelocity(tv[i]) * dt
        t1v[i, :] = t
        fi = i
        @show i, t[1], Er, ptgt
    end
    @show i
    return t1v[1:fi, :], tv[1:fi]
end

function CEconvergence(ntrajs = 5, niter = 5)
    UASPos = [800, 450]
    LPos = [600, 600]
    ts = 0
    PNRStat = false
    np = [1, 2]
    ptgt = 2
    dt = 1
    N = 100
    Ns = 5
    PosSamples = zeros(N, length(p))
    OptTimeSample = zeros(length(p))
    gp = nothing
    resv = zeros(ntrajs, niter)
    Threads.@threads for j = 1:ntrajs
        μ = 60 * ones(2) .+ 2 * randn(2)
        Σ = 10 * ones(2) .+ 1 * randn(2)
        for i = 1:niter
            #sample rendezvous candidates
            UASPos = [800, 450] .+ 50 * randn(2)
            TimeSamples = SampleTime(μ, Σ, N)
            elites, ptgt =
                rankMultiPath(TimeSamples, UASPos, Ns, np, gp, 0.0, 0.0)
            CEM(μ, Σ, np, elites, OptTimeSample, TimeSamples)
            resv[j, i] = Σ[1]
        end
        @show j
    end
    pp = plot(
        1:niter,
        resv',
        linealpha = 0.1,
        lw = 2,
        label = "",
        linecolor = :green,
        ylims = (0, 6),
    )
    resm = mean(resv, dims = 1)'
    pp = plot!(
        1:niter,
        resm,
        lw = 3,
        linecolor = :black,
        label = L"\textrm{Mean of Trials}",
    )
    pp = hline!(
        [1.0],
        linestyle = :dash,
        linecolor = :red,
        formatter = :latex,
        title = L"\textrm{Importance Sampling Rate of Convergence}",
        xlabel = L"\textrm{Iteration}",
        ylabel = L"\Sigma_\mathcal{A}",
        label = L"\lambda",
    )
    display(pp)
    savefig("ce_con.pdf")
    return resm, resv
end

function iterDriverAndCE(tt = 20, ti = 5, dt = 1)
    #first build an initial dataset and gp
    t = range(0.0, stop = ti, step = dt)
    li = length(t)
    D = zeros(li, 2)
    D[:, 1] = VelocityPrior.(t)
    D[:, 2] = Deviation.(VelocityPrior.(t)) + NoiseStd .* randn(li)
    gp = LearnDeviationFunction(D, true, "DTC")
    #now we move the driver, collect new data, update gp and sample on the gp
    t = range(0.0, stop = tt, step = dt)
    l = length(t)
    D = [D; zeros(l, 2)]
    PosSamples = zeros(N, length(p))
    OptTimeSample = zeros(length(p))
    μ = 60 * ones(2) .+ 2 * randn(2)
    Σ = 10 * ones(2) .+ 1 * randn(2)
    UASPos = [800, 450] .+ 50 * randn(2)
    DriverPos = 0.0
    Σv = zeros(l)
    for i = 1:l
        #sample driver velocity
        DriverVelSample = DriverVelocity(t[i]) + NoiseStd * randn(1)[1]
        DriverDeviationSample = DriverVelSample - VelocityPrior(t[i])
        #add to dataset
        D[li+i, 1] = VelocityPrior(t[i])
        D[li+i, 2] = DriverDeviationSample
        #re-train gp
        gp = LearnDeviationFunction(D[1:(li+i), :], true, "DTC")
        #sample rendezvous candidates
        TimeSamples = SampleTime(μ, Σ, N)
        elites, ptgt =
            rankMultiPath(TimeSamples, UASPos, Ns, [1, 2], gp, t[i], DriverPos)
        CEM(μ, Σ, [1, 2], elites, OptTimeSample, TimeSamples)
        PosSamples = DriverPosition(t[i], TimeSamples, gp) .+ DriverPos
        DriverPos = DriverPos + DriverVelocity(t[i]) * dt
        Σv[i] = Σ[1]
    end
    return Σv[1:end-1]
end

function CEconvergenceMdlUpdate(ntrajs = 5, niter = 50)
    #this variant updates driver model
    ts = 0
    PNRStat = false
    np = [1, 2]
    ptgt = 2
    dt = 1
    N = 100
    Ns = 5
    PosSamples = zeros(N, length(p))
    OptTimeSample = zeros(length(p))
    gp = nothing
    resv = zeros(ntrajs, niter)
    Threads.@threads for j = 1:ntrajs
        Σv = iterDriverAndCE(niter, 5, 1)
        resv[j, :] = Σv'
        @show j
    end
    resm = mean(resv, dims = 1)'
    pp1 = plot(
        1:niter,
        resv',
        linealpha = 0.1,
        lw = 2,
        label = "",
        linecolor = :green,
        formatter = :latex,
    )
    resm = mean(resv, dims = 1)'
    pp1 = plot!(
        1:niter,
        resm,
        lw = 3,
        linecolor = :black,
        label = L"\textrm{Mean of Trials}",
        formatter = :latex,
    )

    pp1 = hline!(
        [1.0],
        linestyle = :dash,
        linecolor = :red,
        formatter = :latex,
        title = L"\textrm{Importance Sampling Performance}",
        xlabel = L"\textrm{Time }[s]",
        ylabel = L"\Sigma_\mathcal{A}",
        label = L"\lambda",
        ylims = (0.75, 1.5),
        xlims = (0, 40),
        xticks = 0:4:40,
    )
    display(pp1)
    savefig("ce_mdl_up.pdf")
    return pp1
end