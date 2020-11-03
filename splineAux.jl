

using Plots, LinearAlgebra
using Dierckx

x = [0., 1., 2., 3., 4.]
y = [-1., 0., 7., 26., 63.]  # x.^3 - 1.
spl = Spline1D(x, y)
xv = -1:0.01:4
plot(xv,spl.(xv))
scatter!(x,y)

u = [1., 2., 3., 4.]
x = [1. 2. 3. 4.; 0. -2. 4. 4.]
spl = ParametricSpline(u, x, k=1, s=size(x, 2))

s = spl(1.0:0.01:4.0)
plot(s[1,:],s[2,:])

using Interpolations

t = 0:.1:1
x = sin.(2π*t)
y = cos.(2π*t)
A = hcat(x,y)

itp = Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), t, 1:2)

tfine = 0:.01:1
xs, ys = [itp(t,1) for t in tfine], [itp(t,2) for t in tfine]
using Plots

scatter(x, y, label="knots")
plot!(xs, ys, label="spline")
