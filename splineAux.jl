

using Plots, LinearAlgebra

function cubic_spline(t, x)

    #create matrix
    N = length(x)
    delta = abs(x[1] - x[2])
    A = zeros(N, N)
    B = zeros(N)
    for i = 1:1:N
        if i == 1
            A[1, 1] = 1
            B[1] = 0
        elseif i == N
            A[N, N] = 1
            B[N] = 0
        else
            A[i, i-1] = 1 / 6
            A[i, i] = 2 / 3
            A[i, i+1] = 1 / 6
            B[i] = (x[i+1] - 2 * x[i] + x[i-1]) / (delta^2)
        end
    end

    #solve matrix
    G = A \ (B)
    j(yy) = min(floor((yy - minimum(x)) / (x[2] - x[1])) + 1, length(x) - 1)
    f(xx) =
        (G[j(xx)] / 6) * (((x[j(xx)+1] - xx)^3) / delta - delta * (x[j(xx)+1] - xx)) +
        (G[j(xx)+1] / 6) * (((xx - x[j(xx)])^3) / delta - delta * (xx - x[j(xx)])) +
        y[j(xx)] * (x[j(xx)+1] - xx) / delta +
        y[j(xx)+1] * (xx - x[j(xx)]) / delta
end
