export solve_qp

using Convex
using Zygote: @adjoint

"""
Solving QP

Given `Q`, `p`, `G`, `h`, `A` and `b`, it solves following convex optimization problem:

```math
min x'Qx + p'x
s.t Gx ≤ h
    Ax = b
```

# Parameters

- `Q`: array of size (`num_batch`, `dim_x`, `dim_x`)
- `p`: array of size (`num_batch`, `dim_x`, `1`)
- `G`: array of size (`num_batch`, `num_ineq`, `dim_x`)
- `h`: array of size (`num_batch`, `num_ineq`, `1`)
- `A`: array of size (`num_batch`, `num_eq`, `dim_x`)
- `b`: array of size (`num_batch`, `num_eq`, `1`)

# Return

- `x`: array of size (`num_batch`, `dim_x`, `1`) which is the solution to the given QP
"""
function solve_qp(Q, p, G, h, A, b)
    # solve qp and return x
    num_batch = size(Q, 1)
    dim_x = size(Q, 2)
    num_eq = size(A, 2)
    num_ineq = size(G, 2)
    X = Array{Float64, 3}(undef, num_batch, dim_x, 1)
    λs = Array{Float64, 3}(undef, num_batch, num_ineq, 1)
    νs = Array{Float64, 3}(undef, num_batch, num_eq, 1)
    for i in 1:num_batch
        x = Variable(dim_x)
    end
    return X, λs, νs
end

@adjoint function solve_qp(Q, p, G, h, A, b)
    X, λs, νs = solve_qp(Q, p, G, h, A, b)
    function backward(ΔX, Δλs)
        # solve dQ, dp, dG, dh, dA, db
        ΔQ = nothing
        Δp = nothing
        ΔG = nothing
        Δh = nothing
        ΔA = nothing
        Δb = nothing
        (nothing, nothing, nothing, nothing, nothing, nothing)
    end
    return (X, λs, νs), backward
end
