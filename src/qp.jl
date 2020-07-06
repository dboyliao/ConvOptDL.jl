export solve_qp

using Convex
using Zygote: @adjoint
using SCS

"""
Solving QP

Given `Q`, `p`, `G`, `h`, `A` and `b`, it solves following convex optimization problem:

```math
min x'Qx + p'x
s.t Gx ≤ h
    Ax = b
```

# Parameters

- `Q`: array of size (`dim_x`, `dim_x`)
- `p`: array of size (`dim_x`,)
- `G`: array of size (`num_ineq`, `dim_x`)
- `h`: array of size (`num_ineq`,)
- `A`: array of size (`num_eq`, `dim_x`)
- `b`: array of size (`num_eq`,)

# Return

- `x`: array of size (`dim_x`,) which is the solution to the given QP
"""
function solve_qp(Q, p, G, h, A, b)
    # solve qp and return x
    dim_x = size(Q, 1)
    num_ineq = size(G, 1)
    x = Variable(dim_x)
    ζs = Variable(num_ineq)
    cond_ζs = (ζs >= 0)
    cond_eq = (A*x == b)
    cond_ineq = (G*x + ζs == h)
    obj = 0.5*quadform(x, Q) + p'x
    problem = minimize(obj, [cond_ζs, cond_ineq, cond_eq])
    solve!(problem, SCS.Optimizer)
    @assert problem.status == Convex.MOI.OPTIMAL
    λs_ = cond_ineq.dual
    νs_ = cond_eq.dual
    return x.value, λs_, νs_
end

function solve_qp_batch(Q, p, G, h, A, b)
    num_batch = size(Q, 1)
    dim_x = size(Q, 2)
    num_eq = size(A, 2)
    num_ineq = size(G, 2)
    X = Array{Float64, 3}(undef, num_batch, dim_x, 1)
    λs = Array{Float64, 3}(undef, num_batch, num_ineq, 1)
    νs = Array{Float64, 3}(undef, num_batch, num_eq, 1)
    for i = 1:num_batch
        Q_ = Q[i, :, :]
        p_ = p[i, :]
        G_ = G[i, :, :]
        h_ = h[i, :, :]
        A_ = A[i, :, :]
        b_ = b[i, :]
        x, λs_, νs_ = solve_qp(Q_, p_, G_, h_, A_, b_)
        X[i, :] .= x
        λs[i, :] .= λs_
        νs[i, :] .= νs_
    end
    return X, λs, νs
end

function _solve_kkt()
end

@adjoint function solve_qp_batch(Q, p, G, h, A, b)
    X, λs, νs = solve_qp_batch(Q, p, G, h, A, b)
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
