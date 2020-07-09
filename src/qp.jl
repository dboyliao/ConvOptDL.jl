export solve_qp, solve_qp_batch

using Convex
using Zygote: @adjoint
using SCS
using LinearAlgebra

"""
Solving QP

Given `Q`, `p`, `G`, `h`, `A` and `b`, it solves following convex optimization problem:

```math
min 0.5*x'Qx + p'x
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
- `λs_`: adjoint values of inequality constraints
- `νs_`: adjoint values of equality constraints
"""
function solve_qp(Q, p, G, h, A, b, optimizer = SCS.Optimizer; kwargs...)
    # solve qp and return x
    dim_x = size(Q, 1)
    num_ineq = size(G, 1)
    x = Variable(dim_x)
    cond_eq = (A * x == b)
    cond_ineq = (G * x ≤ h)
    obj = 0.5 * quadform(x, Q) + p'x
    problem = minimize(obj, [cond_ineq, cond_eq])
    solve!(problem, optimizer; kwargs...)
    @assert problem.status == Convex.MOI.OPTIMAL
    x_ = isa(x.value, AbstractArray) ? dropdims(x.value, dims = 2) : [x.value]
    λs_ = isa(cond_ineq.dual, AbstractArray) ? dropdims(cond_ineq.dual, dims = 2) :
        [cond_ineq.dual]
    # dual value should be postive
    # however, Convex.jl sometime gives negative dual, I don't know why.
    νs_ = isa(cond_eq.dual, AbstractArray) ? dropdims(abs.(cond_eq.dual), dims = 2) :
        [abs(cond_eq.dual)]
    return x_, λs_, νs_
end


"""
Solving QP in batch

# Parameters

- `Q`: array of size (`dim_x`, `dim_x`, `num_batch`)
- `p`: array of size (`dim_x`, `num_batch`)
- `G`: array of size (`num_ineq`, `dim_x`, `num_batch`)
- `h`: array of size (`num_ineq`, `num_batch`)
- `A`: array of size (`num_eq`, `dim_x`, `num_batch`)
- `b`: array of size (`num_eq`, `num_batch`)

# Return

- `X`: array of size (`dim_x`, `num_batch`) which are the solutions to the given QP batch
- `λs`: array of size (`dim_ineq`, `num_batch`) which are the adjoint values of inequality constraints
- `νs`: array of size (`dim_eq`, `num_batch`) which are the adjoint values of equality constraints
"""
function solve_qp_batch(Q, p, G, h, A, b, optimizer = SCS.Optimizer)
    num_batch = size(Q, 3)
    dim_x = size(Q, 1)
    num_eq = size(A, 1)
    num_ineq = size(G, 1)
    X = Array{Float64,3}(undef, dim_x, num_batch)
    λs = Array{Float64,3}(undef, num_ineq, num_batch)
    νs = Array{Float64,3}(undef, num_eq, num_batch)
    @inbounds for i = 1:num_batch
        Q_ = Q[:, :, i]
        p_ = p[:, i]
        G_ = G[:, :, i]
        h_ = h[:, :, i]
        A_ = A[:, :, i]
        b_ = b[:, i]
        x, λs_, νs_ = solve_qp(Q_, p_, G_, h_, A_, b_, optimizer, verbose = false)
        X[:, i] .= x
        λs[:, i] .= λs_
        νs[:, i] .= νs_
    end
    return X, λs, νs
end

@adjoint function solve_qp_batch(Q, p, G, h, A, b, optimizer = SCS.Optimizer)
    X, λs, νs = solve_qp_batch(Q, p, G, h, A, b, optimizer, verbose = false)
    function backward(ΔX, Δλs, Δνs)
        ΔQ = similar(Q)
        Δp = similar(p)
        ΔG = similar(G)
        Δh = similar(h)
        ΔA = similar(A)
        Δb = similar(b)
        # solve dQ, dp, dG, dh, dA, db
        num_batch = size(X, 2)
        for i = 1:num_batch
            x = X[:, i]
            λ = λs[:, i]
            ν = νs[:, i]
            dx, dλ, dν = solve_kkt(Q, G, h, A, ΔX[:, i], λ, x)
            ΔQ[:, :, i] .= 0.5 * (dx * x' + x * dx') # dQ
            Δp[:, i] .= dx # dp
            ΔG[:, :, i] .= Diagonal(λ) * (dλ * x' + λ * dx') # dG
            Δh[:, i] .= -Diagonal(λ) * dλ # dh
            ΔA[:, :, i] .= (dν * x' + ν * dx') # dA
            Δb[:, i] .= -dν # db
        end
        Δoptimizer = nothing
        (ΔQ, Δp, ΔG, Δh, ΔA, Δb, Δoptimizer)
    end
    return (X, λs, νs), backward
end

"""
# TODO
- implement primal-dual interior point method (PDIPM) for solving kkt
    - better GPU utilization
"""
function solve_kkt(Q, G, h, A, λ, x, Δx)
    M = kkt_matrix(Q, G, h, A, λ, x)
    sol = qr(-M, Val(true)) \ Δx
    dx = sol[1:length(x)]
    dλ = sol[length(x)+1:length(x)+length(λ)]
    dν = sol[length(x)+length(λ)+1:end]
    return dx, dλ, dν
end

function kkt_matrix(Q, G, h, A, λ, x)
    return [
        Q G' * Diagonal(λ) A'
        G Diagonal(G * x - h) zeros(size(G, 1), size(A, 1))
        A zeros(size(A, 1), size(h, 1)) zeros(size(A, 1), size(A, 1))
    ]
end
