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
- `G`: array of size (`num_ineq`, `dim_x`), can be empty
- `h`: array of size (`num_ineq`,), can be empty
- `A`: array of size (`num_eq`, `dim_x`), can be empty
- `b`: array of size (`num_eq`,), can be empty

# Return

- `x`: array of size (`dim_x`,) which is the solution to the given QP
- `λs_`: adjoint values of inequality constraints
- `νs_`: adjoint values of equality constraints
"""
function solve_qp(Q, p, G, h, A, b, optimizer = SCS.Optimizer; kwargs...)
    !(isempty(G) ⊻ isempty(h)) ||
        throw(ArgumentError("G and h should be both empty or not empty"))
    !(isempty(A) ⊻ isempty(b)) ||
        throw(ArgumentError("A and b should be both empty or not empty"))
    # solve qp and return x
    dim_x = size(Q, 1)
    x = Variable(dim_x)
    constraints = Dict{String,Constraint}()
    if !isempty(A)
        constraints["Eq"] = (A * x == b)
    end
    if !isempty(G)
        constraints["InEq"] = (G * x ≤ h)
    end
    obj = 0.5 * quadform(x, Q) + p'x
    problem = minimize(obj, collect(values(constraints)))
    solve!(problem, optimizer; kwargs...)
    @assert problem.status == Convex.MOI.OPTIMAL
    x_ = isa(x.value, AbstractArray) ? dropdims(x.value, dims = 2) : [x.value]
    if haskey(constraints, "InEq")
        cond_ineq = constraints["InEq"]
        λs_ = isa(cond_ineq.dual, AbstractArray) ? dropdims(cond_ineq.dual, dims = 2) :
            [cond_ineq.dual]
    else
        λs_ = similar(G, 0)
    end
    if haskey(constraints, "Eq")
        cond_eq = constraints["Eq"]
        νs_ = isa(cond_eq.dual, AbstractArray) ? dropdims(abs.(cond_eq.dual), dims = 2) :
            [abs(cond_eq.dual)]
    else
        νs_ = similar(A, 0)
    end
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
function solve_qp_batch(
    Q::AbstractArray{T,3},
    p::AbstractArray{T,2},
    G::AbstractArray{T,3},
    h::AbstractArray{T,2},
    A::AbstractArray{T,3},
    b::AbstractArray{T,2},
    optimizer = SCS.Optimizer,
) where {T<:Real}
    num_batch = size(Q, 3)
    dim_x = size(Q, 1)
    num_eq = size(A, 1)
    num_ineq = size(G, 1)
    X = similar(p)
    λs = similar(h)
    νs = similar(b)
    for i = 1:num_batch
        @inbounds Q_ = Q[:, :, i]
        @inbounds p_ = p[:, i]
        @inbounds G_ = G[:, :, i]
        @inbounds h_ = h[:, i]
        @inbounds A_ = A[:, :, i]
        @inbounds b_ = b[:, i]
        x, λs_, νs_ = solve_qp(Q_, p_, G_, h_, A_, b_, optimizer, verbose = false)
        @inbounds X[:, i] .= x
        if !isempty(λs_)
            @inbounds λs[:, i] .= λs_
        end
        if !isempty(νs_)
            @inbounds νs[:, i] .= νs_
        end
    end
    return X, λs, νs
end

function solve_qp_batch_back(
    Q::AbstractArray{T,3},
    p::AbstractArray{T,2},
    G::AbstractArray{T,3},
    h::AbstractArray{T,2},
    A::AbstractArray{T,3},
    b::AbstractArray{T,2},
    optimizer = SCS.Optimizer,
) where {T<:Real}
    X, λs, νs = solve_qp_batch(Q, p, G, h, A, b, optimizer)
    num_batch = size(X, 2)
    function backward(Δ)
        ΔX, Δλs, Δνs = Δ
        ΔQ = similar(Q)
        Δp = similar(p)
        ΔG = similar(G)
        Δh = similar(h)
        ΔA = similar(A)
        Δb = similar(b)
        # solve dQ, dp, dG, dh, dA, db
        for i = 1:num_batch
            @inbounds x = X[:, i]
            @inbounds λ = λs[:, i]
            @inbounds ν = νs[:, i]
            dx, dλ, dν = solve_kkt(Q, G, h, A, ΔX[:, i], λ, x)
            @inbounds ΔQ[:, :, i] .= 0.5 * (dx * x' + x * dx') # dQ
            @inbounds Δp[:, i] .= dx # dp
            @inbounds ΔG[:, :, i] .= Diagonal(λ) * (dλ * x' + λ * dx') # dG
            @inbounds Δh[:, i] .= -Diagonal(λ) * dλ # dh
            @inbounds ΔA[:, :, i] .= (dν * x' + ν * dx') # dA
            @inbounds Δb[:, i] .= -dν # db
        end
        Δoptimizer = nothing
        (ΔQ, Δp, ΔG, Δh, ΔA, Δb, Δoptimizer)
    end
    return (X, λs, νs), backward
end


function solve_kkt(Q, G, h, A, λ, x, Δx)
    # TODO: G, h or A are empty
    # TODO: implement primal-dual interior point method (PDIPM) for solving kkt, better GPU utilization
    M = kkt_matrix(Q, G, h, A, λ, x)
    sol = pinv(-M) * Δx
    dx = sol[1:length(x)]
    dλ = sol[length(x)+1:length(x)+length(λ)]
    dν = sol[length(x)+length(λ)+1:end]
    return dx, dλ, dν
end

function kkt_matrix(Q, G, h, A, λ, x)
    # TODO: G, h or A are empty
    return [
        Q G' * Diagonal(λ) A'
        G Diagonal(G * x - h) zeros(size(G, 1), size(A, 1))
        A zeros(size(A, 1), size(h, 1)) zeros(size(A, 1), size(A, 1))
    ]
end
