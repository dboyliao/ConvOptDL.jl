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

    isempty(A) || (constraints["Eq"] = (A * x == b))
    isempty(G) || (constraints["InEq"] = (G * x ≤ h))

    obj = 0.5 * quadform(x, Q) + p'x
    problem = minimize(obj, collect(values(constraints)))
    null_dest = Sys.isunix() ? "/dev/null" : "nul" # assume windows
    open(null_dest, "w") do stream_null
        redirect_stdout(stream_null) do
            redirect_stderr(stream_null) do
                solve!(problem, optimizer; kwargs...)
            end
        end
    end
    # problem.status == Convex.MOI.OPTIMAL || @warn "Non optimal status: $(problem.status)"
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
        νs_ = isa(cond_eq.dual, AbstractArray) ? dropdims(cond_eq.dual, dims = 2) :
            [cond_eq.dual]
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
    @inbounds for i = 1:num_batch
        Q_ = @view Q[:, :, i]
        G_ = @view G[:, :, i]
        p_ = @view p[:, i]
        h_ = @view h[:, i]
        A_ = @view A[:, :, i]
        b_ = @view b[:, i]
        x, λs_, νs_ = solve_qp(Q_, p_, G_, h_, A_, b_, optimizer)
        view(X, :, i) .= x
        isempty(λs_) || (view(λs, :, i) .= λs_)
        isempty(νs_) || (view(νs, :, i) .= νs_)
    end
    return X, λs, νs
end

@adjoint function solve_qp_batch(
    Q::AbstractArray{T,3},
    p::AbstractArray{T,2},
    G::AbstractArray{T,3},
    h::AbstractArray{T,2},
    A::AbstractArray{T,3},
    b::AbstractArray{T,2},
    optimizer = SCS.Optimizer,
) where {T<:Real}
    return solve_qp_batch_with_back(Q, p, G, h, A, b, optimizer)
end

function solve_qp_batch_with_back(
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
        @inbounds for i = 1:num_batch
            ν_ = @view νs[:, i]
            Q_ = @view Q[:, :, i]
            G_ = @view G[:, :, i]
            h_ = @view h[:, i]
            A_ = @view A[:, :, i]
            ΔX_ = @view ΔX[:, i]
            λ_ = @view λs[:, i]
            x_ = @view X[:, i]
            dx, dλ, dν = solve_kkt(Q_, G_, h_, A_, λ_, x_, ΔX_)
            view(ΔQ, :, :, i) .= 0.5 * (dx * x_' + x_ * dx') # dQ
            view(Δp, :, i) .= dx # dp
            view(ΔG, :, :, i) .= Diagonal(λ_) * (dλ * x_' + λ_ * dx') # dG
            view(Δh, :, i) .= -Diagonal(λ_) * dλ # dh
            view(ΔA, :, :, i) .= (dν * x_' + ν_ * dx') # dA
            view(Δb, :, i) .= -dν # db
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
    ndim = size(M, 1)
    Δl = [Δx; zeros(ndim - size(Δx, 1))]
    sol = pinv(-M) * Δl
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
