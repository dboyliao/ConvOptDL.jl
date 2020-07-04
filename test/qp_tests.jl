using Test
using Convex: MOI
import SCS

# https://optimization.mccormick.northwestern.edu/index.php/Quadratic_programming#Numerical_example
Q = Float64[[3 1]; [1 1]]
p = Float64[1, 6]
G = Float64[[-2 -3]; [-1 0]; [0 -1]]
h = Float64[-4, 0, 0]
A = Float64[1 1]
b = Float64[1.5]

@testset "qp.jl: Solve QP Test 1" begin
    x, λ, ν = solve_qp(Q, p, G, h, A, b)
    @test size(x) == (2,)
    @test size(λ) == (3,)
    @test size(ν) == (1,)
    @test x ≈ [0.5, 1.0] atol = 1e-5
    @test λ ≈ [4.0, 0.0, 0.0] atol = 2e-5
    @test ν ≈ [4.5] atol = 3e-5
end

@testset "qp.jl: Solve QP Test 2" begin
    optimizer = SCS.Optimizer()
    MOI.set(optimizer, MOI.RawParameter("rho_x"), 1e-5)
    x, λ, ν = solve_qp(Q, p, G, h, A, b, optimizer, verbose = false)
    @test size(x) == (2,)
    @test size(λ) == (3,)
    @test size(ν) == (1,)
    @test x ≈ [0.5, 1.0] atol = 1e-10
    @test λ ≈ [4.0, 0.0, 0.0] atol = 5e-10
    @test ν ≈ [4.5] atol = 5e-10
end

@testset "qp.jl: Solve QP Test 3" begin
    _, λ, ν = solve_qp(Q, p, G, h, Float64[], Float64[])
    @test isempty(ν)
    @test !isempty(λ)
end

@testset "qp.jl: Solve QP Test 4" begin
    _, λ, ν = solve_qp(Q, p, Float64[], Float64[], A, b)
    @test isempty(λ)
    @test !isempty(ν)
end

@testset "qp.jl: Solve Batch QP" begin
    Qs = reshape(repeat(Q, 1, 5), 2, 2, 5)
    ps = repeat(p, 1, 5)
    Gs = reshape(repeat(G, 1, 5), 3, 2, 5)
    hs = repeat(h, 1, 5)
    As = reshape(repeat(A, 1, 5), 1, 2, 5)
    bs = repeat(b, 1, 5)
    X, λs, νs = solve_qp_batch(Qs, ps, Gs, hs, As, bs)
    @test size(X) == (2, 5)
    @test size(λs) == (3, 5)
    @test size(νs) == (1, 5)
    @test X ≈ repeat([0.5, 1.0], 1, 5) atol = 5e-6
    @test λs ≈ repeat([4.0, 0.0, 0.0], 1, 5) atol = 5e-5
    @test νs ≈ repeat([4.5], 1, 5) atol = 1e-4
end
