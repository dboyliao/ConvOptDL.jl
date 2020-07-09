using Test
using Convex:MOI
import SCS

@testset "qp.jl: Solve QP Test 1" begin
    Q = Float64[[3 1]; [1 1]]
    p = Float64[1, 6]
    G = Float64[[-2 -3]; [-1 0]; [0 -1]]
    h = Float64[-4, 0, 0]
    A = Float64[1 1]
    b = Float64[1.5]
    x, λ, ν = solve_qp(
        Q,
        p,
        G,
        h,
        A,
        b,
    )
    @test size(x) == (2,)
    @test size(λ) == (3,)
    @test size(ν) == (1,)
    @test x ≈ [0.5, 1.0] atol = 1e-5
    @test λ ≈ [4.0, 0.0, 0.0] atol = 2e-5
    @test ν ≈ [4.5] atol = 3e-5
end

@testset "qp.jl: Solve QP Test 2" begin
    Q = Float64[[3 1]; [1 1]]
    p = Float64[1, 6]
    G = Float64[[-2 -3]; [-1 0]; [0 -1]]
    h = Float64[-4, 0, 0]
    A = Float64[1 1]
    b = Float64[1.5]
    optimizer = SCS.Optimizer()
    MOI.set(optimizer, MOI.RawParameter("rho_x"), 1e-5)
    x, λ, ν = solve_qp(
        Q,
        p,
        G,
        h,
        A,
        b,
        optimizer,
    )
    @test size(x) == (2,)
    @test size(λ) == (3,)
    @test size(ν) == (1,)
    @test x ≈ [0.5, 1.0] atol = 1e-10
    @test λ ≈ [4.0, 0.0, 0.0] atol = 5e-10
    @test ν ≈ [4.5] atol = 5e-10
end
