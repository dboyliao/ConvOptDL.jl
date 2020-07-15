using Test
using Zygote
using Serialization
using ConvOptDL: solve_qp_batch_with_back

# https://optimization.mccormick.northwestern.edu/index.php/Quadratic_programming#Numerical_example
Qs = reshape([[3.0 1.0]; [1.0 1.0]], 2, 2, 1)
ps = reshape([1.0, 6.0], 2, 1)
Gs = reshape([[-2.0 -3.0]; [-1.0 0.0]; [0.0 -1.0]], 3, 2, 1)
hs = reshape([-4.0, 0.0, 0.0], 3, 1)
As = reshape([1.0 1.0], 1, 2, 1)
bs = reshape([1.5], 1, 1)


@testset "QP Adjoint Test 1" begin
    _, back = solve_qp_batch_with_back(Qs, ps, Gs, hs, As, bs)
    ΔX = ones(2, 1)
    Δλs = zeros(3, 1)
    Δνs = zeros(1, 1)
    Δ = (ΔX, Δλs, Δνs)
    ΔQs, Δps, ΔGs, Δhs, ΔAs, Δbs, _ = back(Δ)
    ref_grads = Serialization.deserialize("ref_grads.jls")
    @test ΔQs ≈ ref_grads["dQs"] atol=1e-16
    @test Δps ≈ ref_grads["dps"] atol=1e-16
    @test ΔGs ≈ ref_grads["dGs"] atol=1e-14
    @test Δhs ≈ ref_grads["dhs"] atol=1e-16
    @test ΔAs ≈ ref_grads["dAs"] atol=1e-5
    @test Δbs ≈ ref_grads["dbs"] atol=1e-16
end
