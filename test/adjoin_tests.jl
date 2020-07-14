using Test
using Zygote

# https://optimization.mccormick.northwestern.edu/index.php/Quadratic_programming#Numerical_example
Qs = reshape([[3.0 1.0]; [1.0 1.0]], 2, 2, 1)
ps = reshape([1.0, 6.0], 2, 1)
Gs = reshape([[-2.0 -3.0]; [-1.0 0.0]; [0.0 -1.0]], 3, 2, 1)
hs = reshape([-4.0, 0.0, 0.0], 3, 1)
As = reshape([1.0 1.0], 1, 2, 1)
bs = reshape([1.5], 1, 1)


# @testset "QP Sovler Adjoint" begin
#     _, back = pullback(solve_qp_batch, Qs, ps, Gs, hs, As, bs)
# end
