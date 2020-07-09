using Test
using ConvOptDL.Utils

@testset "Utils Test" begin
    A = hcat([ones(4)*i for i in 1:5]...)
    B = hcat([ones(3)*i for i in 1:5]...)
    refC = ones(4, 3, 5)
    for i in 1:5
        refC[:, :, i] .*= i^2
    end
    @test batch_outer(A, B) == refC
end