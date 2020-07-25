using Test
using ConvOptDL.Utils

@testset "Utils Test (batched_outer)" begin
    A = hcat([ones(4)*i for i in 1:5]...)
    B = hcat([ones(3)*i for i in 1:5]...)
    refC = ones(4, 3, 5)
    for i in 1:5
        refC[:, :, i] .*= i^2
    end
    @test batched_outer(A, B) == refC
end

@testset "Utils Test (batched_kronecker)" begin
    A = [
        1 2;
        3 4;
    ]
    B = ones(3, 2)
    ans_ = repeat(repeat(A, inner=(3, 2)), outer=(1, 1, 5))
    @test batched_kronecker(repeat(A, outer=(1, 1, 5)), repeat(B, outer=(1, 1, 5))) == ans_
end