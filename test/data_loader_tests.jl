using ConvOptDL
using Test
using StatsBase

@testset "data_loader.jl" begin
    dloader = FewShotDataLoader("./test_data.jls")
    meta_sample = sample(dloader, 2)
    @test size(meta_sample) == 2
end