using ConvOptDL
using Test
using StatsBase

@testset "data_loader.jl" begin
    # Write your tests here.
    dloader = FewShotDataLoader("./test_data.jls")
    meta_samples = sample(dloader, 2)
    @test length(meta_samples) == 2
    meta_sample = sample(dloader)
    @test intersect(meta_sample.train_labels, meta_sample.test_labels) |> isempty
end