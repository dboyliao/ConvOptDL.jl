using ConvOptDL
using Test
using StatsBase

@testset "data_loader.jl" begin
    dloader = FewShotDataLoader("./test_data.jls")
    batch = sample(dloader, 2, support_n_ways=3, support_k_shots=5)
    @test size(batch) == 2
    @test size(batch.support_samples) == (32, 32, 3, 15, 2)
    @test size(batch.support_labels) == (15, 2)
    try
        _ = sample(dloader, 5, support_n_ways=3, support_k_shots=5, query_n_ways=5, query_k_shots=2)
    catch err
        @test err.msg == false
    end
end