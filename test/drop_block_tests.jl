using Test
using ConvOptDL: _get_block_mask, DropBlock, dropblock
using Flux
using Zygote: pullback

@testset "DropBlock Test" begin
    mask = reshape(
        [   
            [0, 0, 0, 0, 0]
            [0, 1, 0, 0, 0]
            [0, 0, 0, 0, 0]
            [0, 1, 0, 0, 0]
            [0, 0, 0, 0, 0]
        ],
        5,
        5,
        1,
        1,
    )
    expected = reshape(
        [
            [0, 0, 0, 1, 1]
            [0, 0, 0, 1, 1]
            [0, 0, 0, 1, 1]
            [0, 0, 0, 1, 1]
            [0, 0, 0, 1, 1]
        ],
        5,
        5,
        1,
        1,
    )
    block = DropBlock(3, 0.1)
    block_mask = _get_block_mask(block, rand(5, 5, 1, 1), mask)
    @test block_mask == expected

    testmode!(block)
    x = rand(5, 5, 3, 10)
    @test block(x) == x

    (_, block_mask), back = pullback(dropblock, x, block)
    Δy = ones(size(x))
    Δx, _  = back((Δy, nothing))
    @test Δx == Δy .* block_mask

    block = DropBlock(3, 0)
    testmode!(block)
    Δy, = gradient((x) -> sum(block(x)), x)
    Δŷ, = gradient(sum, x)
    @test Δy == Δŷ

    trainmode!(block)
    Δy, = gradient((x) -> sum(block(x)), x)
    Δŷ, = gradient(sum, x)
    @test Δy != Δŷ
end
