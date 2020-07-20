using Flux

export BasicBlock

struct BasicBlock
    pre_block::Any
    downsample_block::Any
    drop_block::Any
end

function BasicBlock(
    chs::Pair{<:Integer,<:Integer};
    init = Flux.kaiming_normal,
    drop_prob = 0.0,
    use_dropblock = false,
    block_size = 3,
    pool_stride = 2,
)
    in_channel, out_channel = chs
    pre_block = Chain(
        Conv((3, 3), in_channel => out_channel, init = init, stride = 1, pad = 1),
        BatchNorm(out_channel, (x) -> leakyrelu(x, Float32(0.1))),
        Conv((3, 3), out_channel => out_channel, init = init, stride = 1, pad = 1),
        BatchNorm(out_channel, (x) -> leakyrelu(x, Float32(0.1))),
        Conv((3, 3), out_channel => out_channel, init = init, stride = 1, pad = 1),
        BatchNorm(out_channel),
        (x) -> leakyrelu.(x, Float32(0.1)),
        MaxPool((pool_stride, pool_stride), pad = SamePad(), stride = pool_stride),
    )
    downsample_block = in_channel != out_channel || pool_stride != 1 ?
        Chain(
        Conv((1, 1), in_channel => out_channel, stride = pool_stride, pad = SamePad()),
        BatchNorm(out_channel),
    ) :
        identity
    drop_block = use_dropblock ? DropBlock(block_size, 1 - drop_prob) : Dropout(drop_prob)
    if drop_prob > 0
        Flux.trainmode!(drop_block)
    else
        Flux.testmode!(drop_block)
    end
    BasicBlock(pre_block, downsample_block, drop_block)
end

function (block::BasicBlock)(x)
    residule = block.downsample_block(x)
    out = block.pre_block(x)
    out = out .+ residule
    out = block.drop_block(out)
    out
end

Flux.testmode!(m::BasicBlock, mode = true) = (
    map(
        x -> Flux.testmode!(x, mode),
        (m.pre_block, m.downsample_block, m.post_block, m.drop_block),
    );
    m
)

Flux.functor(::Type{<:BasicBlock}, m) =
    (m.pre_block, m.downsample_block, m.post_block, m.drop_block),
    blocks -> BasicBlock(blocks...)
