using Flux

export ResNetBasicBlock

struct ResNetBasicBlock
    pre_block::Any
    downsample_block::Any
    post_block::Any
    drop_block::Any
end

function ResNetBasicBlock(
    chs::Pair{<:Integer,<:Integer};
    init = Flux.kaiming_normal,
    drop_prob = 0.0,
    block_size = 3,
    pool_stride = 2,
    downsample = false,
)
    in_channel, out_channel = chs
    pre_block = Chain(
        Conv((3, 3), in_channel => out_channel, init = init, stride = 1, pad = 1),
        BatchNorm(out_channel, (x) -> leakyrelu(x, Float32(0.1))),
        Conv((3, 3), out_channel => out_channel, init = init, stride = 1, pad = 1),
        BatchNorm(out_channel, (x) -> leakyrelu(x, Float32(0.1))),
        Conv((3, 3), out_channel => out_channel, init = init, stride = 1, pad = 1),
        BatchNorm(out_channel),
    )
    downsample_block = downsample ?
        Chain(Conv((1, 1), in_channel => out_channel, stride = 1), BatchNorm(out_channel)) :
        identity
    post_block =
        Chain((x) -> leakyrelu(x, Float32(0.1)), MaxPool((pool_stride, pool_stride), stride = pool_stride))
    drop_block = DropBlock(block_size, 1 - drop_prob)
    if drop_prob > 0
        Flux.trainmode!(drop_block)
    else
        Flux.testmode!(drop_block)
    end
    ResNetBasicBlock(pre_block, downsample_block, post_block, drop_block)
end

function (block::ResNetBasicBlock)(x)
    residule = block.downsample_block(x)
    out = block.pre_block(x)
    out = out .+ residule
    out = block.post_block(out)
    out = block.drop_block(out)
    out
end

Flux.testmode!(m::ResNetBasicBlock, mode = true) = (
    map(
        x -> Flux.testmode!(x, mode),
        (m.pre_block, m.downsample_block, m.post_block, m.drop_block),
    );
    m
)

Flux.functor(::Type{<:ResNetBasicBlock}, m) =
    (m.pre_block, m.downsample_block, m.post_block, m.drop_block),
    blocks -> ResNetBasicBlock(blocks...)
