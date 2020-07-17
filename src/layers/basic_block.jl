using Flux

export ResNetBasicBlock

struct ResNetBasicBlock
    init
    keep_rate::AbstractFloat
    block_size::Integer
    stride::Integer
    pad::Integer

end

function ResNetBasicBlock(init=Flux.kaiming_normal, keep_rate=0.0, block_size=3)
    ResNetBasicBlock(init, keep_rate, block_size, 2, 1)
end

function (block::ResNetBasicBlock)(chs::Pair{Integer, Integer}, downsample=nothing)
    in_channel, out_channel = chs
    layers = [
        Conv((3, 3), in_channel => out_channel, init = block.init, stride = block.stride, pad = block.pad),
        BatchNorm(out_channel, (x) -> leakyrelu(x, 0.1)),
        Conv((3, 3), out_channel => out_channel, init = block.init, stride = block.stride, pad = block.pad),
        BatchNorm(out_channel),
        Conv((3, 3), out_channel => out_channel, init = block.init, stride = block.stride, pad = block.pad),
        BatchNorm(out_channel),
        MaxPool((2, 2)),
    ]
    if !isnothing(downsample)
        
    end
end
