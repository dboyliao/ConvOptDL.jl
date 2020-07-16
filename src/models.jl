using Distributions
export DropBlock

"""
https://arxiv.org/pdf/1810.12890.pdf
"""
mutable struct DropBlock
    block_size::Integer
    training::Bool
    function DropBlock(block_size, training=true)
        new(block_size, training)
    end
end

function _get_mask(block::DropBlock, x, gamma)
    mask = ones(eltype(x), size(x))
    height, width, channel, batch = size(x)
    zero_pos = rand(Bernoulli(gamma), height, width, channel, batch) |> 
        Base.Fix1(findall, (x) -> x > 0)
    height_offset = width_offset = floor(Integer, block.block_size / 2)
    for pos in zero_pos
        c_height, c_width, c_channel, c_batch = pos.I
        height_start = max(c_height-height_offset, 1)
        height_end = min(c_height+height_offset, height)
        width_start = max(c_width-width_offset, 1)
        width_end = min(c_width+width_offset, width)
        mask[
            height_start:height_end,
            width_start:width_end,
            c_channel,
            c_batch,
        ] .= 0
    end
    return mask
end

function (block::DropBlock)(x::AbstractArray{T, 4}, gamma::AbstractFloat) where {T <: Real}
    if block.training
        mask = _get_mask(block, x, gamma)
        ret = x .* mask .* (length(mask) / sum(mask))
    else
        ret = x
    end
    return ret
end