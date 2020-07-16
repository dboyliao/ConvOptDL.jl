using Distributions
using Flux: _isactive
import Flux: testmode!
export DropBlock

"""
https://arxiv.org/pdf/1810.12890.pdf
"""
mutable struct DropBlock
    block_size::Integer
    keep_prop::AbstractFloat
    active::Union{Bool,Nothing}
    function DropBlock(block_size, keep_prop)
        @assert 0 ≤ keep_prop ≤ 1
        new(block_size, keep_prop, nothing)
    end
end

function _get_mask(block::DropBlock, x, gamma)
    mask = ones(eltype(x), size(x))
    height, width, channel, batch = size(x)
    height_offset = width_offset = floor(Integer, block.block_size / 2)
    zero_pos =
        rand(
            Bernoulli(gamma),
            height - height_offset * 2,
            width - width_offset * 2,
            channel,
            batch,
        ) |> Base.Fix1(findall, (x) -> x > 0)
    for pos in zero_pos
        c_height, c_width, c_channel, c_batch = pos.I
        height_start = max(c_height, height_offset)
        height_end = min(height_start + block.block_size - 1, height)
        width_start = max(c_width, width_offset)
        width_end = min(width_start + block.block_size - 1, width)
        mask[height_start:height_end, width_start:width_end, c_channel, c_batch] .= 0
    end
    return mask
end

function (block::DropBlock)(x::AbstractArray{T,4}) where {T<:Real}
    _isactive(block) || return x
    feat_size = size(x, 1) * size(x, 2)
    gamma =
        (1 - block.keep_prop) / block.block_size^2 * feat_size^2 /
        (feat_size - block.block_size + 1)^2
    mask = _get_mask(block, x, gamma)
    x .* mask .* (length(mask) / sum(mask))
end

testmode!(m::DropBlock, mode = true) =
    (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
