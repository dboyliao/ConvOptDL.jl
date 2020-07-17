using Distributions
using Flux: _isactive
using Zygote: @adjoint
import Flux
import Base

export DropBlock, dropblock

"""
https://arxiv.org/pdf/1810.12890.pdf
"""
mutable struct DropBlock
    block_size::Integer
    keep_prop::AbstractFloat
    active::Union{Bool,Nothing}
    function DropBlock(block_size, keep_prop, active = nothing)
        @assert 0 ≤ keep_prop ≤ 1
        new(block_size, keep_prop, active)
    end
end

function _get_block_mask(block::DropBlock, x, mask = nothing)
    feat_size = size(x, 1) * size(x, 2)
    height, width, channel, batch = size(x)
    height_offset = width_offset = floor(Integer, block.block_size / 2)
    gamma =
        (1 - block.keep_prop) / block.block_size^2 * feat_size^2 /
        (feat_size - block.block_size + 1)^2
    if isnothing(mask)
        mask = zeros(size(x))
        mask[
            height_offset+1:height-height_offset,
            width_offset+1:width-width_offset,
            :,
            :,
        ] .= rand(
            Bernoulli(gamma),
            height - height_offset * 2,
            width - width_offset * 2,
            channel,
            batch,
        )
    end
    mask_pos = findall(x -> x > 0, mask)
    block_mask = ones(eltype(x), size(x))
    for pos in mask_pos
        c_height, c_width, c_channel, c_batch = pos.I
        height_start = c_height - height_offset
        height_end = c_height + height_offset
        width_start = c_width - width_offset
        width_end = c_width + width_offset
        block_mask[height_start:height_end, width_start:width_end, c_channel, c_batch] .= 0
    end
    return block_mask
end

function dropblock(x, block::DropBlock)
    block_mask = _get_block_mask(block, x)
    y = x .* block_mask .* (length(block_mask) / sum(block_mask))
    return y, block_mask
end

@adjoint function dropblock(x, block::DropBlock)
    y, block_mask = dropblock(x, block)
    function back(Δ)
        Δy, _ = Δ
        return (Δy .* block_mask, nothing)
    end
    return (y, block_mask), back
end

function (block::DropBlock)(x::AbstractArray{T,4}) where {T<:Real}
    _isactive(block) || return x
    return dropblock(x, block)[1]
end

Flux.testmode!(block::DropBlock, mode = true) =
    (block.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; block)

function Base.show(io::IO, block::DropBlock)
    print(io, "DropBlock($(block.block_size), $(block.keep_prop), $(block.active))")
end
