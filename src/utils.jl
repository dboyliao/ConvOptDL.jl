module Utils

export batch_outer, add_dim, kaiming_normal, kaiming_uniform

function batch_outer(A, B)
    num_batch = size(A)[end]
    size(A)[end] == size(B)[end] || throw(DimensionMismatch("batch size mismatch"))
    As = [reshape(A[:, i], :, 1) for i = 1:num_batch]
    Bs = [B[:, i]' for i = 1:num_batch]
    Cs = As .* Bs
    # return reshape(hcat(Cs...), (size(A, 1), size(B, 1), num_batch))
    C = Array{eltype(Cs[1])}(undef, size(A, 1), size(B, 1), num_batch)
    for i = 1:num_batch
        C[:, :, i] .= Cs[i]
    end
    return C
end

"""
https://stackoverflow.com/a/59849291/3908797
"""
add_dim(x::AbstractArray{T, N}) where {T, N} = reshape(x, Val(N+1))

# copied from https://github.com/FluxML/Flux.jl/blob/master/src/utils.jl
# I have trouble upgrading Flux to 0.11.0
"""
    nfan(n_out, n_in=1) -> Tuple
    nfan(dims...)
    nfan(dims::Tuple)
For a layer characterized by dimensions `dims`, return a tuple `(fan_in, fan_out)`, where `fan_in`
is the number of input neurons connected to an output one, and `fan_out` is the number of output neurons
connected to an input one.
This function is mainly used by weight initializers, e.g., [`kaiming_normal`](@ref Flux.kaiming_normal).
# Examples
```jldoctest
julia> layer = Dense(10, 20)
Dense(10, 20)
julia> Flux.nfan(size(layer.W))
(10, 20)
julia> layer = Conv((3, 3), 2=>10)
Conv((3, 3), 2=>10)
julia> Flux.nfan(size(layer.weight))
(18, 90)
```
"""
nfan() = 1, 1 # fan_in, fan_out
nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end]) # In case of convolution kernels

"""
    kaiming_uniform(dims...; gain = √2)
Return an `Array` of size `dims` containing random variables taken from a uniform distribution in the
interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.
This method is described in [1] and also known as He initialization.
# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.kaiming_uniform(3, 2)
3×2 Array{Float32,2}:
  0.950413   1.27439
  1.4244    -1.28851
 -0.907795   0.0909376
```
# See also
* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)
# References
[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_uniform(dims...; gain = √2)
  bound = Float32(√3 * gain / sqrt(first(nfan(dims...)))) # fan_in
  return (rand(Float32, dims...) .- 0.5f0) .* 2bound
end

"""
    kaiming_normal(dims...; gain = √2)
Return an `Array` of size `dims` containing random variables taken from a normal
distribution with mean 0 and standard deviation `gain * sqrt(fan_in)`.
This method is described in [1] and also known as He initialization.
# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.kaiming_normal(3, 2)
3×2 Array{Float32,2}:
  0.679107  -0.134854
  0.828413   0.586617
 -0.353007   0.297336
```
# See also
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)
# References
[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_normal(dims...; gain = √2f0)
  std = Float32(gain / sqrt(first(nfan(dims...)))) # fan_in
  return randn(Float32, dims...) .* std
end


end
