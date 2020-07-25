module Utils

export batched_outer, batched_kronecker, add_dim

using Flux

"""
# Parameters

- `A`: array of shape (`m`, `num_batch`)
- `B`: array of shape (`n`, `num_batch`)

# Return

- `C`: array of shape (`m`, `n`, `num_batch`), where `C[:, :, i] = A[:, i] * B[:, i]'`

"""
function batched_outer(A, B)
    @assert ndims(A) == 2
    @assert ndims(B) == 2
    size(A, 2) == size(B, 2) || throw(DimensionMismatch("batch size mismatch"))
    num_batch = size(A, 2)
    A′ = reshape(A, :, 1, num_batch)
    B′ = reshape(B, 1, :, num_batch)
    Flux.batched_mul(A′, B′)
end

"""
mat1: (m, n, batch)
mat2: (j, k, batch)
"""
function batched_kronecker(mat1, mat2)
    @assert ndims(mat1) == 3
    @assert ndims(mat2) == 3
    @assert size(mat1, 3) == size(mat2, 3)
    num_batch = size(mat1, 3)
    mat1_flatten = reshape(mat1, :, 1, num_batch)
    mat2_flatten = reshape(mat2, 1, :, num_batch)
    curried_reshape(new_shape...) = (m)-> reshape(m, new_shape...)
    curried_permdims(perms...) = (m) -> permutedims(m, perms)
    Flux.batched_mul(
        mat1_flatten,
        mat2_flatten,
        ) |> 
    curried_reshape(size(mat1)[1:end-1]..., size(mat2)[1:end-1]..., num_batch) |>
    curried_permdims(3, 1, 4, 2, 5) |>
    curried_reshape(size(mat1, 1)*size(mat2, 1), size(mat1, 2)*size(mat2, 2), num_batch)
end

"""
https://stackoverflow.com/a/59849291/3908797
"""
add_dim(x::AbstractArray{T, N}) where {T, N} = reshape(x, Val(N+1))

"""
- `A`: (d, m, batch)
- `B`: (d, n, batch)

return C: (m, n, batch)
"""
gram_matrix(A, B) = Flux.batched_mul(permutedims(A, (2, 1, 3)), B)

end
