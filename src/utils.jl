module Utils

export batch_outer, add_dim

function batch_outer(A, B)
    num_batch = size(A)[end]
    size(A)[end] == size(B)[end] || throw(DimensionMismatch("batch size mismatch"))
    As = [reshape(A[:, i], :, 1) for i = 1:num_batch]
    Bs = [B[:, i]' for i = 1:num_batch]
    Cs = As .* Bs
    # return reshape(hcat(Cs...), (size(A, 1), size(B, 1), num_batch))
    C = similar(A, size(A, 1), size(B, 1), num_batch)
    for i = 1:num_batch
        C[:, :, i] .= Cs[i]
    end
    return C
end

"""
https://stackoverflow.com/a/59849291/3908797
"""
add_dim(x::AbstractArray{T, N}) where {T, N} = reshape(x, Val(N+1))

end
