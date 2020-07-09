module Utils

export batch_outer

function batch_outer(A, B)
    num_batch = size(A)[end]
    size(A)[end] == size(B)[end] || throw(DimensionMismatch("batch size mismatch"))
    As = [reshape(A[:, i], :, 1) for i in 1:num_batch]
    Bs = [B[:, i]' for i in 1:num_batch]
    Cs = As .* Bs
    # return reshape(hcat(Cs...), (size(A, 1), size(B, 1), num_batch))
    C = Array{eltype(Cs[1])}(undef, size(A, 1), size(B, 1), num_batch)
    for i in 1:num_batch
        C[:, :, i] = Cs[i]
    end
    return C
end

end