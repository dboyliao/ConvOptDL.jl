export crammer_svm, embed

using Flux
using LinearAlgebra
import SCS


"""
Implement multi-class kernel-based SVM

# Inputs

- `embed_support`: (`feat_dim`, `n_ways`*`k_shots`, `num_tasks`)
- `batch :: MetaDataSample`

# References

- http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
"""
function crammer_svm(embed_support, batch::MetaDataSample; C_reg = Float32(0.1))
    # reshape (m, n, c, num_samples, num_tasks) to (m, n, c, num_samples * num_tasks)
    n_support = batch.support_n_ways * batch.support_k_shots
    # construct dual QP problem: finding Q, p, G, h, A, b
    # (`n_ways`*`k_shots`, `n_ways`*`k_shots`, `num_tasks`)
    kernel_mat = Utils.gram_matrix(embed_support, embed_support)
    Q = Utils.batched_kronecker(
        kernel_mat,
        repeat(
            Array{Float32}(I, batch.support_n_ways, batch.support_n_ways),
            outer = (1, 1, size(batch)),
        ),
    )
    p = Float32.(-1 * Utils.onehot(batch.support_labels))

    support_labels_onehot = Utils.onehot(batch.support_labels)
    G = repeat(
        Array{Float32}(
            I,
            n_support * batch.support_n_ways,
            n_support * batch.support_n_ways,
        ),
        outer = (1, 1, size(batch)),
    )
    h = C_reg * support_labels_onehot

    A = Utils.batched_kronecker(
        repeat(Array{Float32}(I, n_support, n_support), outer = (1, 1, size(batch))),
        ones(Float32, 1, batch.support_n_ways, size(batch)),
    )
    b = zeros(Float32, n_support, size(batch))
    return Q, p, G, h, A, b
end
