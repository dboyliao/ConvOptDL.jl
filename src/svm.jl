export crammer_svm

using Flux

"""
Implement multi-class kernel-based SVM

# Inputs

- `batch`(::MetaDataSample) 

# References

- http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
"""
function crammer_svm(model, batch::MetaDataSample)
    support_samples = _format_batch(batch, support = Val(true))
    query_samples = _format_batch(batch, support = Val(false))
    # `labels_support`: array of shape (`n_ways`*`k_shots`, `tasks_per_batch`)
    # `embed_support`: (`feat_dim`, `n_ways`*`k_shots`, `num_tasks`)
    embed_support = reshape(
        model(support_samples),
        :,
        batch.support_n_ways * batch.support_k_shots,
        size(batch),
    )
    # `embed_query`: (`feat_dim`, `n_ways`*`k_shots`, `num_tasks`)
    embed_query = reshape(
        model(query_samples),
        :,
        batch.query_n_ways * batch.query_k_shots,
        size(batch),
    )
    # construct dual QP problem
    # (`n_ways`*`k_shots`, `n_ways`*`k_shots`)
    kernel_mat = Flux.batched_mul(permutedims(embed_support, (2, 1, 3)), embed_support)
    # solve QP
    # compute compatibility
    return nothing
end

_format_batch(batch; support::Val{true}) =
    reshape(batch.support_samples, size(batch.support_samples)[1:end-2]..., :)
_format_batch(batch; support::Val{false}) =
    reshape(batch.query_samples, size(batch.query_samples)[1:end-2]..., :)
