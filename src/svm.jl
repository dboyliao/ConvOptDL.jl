export crammer_svm

using Flux

_format_batch(batch) = reshape(batch, size(batch)[1:end-2]..., :)

"""
Implement multi-class kernel-based SVM

# Inputs

- `batch`(::MetaDataSample) 

# References

- http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
"""
function crammer_svm(model, batch::MetaDataSample)
    # reshape (m, n, c, num_samples, num_tasks) to (m, n, c, num_samples * num_tasks)
    support_samples = _format_batch(batch.support_samples)
    query_samples = _format_batch(batch.query_samples)
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
    # (`n_ways`*`k_shots`, `n_ways`*`k_shots`, `num_tasks`)
    kernel_mat = Utils.gram_matrix(embed_support, embed_support)
    # solve QP
    # compute compatibility
    return nothing
end
