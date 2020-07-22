export crammer_svm

using Flux

"""
Implement multi-class kernel-based SVM

# Inputs

- `batch`(::MetaDataSample) 

# References

- http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
"""
function crammer_svm(batch::MetaDataSample)
    # Note: sample_per_task = n_ways * k_shots
    # `embed_support`: array of shape (`embed_dim`, `samples_per_task`, `tasks_per_batch`)
    # `labels_support`: array of shape (`sample_per_task`, `tasks_per_batch`)
    # `embed_query`: array of shape (`embed_dim`, `sample_per_task`, `tasks_per_batch`)

    # construct dual QP problem
    # solve QP
    # compute compatibility
    return nothing
end
