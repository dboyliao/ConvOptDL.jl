export crammer_svm

using Flux

"""
Implement multi-class kernel-based SVM

# Inputs

- `embed_support`: array of shape (`tasks_per_batch`, `sample_per_task`, `embed_dim`)
- `labels_support`: array of shape (`tasks_per_batch`, `sample_per_task`)
- `embed_query`: array of shape (`tasks_per_batch`, `sample_per_task`, `embed_dim`)

# References

- http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
"""
function crammer_svm(embed_support, labels_support, embed_query)
    # construct dual QP problem
    # solve QP
    # compute compatibility
    return nothing
end
