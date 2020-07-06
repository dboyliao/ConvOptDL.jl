export crammer_svm

include("qp.jl")

"""
Implement multi-class kernel-based SVM

# References

- http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
"""
function crammer_svm(embed_support, labels_support, embed_query)
    # construct dual QP problem
    # solve QP
    # compute compatibility
    return nothing
end

@adjoint crammer_svm(embed_support, labels_support, embed_query) = crammer_svm(embed_support, labels_support, embed_query), (Δ) -> (nothing, nothing, embed_query)