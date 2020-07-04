export crammer_svm

using Zygote
using JuMP

# @adjoint foo(x)
#   y = g(x)
#   z = h(y)
#   return z, (Δ) -> Δ + y
# end
"""
Implement multi-class kernel-based SVM

# References

- http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
"""
@adjoint crammer_svm(embed_support, labels_support, embed_query)
    return nothing, (Δ) -> (nothing, nothing, embed_query)
end