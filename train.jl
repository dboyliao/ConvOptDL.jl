using ConvOptDL
using ConvOptDL.Utils
using StatsBase


function main()
    model = resnet12()
    dloader = FewShotDataLoader("./test/test_data.jls")
    batch = sample(dloader, 8, support_n_ways=5, support_k_shots=5)
    embed_support, Q, p, G, h, A, b = crammer_svm(model, batch)

    # solve QP
    sol = solve_qp_batch(Q, p, G, h, A, b)
    # compute compatibility
    embed_query = embed(model, batch, support=Val(false)) # (`feat_dim`, `n_ways`*`k_shots`, `num_tasks`)
    compatibility = ConvOptDL.Utils.gram_matrix(embed_query, embed_support)
end

if nameof(@__MODULE__) == :Main
    main()
end