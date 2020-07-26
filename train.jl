using ConvOptDL
using ConvOptDL.Utils
using StatsBase
using Flux: gradient, update!

function loss_log_softmax(logits, one_hot_vec) end

_format_batch(batch) = reshape(Float32.(batch), size(batch)[1:end-2]..., :)

function train!(loss, model, batch, opt)
    # https://fluxml.ai/Flux.jl/stable/training/training/#Custom-Training-loops-1
    ps = Flux.params(model)

    local meta_loss
    gs = Flux.gradient(ps) do
        # (`feat_dim`, `n_ways`*`k_shots`, `num_tasks`)
        embed_query = reshape(
            model(_format_batch(batch.support_samples)),
            :,
            batch.support_n_ways * batch.support_k_shots,
            size(batch),
        )
        embed_support = reshape(
            model(_format_batch(batch.query_samples)),
            :,
            batch.query_n_ways * batch.query_k_shots,
            size(batch),
        )
        Q, p, G, h, A, b = crammer_svm(embed_support, batch)
        # solve QP
        sol = solve_qp_batch(Q, p, G, h, A, b)
        # compatibility
        compatibility = ConvOptDL.Utils.gram_matrix(embed_query, embed_support)
        # smoothed onehot encoding
        onehot_vec = ConvOptDL.Utils.onehot(batch.query_labels)
        meta_loss = loss(compatibility, onehot_vec)
        return meta_loss
    end
    update!(opt, ps, gs)
end

if nameof(@__MODULE__) == :Main
    model = resnet12()
    dloader = FewShotDataLoader("./test/test_data.jls")
    batch = sample(dloader, 8, support_n_ways = 5, support_k_shots = 5)
    opt = Flux.Optimise.Descent(0.1)
    train!(loss_log_softmax, model, batch, opt)
end
