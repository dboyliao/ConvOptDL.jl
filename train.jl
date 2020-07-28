using ConvOptDL
using ConvOptDL.Utils
using StatsBase
using LinearAlgebra: I
using ArgParse
using Serialization
using Pipe: @pipe
using Flux: update!
using Statistics: mean
import Flux

function loss_log_softmax(logits, one_hot_vec)
    log_prob = log.(Flux.softmax(logits, dims = 1))
    loss = @pipe -(log_prob .* one_hot_vec) |> sum(_, dims = 1)
    mean(loss)
end

_format_batch(batch) = reshape(Float32.(batch), size(batch)[1:end-2]..., :)

function train!(loss, model, batch, opt)
    # https://fluxml.ai/Flux.jl/stable/training/training/#Custom-Training-loops-1
    ps = Flux.params(model)
    # smoothed onehot encoding
    # onehot is not differentiable, moving out of the do block
    query_onehot = @pipe ConvOptDL.Utils.onehot(batch.query_labels) |>
          reshape(_, batch.support_n_ways, :) |>
          _ .* (1 - 5f-2) .+ 5f-2 * (1 .- _) ./ (batch.support_n_ways - 1f0)
    support_onehot = ConvOptDL.Utils.onehot(batch.support_labels)

    local meta_loss
    gs = Flux.gradient(ps) do
        # (`feat_dim`, `n_ways`*`k_shots`, `num_tasks`)
        embed_query = reshape(
            model(_format_batch(batch.query_samples)),
            :,
            batch.n_query,
            size(batch),
        )
        embed_support = reshape(
            model(_format_batch(batch.support_samples)),
            :,
            batch.n_support,
            size(batch),
        )
        Q, p, G, h, A, b = crammer_svm(embed_support, support_onehot, batch)
        Q += repeat(
            Array{Float32}(
                I,
                batch.support_n_ways * batch.n_support,
                batch.support_n_ways * batch.n_support,
            ),
            outer = (1, 1, size(batch)),
        )
        # solve QP
        X, λ, ν = solve_qp_batch(Q, p, G, h, A, b)
        α = @pipe X |>
              reshape(_, 1, batch.n_support, batch.support_n_ways, size(batch)) |>
              repeat(_, outer = (batch.n_query, 1, 1, 1)) |>
              permutedims(_, (3, 2, 1, 4))
        # logits
        logits = @pipe ConvOptDL.Utils.gram_matrix(embed_query, embed_support) |>
              reshape(_, 1, size(_)...) |>
              repeat(_, outer = (5, 1, 1, 1)) |>
              _ .* α |>
              sum(_, dims = 2) |>
              dropdims(_, dims = 2) |>
              reshape(_, batch.support_n_ways, :)
        meta_loss = loss(logits, query_onehot)
        return meta_loss
    end
    update!(opt, ps, gs)
    return meta_loss
end

function parse_opts()
    s = ArgParseSettings("Meta Convex Optimization Learning training")
    @add_arg_table! s begin
        "--batch-size"
        arg_type = Int64
        metavar = "INT"
        default = 8
        "--batches-per-episode"
        arg_type = Int64
        metavar = "INT"
        default = 200
        "--log-period"
        help="print log for every given number of batches"
        arg_type = Int64
        metavar = "INT"
        default = 25
        "--num-episodes"
        arg_type = Int64
        metavar = "INT"
        default = 50
        "--learning-rate"
        arg_type = Float32
        default = 1f-1
        "-o"
        arg_type = String
        default = "model.jls"
        "data_file"
        arg_type = String
        default = "./test/test_data.jls"
    end
    args = parse_args(s)
    return args
end

if nameof(@__MODULE__) == :Main
    args = parse_opts()
    println("Training Options:")
    for (key, value) in args
        println("$(key) := $(value)")
    end
    batch_size = args["batch-size"]
    batches_per_episode = args["batches-per-episode"]
    log_period = args["log-period"]
    num_episodes = args["num-episodes"]
    data_file = args["data_file"]
    out_model_file = args["o"]
    lr = args["learning-rate"]
    model_name, _ = splitext(out_model_file)
    model = resnet12()
    dloader = FewShotDataLoader(data_file)
    opt = Flux.Optimise.Descent(lr)
    record = Dict()
    for episode = 1:num_episodes
        meta_losses = []
        total_time = 0
        for i = 1:batches_per_episode
            time_start = time()
            batch = sample(dloader, batch_size, support_n_ways = 5, support_k_shots = 5)
            meta_loss = train!(loss_log_softmax, model, batch, opt)
            time_end = time()
            push!(meta_losses, meta_loss)
            total_time += (time_end - time_start)
            if i % log_period == 0
                println("mean meta loss: $(mean(meta_losses))")
                println("average execution time: $(total_time / i)")
            end
        end
        record[episode] = meta_losses
        serialize("train_$(model_name)_meta_losses_$(episode).jls", meta_losses)
    end
    serialize("all_train_$(model_name)_meta_losses.jls", record)
    serialize(out_model_file, model)
end
