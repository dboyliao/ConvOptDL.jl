export FewShotDataLoader, sample

using Serialization
import StatsBase

struct FewShotDataLoader
    labels::Any
    samples::Any
    label2Idices::Any

    function FewShotDataLoader(path::String)
        data = deserialize(path)
        @assert typeof(data) <: Dict "$path should containing a Dict, get $(typeof(data))"
        labels = data["labels"]
        samples = data["data"]
        N = size(labels, 1)
        label2Idices = Dict{Int64,Array{Int64,1}}()
        for i = 1:N
            label = labels[i]
            if !haskey(label2Idices, label)
                label2Idices[label] = []
            end
            push!(label2Idices[label], i)
        end
        new(labels, samples, label2Idices)
    end
end

struct MetaDataSample
    train_n_ways::Int64
    train_k_shots::Int64
    test_n_ways::Int64
    test_k_shots::Int64
    train_samples::Any
    train_labels::Any
    test_samples::Any
    test_labels::Any

    function MetaDataSample(
        train_n_ways,
        train_k_shots,
        test_n_ways,
        test_k_shots,
        train_samples,
        train_labels,
        test_samples,
        test_labels,
    )
        @assert train_n_ways > 0 "train_n_ways must be positive: $train_n_ways"
        @assert train_k_shots > 0 "train_k_shots must be positive: $train_k_shots"
        @assert test_n_ways > 0 "test_n_ways must be positive: $test_n_ways"
        @assert test_k_shots > 0 "test_k_shots must be positive: $test_k_shots"
        @assert isempty(intersect(train_labels, test_labels)) "train_labels and test_labels should be disjoint"
        new(
            train_n_ways,
            train_k_shots,
            test_n_ways,
            test_k_shots,
            train_samples,
            train_labels,
            test_samples,
            test_labels,
        )
    end
end

function sample(
    dloader::FewShotDataLoader;
    train_n_ways = 2,
    train_k_shots = 5,
    test_n_ways = train_n_ways,
    test_k_shots = train_k_shots,
)
    uniq_labels = unique(dloader.labels)
    idx_map = Dict([(k, i) for (i, k) in enumerate(uniq_labels)])
    train_target_labels = StatsBase.sample(uniq_labels, train_n_ways, replace = false)
    idxs = [idx_map[label] for label in train_target_labels]
    weight_vs = ones(size(uniq_labels, 1))
    weight_vs[idxs] .= 0
    test_target_labels = StatsBase.sample(
        uniq_labels,
        StatsBase.Weights(weight_vs),
        test_n_ways,
        replace = false,
    )

    train_candidate_idxs =
        vcat([dloader.label2Idices[label] for label in train_target_labels]...)
    test_candidate_idxs =
        vcat([dloader.label2Idices[label] for label in test_target_labels]...)
    train_idxs = StatsBase.sample(
        train_candidate_idxs,
        train_n_ways * train_k_shots,
        replace = false,
    )
    test_idxs =
        StatsBase.sample(test_candidate_idxs, test_n_ways * test_k_shots, replace = false)
    train_samples = dloader.samples[train_idxs]
    train_labels = dloader.labels[train_idxs]
    test_samples = dloader.samples[test_idxs]
    test_labels = dloader.labels[test_idxs]
    MetaDataSample(
        train_n_ways,
        train_k_shots,
        test_n_ways,
        test_k_shots,
        train_samples,
        train_labels,
        test_samples,
        test_labels,
    )
end

function sample(
    dloader::FewShotDataLoader,
    n::Int64;
    train_n_ways = 2,
    train_k_shots = 5,
    test_n_ways = train_n_ways,
    test_k_shots = train_k_shots,
)
    meta_samples = [
        sample(
                dloader,
                train_n_ways = train_n_ways,
                train_k_shots = train_k_shots,
                test_n_ways = train_n_ways,
                test_k_shots = train_k_shots,
            )
        for _ in 1:n
    ]
    meta_samples
end
