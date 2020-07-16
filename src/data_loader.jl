export FewShotDataLoader

using Serialization
import Base: size
import StatsBase: sample, Weights

struct FewShotDataLoader
    labels::AbstractArray{L,1} where {L}
    samples::AbstractArray
    label2Idices::Dict
    dims_feature::Integer

    function FewShotDataLoader(path::String)
        data = deserialize(path)
        @assert typeof(data) <: Dict "$path should containing a Dict, get $(typeof(data))"
        labels = data["labels"]
        samples = data["data"]
        @assert ndims(samples) >= 2 "ndims of samples should be at least 2, get $(ndims(samples))"
        N = size(labels)[end]
        label2Idices = Dict{eltype(labels),Array{Integer,1}}()
        for i = 1:N
            label = labels[i]
            if !haskey(label2Idices, label)
                label2Idices[label] = []
            end
            push!(label2Idices[label], i)
        end
        dims_feature = reduce((a, b) -> a * b, size(samples)[1:end-1])
        new(labels, samples, label2Idices, dims_feature)
    end
end

"""
Fields
======

`support_samples`: array of support samples with size (`feature_dim`, `num_support`, `num_tasks`)
`support_labels`: array of labels of support samples with size (`num_support`, `num_tasks`)
`query_samples`: array of query samples with size (`feature_dim`, `num_query`, `num_tasks`)
`query_labels`: array of labels of query samples with size (`num_query`, `num_tasks`)
"""
struct MetaDataSample{T,L}
    support_n_ways::Integer
    support_k_shots::Integer
    query_n_ways::Integer
    query_k_shots::Integer
    support_samples::AbstractArray{T,3}
    support_labels::AbstractArray{L,2}
    query_samples::AbstractArray{T,3}
    query_labels::AbstractArray{L,2}
end

function MetaDataSample(;
    support_n_ways,
    support_k_shots,
    query_n_ways,
    query_k_shots,
    support_samples,
    support_labels,
    query_samples,
    query_labels,
)
    @assert support_n_ways > 0 "support_n_ways must be positive: $support_n_ways"
    @assert support_k_shots > 0 "support_k_shots must be positive: $support_k_shots"
    @assert query_n_ways > 0 "query_n_ways must be positive: $query_n_ways"
    @assert query_k_shots > 0 "query_k_shots must be positive: $query_k_shots"
    @assert ndims(support_samples) >= 2 "support_samples must be at least 2-dim"
    @assert ndims(query_samples) >= 2 "query_samples must be at least 2-dim"
    @assert size(support_samples) == size(query_samples) "size of support samples and query samples should be the same"
    n_tasks = size(support_samples)[end]
    if ndims(support_samples) > 2
        support_samples =
            reshape(support_samples, :, support_n_ways * support_k_shots, n_tasks)
        query_samples = reshape(query_samples, :, query_n_ways * query_k_shots, n_tasks)
    end
    for i = 1:n_tasks
        support_label = support_labels[:, i]
        query_label = query_labels[:, i]
        @assert isempty(intersect(support_label, query_label)) "support_labels and query_labels is not disjoint at task $i"
    end
    MetaDataSample(
        support_n_ways,
        support_k_shots,
        query_n_ways,
        query_k_shots,
        support_samples,
        support_labels,
        query_samples,
        query_labels,
    )
end

function size(meta_data_sample::MetaDataSample)
    return size(meta_data_sample.support_samples)[end]
end

function _sample(
    dloader::FewShotDataLoader;
    support_n_ways = 2,
    support_k_shots = 5,
    query_n_ways = support_n_ways,
    query_k_shots = support_k_shots,
)
    # find exclusive train/test samples
    uniq_labels = unique(dloader.labels)
    idx_map = Dict([(k, i) for (i, k) in enumerate(uniq_labels)])
    support_target_labels = sample(uniq_labels, support_n_ways, replace = false)
    idxs = [idx_map[label] for label in support_target_labels]
    weight_vs = ones(size(uniq_labels, 1))
    weight_vs[idxs] .= 0
    query_target_labels =
        sample(uniq_labels, Weights(weight_vs), query_n_ways, replace = false)
    support_candidate_idxs =
        vcat([dloader.label2Idices[label] for label in support_target_labels]...)
    query_candidate_idxs =
        vcat([dloader.label2Idices[label] for label in query_target_labels]...)
    train_idxs =
        sample(support_candidate_idxs, support_n_ways * support_k_shots, replace = false)
    test_idxs = sample(query_candidate_idxs, query_n_ways * query_k_shots, replace = false)
    support_samples = reshape(
        dloader.samples[repeat([:], ndims(dloader.samples) - 1)..., train_idxs],
        dloader.dims_feature,
        support_n_ways * support_k_shots,
        1,
    )
    support_labels = Utils.add_dim(dloader.labels[train_idxs,])
    query_samples = reshape(
        dloader.samples[repeat([:], ndims(dloader.samples) - 1)..., test_idxs],
        dloader.dims_feature,
        query_n_ways * query_k_shots,
        1,
    )
    query_labels = Utils.add_dim(dloader.labels[test_idxs])
    return support_samples, support_labels, query_samples, query_labels
end

function sample(
    dloader::FewShotDataLoader;
    support_n_ways = 2,
    support_k_shots = 5,
    query_n_ways = support_n_ways,
    query_k_shots = support_k_shots,
)
    support_samples, support_labels, query_samples, query_labels = _sample(
        dloader,
        support_n_ways = support_n_ways,
        support_k_shots = support_k_shots,
        query_n_ways = query_n_ways,
        query_k_shots = query_k_shots,
    )
    MetaDataSample(
        support_n_ways = support_n_ways,
        support_k_shots = support_k_shots,
        query_n_ways = query_n_ways,
        query_k_shots = query_k_shots,
        support_samples = support_samples,
        support_labels = support_labels,
        query_samples = query_samples,
        query_labels = query_labels,
    )
end

function sample(
    dloader::FewShotDataLoader,
    n_tasks::Integer;
    support_n_ways = 2,
    support_k_shots = 5,
    query_n_ways = support_n_ways,
    query_k_shots = support_k_shots,
)
    support_sets = []
    support_labels = []
    query_sets = []
    query_labels = []
    for i = 1:n_tasks
        support, support_label, query, query_label = _sample(
            dloader,
            support_n_ways = support_n_ways,
            support_k_shots = support_k_shots,
            query_n_ways = query_n_ways,
            query_k_shots = query_k_shots,
        )
        push!(support_sets, support)
        push!(support_labels, support_label)
        push!(query_sets, query)
        push!(query_labels, query_label)
    end
    MetaDataSample(
        support_n_ways = support_n_ways,
        support_k_shots = support_k_shots,
        query_n_ways = query_n_ways,
        query_k_shots = query_k_shots,
        support_samples = cat(support_sets..., dims = Val(3)),
        support_labels = cat(support_labels..., dims = Val(2)),
        query_samples = cat(query_sets..., dims = Val(3)),
        query_labels = cat(query_labels..., dims = Val(2)),
    )
end
