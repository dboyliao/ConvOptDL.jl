export FewShotDataLoader, MetaDataSample

using Serialization
using Random: shuffle!
import StatsBase
import Base

struct FewShotDataLoader
    labels::AbstractArray{L,1} where {L}
    samples::AbstractArray
    label2Idices::Dict

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
        new(labels, samples, label2Idices)
    end
end

function Base.show(io::IO, dloader::FewShotDataLoader)
    print(
        io,
        "FewShotDataLoader(sample:$(size(dloader.samples)), labels:$(typeof(dloader.labels)))",
    )
end

"""
Fields
======

`support_samples`: array of support samples with size (`feature_size`..., `num_support`, `num_tasks`)
`support_labels`: array of labels of support samples with size (`num_support`, `num_tasks`)
`query_samples`: array of query samples with size (`feature_size`..., `num_query`, `num_tasks`)
`query_labels`: array of labels of query samples with size (`num_query`, `num_tasks`)
"""
struct MetaDataSample{T,L}
    support_n_ways::Integer
    support_k_shots::Integer
    query_n_ways::Integer
    query_k_shots::Integer
    support_samples::AbstractArray{T}
    support_labels::AbstractArray{L,2}
    query_samples::AbstractArray{T}
    query_labels::AbstractArray{L,2}
end

function Base.getproperty(m::MetaDataSample, name::Symbol)
    if name == :n_support
        return m.support_k_shots * m.support_n_ways
    elseif name == :n_query
        return m.query_k_shots * m.query_n_ways
    end
    getfield(m, name)
end

function Base.show(io::IO, meta_sample::MetaDataSample)
    print(
        io,
        "MetaDataSample(",
        "support_n_ways: $(meta_sample.support_n_ways), ",
        "support_k_shots: $(meta_sample.support_k_shots), ",
        "query_n_ways: $(meta_sample.query_n_ways), ",
        "query_k_shots: $(meta_sample.query_k_shots), ",
        "support_samples: [$(eltype(meta_sample.support_samples))]$(size(meta_sample.support_samples)), ",
        "support_labels: [$(eltype(meta_sample.support_labels))]$(size(meta_sample.support_labels)), ",
        "query_samples: [$(eltype(meta_sample.query_samples))]$(size(meta_sample.query_samples)), ",
        "query_labels: [$(eltype(meta_sample.query_labels))]$(size(meta_sample.query_labels)))",
    )
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
    @assert ndims(support_samples) >= 3 "support_samples must be at least 3-dim"
    @assert ndims(query_samples) >= 3 "query_samples must be at least 3-dim"
    @assert support_n_ways * support_k_shots == size(support_samples)[end-1]
    @assert query_n_ways * query_k_shots == size(query_samples)[end-1]
    @assert size(support_samples)[1:end-2] == size(query_samples)[1:end-2] "size of support samples and query samples should be the same"
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

function Base.size(meta_data_sample::MetaDataSample)
    return size(meta_data_sample.support_samples)[end]
end

function _sample(
    dloader::FewShotDataLoader;
    support_n_ways = 2,
    support_k_shots = 5,
    query_k_shots = support_k_shots,
)
    # find exclusive train/test samples
    uniq_labels = unique(dloader.labels)
    target_labels = StatsBase.sample(uniq_labels, support_n_ways, replace = false)
    support_idxs = []
    query_idxs = []
    for label in target_labels
        idx_map = Dict([(v, i) for (i, v) in enumerate(dloader.label2Idices[label])])
        selected_idxs = StatsBase.sample(dloader.label2Idices[label], support_k_shots)
        support_idxs = vcat(
            support_idxs,
            selected_idxs,
        )
        weights = ones(length(dloader.label2Idices[label]))
        for idx in selected_idxs
            weights[idx_map[idx]] = 0
        end
        weights = StatsBase.Weights(weights)
        selected_idxs =
            StatsBase.sample(dloader.label2Idices[label], weights, query_k_shots)
        query_idxs = vcat(
            query_idxs,
            selected_idxs
        )
    end
    @assert intersect(support_idxs, query_idxs) |> isempty
    shuffle!(support_idxs)
    shuffle!(query_idxs)
    support_samples = Utils.add_dim(
        dloader.samples[repeat([:], ndims(dloader.samples)-1)..., support_idxs]
    )
    support_labels = Utils.add_dim(dloader.labels[support_idxs])
    query_samples = Utils.add_dim(
        dloader.samples[repeat([:], ndims(dloader.samples)-1)..., query_idxs]
    )
    query_labels = Utils.add_dim(dloader.labels[query_idxs])
    return support_samples, support_labels, query_samples, query_labels
end

function StatsBase.sample(
    dloader::FewShotDataLoader;
    support_n_ways = 2,
    support_k_shots = 5,
    query_k_shots = support_k_shots,
)
    support_samples, support_labels, query_samples, query_labels = _sample(
        dloader,
        support_n_ways = support_n_ways,
        support_k_shots = support_k_shots,
        query_k_shots = query_k_shots,
    )
    MetaDataSample(
        support_n_ways = support_n_ways,
        support_k_shots = support_k_shots,
        query_n_ways = support_n_ways,
        query_k_shots = query_k_shots,
        support_samples = support_samples,
        support_labels = support_labels,
        query_samples = query_samples,
        query_labels = query_labels,
    )
end

function StatsBase.sample(
    dloader::FewShotDataLoader,
    n_tasks::Integer;
    support_n_ways = 2,
    support_k_shots = 5,
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
            query_k_shots = query_k_shots,
        )
        push!(support_sets, support)
        push!(support_labels, support_label)
        push!(query_sets, query)
        push!(query_labels, query_label)
    end
    ndim_sample = ndims(support_sets[1])
    MetaDataSample(
        support_n_ways = support_n_ways,
        support_k_shots = support_k_shots,
        query_n_ways = support_n_ways,
        query_k_shots = query_k_shots,
        support_samples = cat(support_sets..., dims = Val(ndim_sample)),
        support_labels = cat(support_labels..., dims = Val(2)),
        query_samples = cat(query_sets..., dims = Val(ndim_sample)),
        query_labels = cat(query_labels..., dims = Val(2)),
    )
end
