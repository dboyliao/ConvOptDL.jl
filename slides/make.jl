using Remark, FileWatching
using ArgParse

function _parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--watch"
        action = :store_true
        help = "watch files and rebuilt on changes"
    end
    parse_args(s)
end

function main(watch::Bool)
    Remark.slideshow(
        @__DIR__;
        options = Dict("ratio" => "16:9"),
        title = "Meta-Learning and Differentiable Convex Optimization with Flux and Zygote",
    )
    @info "slides build done"
    while watch
        FileWatching.watch_folder(joinpath(@__DIR__, "src"))
        @info "Rebuilt"
        Remark.slideshow(
            @__DIR__;
            options = Dict("ratio" => "16:9"),
            title = "Meta-Learning and Differentiable Convex Optimization with Flux and Zygote",
        )
    end
end

if nameof(@__MODULE__) == :Main
    args = _parse_args()
    main(args["watch"])
end
