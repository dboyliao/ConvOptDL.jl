import Remark

slideshowdir = Remark.slideshow(
    ".",
    options = Dict("ratio" => "16:9"),
    title = "Meta-Learning and Differentiable Convex Optimization with Flux and Zygote",
)

# # Open presentation in default browser.
# Remark.open(slideshowdir)
