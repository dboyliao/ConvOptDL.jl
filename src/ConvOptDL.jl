__precompile__()
module ConvOptDL

include("data_loader.jl")
include("qp.jl")
include("svm.jl")
include("layers/dropblock.jl")
include("layers/basic_block.jl")
include("layers/resnet12.jl")
include("utils.jl")

end
