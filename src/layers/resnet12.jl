export resnet12
using Flux

# follow the implementation of https://github.com/kjunelee/MetaOptNet.git
function resnet12(use_avgpool = true)
    layer1 = BasicBlock(3 => 64, drop_prob=0.1, block_size=3)
    layer2 = BasicBlock(64 => 160, drop_prob=0.1, block_size=3)
    layer3 = BasicBlock(160 => 320, drop_prob=0.1, block_size=3, use_dropblock=true)
    layer4 = BasicBlock(320 => 640, drop_prob=0.1, block_size=3, use_dropblock=true)
    if use_avgpool
        pool = MeanPool(5, pad=1)
    else
        pool = identity
    end
    Chain(layer1, layer2, layer3, layer4, pool, (x)->reshape(x, :, size(x)[end]))
end