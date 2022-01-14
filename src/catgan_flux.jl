module catgan_flux

using NNlib
using Flux
using Zygote
using MLDatasets

include("models.jl")
include("lossfuns.jl")
include("utils.jl")

end # module
