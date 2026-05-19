using Test
using ImageLab

@testset "ImageLab" begin
    include("test_synth.jl")
    include("test_kernels.jl")
    include("test_padding.jl")
    include("test_convolution.jl")
    include("test_io.jl")
end
