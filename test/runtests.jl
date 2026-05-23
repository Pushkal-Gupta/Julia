using Test
using ImageLab

@testset "ImageLab" begin
    include("test_synth.jl")
    include("test_kernels.jl")
    include("test_padding.jl")
    include("test_convolution.jl")
    include("test_separable.jl")
    include("test_performance.jl")
    include("test_linalg_view.jl")
    include("test_viz.jl")
    include("test_edges.jl")
    include("test_canny.jl")
    include("test_filters.jl")
    include("test_features.jl")
    include("test_pyramids.jl")
    include("test_metrics.jl")
    include("test_viz_drawing.jl")
    include("test_io.jl")
    include("test_photos.jl")
    include("test_raytracer.jl")
    include("test_autodiff.jl")
    include("test_flow.jl")
end
