"""
    ImageLab

A first-principles image-processing lab written in Julia. The aim is to learn
the language by building convolution, edge detection, filtering, and broader
computational-imaging machinery from scratch — then compare against the
ecosystem once the fundamentals are solid.

Submodules so far:

- [`Synth`](@ref ImageLab.Synth)        — synthetic image generators
- [`Kernels`](@ref ImageLab.Kernels)    — canonical 2D kernels (box, Gaussian, Sobel, …)
- [`Padding`](@ref ImageLab.Padding)    — border modes (`:zero`, `:replicate`, `:reflect`, `:circular`, `:valid`)
- [`Convolution`](@ref ImageLab.Convolution) — 2D correlation, convolution, 1D + separable variants
- [`Edges`](@ref ImageLab.Edges)        — gradient, Laplacian / LoG / DoG, zero-crossings, thresholds, full Canny
- [`Filters`](@ref ImageLab.Filters)    — median, bilateral, binary dilation
- [`Features`](@ref ImageLab.Features)  — Harris corners, Hough lines, connected components, NCC template match
- [`Pyramids`](@ref ImageLab.Pyramids)  — Gaussian and Laplacian pyramids (Burt-Adelson), perfect reconstruction
- [`Metrics`](@ref ImageLab.Metrics)    — precision / recall / F1, IoU for comparing edge maps
- [`Viz`](@ref ImageLab.Viz)            — normalization, montage, line / point drawing helpers
- [`PNM`](@ref ImageLab.PNM)            — Netpbm (PGM/PPM) readers and writers

Convention: images are `Matrix{<:Real}` with values in `[0, 1]` for floats or
`[0, 255]` for `UInt8`. Row index = y (top → bottom), column index = x
(left → right). This matches Julia's column-major storage and most linear
algebra intuition.
"""
module ImageLab

include("synth.jl")
include("kernels.jl")
include("padding.jl")
include("convolution.jl")
include("edges.jl")
include("filters.jl")
include("features.jl")
include("pyramids.jl")
include("metrics.jl")
include("viz.jl")
include("io.jl")

using .Synth
using .Kernels
using .Padding
using .Convolution
using .Edges
using .Filters
using .Features
using .Pyramids
using .Metrics
using .Viz
using .PNM

export Synth, Kernels, Padding, Convolution, Edges, Filters, Features, Pyramids, Metrics, Viz, PNM

end # module ImageLab
