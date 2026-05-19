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
- [`Convolution`](@ref ImageLab.Convolution) — naive 2D correlation and convolution
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
include("io.jl")

using .Synth
using .Kernels
using .Padding
using .Convolution
using .PNM

export Synth, Kernels, Padding, Convolution, PNM

end # module ImageLab
