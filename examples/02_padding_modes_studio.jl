#!/usr/bin/env julia
# 02_padding_modes_studio.jl
#
# I wanted one image that makes the difference between padding modes
# obvious. So: build an off-center bright object, blur it with a large
# Gaussian (σ=3, kernel reaches ~9 pixels into the border), then run that
# under each of the five non-`:valid` modes. Tile them into a 2×3 montage.
#
#   julia --project=. examples/02_padding_modes_studio.jl
#
# What each mode looks like once it's blurred:
#   :zero        bright objects fade into a dark halo at the border because
#                the world "outside the frame" is taken to be black.
#   :replicate   the edge color extends outward, so the halo doesn't form.
#   :reflect     near-invisible at smooth interiors. Can introduce a fake
#                edge when the actual border had a strong gradient.
#   :symmetric   almost identical to :reflect — the border pixel is included
#                in the mirror.
#   :circular    opposite edges fuse, so a bright top-right block can leak
#                onto the bottom-left. This is what FFT-based convolution
#                does implicitly.

using ImageLab
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution, ImageLab.Viz, ImageLab.PNM

# Off-center bright object → padding choice is visibly different on the
# adjacent border. We tuck a small square toward the top-left.
function studio_image(h, w)
    img = zeros(Float64, h, w)
    img[3:18, 3:18] .= 1.0                       # bright corner block
    img[(h÷2-4):(h÷2+4), (w-12):(w-3)] .= 0.85    # bright right-edge bar
    img[(h-15):(h-3), (w÷2-6):(w÷2+6)] .= 0.5     # mid-tone bottom block
    return img
end

input = studio_image(96, 96)
g1 = Kernels.gaussian1d(15; sigma = 3.0)   # σ=3, so the blur reaches well past
                                           # the border — exactly where padding
                                           # decisions become visible.

modes = (:zero, :replicate, :reflect, :symmetric, :circular)

outdir = joinpath(@__DIR__, "..", "artifacts", "02_padding_modes_studio")
mkpath(outdir)

# Save the input plus one blurred PGM per mode.
PNM.save_pgm(joinpath(outdir, "00_input.pgm"), input)

blurred = Dict{Symbol, Matrix{Float64}}()
for m in modes
    out = separable_correlate2d(input, g1, g1; pad = m)
    blurred[m] = out
    PNM.save_pgm(joinpath(outdir, "blur_$(m).pgm"), out)
end

# Montage: original + the five padded variants in row-major order.
tiles = [input, blurred[:zero], blurred[:replicate],
                blurred[:reflect], blurred[:symmetric], blurred[:circular]]
labels = ["original", ":zero", ":replicate", ":reflect", ":symmetric", ":circular"]
grid = Viz.montage(tiles; cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage.pgm"), grid)

println("Padding mode studio:")
for (i, m) in enumerate(("original", string.(modes)...))
    println("  tile $i  = $m")
end
println("→ $outdir")
println("Open montage.pgm for the side-by-side view.")
