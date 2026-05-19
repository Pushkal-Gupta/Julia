#!/usr/bin/env julia
# 02_padding_modes_studio.jl
#
# Milestone 2 visual deliverable. Apply the same large Gaussian blur to the
# same image under each padding mode, and assemble a 2×3 comparison montage
# so the differences are obvious at a glance.
#
#   julia --project=. examples/02_padding_modes_studio.jl
#
# What to look for:
#   :zero        → bright objects bleed into a dark halo at the border because
#                   the world "outside the frame" is taken to be black.
#   :replicate   → the edge color extends outward; objects near the border keep
#                   their brightness without the halo.
#   :reflect     → near-perfect at smooth interiors; can introduce a fake edge
#                   when the actual border has a strong gradient.
#   :symmetric   → like :reflect but the border pixel is duplicated.
#   :circular    → opposite edges fuse, so a bright top-right corner can leak
#                   onto the bottom-left. The implicit FFT-convolution mode.

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
