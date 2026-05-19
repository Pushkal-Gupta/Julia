#!/usr/bin/env julia
# 05_log_dog_zero_crossings.jl
#
# Second-order edge detection. The intuition: where the first derivative
# *peaks*, the second derivative *crosses zero*. So instead of looking
# for big gradient magnitudes, I look for sign changes in the Laplacian
# (or LoG, or DoG, which are smoothed approximations).
#
# This script sweeps σ for LoG and DoG, extracts zero-crossings, and
# tiles everything into a 3×3 montage so I can see how the scale
# parameter controls what kind of structure each operator picks up.
#
#   julia --project=. examples/05_log_dog_zero_crossings.jl
#
# What I'm expecting:
#   - LoG at small σ picks up fine detail (checker grid).
#   - Larger σ smooths the fine detail away and only the major edges
#     survive — scale-space at its simplest.
#   - DoG with σ₂/σ₁ ≈ 1.6 looks visually similar to LoG at σ ≈ σ₁:
#     this is the classical approximation, and it's much cheaper because
#     both Gaussians are separable.
#   - Zero-crossings give a thin one-pixel-wide edge map without needing
#     a magnitude threshold, but they pick up spurious noise — that's
#     why `min_diff` exists.

using ImageLab
using ImageLab.Synth, ImageLab.Edges, ImageLab.Viz, ImageLab.PNM

# Composite test image: bright disk + mid-gray square + checker + noise.
function studio_image(h, w)
    canvas = Synth.checkerboard(h, w; tile = 16) .* 0.55
    disk   = Synth.circle(h, w; radius = min(h, w) ÷ 5)
    img    = max.(canvas, disk)
    sq     = Synth.square(h, w; side = 28, value = 0.4)
    img    = max.(img, sq)
    return Synth.gaussian_noise(img; sigma = 0.04, seed = 7)
end

input = studio_image(128, 128)

# LoG at three scales.
log_s1 = log_filter(input, 9;  sigma = 1.0, pad = :replicate)
log_s2 = log_filter(input, 13; sigma = 2.0, pad = :replicate)
log_s4 = log_filter(input, 25; sigma = 4.0, pad = :replicate)

# DoG with two σ ratios.
dog_classic = dog_filter(input; sigma1 = 1.0, sigma2 = 1.6, pad = :replicate)  # ratio 1.6 (Marr)
dog_wider   = dog_filter(input; sigma1 = 1.0, sigma2 = 2.5, pad = :replicate)  # picks bigger blobs

# Zero-crossings. `min_diff` is set as a small fraction of the response
# range so I'm not picking up float-noise crossings on flat regions.
function zc_visual(signed)
    floor = 0.02 * maximum(abs, signed)
    return Float64.(zero_crossings(signed; min_diff = floor))
end

zc_log2 = zc_visual(log_s2)
zc_dog  = zc_visual(dog_classic)

# Save individual PGMs.
outdir = joinpath(@__DIR__, "..", "artifacts", "05_log_dog_zero_crossings")
mkpath(outdir)

PNM.save_pgm(joinpath(outdir, "00_input.pgm"),       input)
PNM.save_pgm(joinpath(outdir, "01_log_sigma1.pgm"),  Viz.signed_to_gray(log_s1))
PNM.save_pgm(joinpath(outdir, "02_log_sigma2.pgm"),  Viz.signed_to_gray(log_s2))
PNM.save_pgm(joinpath(outdir, "03_log_sigma4.pgm"),  Viz.signed_to_gray(log_s4))
PNM.save_pgm(joinpath(outdir, "04_dog_1_1p6.pgm"),   Viz.signed_to_gray(dog_classic))
PNM.save_pgm(joinpath(outdir, "05_dog_1_2p5.pgm"),   Viz.signed_to_gray(dog_wider))
PNM.save_pgm(joinpath(outdir, "06_zc_log_sigma2.pgm"), zc_log2)
PNM.save_pgm(joinpath(outdir, "07_zc_dog.pgm"),        zc_dog)

# 3×3 montage. Row 1: scale-space LoG. Row 2: DoG variants + input.
# Row 3: zero-crossings.
tiles = [
    input,                          Viz.signed_to_gray(log_s1),  Viz.signed_to_gray(log_s2),
    Viz.signed_to_gray(log_s4),     Viz.signed_to_gray(dog_classic), Viz.signed_to_gray(dog_wider),
    zc_log2,                        zc_dog,                       Viz.signed_to_gray(log_s2 .- dog_classic),
]
grid = Viz.montage(tiles; cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage.pgm"), grid)

println("Second-order edge studio:")
println("  tiles (row-major):")
println("    input             | LoG σ=1          | LoG σ=2")
println("    LoG σ=4           | DoG σ=1,1.6      | DoG σ=1,2.5")
println("    LoG σ=2 zero-cross| DoG zero-cross   | LoG σ=2 minus DoG")
println("→ $outdir")
println("Open montage.pgm for the side-by-side view.")
