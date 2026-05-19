#!/usr/bin/env julia
# 06_canny_pipeline.jl
#
# I want to see every stage of Canny on a single input, so I can build
# intuition for what each step is actually doing. The pipeline:
#
#   1. Smooth with a Gaussian (kills noise, prevents tiny features from
#      dominating the gradient).
#   2. Compute Sobel gradients.
#   3. Magnitude + direction.
#   4. Non-maximum suppression along the gradient direction (turns fat
#      edges into thin ridges).
#   5. Double threshold → strong / weak / suppressed.
#   6. Hysteresis: weak pixels survive if they connect to strong ones.
#
# This script runs all six on a synthetic input and tiles them into a
# 3×3 montage so I can read down the columns: input → smoothing →
# gradient response → thresholded ridges → final edges.
#
#   julia --project=. examples/06_canny_pipeline.jl
#
# What I'm expecting:
#   - The "magnitude" tile is fat. NMS thins it.
#   - The "strong" tile is sparse — only the brightest gradient peaks.
#     The "weak" tile fills in the breaks.
#   - The "final edges" tile looks like a clean line drawing of the
#     input. That's the whole point.

using ImageLab
using ImageLab.Synth, ImageLab.Edges, ImageLab.Viz, ImageLab.PNM

# An input with a mix of curved and straight edges, plus some noise so
# the pre-smooth step has something to do.
function studio_image(h, w)
    canvas = Synth.checkerboard(h, w; tile = 24) .* 0.45
    disk   = Synth.circle(h, w; radius = min(h, w) ÷ 4)
    img    = max.(canvas, disk)
    sq     = Synth.square(h, w; side = 32, value = 0.7)
    img    = max.(img, sq)
    return Synth.gaussian_noise(img; sigma = 0.06, seed = 11)
end

input = studio_image(128, 128)

# A reasonable middle-of-the-road parameter setting. The studios that
# follow this one (07_canny_parameter_sweep.jl) explore the full grid.
σ, low, high = 1.2, 0.06, 0.18
s = canny_stages(input; sigma = σ, low = low, high = high)

# Convert each stage to a viewable [0, 1] image.
binary(b::BitMatrix) = Float64.(b)

outdir = joinpath(@__DIR__, "..", "artifacts", "06_canny_pipeline")
mkpath(outdir)

PNM.save_pgm(joinpath(outdir, "00_input.pgm"),        input)
PNM.save_pgm(joinpath(outdir, "01_blurred.pgm"),      s.blurred)
PNM.save_pgm(joinpath(outdir, "02_gx.pgm"),           Viz.signed_to_gray(s.gx))
PNM.save_pgm(joinpath(outdir, "03_gy.pgm"),           Viz.signed_to_gray(s.gy))
PNM.save_pgm(joinpath(outdir, "04_magnitude.pgm"),    Viz.normalize01(s.magnitude))
PNM.save_pgm(joinpath(outdir, "05_nms.pgm"),          Viz.normalize01(s.nms))
PNM.save_pgm(joinpath(outdir, "06_strong.pgm"),       binary(s.strong))
PNM.save_pgm(joinpath(outdir, "07_weak.pgm"),         binary(s.weak))
PNM.save_pgm(joinpath(outdir, "08_edges_final.pgm"),  binary(s.edges))

# 3×3 montage: read top-left to bottom-right as the pipeline flows.
tiles = [
    input,                              s.blurred,                         Viz.normalize01(s.magnitude),
    Viz.signed_to_gray(s.gx),           Viz.signed_to_gray(s.gy),          Viz.normalize01(s.nms),
    binary(s.strong),                   binary(s.weak),                    binary(s.edges),
]
grid = Viz.montage(tiles; cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage.pgm"), grid)

n_edges = sum(s.edges)
n_strong = sum(s.strong)
n_weak_recruited = n_edges - n_strong
println("Canny pipeline (σ=$σ, low=$low, high=$high):")
println("  strong pixels:        $n_strong")
println("  weak pixels:          $(sum(s.weak))")
println("  weak → recruited:     $n_weak_recruited")
println("  final edges:          $n_edges  ($(round(100 * n_edges / length(s.edges), digits=2))% of image)")
println("→ $outdir")
println("Open montage.pgm and read it row by row.")
