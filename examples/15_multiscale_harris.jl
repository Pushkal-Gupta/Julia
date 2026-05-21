#!/usr/bin/env julia
# 15_multiscale_harris.jl
#
# A single-scale Harris with a 3×3 window has a strong bias: it
# detects "corners that look like 3×3 corners" — fine for sharp
# small geometry, not so much for a soft curved corner that's
# really 12 pixels across. Multi-scale Harris fixes that by running
# the same detector on every level of a Gaussian pyramid, where a
# 3×3 window at level 2 corresponds to a 12-pixel window in the
# original image.
#
#   julia --project=. examples/15_multiscale_harris.jl
#
# What I expect to see:
#   - The single-scale Harris fires only on the sharp small corners.
#   - The multi-scale variant additionally picks up corners on
#     bigger or softer features that the small window can't see.
#   - Overlay markers are sized by detection level — pixel-sized dots
#     for level 0, larger squares for coarser levels.

using ImageLab
using ImageLab.Synth, ImageLab.Features, ImageLab.Viz, ImageLab.PNM

function studio_image(h, w)
    img = zeros(Float64, h, w)
    # Sharp small rectangle (every corner is "3×3"-sized).
    img[15:35, 15:35] .= 0.9
    # Bigger soft rectangle (corners are wider — best seen at coarser scale).
    big = zeros(h, w)
    big[60:130, 60:140] .= 0.6
    # Soften the big rectangle by smoothing with a synthetic blur via
    # box-style local averaging. I just use successive small averages.
    smoothed = big
    for _ in 1:3
        # Apply a tiny moving average by hand to avoid pulling in extra
        # imports for this synthesis step.
        H2, W2 = size(smoothed)
        out = copy(smoothed)
        for j in 2:W2-1, i in 2:H2-1
            out[i, j] = (smoothed[i-1, j] + smoothed[i+1, j] +
                         smoothed[i, j-1] + smoothed[i, j+1] +
                         smoothed[i, j]) / 5
        end
        smoothed = out
    end
    img = max.(img, smoothed)
    # A small dot — noise-like, shouldn't trigger any corner detection.
    img[160, 160] = 0.85
    return img
end

input = studio_image(192, 192)

# Single-scale Harris (level 0 only).
single = harris_corners(input; sigma = 1.2, k = 0.04,
                         threshold = 0.02, min_distance = 5)

# Multi-scale: 3 pyramid levels.
multi = multiscale_harris_corners(input; levels = 2,
                                   sigma = 1.2, k = 0.04,
                                   threshold = 0.02, min_distance = 5)

println("Single-scale Harris:  $(length(single)) corners")
println("Multi-scale Harris:   $(length(multi)) detections across $(maximum(r.level for r in multi) + 1) levels")
println("  level histogram:")
for L in 0:maximum(r.level for r in multi; init = 0)
    n = count(r -> r.level == L, multi)
    println("    level $L (≈$(2^L)px features): $n")
end
println()

# Overlay: single-scale dots; multi-scale dots sized by level.
single_overlay = input .* 0.5
Viz.mark_points!(single_overlay, single; size = 1, value = 1.0)

multi_overlay = input .* 0.5
for r in multi
    Viz.mark_points!(multi_overlay, [(r.row, r.col)];
                     size = 1 + r.level, value = 1.0)
end

outdir = joinpath(@__DIR__, "..", "artifacts", "15_multiscale_harris")
mkpath(outdir)

PNM.save_pgm(joinpath(outdir, "00_input.pgm"),           input)
PNM.save_pgm(joinpath(outdir, "01_single_scale.pgm"),    single_overlay)
PNM.save_pgm(joinpath(outdir, "02_multi_scale.pgm"),     multi_overlay)

grid = Viz.montage([input, single_overlay, multi_overlay];
                   cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage.pgm"), grid)

println("Montage tiles (row-major): input | single-scale | multi-scale")
println("→ $outdir")
