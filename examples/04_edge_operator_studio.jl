#!/usr/bin/env julia
# 04_edge_operator_studio.jl
#
# I want to see Sobel, Prewitt, Scharr, and Roberts side by side on the
# same input. Each of them detects edges, but they weight the smoothing
# direction differently — Sobel uses [1,2,1], Prewitt [1,1,1], Scharr
# [3,10,3], and Roberts skips smoothing entirely (it's a 2×2 cross).
#
# This script builds a synthetic test image, runs all four, and writes
# a 3×3 montage so the differences are visible at a glance.
#
#   julia --project=. examples/04_edge_operator_studio.jl
#
# What I'm expecting to see:
#   - Sobel / Prewitt / Scharr all look similar; Scharr should have the
#     best rotational symmetry (cleanest ring on the circle).
#   - Roberts is noisier and offset by half a pixel — it's a historical
#     operator, not what you'd use in practice.
#   - The direction tile (bottom-middle) shows gradient angle masked by
#     magnitude — straight edges have constant direction, curved ones
#     fan out smoothly.

using ImageLab
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution
using ImageLab.Edges, ImageLab.Viz, ImageLab.PNM

# Build a synthetic image that's rich enough to stress every operator:
# a circle (curved edges), a rectangle (straight edges + corners), and a
# diagonal line (tests directional sensitivity).
function studio_image(h, w)
    img = zeros(Float64, h, w)
    # Bright filled circle, off-center.
    cy, cx = h * 0.40, w * 0.35
    r = min(h, w) * 0.22
    for j in 1:w, i in 1:h
        if (i - cy)^2 + (j - cx)^2 ≤ r^2
            img[i, j] = 1.0
        end
    end
    # Mid-gray rectangle.
    i0 = round(Int, h * 0.55); i1 = round(Int, h * 0.85)
    j0 = round(Int, w * 0.55); j1 = round(Int, w * 0.90)
    img[i0:i1, j0:j1] .= 0.6
    # Diagonal line (anti-aliasing-free: just stamp 1-pixel-thick).
    for k in 1:min(h, w)
        i = k; j = k
        (1 ≤ i ≤ h && 1 ≤ j ≤ w) && (img[i, j] = max(img[i, j], 0.85))
    end
    return img
end

# Normalize a signed gradient image for inspection — zero pinned to mid-gray.
# Magnitude images are non-negative so we use normalize01 for those.
function direction_to_gray(θ, mag; mag_floor)
    # Map θ to [0, 1) on the half-circle (edge direction is mod π).
    # Pixels with weak magnitude are shown as mid-gray so they don't bias
    # the eye toward "interesting" directions in flat regions.
    out = fill(0.5, size(θ))
    for i in eachindex(θ)
        if mag[i] ≥ mag_floor
            out[i] = mod(θ[i], π) / π
        end
    end
    return out
end

input = studio_image(128, 128)

# All four gradient operators.
sobel_gx,   sobel_gy   = gradient(input, :sobel;   pad = :replicate)
prewitt_gx, prewitt_gy = gradient(input, :prewitt; pad = :replicate)
scharr_gx,  scharr_gy  = gradient(input, :scharr;  pad = :replicate)
roberts_gx, roberts_gy = gradient(input, :roberts; pad = :replicate)

sobel_mag   = gradient_magnitude(sobel_gx, sobel_gy)
prewitt_mag = gradient_magnitude(prewitt_gx, prewitt_gy)
scharr_mag  = gradient_magnitude(scharr_gx, scharr_gy)
roberts_mag = gradient_magnitude(roberts_gx, roberts_gy)

sobel_θ = gradient_direction(sobel_gx, sobel_gy)

# Direction map, masked by magnitude > 50th percentile so I'm only
# looking at directions where edges actually exist.
mag_floor = percentile_threshold(sobel_mag, 0.5)
sobel_dir_gray = direction_to_gray(sobel_θ, sobel_mag; mag_floor = mag_floor)

# Top 10% of Sobel magnitude → a clean binary edge map.
edge_thresh = percentile_threshold(sobel_mag, 0.90)
sobel_thresh = Float64.(threshold_mask(sobel_mag, edge_thresh))

outdir = joinpath(@__DIR__, "..", "artifacts", "04_edge_operator_studio")
mkpath(outdir)

PNM.save_pgm(joinpath(outdir, "00_input.pgm"),         input)
PNM.save_pgm(joinpath(outdir, "01_sobel_gx.pgm"),      Viz.signed_to_gray(sobel_gx))
PNM.save_pgm(joinpath(outdir, "02_sobel_gy.pgm"),      Viz.signed_to_gray(sobel_gy))
PNM.save_pgm(joinpath(outdir, "03_sobel_mag.pgm"),     Viz.normalize01(sobel_mag))
PNM.save_pgm(joinpath(outdir, "04_prewitt_mag.pgm"),   Viz.normalize01(prewitt_mag))
PNM.save_pgm(joinpath(outdir, "05_scharr_mag.pgm"),    Viz.normalize01(scharr_mag))
PNM.save_pgm(joinpath(outdir, "06_roberts_mag.pgm"),   Viz.normalize01(roberts_mag))
PNM.save_pgm(joinpath(outdir, "07_sobel_direction.pgm"), sobel_dir_gray)
PNM.save_pgm(joinpath(outdir, "08_sobel_p90_mask.pgm"),  sobel_thresh)

# Assemble a 3×3 montage of the most informative tiles.
tiles = [
    input,                            Viz.signed_to_gray(sobel_gx),    Viz.signed_to_gray(sobel_gy),
    Viz.normalize01(sobel_mag),       Viz.normalize01(prewitt_mag),    Viz.normalize01(scharr_mag),
    Viz.normalize01(roberts_mag),     sobel_dir_gray,                  sobel_thresh,
]
grid = Viz.montage(tiles; cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage.pgm"), grid)

println("First-order edge operator studio:")
println("  tiles (row-major):")
println("    input            | Sobel gx       | Sobel gy")
println("    Sobel mag        | Prewitt mag    | Scharr mag")
println("    Roberts mag      | Sobel θ (mask) | Sobel ≥ p90 magnitude")
println("→ $outdir")
println("Open montage.pgm for the side-by-side view.")
