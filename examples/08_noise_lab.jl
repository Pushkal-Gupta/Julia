#!/usr/bin/env julia
# 08_noise_lab.jl
#
# A question I've been wanting to ask the pipeline directly: how much
# does the pre-smoothing choice actually matter, and does it depend on
# the kind of noise?
#
# The setup:
#   1. Build a clean synthetic image.
#   2. Run Canny on it — that's my ground-truth edge map.
#   3. Add either Gaussian noise or salt-and-pepper noise.
#   4. Apply each of a few smoothers (or none) to the noisy image.
#   5. Run Canny on each smoothed result with identical parameters.
#   6. Score each output against the ground truth via precision /
#      recall / F1 with a 1-pixel tolerance.
#
#   julia --project=. examples/08_noise_lab.jl
#
# My prediction before running:
#   - Salt-and-pepper: median should win by a wide margin. Gaussian
#     smoothing spreads each speck into a soft blob; median removes
#     individual specks outright.
#   - Gaussian noise: a stronger Gaussian (or bilateral with low
#     intensity sigma) should help most. Median doesn't have a
#     particular reason to help here — it's not a noise model match.
#   - "No smoother": both noise types should produce massively more
#     edges (false positives) than the ground truth.

using ImageLab
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution
using ImageLab.Edges, ImageLab.Filters, ImageLab.Metrics, ImageLab.Viz, ImageLab.PNM
using Printf

# A higher-contrast image with mixed straight + curved edges. The
# "ground-truth" Canny on this is robust.
function studio_image(h, w)
    img = zeros(Float64, h, w)
    # Disk:
    cy, cx = h * 0.35, w * 0.40
    r = min(h, w) * 0.18
    for j in 1:w, i in 1:h
        (i - cy)^2 + (j - cx)^2 ≤ r^2 && (img[i, j] = 1.0)
    end
    # Square:
    i0, i1 = round(Int, h * 0.55), round(Int, h * 0.85)
    j0, j1 = round(Int, w * 0.55), round(Int, w * 0.85)
    img[i0:i1, j0:j1] .= 0.7
    return img
end

# Fixed Canny parameters across the entire experiment so I'm only
# comparing the effect of the smoother.
const CANNY_SIGMA = 1.0
const CANNY_LOW   = 0.06
const CANNY_HIGH  = 0.18
const TOLERANCE   = 1

run_canny(img) = canny(img; sigma = CANNY_SIGMA, low = CANNY_LOW, high = CANNY_HIGH)

# Smoothers I want to compare. Each is a name + a function from image to image.
const SMOOTHERS = [
    ("none",         identity),
    ("box 3×3",      img -> correlate2d(img, box(3); pad = :replicate)),
    ("box 5×5",      img -> correlate2d(img, box(5); pad = :replicate)),
    ("Gaussian σ=1", img -> let g = gaussian1d(7;  sigma = 1.0);
                                separable_correlate2d(img, g, g; pad = :replicate) end),
    ("Gaussian σ=2", img -> let g = gaussian1d(13; sigma = 2.0);
                                separable_correlate2d(img, g, g; pad = :replicate) end),
    ("median 3×3",   img -> median_filter(img; window = 3)),
    ("median 5×5",   img -> median_filter(img; window = 5)),
    ("bilateral",    img -> bilateral_filter(img; window = 7,
                                              sigma_spatial = 2.0,
                                              sigma_intensity = 0.10)),
]

# The clean image and the reference edge map.
clean = studio_image(128, 128)
gt_edges = run_canny(clean)

# Noise variants.
gaussian_noisy = Synth.gaussian_noise(clean; sigma = 0.12, seed = 41)
sp_noisy       = Synth.salt_pepper(clean;    p     = 0.08, seed = 41)

# Score one (noise, smoother) cell. Returns (denoised image, edges, p, r, f1).
function score(noisy, smoother_fn)
    denoised = smoother_fn(noisy)
    edges = run_canny(denoised)
    p, r, f1 = edge_match_stats(edges, gt_edges; tolerance = TOLERANCE)
    return denoised, edges, p, r, f1
end

# Run the full grid.
outdir = joinpath(@__DIR__, "..", "artifacts", "08_noise_lab")
mkpath(outdir)

PNM.save_pgm(joinpath(outdir, "00_clean.pgm"),         clean)
PNM.save_pgm(joinpath(outdir, "01_gt_edges.pgm"),      Float64.(gt_edges))
PNM.save_pgm(joinpath(outdir, "10_noisy_gaussian.pgm"), gaussian_noisy)
PNM.save_pgm(joinpath(outdir, "20_noisy_saltpepper.pgm"), sp_noisy)

function run_block(noisy, label)
    println("\n── $label ── (clean reference has $(sum(gt_edges)) edge pixels)")
    @printf("  %-15s  %-9s  %-8s  %-8s  %-8s\n", "smoother", "edge px", "precision", "recall", "F1")
    println("  ", repeat("-", 58))
    edge_tiles = Matrix{Float64}[]
    for (i, (name, fn)) in enumerate(SMOOTHERS)
        _, edges, p, r, f1 = score(noisy, fn)
        @printf("  %-15s  %-9d  %-8.3f  %-8.3f  %-8.3f\n",
                name, sum(edges), p, r, f1)
        push!(edge_tiles, Float64.(edges))
        # Save a per-cell PGM so I can flip through them individually.
        safename = replace(name, " " => "_", "×" => "x", "σ" => "sigma")
        PNM.save_pgm(joinpath(outdir, "$(label)_$(safename).pgm"), Float64.(edges))
    end
    return edge_tiles
end

gauss_tiles = run_block(gaussian_noisy, "gaussian_noise")
sp_tiles    = run_block(sp_noisy,        "saltpepper_noise")

# Montages: 2 rows × 4 columns per noise type. Rows of 4 smoothers fits
# the 8-entry list neatly.
gauss_grid = Viz.montage(gauss_tiles; cols = 4, gap = 3, background = 0.5)
sp_grid    = Viz.montage(sp_tiles;    cols = 4, gap = 3, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage_gaussian.pgm"),   gauss_grid)
PNM.save_pgm(joinpath(outdir, "montage_saltpepper.pgm"), sp_grid)

# Tile order for both montages (read row-major):
#   none           box 3×3        box 5×5        Gaussian σ=1
#   Gaussian σ=2   median 3×3     median 5×5     bilateral
println()
println("Tile order in both montages (row-major):")
println("  none          | box 3×3      | box 5×5      | Gaussian σ=1")
println("  Gaussian σ=2  | median 3×3   | median 5×5   | bilateral")
println()
println("→ $outdir")
