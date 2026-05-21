#!/usr/bin/env julia
# 14_real_image_pipeline.jl
#
# Run the whole pipeline on a real PNG. If no argument is given, the
# script generates a synthetic test image, saves it as a PNG, and
# loads it back through Photos.load_grayscale — that way it works
# out of the box without me having to ship a sample image. To run on
# my own photo:
#
#   julia --project=. examples/14_real_image_pipeline.jl path/to/photo.png
#
# The script writes:
#   - blur: a separable Gaussian smoothing
#   - canny: the final edge map
#   - canny_overlay: edges overlaid on the input
#   - harris_corners: corners overlaid on the input
#   - pyramid_montage: the Gaussian pyramid as a row of cells
#
# This is the moment where everything I've built so far stops being
# a "synthetic image lab" and becomes an actual image-processing
# pipeline I can point at any image on my disk.

using ImageLab
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution
using ImageLab.Edges, ImageLab.Filters, ImageLab.Features
using ImageLab.Pyramids, ImageLab.Viz, ImageLab.Photos
using Printf

# Resolve the input. If none, fall back to a synthetic PNG we make on
# the fly so the script self-contains.
function sample_image()
    h, w = 192, 192
    img = zeros(Float64, h, w)
    # Rectangles
    img[20:60, 20:80]  .= 0.85
    img[110:160, 90:170] .= 0.55
    # A disk
    cy, cx = 80, 130
    for j in 1:w, i in 1:h
        ((i - cy)^2 + (j - cx)^2 ≤ 18^2) && (img[i, j] = 0.7)
    end
    # A thin diagonal line (tests Hough-style detection later)
    for k in 1:min(h, w)
        if 1 ≤ k ≤ h && 1 ≤ k ≤ w
            img[k, k] = max(img[k, k], 0.9)
        end
    end
    # Background gradient + light noise
    for j in 1:w, i in 1:h
        img[i, j] = max(img[i, j], 0.08 + 0.2 * (j - 1) / (w - 1))
    end
    return Synth.gaussian_noise(img; sigma = 0.03, seed = 13)
end

outdir = joinpath(@__DIR__, "..", "artifacts", "14_real_image_pipeline")
mkpath(outdir)

input_path = if isempty(ARGS)
    fallback = joinpath(outdir, "00_sample_input.png")
    Photos.save_grayscale(fallback, sample_image())
    @info "No argument given — using the auto-generated sample at $fallback"
    fallback
else
    ARGS[1]
end

println("Loading: $input_path")
img = Photos.load_grayscale(input_path)
H, W = size(img)
println("  size: $H × $W")
println("  range: [$(round(minimum(img), digits=3)), $(round(maximum(img), digits=3))]")
println()

# ── 1. Gaussian smoothing (the separable path) ────────────────────────────────
sigma = max(1.0, min(H, W) / 256)   # mildly adaptive: stronger for bigger images
g = gaussian1d(Int(2 * ceil(3 * sigma) + 1); sigma = sigma)
blurred = separable_correlate2d(img, g, g; pad = :replicate)
Photos.save_grayscale(joinpath(outdir, "01_blurred.png"), blurred)

# ── 2. Canny ──────────────────────────────────────────────────────────────────
edges = canny(img; sigma = sigma, low = 0.06, high = 0.18)
@printf("Canny: σ=%.2f, low=0.06, high=0.18 → %d edge pixels (%.2f%% of image)\n",
        sigma, sum(edges), 100 * sum(edges) / length(edges))
Photos.save_grayscale(joinpath(outdir, "02_canny.png"), Float64.(edges))

# Overlay edges on the input — darken the input slightly, draw edges in white.
edges_overlay = img .* 0.6
edges_overlay[edges] .= 1.0
Photos.save_grayscale(joinpath(outdir, "03_canny_overlay.png"), edges_overlay)

# ── 3. Harris corners ─────────────────────────────────────────────────────────
corners = harris_corners(img; sigma = max(sigma, 1.0),
                              threshold = 0.02, min_distance = max(3, Int(round(min(H, W) / 64))))
println("Harris: $(length(corners)) corners")

corners_overlay = img .* 0.6
Viz.mark_points!(corners_overlay, corners; size = max(1, min(H, W) ÷ 96), value = 1.0)
Photos.save_grayscale(joinpath(outdir, "04_harris_overlay.png"), corners_overlay)

# ── 4. Gaussian pyramid row ───────────────────────────────────────────────────
levels = max(1, min(4, Int(floor(log2(min(H, W) / 16)))))
G = gaussian_pyramid(img; levels = levels)
println("Gaussian pyramid: $(length(G)) levels at $(join(["$(size(g))" for g in G], ", "))")

# Pad each pyramid level into a uniform-sized cell, then montage.
function pad_to(small, h, w; bg = 0.5)
    H0, W0 = size(small)
    out = fill(Float64(bg), h, w)
    out[1:H0, 1:W0] .= small
    return out
end
cell_h, cell_w = size(G[1])
pyramid_tiles = [pad_to(g, cell_h, cell_w) for g in G]
pyramid_grid = Viz.montage(pyramid_tiles; cols = length(G), gap = 4, background = 0.5)
Photos.save_grayscale(joinpath(outdir, "05_pyramid.png"), pyramid_grid)

println()
println("→ $outdir")
println("Files: 01_blurred.png, 02_canny.png, 03_canny_overlay.png,")
println("       04_harris_overlay.png, 05_pyramid.png")
