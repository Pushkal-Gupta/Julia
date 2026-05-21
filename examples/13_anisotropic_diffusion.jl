#!/usr/bin/env julia
# 13_anisotropic_diffusion.jl
#
# The point of Perona-Malik in one sentence: it smooths the flat
# regions of an image hard while almost not touching the edges. The
# trick is that the local conductivity in the diffusion equation
# depends on the gradient magnitude — small where gradient is large,
# big where gradient is small.
#
# This script puts three smoothers against each other on a noisy
# image with a strong straight edge:
#
#   - Linear diffusion = Gaussian smoothing. Loses the edge with the
#     noise.
#   - Bilateral. Preserves the edge well; the spatial Gaussian decides
#     the smoothing scale.
#   - Perona-Malik (both conduction functions). Preserves the edge
#     even better and lets you crank the iteration count without
#     blurring it.
#
# Plus a Canny on each result, scored against the ground truth from
# the clean image — the same setup as the noise lab.
#
#   julia --project=. examples/13_anisotropic_diffusion.jl
#
# What I'm expecting:
#   - The "raw" Canny on the noisy input is contaminated with noise
#     edges everywhere.
#   - Gaussian helps but blurs the real edges so Canny localizes them
#     a pixel or two off, hurting precision at the small tolerance.
#   - Bilateral and Perona-Malik should both score above Gaussian.
#     The two of them often tie; on this image we'll see which one
#     edges out.

using ImageLab
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution
using ImageLab.Edges, ImageLab.Filters, ImageLab.Metrics
using ImageLab.Viz, ImageLab.PNM
using Printf

function studio_image(h, w)
    img = zeros(Float64, h, w)
    # A strong rectangle (sharp gradient on every side)
    img[18:60, 22:75] .= 0.85
    # A disk on the right
    cy, cx = 80, 95
    r = 16
    for j in 1:w, i in 1:h
        ((i - cy)^2 + (j - cx)^2 ≤ r^2) && (img[i, j] = 0.6)
    end
    # A faint mid-gray ramp at the bottom (so I can see what's preserved)
    for j in 1:w, i in 80:110
        img[i, j] = max(img[i, j], 0.15 + (i - 80) / 60)
    end
    return img
end

clean = studio_image(128, 128)
noisy = Synth.gaussian_noise(clean; sigma = 0.10, seed = 7)

# Same Canny parameters everywhere — only the pre-smoother changes.
const CANNY_SIGMA, CANNY_LOW, CANNY_HIGH, TOLERANCE = 1.0, 0.06, 0.18, 1
run_canny(img) = canny(img; sigma = CANNY_SIGMA, low = CANNY_LOW, high = CANNY_HIGH)
gt_edges = run_canny(clean)

# Smoothers under test. Each is (name, fn).
g1 = gaussian1d(13; sigma = 2.0)
smoothers = [
    ("raw (no smooth)",   identity),
    ("Gaussian σ=2",      img -> separable_correlate2d(img, g1, g1; pad = :replicate)),
    ("bilateral",         img -> bilateral_filter(img; window = 7,
                                                     sigma_spatial = 2.0,
                                                     sigma_intensity = 0.10)),
    ("PM exp K=0.05",     img -> perona_malik(img; iterations = 40, K = 0.05,
                                                   mode = :exponential, lambda = 0.2)),
    ("PM exp K=0.15",     img -> perona_malik(img; iterations = 40, K = 0.15,
                                                   mode = :exponential, lambda = 0.2)),
    ("PM rational K=0.10", img -> perona_malik(img; iterations = 40, K = 0.10,
                                                    mode = :rational,    lambda = 0.2)),
]

results = []
for (name, fn) in smoothers
    denoised = fn(noisy)
    edges = run_canny(denoised)
    p, r, f1 = edge_match_stats(edges, gt_edges; tolerance = TOLERANCE)
    push!(results, (name = name, denoised = denoised, edges = edges,
                    p = p, r = r, f1 = f1))
end

# Print the table.
println("Anisotropic-diffusion comparison on a noisy image (σ=0.10):")
println("  Canny: σ=$CANNY_SIGMA, low=$CANNY_LOW, high=$CANNY_HIGH; tolerance=$TOLERANCE px")
println("  ground-truth edges: $(sum(gt_edges))")
println()
@printf("%-22s  %-9s  %-8s  %-8s  %s\n",
        "smoother", "edge px", "precision", "recall", "F1")
println(repeat("-", 60))
for r in results
    @printf("%-22s  %-9d  %-8.3f  %-8.3f  %.3f\n",
            r.name, sum(r.edges), r.p, r.r, r.f1)
end
println()

# Save artifacts: denoised image and edges per cell, plus a montage.
outdir = joinpath(@__DIR__, "..", "artifacts", "13_anisotropic_diffusion")
mkpath(outdir)

PNM.save_pgm(joinpath(outdir, "00_clean.pgm"),       clean)
PNM.save_pgm(joinpath(outdir, "00_gt_edges.pgm"),    Float64.(gt_edges))
PNM.save_pgm(joinpath(outdir, "01_noisy.pgm"),       noisy)

for (k, r) in enumerate(results)
    tag = lpad(string(k), 2, '0')
    safename = replace(r.name, " " => "_", "=" => "", "(" => "", ")" => "", "σ" => "s")
    PNM.save_pgm(joinpath(outdir, "$(tag)_denoised_$(safename).pgm"), r.denoised)
    PNM.save_pgm(joinpath(outdir, "$(tag)_edges_$(safename).pgm"),    Float64.(r.edges))
end

# Two montages: one of denoised images, one of edge maps. 2 rows × 3 cols.
denoised_tiles = [r.denoised   for r in results]
edge_tiles     = [Float64.(r.edges) for r in results]
PNM.save_pgm(joinpath(outdir, "montage_denoised.pgm"),
             Viz.montage(denoised_tiles; cols = 3, gap = 4, background = 0.5))
PNM.save_pgm(joinpath(outdir, "montage_edges.pgm"),
             Viz.montage(edge_tiles; cols = 3, gap = 4, background = 0.5))

println("Tile order in both montages (row-major):")
for (k, r) in enumerate(results)
    @printf("  %d: %s\n", k, r.name)
end
println("→ $outdir")
