#!/usr/bin/env julia
# 16_laplacian_blending.jl
#
# The classical "two faces, one line down the middle" demo, the
# application that motivated Burt and Adelson's pyramid algorithm in
# 1983. I take two images with different textures and intensities,
# define a mask that's 1 on the left and 0 on the right, and blend
# them two ways:
#
#   - Naive: A · mask + B · (1 − mask). Sharp seam where the mask is
#     sharp. Each pixel jumps from one image's content to the other
#     in a single step.
#   - Laplacian-pyramid blending: blend each frequency band at its
#     own spatial scale. High-frequency detail uses a sharp mask;
#     low-frequency intensity uses a smoothed mask. The result has
#     no visible seam.
#
#   julia --project=. examples/16_laplacian_blending.jl
#
# What I'm looking for:
#   - The naive blend shows a hard line in the middle.
#   - The Laplacian blend transitions over ~16 pixels of width
#     (proportional to the number of pyramid levels) but preserves
#     fine detail from each side.

using ImageLab
using ImageLab.Synth, ImageLab.Pyramids, ImageLab.Viz, ImageLab.PNM

# Two distinct synthetic "images" that look different on either side
# of a vertical boundary. Side A is a high-frequency checkerboard;
# side B is a smooth gradient.
function image_A(h, w)
    canvas = Synth.checkerboard(h, w; tile = 12) .* 0.7 .+ 0.1
    disk   = Synth.circle(h, w; radius = 22)
    return max.(canvas, 0.8 .* disk)
end

function image_B(h, w)
    g = Synth.ramp(h, w; axis = :y)
    sq = Synth.square(h, w; side = 40, value = 0.5)
    return max.(g .* 0.7, sq)
end

# Build the inputs and the mask.
A = image_A(128, 128)
B = image_B(128, 128)
mask = zeros(Float64, 128, 128)
mask[:, 1:64] .= 1.0           # left half from A, right half from B — sharp seam

LEVELS = 5

# Naive linear blend (this is what produces the visible seam).
naive = mask .* A .+ (1 .- mask) .* B

# Multi-band Laplacian blend (no visible seam).
blended = laplacian_blend(A, B, mask; levels = LEVELS)

# A second demo: a circular mask. Cuts a disk out of A and pastes it
# onto B; the soft mask edge avoids any rim artifact in the result.
soft_mask = zeros(Float64, 128, 128)
cy, cx, r = 64, 64, 36
for j in 1:128, i in 1:128
    if (i - cy)^2 + (j - cx)^2 ≤ r^2
        soft_mask[i, j] = 1.0
    end
end

naive_disk    = soft_mask .* A .+ (1 .- soft_mask) .* B
blended_disk  = laplacian_blend(A, B, soft_mask; levels = LEVELS)

# Save the gallery.
outdir = joinpath(@__DIR__, "..", "artifacts", "16_laplacian_blending")
mkpath(outdir)

PNM.save_pgm(joinpath(outdir, "00_A.pgm"),                 A)
PNM.save_pgm(joinpath(outdir, "01_B.pgm"),                 B)
PNM.save_pgm(joinpath(outdir, "02_mask_vertical.pgm"),     mask)
PNM.save_pgm(joinpath(outdir, "03_naive_vertical.pgm"),    clamp.(naive, 0, 1))
PNM.save_pgm(joinpath(outdir, "04_blended_vertical.pgm"),  clamp.(blended, 0, 1))
PNM.save_pgm(joinpath(outdir, "05_mask_disk.pgm"),         soft_mask)
PNM.save_pgm(joinpath(outdir, "06_naive_disk.pgm"),        clamp.(naive_disk, 0, 1))
PNM.save_pgm(joinpath(outdir, "07_blended_disk.pgm"),      clamp.(blended_disk, 0, 1))

# Two side-by-side comparison montages.
mont_vertical = Viz.montage(
    [A, B, mask,
     clamp.(naive, 0, 1), clamp.(blended, 0, 1), abs.(blended .- naive)];
    cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage_vertical.pgm"), mont_vertical)

mont_disk = Viz.montage(
    [A, B, soft_mask,
     clamp.(naive_disk, 0, 1), clamp.(blended_disk, 0, 1), abs.(blended_disk .- naive_disk)];
    cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage_disk.pgm"), mont_disk)

# Quick numeric witness: peak difference at the seam.
seam_col = 64
naive_jump = maximum(abs.(diff(naive[:, seam_col - 1:seam_col + 1]; dims = 2)))
blended_jump = maximum(abs.(diff(blended[:, seam_col - 1:seam_col + 1]; dims = 2)))

println("Laplacian blending demo (levels = $LEVELS):")
println("  vertical seam at column $seam_col")
println("    naive    : max one-pixel jump across the seam  = ", round(naive_jump, digits = 3))
println("    blended  : max one-pixel jump across the seam  = ", round(blended_jump, digits = 3))
println()
println("Montage tile order (row-major):")
println("    A           | B            | mask")
println("    naive blend | pyramid blend | |difference|")
println("→ $outdir")
