#!/usr/bin/env julia
# 23_pyramidal_horn_schunck.jl
#
# Pyramidal Horn-Schunck. The coarse-to-fine driver from
# `lucas_kanade_pyramid` is algorithm-agnostic — wrap it around
# `horn_schunck` instead of `lucas_kanade` and HS gets the same
# big-motion fix.
#
#   julia --project=. examples/23_pyramidal_horn_schunck.jl
#
# What this produces in artifacts/23_pyramidal_horn_schunck/:
#   01_big_motion.ppm        — plain HS (left) vs pyramidal HS (right)
#                               on a 4-pixel translation
#   02_motion_sweep.pgm      — recovered |flow| at shifts of 1, 2, 4, 6 px
#   03_warped_residual.pgm   — |warp(I₂, recovered_flow) − I₁|
#
# Why I lowered the test pattern's spatial frequency from the previous
# chunks: the Gaussian pyramid's 5-tap [1, 4, 6, 4, 1] / 16 filter
# isn't a perfect low-pass. A sinusoid at frequency 0.04 cycles/pixel
# becomes ~0.32 cycles/pixel after three pyramid reductions, which is
# past Nyquist for the 5-tap filter and aliases. At 0.025 cycles/pixel
# in the original, three reductions land at 0.2 — well inside the
# safe band. The fix isn't HS-specific; LK pyramid would alias the
# same way if I pushed it harder.

using ImageLab
using ImageLab.Flow
using ImageLab.Viz, ImageLab.PNM
using Printf

outdir = joinpath(@__DIR__, "..", "artifacts", "23_pyramidal_horn_schunck")
mkpath(outdir)

function low_freq_frame(H, W; shift_u = 0.0, shift_v = 0.0)
    img = zeros(H, W)
    for j in 1:W, i in 1:H
        x = j - shift_u
        y = i - shift_v
        img[i, j] = 0.5 + 0.4 * sin(2π * 0.025 * x) * cos(2π * 0.025 * y)
    end
    return img
end

H, W = 192, 192

# ── 1. Big-motion test: 4-pixel translation ────────────────────────────────
println("== Translation (u, v) = (4.0, -2.0), well past plain HS's regime ==")
img1 = low_freq_frame(H, W)
img2 = low_freq_frame(H, W; shift_u = 4.0, shift_v = -2.0)

hs    = horn_schunck(img1, img2; alpha = 0.1, iterations = 300)
hsp   = horn_schunck_pyramid(img1, img2;
                             levels = 3, alpha = 0.1, iters_per_level = 150)

inner = (60:H-60, 60:W-60)
function mean_inside(field)
    sum(field[inner...]) / length(field[inner...])
end

@printf("  plain HS    : recovered (u, v) = (%+.3f, %+.3f)\n",
        mean_inside(hs.u),  mean_inside(hs.v))
@printf("  pyramid HS  : recovered (u, v) = (%+.3f, %+.3f)\n",
        mean_inside(hsp.u), mean_inside(hsp.v))

Rh,  Gh,  Bh  = flow_to_rgb(hs.u,  hs.v;  max_mag = 5.0)
Rhp, Ghp, Bhp = flow_to_rgb(hsp.u, hsp.v; max_mag = 5.0)
PNM.save_ppm(joinpath(outdir, "01_big_motion.ppm"),
             [Rh Rhp], [Gh Ghp], [Bh Bhp])

# ── 2. Motion sweep — recover increasingly large motions ────────────────────
println()
println("== Motion sweep — pyramid HS at shifts of 1, 2, 4, 6 pixels ==")
sweep_tiles = Matrix{Float64}[]
for shift in (1.0, 2.0, 4.0, 6.0)
    img2_k = low_freq_frame(H, W; shift_u = shift)
    f = horn_schunck_pyramid(img1, img2_k; levels = 3, alpha = 0.1,
                             iters_per_level = 150)
    @printf("  shift = %3.1f  →  recovered u = %+.3f\n",
            shift, mean_inside(f.u))
    R, G, B = flow_to_rgb(f.u, f.v; max_mag = 7.0)
    push!(sweep_tiles, 0.299 .* R .+ 0.587 .* G .+ 0.114 .* B)
end
PNM.save_pgm(joinpath(outdir, "02_motion_sweep.pgm"),
             Viz.montage(sweep_tiles; cols = 4, gap = 4, background = 0.0))

# ── 3. How well does the recovered flow reconstruct I₁? ─────────────────────
warped_back = warp_bilinear(img2, hsp.u, hsp.v)
residual = abs.(warped_back .- img1)
@printf("\n  mean |warp(I₂, flow) − I₁| inside the test region: %.4f\n",
        sum(residual[inner...]) / length(residual[inner...]))
PNM.save_pgm(joinpath(outdir, "03_warped_residual.pgm"),
             Viz.normalize01(residual))

println()
println("→ $outdir")
println("Files:")
println("  01_big_motion.ppm        — plain HS (left) vs pyramidal HS (right)")
println("  02_motion_sweep.pgm      — pyramid HS at 1, 2, 4, 6 px translations")
println("  03_warped_residual.pgm   — |warp(I₂, flow) − I₁|")
