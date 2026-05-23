#!/usr/bin/env julia
# 21_pyramidal_optical_flow.jl
#
# Pyramidal Lucas-Kanade. The fix for the small-motion limit of
# plain LK. Build Gaussian pyramids of both frames, run LK at the
# coarsest level (where a big motion looks small), upsample the
# estimate, warp frame 2 by it, run LK on the residual, repeat. At
# every level the residual motion is well under a pixel, so the
# OFC linearization is valid even when the underlying motion is
# several pixels.
#
#   julia --project=. examples/21_pyramidal_optical_flow.jl
#
# What this produces in artifacts/21_pyramidal_optical_flow/:
#   01_big_translation.ppm   — frame A, frame B, recovered flow montage
#                               for a 4-pixel translation
#   02_plain_vs_pyramid.ppm  — same input, plain LK vs pyramid LK
#                               flow visualizations side by side
#   03_level_by_level.ppm    — pyramid-LK flow estimates at each
#                               level, finest on the right
#   04_warped_minus_target.pgm — how close warp(I₂, recovered_flow)
#                                 is to I₁ after the pyramid sweeps

using ImageLab
using ImageLab.Flow
using ImageLab.Pyramids
using ImageLab.Viz, ImageLab.PNM
using Printf

outdir = joinpath(@__DIR__, "..", "artifacts", "21_pyramidal_optical_flow")
mkpath(outdir)

# Continuous textured pattern — same one I used in the plain-LK demo.
function textured_frame(H, W; shift_u = 0.0, shift_v = 0.0)
    img = zeros(H, W)
    for j in 1:W, i in 1:H
        x = j - shift_u
        y = i - shift_v
        img[i, j] = 0.5 + 0.4 * sin(2π * 0.04 * x) * cos(2π * 0.035 * y)
    end
    return img
end

H, W = 128, 128
true_u, true_v = 4.0, -2.5

# ── 1. The big-motion test case ─────────────────────────────────────────────
println("== Translation ($(true_u), $(true_v)) — well outside plain-LK range ==")
img1 = textured_frame(H, W)
img2 = textured_frame(H, W; shift_u = true_u, shift_v = true_v)

flow_plain = lucas_kanade(img1, img2; window_size = 15, sigma = 2.0)
flow_pyr   = lucas_kanade_pyramid(img1, img2;
                                  levels = 4, window_size = 15, sigma = 2.0,
                                  iters_per_level = 3)

inner = (40:H-40, 40:W-40)
function mean_inside(field)
    sum(field[inner...]) / length(field[inner...])
end

@printf("  plain LK:    recovered (u, v) = (%+.3f, %+.3f)\n",
        mean_inside(flow_plain.u), mean_inside(flow_plain.v))
@printf("  pyramid LK:  recovered (u, v) = (%+.3f, %+.3f)\n",
        mean_inside(flow_pyr.u),   mean_inside(flow_pyr.v))

# ── 2. Save montages ────────────────────────────────────────────────────────
PNM.save_pgm(joinpath(outdir, "01_big_translation.pgm"),
             Viz.montage([img1, img2]; cols = 2, gap = 4, background = 0.5))

# Plain vs pyramid colour-coded flow, with the same colour scale.
max_mag = 6.0
Rp, Gp, Bp = flow_to_rgb(flow_plain.u, flow_plain.v; max_mag = max_mag)
Ry, Gy, By = flow_to_rgb(flow_pyr.u,   flow_pyr.v;   max_mag = max_mag)
PNM.save_ppm(joinpath(outdir, "02_plain_vs_pyramid.ppm"),
             [Rp Ry], [Gp Gy], [Bp By])

# ── 3. Level-by-level: re-run the pyramid sweep manually so I can snapshot
#       the flow estimate after each level. This is just a hand-rolled
#       copy of what `lucas_kanade_pyramid` does internally — normal
#       callers should just use that function.
println()
println("== Per-level convergence ==")
level_tiles = let levels = 4
    pyr1 = gaussian_pyramid(img1; levels = levels)
    pyr2 = gaussian_pyramid(img2; levels = levels)
    L = length(pyr1)
    cH, cW = size(pyr1[L])
    u = zeros(cH, cW)
    v = zeros(cH, cW)
    tiles = Matrix{Float64}[]
    for k in L:-1:1
        for _ in 1:3
            warped = warp_bilinear(pyr2[k], u, v)
            r = lucas_kanade(pyr1[k], warped; window_size = 15, sigma = 2.0)
            u .+= r.u
            v .+= r.v
        end
        @printf("  level %d (%3d × %3d): mean (u, v) = (%+.3f, %+.3f)\n",
                k - 1, size(pyr1[k])...,
                sum(u) / length(u), sum(v) / length(v))
        # Resample the flow back to full resolution for visualization.
        u_disp = repeat(u, inner = (H ÷ size(u, 1), W ÷ size(u, 2)))[1:H, 1:W]
        v_disp = repeat(v, inner = (H ÷ size(v, 1), W ÷ size(v, 2)))[1:H, 1:W]
        Rk, Gk, Bk = flow_to_rgb(u_disp, v_disp; max_mag = max_mag)
        push!(tiles, 0.299 .* Rk .+ 0.587 .* Gk .+ 0.114 .* Bk)
        if k > 1
            next_H, next_W = size(pyr1[k - 1])
            u_new = zeros(next_H, next_W)
            v_new = zeros(next_H, next_W)
            for j in 1:next_W, i in 1:next_H
                si = next_H > 1 ? 1 + (i - 1) * (size(u, 1) - 1) / (next_H - 1) : 1.0
                sj = next_W > 1 ? 1 + (j - 1) * (size(u, 2) - 1) / (next_W - 1) : 1.0
                i0 = floor(Int, si); j0 = floor(Int, sj)
                ai = si - i0;        aj = sj - j0
                i0c = clamp(i0, 1, size(u, 1)); i1c = clamp(i0 + 1, 1, size(u, 1))
                j0c = clamp(j0, 1, size(u, 2)); j1c = clamp(j0 + 1, 1, size(u, 2))
                u_new[i, j] = 2 * ((1-ai)*(1-aj)*u[i0c, j0c] + (1-ai)*aj*u[i0c, j1c] +
                                   ai*(1-aj)*u[i1c, j0c] + ai*aj*u[i1c, j1c])
                v_new[i, j] = 2 * ((1-ai)*(1-aj)*v[i0c, j0c] + (1-ai)*aj*v[i0c, j1c] +
                                   ai*(1-aj)*v[i1c, j0c] + ai*aj*v[i1c, j1c])
            end
            u = u_new
            v = v_new
        end
    end
    tiles
end

PNM.save_pgm(joinpath(outdir, "03_level_by_level.pgm"),
             Viz.montage(level_tiles; cols = length(level_tiles),
                         gap = 4, background = 0.0))

# ── 4. Warping I₂ back by the recovered flow should produce ~I₁ ─────────────
warped_final = warp_bilinear(img2, flow_pyr.u, flow_pyr.v)
residual = abs.(warped_final .- img1)
@printf("\n  mean residual |warp(I₂, flow) − I₁| (interior): %.4f\n",
        sum(residual[inner...]) / length(residual[inner...]))
PNM.save_pgm(joinpath(outdir, "04_warped_minus_target.pgm"),
             Viz.normalize01(residual))

println()
println("→ $outdir")
println("Files:")
println("  01_big_translation.pgm    — frame A and frame B side by side")
println("  02_plain_vs_pyramid.ppm   — plain LK (left) vs pyramidal LK (right)")
println("  03_level_by_level.pgm     — flow estimate at each pyramid level")
println("  04_warped_minus_target.pgm — |warp(I₂, recovered_flow) − I₁|")
