#!/usr/bin/env julia
# 22_horn_schunck.jl
#
# Horn & Schunck dense optical flow — the 1981 contemporary of LK.
# Same `Ix`, `Iy`, `It` ingredients, completely different recipe:
# minimize a global cost functional with a smoothness prior, instead
# of a per-pixel local window. The result is a *dense* flow field
# that fills in textureless regions by diffusion from the textured
# ones nearby, where LK has to return zero.
#
#   julia --project=. examples/22_horn_schunck.jl
#
# What this produces in artifacts/22_horn_schunck/:
#   01_alpha_sweep.pgm        — same shift, four `α` values side by side
#   02_lk_vs_hs.ppm           — LK (left) vs HS (right) on a textured-patch
#                               scene, showing dense propagation
#   03_iteration_trace.pgm    — HS estimate after 10, 50, 200, 1000 iterations
#   04_outside_propagation.pgm — flow magnitude radially outward from a
#                                 textured patch, in the flat region

using ImageLab
using ImageLab.Flow
using ImageLab.Viz, ImageLab.PNM
using Printf

outdir = joinpath(@__DIR__, "..", "artifacts", "22_horn_schunck")
mkpath(outdir)

function sinusoid_frame(H, W; freq = 0.04, shift_u = 0.0, shift_v = 0.0)
    img = zeros(H, W)
    for j in 1:W, i in 1:H
        x = j - shift_u
        y = i - shift_v
        img[i, j] = 0.5 + 0.4 * sin(2π * freq * x) * cos(2π * freq * y)
    end
    return img
end

H, W = 128, 128

# ── 1. α sweep on a uniform-motion textured frame ──────────────────────────
println("== α sweep on a textured pair, true shift (0.5, 0.3) ==")
img1 = sinusoid_frame(H, W)
img2 = sinusoid_frame(H, W; shift_u = 0.5, shift_v = 0.3)

alpha_tiles = Matrix{Float64}[]
for α in (0.05, 0.1, 0.5, 2.0)
    hs = horn_schunck(img1, img2; alpha = α, iterations = 400)
    mu = sum(hs.u[30:H-30, 30:W-30]) / (length(hs.u[30:H-30, 30:W-30]))
    mv = sum(hs.v[30:H-30, 30:W-30]) / (length(hs.v[30:H-30, 30:W-30]))
    @printf("  α = %4.2f  →  mean (u, v) = (%+.3f, %+.3f)\n", α, mu, mv)
    R, G, B = flow_to_rgb(hs.u, hs.v; max_mag = 1.0)
    push!(alpha_tiles, 0.299 .* R .+ 0.587 .* G .+ 0.114 .* B)
end
PNM.save_pgm(joinpath(outdir, "01_alpha_sweep.pgm"),
             Viz.montage(alpha_tiles; cols = 4, gap = 4, background = 0.0))

# ── 2. LK vs HS on a textured patch in a flat field ────────────────────────
println()
println("== LK vs HS: textured patch moves inside a flat background ==")
img3 = fill(0.5, H, W)
img3[40:90, 40:90] .= sinusoid_frame(51, 51; freq = 0.04)
img4 = fill(0.5, H, W)
img4[40:90, 40:90] .= sinusoid_frame(51, 51; freq = 0.04, shift_u = 0.5)

lk = lucas_kanade(img3, img4; window_size = 15, sigma = 2.0)
hs = horn_schunck(img3, img4; alpha = 0.5, iterations = 3000)

@printf("  LK on patch (interior): u = %+.3f  (should be ~0.5)\n",
        sum(lk.u[55:75, 55:75]) / length(lk.u[55:75, 55:75]))
@printf("  LK in flat region:      u = %+.3f  (no texture → 0)\n",
        sum(abs, lk.u[5:25, 55:75]) / length(lk.u[5:25, 55:75]))
@printf("  HS on patch (interior): u = %+.3f\n",
        sum(hs.u[55:75, 55:75]) / length(hs.u[55:75, 55:75]))
@printf("  HS in flat region:      u = %+.3f  (diffused outward)\n",
        sum(hs.u[5:25, 55:75]) / length(hs.u[5:25, 55:75]))

Rl, Gl, Bl = flow_to_rgb(lk.u, lk.v; max_mag = 1.0)
Rh, Gh, Bh = flow_to_rgb(hs.u, hs.v; max_mag = 1.0)
PNM.save_ppm(joinpath(outdir, "02_lk_vs_hs.ppm"),
             [Rl Rh], [Gl Gh], [Bl Bh])

# ── 3. Iteration trace: watching the diffusion happen ──────────────────────
println()
println("== HS iteration trace, same textured-patch input ==")
iter_tiles = Matrix{Float64}[]
for iters in (10, 50, 200, 1000)
    hs_k = horn_schunck(img3, img4; alpha = 0.5, iterations = iters)
    mu_in  = sum(hs_k.u[55:75, 55:75]) / length(hs_k.u[55:75, 55:75])
    mu_out = sum(hs_k.u[5:25,  55:75]) / length(hs_k.u[5:25,  55:75])
    @printf("  iters = %4d  →  inside = %+.3f   outside = %+.3f\n",
            iters, mu_in, mu_out)
    R, G, B = flow_to_rgb(hs_k.u, hs_k.v; max_mag = 1.0)
    push!(iter_tiles, 0.299 .* R .+ 0.587 .* G .+ 0.114 .* B)
end
PNM.save_pgm(joinpath(outdir, "03_iteration_trace.pgm"),
             Viz.montage(iter_tiles; cols = 4, gap = 4, background = 0.0))

# ── 4. Propagation profile: how far does flow diffuse from the patch ───────
# Plot mean recovered |u| as a function of distance from the moving patch's
# left edge, on the row through the centre.
hs_final = horn_schunck(img3, img4; alpha = 0.5, iterations = 5000)
profile = [sum(abs, hs_final.u[i, 55:75]) / 21 for i in 1:H]
# Render as a 1×H strip stretched to a readable height.
profile_img = repeat(reshape(profile, 1, :), 32, 1)
PNM.save_pgm(joinpath(outdir, "04_outside_propagation.pgm"),
             Viz.normalize01(profile_img))

println()
println("→ $outdir")
println("Files:")
println("  01_alpha_sweep.pgm        — same input, α = 0.05 / 0.1 / 0.5 / 2.0")
println("  02_lk_vs_hs.ppm           — LK (left) vs HS (right) on textured-patch input")
println("  03_iteration_trace.pgm    — HS at 10 / 50 / 200 / 1000 iterations")
println("  04_outside_propagation.pgm — |u| as a function of row (1×H strip)")
