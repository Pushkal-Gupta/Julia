#!/usr/bin/env julia
# 24_color_processing.jl
#
# Colour image processing. Up to now everything in the repo has
# been grayscale `Matrix{Float64}` — this script exercises the
# `Color` submodule: RGB ↔ HSV, RGB ↔ YCbCr, Rec. 709 luminance,
# per-channel filter application, and the Di Zenzo colour gradient
# magnitude.
#
#   julia --project=. examples/24_color_processing.jl
#
# What this produces in artifacts/24_color_processing/:
#   01_synth_color.ppm           — the synthetic RGB test image
#   02_hsv_planes.pgm            — H, S, V montage (gray-scaled)
#   03_ycbcr_planes.pgm          — Y, Cb, Cr montage
#   04_luminance.pgm             — Rec. 709 luminance (single plane)
#   05_per_channel_blur.ppm      — Gaussian blur applied per channel
#   06_color_gradient.pgm        — Di Zenzo |∇I|_color
#   07_naive_vs_dizenzo.pgm      — averaged per-channel grad vs Di Zenzo
#                                   (side-by-side, equal scale)

using ImageLab
using ImageLab.Color
using ImageLab.Convolution: correlate2d
using ImageLab.Kernels: gaussian, sobel_x, sobel_y
using ImageLab.Viz, ImageLab.PNM, ImageLab.Photos
using Printf

outdir = joinpath(@__DIR__, "..", "artifacts", "24_color_processing")
mkpath(outdir)

H, W = 192, 192

# ── 1. Synthesize a small RGB test image ────────────────────────────────────
# Build colour bars + a few coloured shapes + an opposite-sign-edge case
# (R bright on the left, G bright on the right) so the Di Zenzo gradient
# has something interesting to find.
function colour_test_image(H, W)
    R = zeros(H, W); G = zeros(H, W); B = zeros(H, W)
    # Sky/background — a soft blue
    R .= 0.15; G .= 0.3; B .= 0.55
    # Red square upper-left
    R[20:70, 20:70] .= 0.9; G[20:70, 20:70] .= 0.15; B[20:70, 20:70] .= 0.15
    # Green circle upper-right
    for i in 1:H, j in 1:W
        if (i - 45)^2 + (j - 140)^2 ≤ 25^2
            R[i, j] = 0.1; G[i, j] = 0.85; B[i, j] = 0.15
        end
    end
    # Two-colour transition: bright red on left half of a strip, bright
    # green on right half, on a row in the lower image. The naive
    # average-of-gradient-magnitudes would partly cancel here; Di Zenzo
    # picks up the full step.
    R[120:140, 20:96]  .= 0.9; G[120:140, 20:96]  .= 0.1
    R[120:140, 97:172] .= 0.1; G[120:140, 97:172] .= 0.9
    # Yellow rectangle bottom-right
    R[150:175, 110:170] .= 0.9; G[150:175, 110:170] .= 0.85; B[150:175, 110:170] .= 0.1
    return (R, G, B)
end

R, G, B = colour_test_image(H, W)
PNM.save_ppm(joinpath(outdir, "01_synth_color.ppm"), R, G, B)

# ── 2. HSV decomposition ────────────────────────────────────────────────────
println("== HSV decomposition ==")
Hh, Sh, Vh = rgb_to_hsv(R, G, B)
@printf("  H ∈ [%.3f, %.3f]   S ∈ [%.3f, %.3f]   V ∈ [%.3f, %.3f]\n",
        extrema(Hh)..., extrema(Sh)..., extrema(Vh)...)
PNM.save_pgm(joinpath(outdir, "02_hsv_planes.pgm"),
             Viz.montage([Hh, Sh, Vh]; cols = 3, gap = 4, background = 0.0))
# Round-trip sanity check
Rb, Gb, Bb = hsv_to_rgb(Hh, Sh, Vh)
@printf("  RGB → HSV → RGB max error: %.2e\n",
        max(maximum(abs, R .- Rb),
            maximum(abs, G .- Gb),
            maximum(abs, B .- Bb)))

# ── 3. YCbCr decomposition ──────────────────────────────────────────────────
println()
println("== YCbCr decomposition (BT.601) ==")
Y, Cb, Cr = rgb_to_ycbcr(R, G, B)
@printf("  Y  ∈ [%.3f, %.3f]   Cb ∈ [%.3f, %.3f]   Cr ∈ [%.3f, %.3f]\n",
        extrema(Y)..., extrema(Cb)..., extrema(Cr)...)
PNM.save_pgm(joinpath(outdir, "03_ycbcr_planes.pgm"),
             Viz.montage([Y, Cb, Cr]; cols = 3, gap = 4, background = 0.5))

# ── 4. Rec. 709 luminance ───────────────────────────────────────────────────
lum = rgb_to_luminance(R, G, B)
PNM.save_pgm(joinpath(outdir, "04_luminance.pgm"), lum)

# ── 5. Per-channel Gaussian blur ────────────────────────────────────────────
gk = gaussian(11; sigma = 2.5)
Rb_blur, Gb_blur, Bb_blur = apply_per_channel(
    (c; pad) -> correlate2d(c, gk; pad = pad),
    R, G, B; pad = :replicate)
PNM.save_ppm(joinpath(outdir, "05_per_channel_blur.ppm"),
             Rb_blur, Gb_blur, Bb_blur)

# ── 6. Di Zenzo colour gradient ─────────────────────────────────────────────
println()
println("== Colour gradient (Di Zenzo) ==")
cg = color_gradient_magnitude(R, G, B)
@printf("  |∇I|_color: max = %.3f  mean = %.4f\n", maximum(cg), sum(cg) / length(cg))
PNM.save_pgm(joinpath(outdir, "06_color_gradient.pgm"),
             Viz.normalize01(cg))

# ── 7. Side-by-side: naive average vs Di Zenzo ──────────────────────────────
# The "naive" approach: per-channel gradient magnitudes, averaged.
function naive_color_gradient(R, G, B)
    sx, sy = sobel_x(), sobel_y()
    Rx = correlate2d(R, sx) ./ 8; Ry = correlate2d(R, sy) ./ 8
    Gx = correlate2d(G, sx) ./ 8; Gy = correlate2d(G, sy) ./ 8
    Bx = correlate2d(B, sx) ./ 8; By = correlate2d(B, sy) ./ 8
    return (sqrt.(Rx .^ 2 .+ Ry .^ 2) .+
            sqrt.(Gx .^ 2 .+ Gy .^ 2) .+
            sqrt.(Bx .^ 2 .+ By .^ 2)) ./ 3
end
naive = naive_color_gradient(R, G, B)
@printf("  naive average:  max = %.3f  mean = %.4f\n", maximum(naive), sum(naive) / length(naive))

# Save both on the SAME intensity scale so the eye can compare.
common_max = max(maximum(naive), maximum(cg))
PNM.save_pgm(joinpath(outdir, "07_naive_vs_dizenzo.pgm"),
             Viz.montage([naive ./ common_max, cg ./ common_max];
                         cols = 2, gap = 4, background = 0.0))

# Inspect the red-green-strip region where the methods most diverge.
strip_naive_max = maximum(naive[120:140, 90:105])
strip_cg_max    = maximum(cg[120:140,    90:105])
@printf("  red-→-green strip max:  naive = %.3f   Di Zenzo = %.3f   (ratio %.2f×)\n",
        strip_naive_max, strip_cg_max, strip_cg_max / strip_naive_max)

println()
println("→ $outdir")
println("Files:")
println("  01_synth_color.ppm        — the synthetic RGB input")
println("  02_hsv_planes.pgm         — H, S, V montage")
println("  03_ycbcr_planes.pgm       — Y, Cb, Cr montage")
println("  04_luminance.pgm          — Rec. 709 luminance")
println("  05_per_channel_blur.ppm   — Gaussian blur per channel")
println("  06_color_gradient.pgm     — Di Zenzo |∇I|_color")
println("  07_naive_vs_dizenzo.pgm   — naive average (left) vs Di Zenzo (right)")
