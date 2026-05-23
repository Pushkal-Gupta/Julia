#!/usr/bin/env julia
# 20_optical_flow.jl
#
# Lucas-Kanade optical flow on a pair of synthetic frames. The point
# of this script is to show:
#
#   1. that the algorithm recovers a known continuous translation,
#   2. that the standard flow visualization (hue = direction,
#      saturation = magnitude) is readable,
#   3. that confidence drops to zero in flat regions (no texture =
#      no constraint to solve),
#   4. and that pure rotation flow gets recovered locally even
#      though the global motion isn't translational.
#
#   julia --project=. examples/20_optical_flow.jl
#
# What this produces in artifacts/20_optical_flow/:
#   01_frames.pgm             — frame 1 and frame 2 side by side
#   02_translation_flow.ppm   — recovered flow on the translation pair
#   03_translation_compare.ppm — ground truth (left) vs recovered (right)
#   04_rotation_flow.ppm      — recovered flow on a rotating frame pair
#   05_confidence.pgm         — confidence map (textured vs flat regions)
#   06_color_wheel.ppm        — the colour key for reading the flow images

using ImageLab
using ImageLab.Flow
using ImageLab.Viz, ImageLab.PNM
using Printf

outdir = joinpath(@__DIR__, "..", "artifacts", "20_optical_flow")
mkpath(outdir)

# ── Synthetic frame: a continuous textured pattern that's a true
#    bandlimited signal, so subpixel shifts are well-defined.
function textured_frame(H, W; shift_u = 0.0, shift_v = 0.0)
    # A low-frequency sin·cos pattern. Higher-frequency content would
    # give richer texture but also more OFC-linearization bias for the
    # shifts I'm using here — that's a real trade-off, and plain LK is
    # accurate only when the dominant period in the image is much
    # larger than the inter-frame displacement.
    img = zeros(H, W)
    for j in 1:W, i in 1:H
        x = j - shift_u
        y = i - shift_v
        img[i, j] = 0.5 + 0.4 * sin(2π * 0.04 * x) * cos(2π * 0.035 * y)
    end
    return img
end

function rotation_frame(H, W; angle = 0.0)
    # Rotate the same textured pattern around the image centre.
    cx, cy = (W + 1) / 2, (H + 1) / 2
    c, s = cos(angle), sin(angle)
    img = zeros(H, W)
    for j in 1:W, i in 1:H
        # Inverse-warp: where in the source does this destination pixel come from?
        dx = j - cx
        dy = i - cy
        sx = cx + c * dx + s * dy
        sy = cy - s * dx + c * dy
        img[i, j] = 0.5 + 0.4 * sin(2π * 0.04 * sx) * cos(2π * 0.035 * sy)
    end
    return img
end

H, W = 128, 128

# ── 1. Translation: a sub-pixel shift in the small-motion regime ───────────
# Plain LK linearizes the optical-flow constraint, which is only accurate
# for shifts well below 1 pixel. (1.5, 0.5) gives visible bias; (0.7, 0.3)
# stays comfortably inside the linearization.
const TRUE_U, TRUE_V = 0.7, 0.3
println("== Translation by (u, v) = ($(TRUE_U), $(TRUE_V)) ==")
frame_a = textured_frame(H, W)
frame_b = textured_frame(H, W; shift_u = TRUE_U, shift_v = TRUE_V)
flow_t = lucas_kanade(frame_a, frame_b; window_size = 15, sigma = 2.0)

inner_u = flow_t.u[30:H-30, 30:W-30]
inner_v = flow_t.v[30:H-30, 30:W-30]
@printf("  recovered (mean over interior): u = %.3f  v = %.3f  (true %.3f, %.3f)\n",
        sum(inner_u) / length(inner_u),
        sum(inner_v) / length(inner_v),
        TRUE_U, TRUE_V)
@printf("  recovered (max magnitude in interior): %.3f\n",
        maximum(flow_magnitude(flow_t)[30:H-30, 30:W-30]))

PNM.save_pgm(joinpath(outdir, "01_frames.pgm"),
             Viz.montage([frame_a, frame_b]; cols = 2, gap = 4, background = 0.5))

R, G, B = flow_to_rgb(flow_t.u, flow_t.v; max_mag = 1.0)
PNM.save_ppm(joinpath(outdir, "02_translation_flow.ppm"), R, G, B)

# Side-by-side: ground truth vs recovered.
gtR, gtG, gtB = flow_to_rgb(fill(TRUE_U, H, W), fill(TRUE_V, H, W); max_mag = 1.0)
PNM.save_ppm(joinpath(outdir, "03_translation_compare.ppm"),
             [gtR R], [gtG G], [gtB B])

# ── 2. Rotation: ~1° around the image centre ────────────────────────────────
# At the image edges (radius ~60 px from centre) a 1° rotation gives
# ~1 pixel of motion, still within LK's small-motion regime.
println()
println("== Rotation by ~1° around image centre ==")
angle = deg2rad(1.0)
frame_c = textured_frame(H, W)
frame_d = rotation_frame(H, W; angle = angle)
flow_r = lucas_kanade(frame_c, frame_d; window_size = 15, sigma = 2.0)

# Spot-check the recovered flow at four positions.
# A 2° CCW rotation has (u, v) = (-dy·sin θ, dx·sin θ) ≈ (-dy·0.035, dx·0.035) at small angle.
println("  sample point checks (true_u, true_v) vs (rec_u, rec_v):")
for (i, j) in ((20, W÷2), (H÷2, W-20), (H-20, W÷2), (H÷2, 20))
    dx = j - (W + 1) / 2
    dy = i - (H + 1) / 2
    # The frame is INVERSE-warped, so the displacement of pixel content is:
    true_u = -dy * sin(angle) + dx * (cos(angle) - 1)
    true_v =  dx * sin(angle) + dy * (cos(angle) - 1)
    @printf("    (%3d, %3d) true (%+.2f, %+.2f) | recovered (%+.2f, %+.2f)\n",
            i, j, true_u, true_v, flow_r.u[i, j], flow_r.v[i, j])
end

R, G, B = flow_to_rgb(flow_r.u, flow_r.v; max_mag = 3.0)
PNM.save_ppm(joinpath(outdir, "04_rotation_flow.ppm"), R, G, B)

# ── 3. Confidence: a textured patch in a sea of flat ────────────────────────
println()
println("== Confidence: textured patch in a flat field ==")
flat = fill(0.5, H, W)
flat[40:88, 40:88] .= textured_frame(49, 49)
flat_shifted = copy(flat)
flat_shifted[40:88, 41:89] .= flat[40:88, 40:88]   # shift the patch by 1 col

flow_c = lucas_kanade(flat, flat_shifted; window_size = 11, sigma = 2.0,
                     det_threshold = 1e-12)
@printf("  mean confidence inside textured patch: %.3e\n",
        sum(flow_c.confidence[50:78, 50:78]) / (29 * 29))
@printf("  mean confidence in flat background:    %.3e\n",
        sum(flow_c.confidence[1:20, 1:20]) / (20 * 20))

PNM.save_pgm(joinpath(outdir, "05_confidence.pgm"),
             Viz.normalize01(flow_c.confidence))

# ── 4. Colour wheel — the key for reading the flow visualizations ──────────
wheel_size = 96
wheel_u = zeros(wheel_size, wheel_size)
wheel_v = zeros(wheel_size, wheel_size)
for j in 1:wheel_size, i in 1:wheel_size
    dx = j - (wheel_size + 1) / 2
    dy = i - (wheel_size + 1) / 2
    r = sqrt(dx^2 + dy^2)
    if r ≤ wheel_size / 2
        wheel_u[i, j] = dx
        wheel_v[i, j] = dy
    end
end
wR, wG, wB = flow_to_rgb(wheel_u, wheel_v)
PNM.save_ppm(joinpath(outdir, "06_color_wheel.ppm"), wR, wG, wB)

println()
println("→ $outdir")
println("Files:")
println("  01_frames.pgm              — frame A and frame B, side by side")
println("  02_translation_flow.ppm    — recovered flow (translation)")
println("  03_translation_compare.ppm — ground truth (left) vs recovered (right)")
println("  04_rotation_flow.ppm       — recovered flow (rotation)")
println("  05_confidence.pgm          — structure-tensor determinant")
println("  06_color_wheel.ppm         — the colour key for reading these")
