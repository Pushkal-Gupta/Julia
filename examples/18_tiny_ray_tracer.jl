#!/usr/bin/env julia
# 18_tiny_ray_tracer.jl
#
# Step out of pure image-processing land for a chunk and use Julia for
# something completely different: a small Whitted-style ray tracer that
# still hands its result back as a `Matrix{Float64}` so `Photos` and
# `PNM` can save it.
#
#   julia --project=. examples/18_tiny_ray_tracer.jl
#
# What this produces in artifacts/18_tiny_ray_tracer/:
#   01_basic_scene.ppm    — the standard test scene (red + blue + chrome
#                            spheres over a checker floor, warm key light)
#   02_basic_scene.png    — the same render via Photos (uses FileIO + ImageIO)
#   03_reflection_depth.ppm — montage at depth = 0, 1, 2, 3 to show how each
#                              extra bounce adds detail to the mirror sphere
#   04_camera_orbit.ppm   — three angles around the scene, side by side
#   05_studio_grayscale.pgm — the luminance of the basic render, for
#                              feeding back into the rest of ImageLab
#                              (edge detection on a ray-traced image, etc.)
#
# What I'm looking for:
#   - The chrome sphere shows recognizable copies of the red and blue
#     spheres on its surface.
#   - The checkerboard wraps perspective-correctly around the floor.
#   - Shadows under each sphere darken the floor by ~50%.
#   - At depth = 0, the mirror sphere is black (no reflection, no
#     diffuse). Each extra bounce makes it more believable.

using ImageLab
using ImageLab.RayTracer
using ImageLab.Viz, ImageLab.PNM, ImageLab.Photos
using Printf

outdir = joinpath(@__DIR__, "..", "artifacts", "18_tiny_ray_tracer")
mkpath(outdir)

W, H = 384, 256

# ── 1. The reference scene ──────────────────────────────────────────────────
println("== Rendering basic scene at $(W)×$(H) ==")
scene, cam = basic_scene()
t0 = time()
R, G, B = render(scene, cam; width = W, height = H, depth = 3)
@printf("  done in %.2fs (%.1f µs / pixel)\n",
        time() - t0,
        1e6 * (time() - t0) / (W * H))
PNM.save_ppm(joinpath(outdir, "01_basic_scene.ppm"), R, G, B)
Photos.save_rgb_planes(joinpath(outdir, "02_basic_scene.png"), R, G, B)

# ── 2. Reflection depth study ───────────────────────────────────────────────
println()
println("== Reflection depth = 0, 1, 2, 3 ==")
depth_tiles = Matrix{Float64}[]
for d in 0:3
    Rd, Gd, Bd = render(scene, cam; width = W ÷ 2, height = H ÷ 2, depth = d)
    # Convert to luminance for the montage (montage takes 2D arrays).
    lum = 0.299 .* Rd .+ 0.587 .* Gd .+ 0.114 .* Bd
    push!(depth_tiles, lum)
    @printf("  depth = %d : mean luminance = %.3f, max = %.3f\n",
            d, sum(lum) / length(lum), maximum(lum))
end
PNM.save_pgm(joinpath(outdir, "03_reflection_depth.pgm"),
             Viz.montage(depth_tiles; cols = 4, gap = 4, background = 0.0))

# ── 3. Camera orbit ─────────────────────────────────────────────────────────
println()
println("== Three camera angles ==")
function orbit_camera(angle_deg, radius = 4.0, height = 0.7, look_y = -0.2,
                      look_z = 5.0)
    θ = deg2rad(angle_deg)
    return Camera(V3(radius * sin(θ), height, look_z - radius * cos(θ)),
                  V3(0.0, look_y, look_z),
                  V3(0.0, 1.0, 0.0), 55.0)
end

orbit_tiles = Matrix{Float64}[]
for ang in (-25.0, 0.0, 25.0)
    cam_o = orbit_camera(ang)
    Ro, Go, Bo = render(scene, cam_o; width = W ÷ 2, height = H ÷ 2, depth = 3)
    push!(orbit_tiles, 0.299 .* Ro .+ 0.587 .* Go .+ 0.114 .* Bo)
end
PNM.save_pgm(joinpath(outdir, "04_camera_orbit.pgm"),
             Viz.montage(orbit_tiles; cols = 3, gap = 4, background = 0.0))

# ── 4. Hand the result back to the rest of ImageLab ─────────────────────────
# Once a render is just a matrix, the whole pipeline applies. Take the
# luminance channel and save it as a PGM — from here I could run Canny on
# it, slap a Gaussian pyramid on it, etc.
lum = 0.299 .* R .+ 0.587 .* G .+ 0.114 .* B
PNM.save_pgm(joinpath(outdir, "05_studio_grayscale.pgm"), lum)

println()
println("→ $outdir")
println("Files:")
println("  01_basic_scene.ppm        — PPM via the pure-Julia writer")
println("  02_basic_scene.png        — same render via Photos / FileIO")
println("  03_reflection_depth.pgm   — depth = 0..3 montage (luminance)")
println("  04_camera_orbit.pgm       — three camera angles")
println("  05_studio_grayscale.pgm   — luminance of the basic render")
