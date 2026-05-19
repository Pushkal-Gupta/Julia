#!/usr/bin/env julia
# 01_first_convolutions.jl
#
# First proper sanity check for the convolution engine. I build a
# synthetic image (checkerboard with a bright disk on top), run a handful
# of canonical filters over it, and dump one PGM per stage into
# ./artifacts/01_first_convolutions/. Open them in Preview to look.
#
#   julia --project=. examples/01_first_convolutions.jl
#
# What I expect to see:
#   blur:       edges soften; the checker grid loses contrast.
#   sobel_x:    vertical edges (sides of the disk + checker columns) light
#               up. Horizontal lines are invisible to it.
#   sobel_y:    opposite — horizontal lines light up.
#   grad_mag:   all edges visible regardless of orientation. This is the
#               "everywhere there's a change" image.
#   laplacian:  zero-crossings sit on the edges; signed output.
#   sharpened:  edges over-emphasised; haloing if I push strength too far.

using ImageLab
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution, ImageLab.PNM

# Helper: rescale a (possibly signed) matrix into [0, 1] for visualization.
function visualize(x::AbstractMatrix{<:Real})
    lo, hi = extrema(x)
    hi == lo && return zeros(size(x))
    return (x .- lo) ./ (hi - lo)
end

# Reproducible artifacts directory next to this script.
outdir = joinpath(@__DIR__, "..", "artifacts", "01_first_convolutions")
mkpath(outdir)
save(name, img) = PNM.save_pgm(joinpath(outdir, name * ".pgm"), img)

# ── 1. Build a composite synthetic image ──────────────────────────────────────
# A circle inside a checkerboard quadrant. Rich enough to exercise every
# operator: straight edges, curved edges, and uniform regions.
canvas = Synth.checkerboard(128, 128; tile = 16)
disk   = Synth.circle(128, 128; radius = 36, value = 1.0, bg = 0.0)
img = max.(canvas .* 0.6, disk)   # disk overlays the checkerboard at higher brightness

save("00_input", img)

# ── 2. Smoothing: box vs Gaussian ─────────────────────────────────────────────
box_blur = correlate2d(img, Kernels.box(5); pad = :replicate)
gauss    = correlate2d(img, Kernels.gaussian(5; sigma = 1.0); pad = :replicate)
save("01_box_blur",      box_blur)
save("02_gaussian_blur", gauss)

# ── 3. Gradient operators (run on the Gaussian-smoothed image) ────────────────
# Smoothing first matters: derivatives amplify noise; the Gaussian is the
# textbook regularizer.
gx = correlate2d(gauss, Kernels.sobel_x(); pad = :replicate)
gy = correlate2d(gauss, Kernels.sobel_y(); pad = :replicate)
mag = sqrt.(gx .^ 2 .+ gy .^ 2)

save("03_sobel_x",       visualize(gx))
save("04_sobel_y",       visualize(gy))
save("05_gradient_mag",  visualize(mag))

# ── 4. Laplacian: the second-order alternative ────────────────────────────────
lap = correlate2d(gauss, Kernels.laplacian8(); pad = :replicate)
save("06_laplacian", visualize(lap))

# ── 5. Sharpening: the opposite of blurring ───────────────────────────────────
sharp = correlate2d(img, Kernels.sharpen(0.6); pad = :replicate)
save("07_sharpened", clamp.(sharp, 0.0, 1.0))

println("Wrote $(length(readdir(outdir))) PGM files to $outdir")
println("Open them in Preview (or `qlmanage -p artifacts/01_first_convolutions/*.pgm`)")
