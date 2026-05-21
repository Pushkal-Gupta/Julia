#!/usr/bin/env julia
# 19_differentiable_filters.jl
#
# The bridge from classical to learned CV. Every filter kernel in
# this repo so far is a hand-written set of numbers — Sobel,
# Prewitt, Scharr, Laplacian. The point of this script is to show
# that those same numbers can be *learned*: pick a loss, take the
# gradient w.r.t. the kernel coefficients via ForwardDiff, and SGD
# converges to (almost exactly) the hand-written kernel.
#
#   julia --project=. examples/19_differentiable_filters.jl
#
# What the script does:
#   1. Generate a uniform-noise training image (so every local 3x3
#      patch is distinct → the 9 kernel coefficients are fully
#      observable from one input/target pair).
#   2. Build target outputs for several classical filters (Sobel-x,
#      Sobel-y, Laplacian-4, sharpen).
#   3. For each target, run SGD on a random-initialized kernel and
#      print how close the learned kernel got to the true one.
#   4. Save a montage that shows learned-vs-true filter outputs on a
#      *different* test image, so I can eyeball the transfer.
#
# What I expect:
#   - Loss drops by 8+ orders of magnitude in 800 iterations at
#     `lr = 0.1`. Final L∞ kernel error around `1e-4`.
#   - The learned and true filter outputs look visually identical on
#     the test image.

using ImageLab
using ImageLab.AutoDiff
using ImageLab.Kernels, ImageLab.Convolution, ImageLab.Synth, ImageLab.Viz, ImageLab.PNM
using Random
using Printf

outdir = joinpath(@__DIR__, "..", "artifacts", "19_differentiable_filters")
mkpath(outdir)

# ── 1. Training image (noise → fully observable kernel) ─────────────────────
Random.seed!(42)
train_img = rand(64, 64)

# ── 2. Test image — a real-looking synthetic so the visualizations show
#       structure rather than just noise.
test_img = zeros(96, 96)
test_img[20:50, 20:50] .= 1.0                       # square
for i in 1:96, j in 1:96
    (i - 70)^2 + (j - 70)^2 ≤ 18^2 && (test_img[i, j] = 0.8)
end
for k in 0:1, c in 1:96                              # two diagonal lines
    j = clamp(c + 10k, 1, 96)
    test_img[c, j] = 0.6
end

# ── 3. Sweep over target operators ──────────────────────────────────────────
targets = [
    ("sobel_x",     sobel_x()),
    ("sobel_y",     sobel_y()),
    ("laplacian4",  laplacian4()),
    ("sharpen",     sharpen()),
]

println("== Learning classical kernels from input/target pairs ==")
@printf("  training image: %d×%d uniform noise\n", size(train_img)...)
@printf("  optimizer: vanilla SGD, lr = 0.1, iterations = 800\n\n")

results = NamedTuple[]
for (name, K_true) in targets
    target = correlate2d(train_img, K_true; pad = :replicate)
    Random.seed!(0)
    learned, hist = AutoDiff.fit_kernel(train_img, target;
                                        iterations = 800, lr = 0.1)
    err = maximum(abs, learned .- K_true)
    @printf("  %-12s  loss[1]=%.3e → loss[end]=%.3e   L∞ kernel err = %.2e\n",
            name, hist[1], hist[end], err)
    push!(results, (name = name, K_true = K_true,
                    K_learn = learned, hist = hist))
end

# ── 4. Eyeball the transfer on a test image ─────────────────────────────────
# For each filter, render the true output and the learned output side by
# side. If autodiff actually recovered the kernel, the two should be
# pixel-identical (up to ~1e-4).
tiles = Matrix{Float64}[]
for r in results
    out_true  = correlate2d(test_img, r.K_true;  pad = :replicate)
    out_learn = correlate2d(test_img, r.K_learn; pad = :replicate)
    push!(tiles, Viz.signed_to_gray(out_true))
    push!(tiles, Viz.signed_to_gray(out_learn))
end
PNM.save_pgm(joinpath(outdir, "01_true_vs_learned.pgm"),
             Viz.montage(tiles; cols = 2, gap = 4, background = 0.5))

# ── 5. Loss curves (Sobel-x for the visual) ─────────────────────────────────
# Render the loss history of the Sobel-x fit as a tiny 1D image so I can
# see the descent shape. Y-axis is log10(loss); X-axis is iteration.
function loss_strip(hist; height = 64)
    L = length(hist)
    img = fill(1.0, height, L)
    log_loss = log10.(max.(hist, 1e-14))
    lo, hi = minimum(log_loss), maximum(log_loss)
    for i in 1:L
        row = clamp(round(Int, (hi - log_loss[i]) / (hi - lo) * (height - 1)) + 1,
                    1, height)
        img[row, i] = 0.0
    end
    return img
end
PNM.save_pgm(joinpath(outdir, "02_loss_curve_sobel_x.pgm"),
             loss_strip(results[1].hist))

# ── 6. Verbose print of the learned Sobel-x ─────────────────────────────────
println()
println("Learned Sobel-x kernel (vs hand-written):")
for i in 1:3
    @printf("  %+.4f  %+.4f  %+.4f      |   %+.0f  %+.0f  %+.0f\n",
            results[1].K_learn[i, 1], results[1].K_learn[i, 2],
            results[1].K_learn[i, 3],
            results[1].K_true[i, 1],  results[1].K_true[i, 2],
            results[1].K_true[i, 3])
end

println()
println("→ $outdir")
println("Files:")
println("  01_true_vs_learned.pgm     — true (left) vs learned (right) for 4 kernels")
println("  02_loss_curve_sobel_x.pgm  — log-loss vs iteration for the Sobel-x fit")
