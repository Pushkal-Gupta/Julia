#!/usr/bin/env julia
# 11_performance_lab.jl
#
# Time four implementations of 2D correlation against each other across
# a sweep of kernel sizes, so I can see the actual crossover points
# instead of guessing from textbook complexity numbers.
#
# The contenders:
#   naive       — explicit padding copy + nested loop. The current
#                 `correlate2d`.
#   inline      — no padding copy. Interior pixels go through the same
#                 inner loop; border pixels resolve out-of-bounds reads
#                 on the fly.
#   separable   — two 1D passes (only valid for rank-1 kernels — I use
#                 a Gaussian for this whole benchmark).
#   fft         — FFT-based via FFTW. O(N log N) regardless of k.
#
# Using BenchmarkTools so the numbers are robust (multiple runs, GC
# settled, minimum reported).
#
#   julia --project=. examples/11_performance_lab.jl
#
# What I'm expecting:
#   - naive grows as k². Separable grows as k. FFT is roughly flat in k.
#   - Inline should be very close to naive — same inner work; the only
#     difference is who owns the padding copy. Smaller memory traffic
#     could help on the bigger sizes.
#   - FFT wins for big k (probably k ≥ 21 or so on this image size).
#   - Separable wins basically always once k ≥ 5.

using ImageLab
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution
using BenchmarkTools
using Printf

const IMG_H, IMG_W = 384, 384
const KERNEL_SIZES = (3, 5, 9, 15, 21, 31)

img = Float64.(Synth.checkerboard(IMG_H, IMG_W; tile = 24)) .+
      0.2 .* rand(IMG_H, IMG_W)

# Helper: BenchmarkTools min time in milliseconds.
ms(b) = minimum(b.times) / 1e6

println("Performance lab on $(IMG_H)×$(IMG_W) image, BenchmarkTools min:\n")
header = @sprintf("%-3s  %-9s  %-9s  %-11s  %-7s   %s",
                  "k", "naive", "inline", "separable", "fft", "(speedups vs naive)")
println(header)
println(repeat("-", length(header) + 30))

rows = NamedTuple{(:k, :naive, :inline, :sep, :fft), NTuple{5, Float64}}[]

for k in KERNEL_SIZES
    sigma = k / 6
    K2d = gaussian(k; sigma = sigma)
    g1d = gaussian1d(k; sigma = sigma)

    # Warm up each (compile, allocate plans on the FFT path, etc.).
    correlate2d(img, K2d; pad = :replicate)
    correlate2d_inline(img, K2d; pad = :replicate)
    separable_correlate2d(img, g1d, g1d; pad = :replicate)
    fft_correlate2d(img, K2d; pad = :zero)

    b_naive  = @benchmark correlate2d($img, $K2d; pad = :replicate) samples=8 evals=1
    b_inline = @benchmark correlate2d_inline($img, $K2d; pad = :replicate) samples=8 evals=1
    b_sep    = @benchmark separable_correlate2d($img, $g1d, $g1d; pad = :replicate) samples=8 evals=1
    # FFT benchmarked against :zero (the only mode it shares with the others
    # at the API level without extra padding work).
    b_fft    = @benchmark fft_correlate2d($img, $K2d; pad = :zero) samples=8 evals=1

    t_naive  = ms(b_naive)
    t_inline = ms(b_inline)
    t_sep    = ms(b_sep)
    t_fft    = ms(b_fft)

    push!(rows, (k = k, naive = t_naive, inline = t_inline, sep = t_sep, fft = t_fft))
    @printf("%-3d  %7.2f ms %7.2f ms %9.2f ms %5.2f ms     inline %4.2fx, separable %4.2fx, fft %4.2fx\n",
            k, t_naive, t_inline, t_sep, t_fft,
            t_naive / t_inline, t_naive / t_sep, t_naive / t_fft)
end

# Save the table to disk so I can refer back to it.
outdir = joinpath(@__DIR__, "..", "artifacts", "11_performance_lab")
mkpath(outdir)
open(joinpath(outdir, "timing.txt"), "w") do io
    println(io, "Performance lab on $(IMG_H)×$(IMG_W) image, BenchmarkTools min in ms")
    println(io, "")
    @printf(io, "%-3s  %-12s  %-12s  %-12s  %-10s\n",
            "k", "naive (ms)", "inline (ms)", "separable", "fft (ms)")
    println(io, repeat("-", 60))
    for r in rows
        @printf(io, "%-3d  %-12.2f  %-12.2f  %-12.2f  %-10.2f\n",
                r.k, r.naive, r.inline, r.sep, r.fft)
    end
    println(io, "\nSpeedups vs naive:")
    @printf(io, "%-3s  %-12s  %-12s  %-12s\n", "k", "inline", "separable", "fft")
    for r in rows
        @printf(io, "%-3d  %-12.2f  %-12.2f  %-12.2f\n",
                r.k, r.naive / r.inline, r.naive / r.sep, r.naive / r.fft)
    end
end
println()
println("→ $outdir/timing.txt")
