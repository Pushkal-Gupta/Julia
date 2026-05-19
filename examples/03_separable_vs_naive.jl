#!/usr/bin/env julia
# 03_separable_vs_naive.jl
#
# Quick timing check: naive 2D Gaussian vs the separable two-pass version
# across kernel sizes. Theory says the speedup should be ~k/2 since naive
# does k² mults per pixel and separable does 2k. I want to see that on
# real numbers, and check that the two outputs agree to float precision.
#
#   julia --project=. examples/03_separable_vs_naive.jl
#
# What I expect:
#   - Outputs numerically equal (max abs diff ~1e-15, pure rounding noise).
#   - Speedup ~1× at k=3 (padding overhead dominates), growing toward
#     k/2 as the kernel gets larger.
#
# I'm using `@elapsed` here rather than BenchmarkTools so the script stays
# dependency-free. A proper benchmark suite comes later.

using Printf
using ImageLab
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution, ImageLab.PNM

const REPS = 3
const SIZE = 384  # image side; small enough for k=21 to stay tolerable on a laptop

function timeit(f, args...; reps = REPS)
    f(args...)                     # warm up (compile + cache)
    best = Inf
    for _ in 1:reps
        t = @elapsed f(args...)
        best = min(best, t)
    end
    return best
end

img = Float64.(Synth.checkerboard(SIZE, SIZE; tile = 24)) .+
      0.2 .* rand(SIZE, SIZE)

outdir = joinpath(@__DIR__, "..", "artifacts", "03_separable_vs_naive")
mkpath(outdir)

results = NamedTuple{(:k, :sigma, :naive_ms, :separable_ms, :speedup, :max_diff),
                     Tuple{Int, Float64, Float64, Float64, Float64, Float64}}[]

for k in (3, 5, 9, 15, 21)
    sigma = k / 6
    K2d = Kernels.gaussian(k; sigma = sigma)
    g1d = Kernels.gaussian1d(k; sigma = sigma)

    t_naive = timeit(correlate2d, img, K2d; reps = REPS)
    # `timeit` with kwargs doesn't pass them through, so wrap:
    naive_fn(I, K) = correlate2d(I, K; pad = :replicate)
    sep_fn(I, k1)  = separable_correlate2d(I, k1, k1; pad = :replicate)

    t_naive = timeit(naive_fn, img, K2d)
    t_sep   = timeit(sep_fn,   img, g1d)

    out_naive = naive_fn(img, K2d)
    out_sep   = sep_fn(img, g1d)
    maxdiff = maximum(abs.(out_naive .- out_sep))

    push!(results, (
        k = k, sigma = sigma,
        naive_ms = 1000 * t_naive, separable_ms = 1000 * t_sep,
        speedup = t_naive / t_sep, max_diff = maxdiff,
    ))
end

# Print + save the table.
header = @sprintf("%-3s  %-6s  %10s  %10s  %8s  %12s", "k", "σ", "naive (ms)", "separable", "speedup", "|Δ|_max")
sep    = "─"^length(header)
lines = String[header, sep]
for r in results
    push!(lines, @sprintf("%-3d  %-6.2f  %10.2f  %10.2f  %8.2fx  %12.2e",
        r.k, r.sigma, r.naive_ms, r.separable_ms, r.speedup, r.max_diff))
end
report = join(lines, "\n") * "\n"
println("\nGaussian blur on $(SIZE)×$(SIZE) image, $(REPS) reps, best wall-clock:\n")
print(report)
open(joinpath(outdir, "timing.txt"), "w") do io
    write(io, "Gaussian blur on $(SIZE)x$(SIZE) image, $(REPS) reps, best wall-clock\n")
    write(io, report)
end

# Save one pair of outputs as a visual sanity check.
PNM.save_pgm(joinpath(outdir, "naive_k15.pgm"),
             correlate2d(img, Kernels.gaussian(15); pad = :replicate))
PNM.save_pgm(joinpath(outdir, "separable_k15.pgm"),
             separable_correlate2d(img, Kernels.gaussian1d(15), Kernels.gaussian1d(15); pad = :replicate))

println("→ $outdir/timing.txt")
