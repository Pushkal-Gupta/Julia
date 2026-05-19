#!/usr/bin/env julia
# 07_canny_parameter_sweep.jl
#
# Same input, nine parameter settings. Rows vary σ (the pre-smoothing
# scale); columns vary (low, high) thresholds from loose to tight.
#
#   julia --project=. examples/07_canny_parameter_sweep.jl
#
# What I'm looking for:
#   - As σ grows: noise vanishes, but fine structure (the checker grid)
#     does too. Edges get smoother / blurrier in position.
#   - As thresholds tighten: only the strongest edges survive; weaker
#     ones drop out first.
#   - The "sweet spot" depends on the image. Real-world Canny tuning
#     is mostly looking at outputs like this and picking what looks
#     right.

using ImageLab
using ImageLab.Synth, ImageLab.Edges, ImageLab.Viz, ImageLab.PNM
using Printf

function studio_image(h, w)
    canvas = Synth.checkerboard(h, w; tile = 18) .* 0.45
    disk   = Synth.circle(h, w; radius = min(h, w) ÷ 4)
    sq     = Synth.square(h, w; side = 32, value = 0.7)
    img    = max.(canvas, disk, sq)
    return Synth.gaussian_noise(img; sigma = 0.08, seed = 23)
end

input = studio_image(128, 128)

# Grid: rows = σ, columns = (low, high) pairs.
sigmas = (0.8, 1.6, 3.0)
thresholds = ((0.04, 0.12),  # loose: keep lots of edges
              (0.08, 0.20),  # medium
              (0.15, 0.35))  # tight: only the strongest

outdir = joinpath(@__DIR__, "..", "artifacts", "07_canny_parameter_sweep")
mkpath(outdir)

# Save the input for reference.
PNM.save_pgm(joinpath(outdir, "00_input.pgm"), input)

# Compute every cell.
tiles = Matrix{Matrix{Float64}}(undef, length(sigmas), length(thresholds))
for (i, σ) in enumerate(sigmas), (j, (lo, hi)) in enumerate(thresholds)
    edges = canny(input; sigma = σ, low = lo, high = hi)
    tiles[i, j] = Float64.(edges)
    label = @sprintf("sigma%.1f_low%.2f_high%.2f.pgm", σ, lo, hi)
    PNM.save_pgm(joinpath(outdir, label), tiles[i, j])
end

# Flatten in row-major order: rows of σ, columns of (low, high).
flat_tiles = [tiles[i, j] for i in 1:length(sigmas) for j in 1:length(thresholds)]
grid = Viz.montage(flat_tiles; cols = length(thresholds), gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage.pgm"), grid)

# Print the legend.
println("Canny parameter sweep — rows are σ, columns are (low, high):")
println()
@printf("%-6s | %-18s | %-18s | %-18s\n", "σ \\ τ",
        "low=0.04 high=0.12", "low=0.08 high=0.20", "low=0.15 high=0.35")
println(repeat("-", 70))
for (i, σ) in enumerate(sigmas)
    counts = [sum(tiles[i, j] .> 0.5) for j in 1:length(thresholds)]
    @printf("σ=%-4.1f | %-18s | %-18s | %-18s\n", σ,
            "$(counts[1]) edge px", "$(counts[2]) edge px", "$(counts[3]) edge px")
end
println()
println("→ $outdir")
println("Open montage.pgm to read across the grid.")
