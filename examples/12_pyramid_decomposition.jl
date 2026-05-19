#!/usr/bin/env julia
# 12_pyramid_decomposition.jl
#
# Take one image, build a 5-level Gaussian pyramid and a 5-level
# Laplacian pyramid, then reconstruct from the Laplacian pyramid and
# check that I get the original back.
#
# What I'm looking at:
#   - The Gaussian pyramid shows the same image at progressively
#     coarser resolutions. The smallest level is just the average
#     low-frequency content.
#   - The Laplacian pyramid shows the band-pass detail at each scale.
#     Most of the energy is in the higher-frequency levels (smaller
#     k); the smallest entry is the same as the smallest Gaussian
#     (the residual).
#   - Reconstruction should match the original to ~1e-10. That's the
#     whole pedagogical point of Laplacian pyramids — they're an
#     exact, invertible multi-scale decomposition.
#
#   julia --project=. examples/12_pyramid_decomposition.jl
#
# Each level is a different size, so I pad them up to the original
# image size when assembling the montage.

using ImageLab
using ImageLab.Synth, ImageLab.Pyramids, ImageLab.Viz, ImageLab.PNM
using Printf

function studio_image(h, w)
    img = Synth.circle(h, w; radius = min(h, w) ÷ 3)
    sq  = Synth.square(h, w; side = 36, value = 0.6)
    img = max.(img, sq)
    canvas = Synth.checkerboard(h, w; tile = 18) .* 0.35
    return max.(img, canvas)
end

input = studio_image(128, 128)

LEVELS = 4   # gives 5 levels total (0..4)
G = gaussian_pyramid(input;  levels = LEVELS)
L = laplacian_pyramid(input; levels = LEVELS)
reconstructed = reconstruct_laplacian_pyramid(L)

reconstruction_error = maximum(abs.(reconstructed .- input))

outdir = joinpath(@__DIR__, "..", "artifacts", "12_pyramid_decomposition")
mkpath(outdir)

# Save every level at its native resolution.
for (k, level) in enumerate(G)
    PNM.save_pgm(joinpath(outdir, "gauss_$(k-1)_$(size(level, 1))x$(size(level, 2)).pgm"), level)
end
for (k, level) in enumerate(L[1:end-1])
    PNM.save_pgm(joinpath(outdir, "lapl_$(k-1)_$(size(level, 1))x$(size(level, 2)).pgm"),
                 Viz.signed_to_gray(level))
end
PNM.save_pgm(joinpath(outdir, "lapl_residual.pgm"), L[end])
PNM.save_pgm(joinpath(outdir, "reconstructed.pgm"), reconstructed)
PNM.save_pgm(joinpath(outdir, "00_input.pgm"), input)

# Pad each pyramid level into 128×128 cells for the montage.
H, W = size(input)
function pad_to(img, h, w; bg = 0.5)
    H0, W0 = size(img)
    (H0 == h && W0 == w) && return img
    out = fill(Float64(bg), h, w)
    i0 = (h - H0) ÷ 2 + 1
    j0 = (w - W0) ÷ 2 + 1
    out[i0:i0 + H0 - 1, j0:j0 + W0 - 1] .= img
    return out
end

# Top row: Gaussian levels 0..4.
# Bottom row: Laplacian levels 0..3 + residual.
gauss_tiles = [pad_to(g, H, W) for g in G]
lapl_tiles  = [pad_to(Viz.signed_to_gray(L[k]), H, W) for k in 1:LEVELS]
push!(lapl_tiles, pad_to(L[end], H, W))    # residual (the small Gaussian)

tiles = vcat(gauss_tiles, lapl_tiles)
grid = Viz.montage(tiles; cols = LEVELS + 1, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage.pgm"), grid)

println("Pyramid decomposition:")
println("  input size: $(size(input))")
println("  levels (full + residual): $(length(G))")
for (k, g) in enumerate(G)
    @printf("    G_%d: %dx%d   L_%d: %dx%d\n", k - 1, size(g, 1), size(g, 2),
            k - 1, size(L[k], 1), size(L[k], 2))
end
@printf("  reconstruction error (max abs): %.2e\n", reconstruction_error)
println()
println("Montage layout:")
println("  row 1 = Gaussian pyramid levels G_0, G_1, ..., G_$LEVELS")
println("  row 2 = Laplacian detail levels + the residual")
println("→ $outdir")
