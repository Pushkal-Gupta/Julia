#!/usr/bin/env julia
# 17_convolution_as_linear_algebra.jl
#
# Convolution is matrix-vector multiplication with a structured
# matrix. This script makes that statement concrete:
#
#   1. Build the Toeplitz convolution matrix for a small kernel.
#      Print it. See the band structure (constant along diagonals).
#   2. Build the circulant matrix (same idea, with the rows wrapping).
#      Print it.
#   3. Compute the matrix's eigenvalues two ways: explicitly via
#      `eigvals`, and via the DFT of the first row. They match,
#      because the DFT diagonalizes circulant matrices.
#   4. Save a PGM "image" of each matrix's structure so I can look
#      at the band pattern.
#   5. Compute the eigenvalue magnitudes of a Gaussian convolution
#      operator. This *is* the filter's frequency response.
#
#   julia --project=. examples/17_convolution_as_linear_algebra.jl
#
# What I'm expecting:
#   - Both matrices show diagonal stripes — the Toeplitz one's bands
#     stop at the edges, the circulant one's wrap around.
#   - The eigenvalue magnitudes of the Gaussian look like a
#     low-pass response: largest near the DC component, decaying for
#     higher frequencies. That decay shape is exactly the Gaussian's
#     frequency response.

using ImageLab
using ImageLab.Kernels, ImageLab.Convolution, ImageLab.LinAlgView
using ImageLab.Viz, ImageLab.PNM
using LinearAlgebra: eigvals
using FFTW: fft
using Printf

outdir = joinpath(@__DIR__, "..", "artifacts", "17_convolution_as_linear_algebra")
mkpath(outdir)

# ── 1. Print a tiny Toeplitz matrix ──────────────────────────────────────────
println("== Toeplitz convolution matrix (n=8, kernel=[1, 2, 1]/4) ==\n")
K = [0.25, 0.5, 0.25]
T = toeplitz_conv_matrix(K, 8)
for i in 1:8
    for j in 1:8
        @printf("%5.2f  ", T[i, j])
    end
    println()
end
println()

# ── 2. Print the circulant matrix (same kernel) ──────────────────────────────
println("== Circulant convolution matrix (n=8, same kernel) ==\n")
C = circulant_conv_matrix(K, 8)
for i in 1:8
    for j in 1:8
        @printf("%5.2f  ", C[i, j])
    end
    println()
end
println()

# Witness: each row of C is the previous row shifted by 1 (wrap).
println("Circulant shift check:")
println("  row 1:  ", round.(C[1, :], digits = 2))
println("  row 2:  ", round.(C[2, :], digits = 2),
        "   (= circshift(row 1, 1))")
println()

# ── 3. Eigenvalues two ways ──────────────────────────────────────────────────
println("== Eigenvalues of the 8×8 circulant matrix ==")
λ_explicit = sort(abs.(eigvals(C)))
λ_fft      = sort(abs.(circulant_eigenvalues(K, 8)))
println("  via LinearAlgebra.eigvals(C):  $(round.(λ_explicit, digits = 4))")
println("  via DFT of first row:          $(round.(λ_fft,      digits = 4))")
@printf("  L∞ difference: %.2e\n\n", maximum(abs.(λ_explicit .- λ_fft)))

# ── 4. Save matrix structures as PGMs ────────────────────────────────────────
# Render T and C as normalized grayscale images so I can eyeball the band
# pattern at a larger size.
T_big = toeplitz_conv_matrix(gaussian1d(11; sigma = 1.5), 64)
C_big = circulant_conv_matrix(gaussian1d(11; sigma = 1.5), 64)
PNM.save_pgm(joinpath(outdir, "01_toeplitz_64.pgm"),  Viz.normalize01(T_big))
PNM.save_pgm(joinpath(outdir, "02_circulant_64.pgm"), Viz.normalize01(C_big))

# Side-by-side montage.
PNM.save_pgm(joinpath(outdir, "matrix_structures.pgm"),
             Viz.montage([Viz.normalize01(T_big), Viz.normalize01(C_big)];
                          cols = 2, gap = 4, background = 0.5))

# ── 5. Eigenvalues of a Gaussian = its frequency response ────────────────────
n = 128
g = gaussian1d(11; sigma = 1.5)
λ = circulant_eigenvalues(g, n)
println("== Eigenvalues of a Gaussian-smoothing circulant matrix (n=$n) ==")
println("  |λ| at DC (k = 0):     $(round(abs(λ[1]),  digits = 4))")
println("  |λ| at Nyquist (k = $(n÷2)):  $(round(abs(λ[n ÷ 2 + 1]), digits = 4))")
println()
println("The DC eigenvalue should be ~1 (the Gaussian sums to ~1).")
println("The Nyquist eigenvalue measures how much of the highest")
println("frequency this filter passes. For a strong low-pass it should be")
println("near zero — and it is.")
println()

# Render the eigenvalue magnitudes as a 1D plot (1×n PGM).
freq_response = abs.(λ) ./ maximum(abs.(λ))      # normalize to [0, 1]
strip_img = repeat(reshape(Float64.(freq_response), 1, :), 16, 1)
PNM.save_pgm(joinpath(outdir, "03_gaussian_freq_response.pgm"), strip_img)

println("→ $outdir")
println("Files:")
println("  01_toeplitz_64.pgm           — Toeplitz matrix as an image")
println("  02_circulant_64.pgm          — circulant matrix as an image")
println("  matrix_structures.pgm        — side-by-side")
println("  03_gaussian_freq_response.pgm — |λ| across frequency bins")
