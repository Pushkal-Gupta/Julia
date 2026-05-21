using Test
using ImageLab.Kernels, ImageLab.Convolution, ImageLab.LinAlgView
using LinearAlgebra: diag, eigvals, norm
using FFTW: fft, ifft

@testset "Convolution as matrix-vector multiplication" begin
    @testset "Toeplitz matrix matches correlate1d under :zero padding" begin
        v = collect(Float64, 1:12)
        K = [0.25, 0.5, 0.25]
        M = toeplitz_conv_matrix(K, length(v))
        @test M * v ≈ correlate1d(v, K; pad = :zero) atol = 1e-12
    end

    @testset "matrix is genuinely Toeplitz (constant along diagonals)" begin
        K = [0.2, 0.3, 0.1, 0.3, 0.1]
        M = toeplitz_conv_matrix(K, 8)
        # Check a few interior diagonals.
        for offset in -2:2
            d = diag(M, offset)
            # Only interior entries are bound to be equal; near the edges
            # the Toeplitz pattern is truncated by the :zero padding.
            interior = d[2:end-1]
            @test all(x -> x ≈ first(interior), interior) skip = isempty(interior)
        end
    end

    @testset "Circulant matrix matches correlate1d under :circular padding" begin
        v = randn(16)
        K = [0.1, 0.2, 0.4, 0.2, 0.1]
        M = circulant_conv_matrix(K, length(v))
        @test M * v ≈ correlate1d(v, K; pad = :circular) atol = 1e-12
    end

    @testset "Circulant matrix is shift-symmetric (every row is the previous shifted)" begin
        K = [0.3, 0.4, 0.3]
        M = circulant_conv_matrix(K, 8)
        # Row 2 should be row 1 shifted right by one position with wrap.
        for i in 2:size(M, 1)
            @test M[i, :] ≈ circshift(M[i - 1, :], 1) atol = 1e-12
        end
    end
end

@testset "DFT diagonalizes circulant convolution" begin
    @testset "eigenvalues from FFT match the matrix's spectrum" begin
        K = [0.2, 0.5, 0.2]   # rough triangular smoother
        n = 32
        # The eigenvalues from my circulant_eigenvalues helper:
        λ = circulant_eigenvalues(K, n)
        # …should equal the eigenvalues of the explicit matrix (up to
        # reordering, since LAPACK doesn't sort them).
        M = circulant_conv_matrix(K, n)
        λ_explicit = eigvals(M)
        # Sort by magnitude and compare.
        @test sort(abs.(λ))  ≈ sort(abs.(λ_explicit)) atol = 1e-9
    end

    @testset "convolution via FFT-diagonalized circulant matches direct" begin
        # Verify the diagonalization identity at the level of vectors:
        #   conv(v, K)  ==  ifft( fft(v) .* λ )
        # where λ are the circulant eigenvalues.
        K = [0.1, 0.2, 0.4, 0.2, 0.1]
        v = randn(64)
        λ = circulant_eigenvalues(K, length(v))
        from_fft = real.(ifft(fft(v) .* λ))
        from_direct = correlate1d(v, K; pad = :circular)
        @test from_fft ≈ from_direct atol = 1e-9
    end
end

@testset "Bad inputs are rejected" begin
    @test_throws ArgumentError toeplitz_conv_matrix([0.5, 0.5], 5)         # even kernel
    @test_throws ArgumentError circulant_conv_matrix([0.5, 0.5], 5)        # even kernel
    @test_throws ArgumentError circulant_eigenvalues([0.5, 0.5], 5)        # even kernel
    @test_throws ArgumentError circulant_eigenvalues([0.2, 0.6, 0.2], 2)   # n < klen
end
