using Test
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution

@testset "1D correlation" begin
    @testset "identity on a vector" begin
        v = collect(1.0:7.0)
        @test correlate1d(v, [0.0, 1.0, 0.0]; pad = :zero) ≈ v
        @test correlate1d(v, [0.0, 1.0, 0.0]; pad = :replicate) ≈ v
    end

    @testset "1D box smoothing of a constant signal is constant" begin
        v = fill(0.7, 10)
        @test correlate1d(v, fill(1/3, 3); pad = :replicate) ≈ v
    end

    @testset ":valid shrinks; padded modes preserve length" begin
        v = rand(20)
        k = [0.2, 0.2, 0.2, 0.2, 0.2]
        @test length(correlate1d(v, k; pad = :valid)) == 16
        @test length(correlate1d(v, k; pad = :zero)) == 20
    end

    @testset "1D convolution flips the kernel" begin
        v = zeros(7); v[4] = 1.0  # impulse at index 4
        k = [1.0, 2.0, 3.0]
        # Correlation impulse-response = kernel reversed
        @test correlate1d(v, k; pad = :zero)[3:5] == [3.0, 2.0, 1.0]
        # Convolution impulse-response = kernel as-is
        @test convolve1d(v, k; pad = :zero)[3:5] == [1.0, 2.0, 3.0]
    end

    @testset "1D on matrix axes" begin
        A = reshape(collect(1.0:20.0), 4, 5)
        k = [0.25, 0.5, 0.25]
        h = correlate1d(A, k; axis = :horizontal, pad = :replicate)
        v = correlate1d(A, k; axis = :vertical, pad = :replicate)
        @test size(h) == size(A)
        @test size(v) == size(A)
        # :x / :y are aliases.
        @test correlate1d(A, k; axis = :x, pad = :replicate) ≈ h
        @test correlate1d(A, k; axis = :y, pad = :replicate) ≈ v
    end

    @testset "rejects bad inputs" begin
        @test_throws ArgumentError correlate1d(rand(10), rand(4))      # even kernel
        @test_throws ArgumentError correlate1d(rand(10, 10), rand(3); axis = :diagonal)
    end
end

@testset "separable equals naive 2D for rank-1 kernels" begin
    A = Synth.circle(24, 32; radius = 8) .+ 0.3 .* rand(24, 32)
    sigma = 1.4
    g1 = gaussian1d(7; sigma = sigma)

    @testset "Gaussian: separable ≈ 2D Gaussian" begin
        ref = correlate2d(A, gaussian(7; sigma = sigma); pad = :replicate)
        sep = separable_correlate2d(A, g1, g1; pad = :replicate)
        @test sep ≈ ref atol = 1e-10
    end

    @testset "Box: separable ≈ 2D box" begin
        ref = correlate2d(A, box(5); pad = :replicate)
        sep = separable_correlate2d(A, box1d(5), box1d(5); pad = :replicate)
        @test sep ≈ ref atol = 1e-10
    end

    @testset "Sobel-x: separable ≈ 2D Sobel-x" begin
        ref = correlate2d(A, sobel_x(); pad = :replicate)
        kx, ky = sobel_x_separable()
        sep = separable_correlate2d(A, kx, ky; pad = :replicate)
        @test sep ≈ ref atol = 1e-10
    end

    @testset "Prewitt-y and Scharr-x match too" begin
        for (K2d, (kx, ky)) in (
            (prewitt_y(), prewitt_y_separable()),
            (scharr_x(),  scharr_x_separable()),
        )
            ref = correlate2d(A, K2d; pad = :reflect)
            sep = separable_correlate2d(A, kx, ky; pad = :reflect)
            @test sep ≈ ref atol = 1e-10
        end
    end
end

@testset "factor_separable" begin
    @testset "recovers a rank-1 kernel" begin
        K = sobel_x()
        f = factor_separable(K)
        @test f !== nothing
        kx, ky = f
        @test ky * kx' ≈ K atol = 1e-10
    end

    @testset "returns nothing for a genuinely 2D kernel" begin
        @test factor_separable(laplacian4()) === nothing
        @test factor_separable(laplacian8()) === nothing
    end

    @testset "recovered factors work in separable_correlate2d" begin
        A = rand(20, 20)
        K = gaussian(5; sigma = 1.0)
        kx, ky = factor_separable(K)
        @test separable_correlate2d(A, kx, ky; pad = :reflect) ≈
              correlate2d(A, K; pad = :reflect) atol = 1e-10
    end
end
