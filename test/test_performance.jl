using Test
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution

@testset "Inline correlation (no padding copy)" begin
    @testset "matches the reference correlate2d on every padding mode" begin
        img = Synth.checkerboard(32, 32; tile = 4)
        K   = gaussian(5; sigma = 1.0)
        for mode in (:zero, :replicate, :reflect, :symmetric, :circular)
            ref = correlate2d(img, K; pad = mode)
            inl = correlate2d_inline(img, K; pad = mode)
            @test inl ≈ ref atol = 1e-12
        end
    end

    @testset "works on a non-square image with a non-symmetric kernel" begin
        img = Synth.circle(24, 36; radius = 8) .+ 0.2 .* rand(24, 36)
        K   = sobel_x()
        for mode in (:zero, :replicate, :reflect, :symmetric, :circular)
            @test correlate2d_inline(img, K; pad = mode) ≈
                  correlate2d(img, K; pad = mode) atol = 1e-12
        end
    end

    @testset ":valid is rejected" begin
        @test_throws ArgumentError correlate2d_inline(rand(8, 8), gaussian(3); pad = :valid)
    end
end

@testset "FFT correlation" begin
    @testset ":zero matches the direct correlate2d on a symmetric kernel" begin
        img = Synth.circle(48, 64; radius = 12) .+ 0.1 .* rand(48, 64)
        K = gaussian(7; sigma = 1.2)
        ref = correlate2d(img, K; pad = :zero)
        fft = fft_correlate2d(img, K; pad = :zero)
        @test fft ≈ ref atol = 1e-8
    end

    @testset ":zero matches on an asymmetric kernel too" begin
        img = rand(40, 56)
        K = sobel_x()
        ref = correlate2d(img, K; pad = :zero)
        fft = fft_correlate2d(img, K; pad = :zero)
        @test fft ≈ ref atol = 1e-9
    end

    @testset ":circular matches the direct :circular path" begin
        img = rand(32, 40)
        K = gaussian(5; sigma = 1.0)
        ref = correlate2d(img, K; pad = :circular)
        fft = fft_correlate2d(img, K; pad = :circular)
        @test fft ≈ ref atol = 1e-9
    end

    @testset "fft_convolve2d matches direct convolve2d on :zero" begin
        img = Synth.checkerboard(32, 32; tile = 4)
        K = sobel_y()
        ref = convolve2d(img, K; pad = :zero)
        fft = fft_convolve2d(img, K; pad = :zero)
        @test fft ≈ ref atol = 1e-9
    end

    @testset "unsupported pad modes are rejected" begin
        @test_throws ArgumentError fft_correlate2d(rand(8, 8), gaussian(3); pad = :reflect)
        @test_throws ArgumentError fft_correlate2d(rand(8, 8), gaussian(3); pad = :replicate)
    end

    @testset "even kernel rejected" begin
        @test_throws ArgumentError fft_correlate2d(rand(8, 8), rand(2, 2))
    end
end
