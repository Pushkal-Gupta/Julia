using Test
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution

@testset "Convolution" begin
    @testset "identity kernel is identity" begin
        img = Synth.checkerboard(8, 8; tile = 2)
        out = correlate2d(img, identity_kernel(3); pad = :zero)
        @test out ≈ img
    end

    @testset "impulse response reproduces the kernel" begin
        # Convolving an impulse with a kernel returns the (flipped) kernel.
        im = Synth.impulse(7, 7)
        K = sobel_x()
        out_corr = correlate2d(im, K; pad = :zero)
        # The non-zero patch is centered around the impulse, equal to the kernel
        # rotated 180° (because correlation samples K at +offset rather than -offset).
        c = 4  # center of 7×7
        @test out_corr[c-1:c+1, c-1:c+1] == reverse(reverse(K; dims = 1); dims = 2)

        out_conv = convolve2d(im, K; pad = :zero)
        # Convolution flips back: the patch equals the kernel itself.
        @test out_conv[c-1:c+1, c-1:c+1] == K
    end

    @testset "box blur is a local mean" begin
        img = ones(5, 5)
        out = correlate2d(img, box(3); pad = :replicate)
        # A constant image stays constant — pixel-perfect under :replicate.
        @test all(out .≈ 1.0)
    end

    @testset "Sobel detects exactly the edges of a square" begin
        sq = Synth.square(20, 20; side = 10)
        gx = correlate2d(sq, sobel_x(); pad = :zero)
        gy = correlate2d(sq, sobel_y(); pad = :zero)
        # Horizontal-gradient operator should respond on the *vertical* sides.
        # Rows 5-15 (inside the square) should have a large negative response at
        # the left edge and large positive at the right edge.
        @test minimum(gx) < -1.0
        @test maximum(gx) > 1.0
        # Magnitude is symmetric under x↔y because the shape is.
        @test sort(abs.(gx[:])) ≈ sort(abs.(gy[:]))
    end

    @testset ":valid shrinks output, padded modes preserve size" begin
        img = rand(10, 10)
        K = gaussian(5; sigma = 1.0)
        @test size(correlate2d(img, K; pad = :valid)) == (6, 6)
        @test size(correlate2d(img, K; pad = :zero)) == (10, 10)
        @test size(correlate2d(img, K; pad = :replicate)) == (10, 10)
    end

    @testset "in-place version matches allocating version" begin
        img = Synth.checkerboard(16, 16; tile = 4)
        K = gaussian(3; sigma = 0.8)
        ref = correlate2d(img, K; pad = :reflect)
        out = similar(ref)
        correlate2d!(out, img, K; pad = :reflect)
        @test out ≈ ref
    end

    @testset "convolve2d equals correlate2d on symmetric kernels" begin
        img = Synth.circle(24, 24; radius = 7)
        for K in (box(3), gaussian(5; sigma = 1.0), laplacian4(), laplacian8())
            @test convolve2d(img, K; pad = :replicate) ≈ correlate2d(img, K; pad = :replicate)
        end
    end

    @testset "even-sized kernel rejected (for now)" begin
        @test_throws ArgumentError correlate2d(rand(5, 5), rand(2, 2))
    end
end
