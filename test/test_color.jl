using Test
using ImageLab
using ImageLab.Color
using ImageLab.Convolution: correlate2d
using ImageLab.Kernels: gaussian

@testset "Color" begin

    @testset "rgb_to_hsv on canonical colours" begin
        # Pure red, green, blue, white, black, grey
        R = reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.5], 1, 6)
        G = reshape([0.0, 1.0, 0.0, 1.0, 0.0, 0.5], 1, 6)
        B = reshape([0.0, 0.0, 1.0, 1.0, 0.0, 0.5], 1, 6)
        H, S, V = rgb_to_hsv(R, G, B)
        # Red:    H = 0
        @test H[1, 1] ≈ 0.0
        @test S[1, 1] ≈ 1.0
        @test V[1, 1] ≈ 1.0
        # Green:  H = 1/3
        @test H[1, 2] ≈ 1/3
        # Blue:   H = 2/3
        @test H[1, 3] ≈ 2/3
        # White:  S = 0, V = 1
        @test S[1, 4] ≈ 0.0
        @test V[1, 4] ≈ 1.0
        # Black:  V = 0
        @test V[1, 5] ≈ 0.0
        # Grey:   S = 0, V = 0.5
        @test S[1, 6] ≈ 0.0
        @test V[1, 6] ≈ 0.5
    end

    @testset "RGB → HSV → RGB round trip" begin
        # Random RGB image
        import Random
        Random.seed!(7)
        R = rand(16, 16); G = rand(16, 16); B = rand(16, 16)
        H, S, V = rgb_to_hsv(R, G, B)
        Rb, Gb, Bb = hsv_to_rgb(H, S, V)
        @test maximum(abs, R .- Rb) < 1e-10
        @test maximum(abs, G .- Gb) < 1e-10
        @test maximum(abs, B .- Bb) < 1e-10
    end

    @testset "RGB → YCbCr → RGB round trip" begin
        import Random
        Random.seed!(11)
        R = rand(16, 16); G = rand(16, 16); B = rand(16, 16)
        Y, Cb, Cr = rgb_to_ycbcr(R, G, B)
        Rb, Gb, Bb = ycbcr_to_rgb(Y, Cb, Cr)
        # The published BT.601 inverse coefficients are truncated to
        # six decimals — they aren't the exact inverse of the
        # forward matrix, so the round-trip error is ~1e-7, not at
        # machine precision.
        @test maximum(abs, R .- Rb) < 1e-6
        @test maximum(abs, G .- Gb) < 1e-6
        @test maximum(abs, B .- Bb) < 1e-6
    end

    @testset "YCbCr neutrality of grey" begin
        # A grey image (R = G = B) should have Cb = Cr = 0.5 (the
        # "no colour" point of the centred chrominance encoding).
        grey = fill(0.4, 8, 8)
        Y, Cb, Cr = rgb_to_ycbcr(grey, grey, grey)
        @test all(Y .≈ 0.4)
        @test all(Cb .≈ 0.5)
        @test all(Cr .≈ 0.5)
    end

    @testset "rgb_to_luminance" begin
        # Rec. 709 weights sum to 1
        R = ones(4, 4); G = ones(4, 4); B = ones(4, 4)
        @test all(rgb_to_luminance(R, G, B) .≈ 1.0)
        # Pure green is the brightest of the primaries
        R = fill(0.0, 4, 4); G = fill(1.0, 4, 4); B = fill(0.0, 4, 4)
        @test all(rgb_to_luminance(R, G, B) .≈ 0.7152)
    end

    @testset "apply_per_channel" begin
        # Gaussian-blur each channel with the same kernel
        import Random
        Random.seed!(13)
        R = rand(32, 32); G = rand(32, 32); B = rand(32, 32)
        kern = gaussian(5; sigma = 1.0)
        Rb, Gb, Bb = apply_per_channel((c; pad) -> correlate2d(c, kern; pad = pad),
                                       R, G, B; pad = :replicate)
        @test size(Rb) == size(R)
        @test size(Gb) == size(G)
        @test size(Bb) == size(B)
        # The blurred output should match what I'd get by calling correlate2d
        # directly on each channel.
        @test maximum(abs, Rb .- correlate2d(R, kern; pad = :replicate)) < 1e-12
    end

    @testset "color_gradient_magnitude reduces to scalar for greyscale" begin
        import Random
        Random.seed!(17)
        grey = rand(64, 64)
        # Same image on all three channels — the colour gradient should
        # equal √3 times the per-channel gradient magnitude (since the
        # structure tensor sum stacks three identical contributions).
        cg = color_gradient_magnitude(grey, grey, grey)
        from_one = sqrt.(
            (correlate2d(grey, ImageLab.Kernels.sobel_x()) ./ 8) .^ 2 .+
            (correlate2d(grey, ImageLab.Kernels.sobel_y()) ./ 8) .^ 2)
        @test maximum(abs, cg .- sqrt(3) .* from_one) < 1e-10
    end

    @testset "color_gradient_magnitude catches edges plain averaging misses" begin
        # Two channels with edges of opposite sign. Per-channel averaged
        # gradient would partly cancel; Di Zenzo should pick up the full
        # colour edge because it sums squared gradient components.
        H, W = 32, 32
        R = zeros(H, W); G = zeros(H, W)
        R[:, 1:16] .= 1.0; G[:, 17:32] .= 1.0   # opposite-sign edges at column 16-17
        B = zeros(H, W)
        cg = color_gradient_magnitude(R, G, B)
        # The strong edge should land near column 16.
        @test maximum(cg[:, 14:18]) > 0.2
        # Away from the edge it should be ~zero.
        @test maximum(cg[:, 1:8]) < 1e-10
        @test maximum(cg[:, 25:32]) < 1e-10
    end

    @testset "size-mismatch errors" begin
        a = zeros(8, 8); b = zeros(8, 9)
        @test_throws DimensionMismatch rgb_to_hsv(a, b, a)
        @test_throws DimensionMismatch hsv_to_rgb(a, b, a)
        @test_throws DimensionMismatch rgb_to_ycbcr(a, b, a)
        @test_throws DimensionMismatch ycbcr_to_rgb(a, b, a)
        @test_throws DimensionMismatch rgb_to_luminance(a, b, a)
        @test_throws DimensionMismatch color_gradient_magnitude(a, b, a)
        @test_throws DimensionMismatch apply_per_channel(identity, a, b, a)
    end
end
