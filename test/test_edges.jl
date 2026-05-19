using Test
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Edges

@testset "Edges" begin
    @testset "gradient on a constant image is zero" begin
        flat = fill(0.4, 12, 16)
        for op in (:sobel, :prewitt, :scharr, :roberts)
            gx, gy = gradient(flat, op; pad = :replicate)
            @test all(abs.(gx) .< 1e-12)
            @test all(abs.(gy) .< 1e-12)
        end
    end

    @testset "Sobel x on a vertical edge gives a strong vertical-edge response" begin
        # Half-black, half-white image. Vertical step at column 8.
        img = zeros(12, 16)
        img[:, 9:end] .= 1.0
        gx, gy = gradient(img, :sobel; pad = :replicate)
        # gx should peak around the step.
        @test maximum(gx) > 1.0
        # gy should be ~ zero everywhere (no horizontal change).
        @test maximum(abs, gy) < 1e-12
    end

    @testset "magnitude is rotationally consistent on a circle" begin
        img = Synth.circle(64, 64; radius = 18)
        gx, gy = gradient(img, :scharr; pad = :replicate)
        m = gradient_magnitude(gx, gy)
        # The magnitude image should be symmetric — comparing it to its
        # 90°-rotated version on a perfect circle, both should match closely
        # away from float rounding at sub-pixel edges.
        rotated = permutedims(m)
        @test maximum(abs.(m .- rotated)) < 0.1   # generous; not pixel-perfect
    end

    @testset "direction on a vertical-edge image points horizontally" begin
        img = zeros(32, 32)
        img[:, 17:end] .= 1.0
        gx, gy = gradient(img, :sobel; pad = :replicate)
        θ = gradient_direction(gx, gy)
        # Look at θ at the step itself, away from the borders.
        center_θ = θ[15:18, 16:17]
        # Gradient points in +x direction → angle near 0 (or near π for the
        # other side of a step; for a black→white step, gx is positive).
        @test all(abs.(center_θ) .< 0.1)
    end

    @testset "quantize_direction maps angles to 4 sectors" begin
        @test quantize_direction(0.0)       == 0       # horizontal
        @test quantize_direction(π/4)       == 1       # NE diagonal
        @test quantize_direction(π/2)       == 2       # vertical
        @test quantize_direction(3π/4)      == 3       # NW diagonal
        @test quantize_direction(-π/4)      == 3       # NW (mod π wraps)
        @test quantize_direction(π - 0.01)  == 0       # back to horizontal
        @test quantize_direction(π + π/4)   == 1       # mod π normalization
    end

    @testset "log_filter on an impulse returns the LoG kernel (sign-aware)" begin
        n = 7
        sigma = 1.0
        im = Synth.impulse(11, 11)
        out = log_filter(im, n; sigma = sigma, pad = :zero)
        # Correlation impulse response is the kernel rotated 180°. LoG is
        # rotationally symmetric so this is identical to the kernel itself.
        K = laplacian_of_gaussian(n; sigma = sigma)
        c = 6   # center of 11×11
        r = n ÷ 2
        @test out[c-r:c+r, c-r:c+r] ≈ K atol = 1e-12
    end

    @testset "dog_filter on a constant image is zero" begin
        flat = fill(0.7, 24, 24)
        out = dog_filter(flat; sigma1 = 1.0, sigma2 = 1.6, pad = :replicate)
        @test maximum(abs.(out)) < 1e-12
    end

    @testset "zero_crossings finds a sign change" begin
        signed = [-1.0  -1.0   1.0   1.0;
                  -1.0  -1.0   1.0   1.0;
                  -1.0  -1.0   1.0   1.0]
        zc = zero_crossings(signed)
        # The boundary sits between columns 2 and 3. With my 4-neighbor test,
        # column 2 (interior cells) should be marked: they have a positive
        # neighbor on the right.
        @test zc[2, 2] == true
        @test zc[2, 3] == true   # also marked from the other side
        # Strictly interior columns away from the crossing are not.
        @test zc[2, 1] == false   # but only the (2,1) is at the border row, which we skip
    end

    @testset "threshold helpers" begin
        m = [0.0 0.2; 0.6 0.9]
        @test threshold_mask(m, 0.5) == [false false; true true]
        @test percentile_threshold(m, 0.0)  == 0.0
        @test percentile_threshold(m, 1.0)  == 0.9
        @test_throws ArgumentError percentile_threshold(m, 1.5)
    end

    @testset "unknown operator rejected" begin
        @test_throws ArgumentError gradient(rand(4, 4), :nonsense)
    end
end
