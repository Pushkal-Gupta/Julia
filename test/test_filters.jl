using Test
using ImageLab.Synth, ImageLab.Filters, ImageLab.Kernels, ImageLab.Convolution

@testset "median_filter" begin
    @testset "constant image stays constant" begin
        flat = fill(0.6, 12, 8)
        @test median_filter(flat; window = 3) ≈ flat
    end

    @testset "single salt pixel disappears" begin
        img = fill(0.3, 9, 9)
        img[5, 5] = 1.0   # the salt
        out = median_filter(img; window = 3)
        @test out[5, 5] ≈ 0.3   # neighborhood median is 0.3, salt outvoted 8 to 1
    end

    @testset "median beats box on salt-and-pepper" begin
        base = fill(0.5, 64, 64)
        noisy = Synth.salt_pepper(base; p = 0.10, seed = 5)
        median_out = median_filter(noisy;       window = 3)
        box_out    = correlate2d(noisy, box(3); pad = :replicate)
        # Total recovery error (L1 distance to the clean image): median wins.
        @test sum(abs.(median_out .- base)) < sum(abs.(box_out .- base))
        # And by a healthy margin — at least 3× better.
        @test sum(abs.(median_out .- base)) < sum(abs.(box_out .- base)) / 3
    end

    @testset "edges survive median better than box-style averaging" begin
        img = zeros(10, 10); img[:, 6:end] .= 1.0
        med = median_filter(img; window = 3)
        # The step should still be sharp: column 5 stays 0, column 6 stays 1.
        @test all(med[:, 5] .== 0.0)
        @test all(med[:, 6] .== 1.0)
    end

    @testset "rejects even window" begin
        @test_throws ArgumentError median_filter(rand(5, 5); window = 4)
    end
end

@testset "bilateral_filter" begin
    @testset "constant image stays constant" begin
        flat = fill(0.5, 12, 12)
        @test bilateral_filter(flat; window = 5,
                               sigma_spatial = 1.5, sigma_intensity = 0.1) ≈ flat
    end

    @testset "preserves a sharp edge better than a comparable Gaussian" begin
        img = zeros(20, 20); img[:, 11:end] .= 1.0
        bilateral = bilateral_filter(img; window = 7,
                                     sigma_spatial = 2.0, sigma_intensity = 0.05)
        # Compare the gap between the two sides of the step at the center row.
        # A pure Gaussian with σ=2 would soften this; bilateral with a small
        # intensity sigma should keep it close to (0, 1).
        @test bilateral[10, 10] < 0.05
        @test bilateral[10, 11] > 0.95
    end

    @testset "rejects bad params" begin
        @test_throws ArgumentError bilateral_filter(rand(5, 5); window = 4)
        @test_throws ArgumentError bilateral_filter(rand(5, 5);
                                                    sigma_spatial = -1.0)
    end
end

@testset "perona_malik" begin
    @testset "constant image stays constant" begin
        flat = fill(0.4, 16, 16)
        out = perona_malik(flat; iterations = 30, K = 0.1, lambda = 0.2)
        @test all(abs.(out .- 0.4) .< 1e-12)
    end

    @testset "zero iterations is the identity" begin
        img = rand(8, 12)
        @test perona_malik(img; iterations = 0) ≈ img
    end

    @testset "preserves a sharp step better than linear diffusion" begin
        # The whole point of Perona-Malik: an edge survives many iterations
        # that a Gaussian (i.e. linear diffusion) of comparable effective
        # smoothing would have blurred.
        img = zeros(20, 20); img[:, 11:end] .= 1.0
        # Small K → edges treated as "real" → preserved.
        diffused = perona_malik(img; iterations = 50, K = 0.05, lambda = 0.2)
        # Step should still be near-perfect at the boundary.
        @test all(diffused[:, 10] .< 0.10)
        @test all(diffused[:, 11] .> 0.90)
    end

    @testset "large K behaves like linear diffusion (loses the edge)" begin
        # With K large compared to the step (gradient ≈ 1), the diffusivity
        # is essentially constant and we get linear (Gaussian-like) diffusion.
        img = zeros(20, 20); img[:, 11:end] .= 1.0
        out_big_K = perona_malik(img; iterations = 50, K = 100.0, lambda = 0.2)
        # The edge should now be substantially blurred.
        @test out_big_K[10, 10] > 0.10
        @test out_big_K[10, 11] < 0.90
    end

    @testset "rational and exponential modes both run and preserve edges" begin
        img = zeros(16, 16); img[:, 9:end] .= 1.0
        a = perona_malik(img; iterations = 20, K = 0.1, mode = :exponential)
        b = perona_malik(img; iterations = 20, K = 0.1, mode = :rational)
        @test size(a) == size(img)
        @test size(b) == size(img)
        @test all(a[:, 8] .< 0.10)
        @test all(b[:, 8] .< 0.10)
    end

    @testset "removes Gaussian noise from a flat region" begin
        base = fill(0.5, 32, 32)
        noisy = base .+ 0.08 .* randn(32, 32)
        smoothed = perona_malik(noisy; iterations = 25, K = 0.10, lambda = 0.2)
        # Total L1 deviation from the clean ground truth should drop dramatically.
        @test sum(abs.(smoothed .- base)) < sum(abs.(noisy .- base)) / 3
    end

    @testset "rejects bad parameters" begin
        @test_throws ArgumentError perona_malik(rand(5, 5); lambda = 0.5)
        @test_throws ArgumentError perona_malik(rand(5, 5); K = -0.1)
        @test_throws ArgumentError perona_malik(rand(5, 5); mode = :nonsense)
        @test_throws ArgumentError perona_malik(rand(5, 5); iterations = -1)
    end
end

@testset "binary_dilate" begin
    @testset "radius 0 is identity" begin
        m = BitMatrix([false true; true false])
        @test binary_dilate(m; radius = 0) == m
    end

    @testset "radius 1 expands one pixel each way" begin
        m = falses(5, 5); m[3, 3] = true
        d = binary_dilate(m; radius = 1)
        # 8-connected expansion → a 3×3 block centered on (3, 3) is now true.
        @test sum(d) == 9
        @test all(d[2:4, 2:4])
        @test d[1, 1] == false
    end

    @testset "successive iterations match radius=n directly" begin
        m = falses(7, 7); m[4, 4] = true
        d2 = binary_dilate(m; radius = 2)
        # Manually doing it twice should give the same result.
        d11 = binary_dilate(binary_dilate(m; radius = 1); radius = 1)
        @test d2 == d11
    end

    @testset "negative radius is rejected" begin
        @test_throws ArgumentError binary_dilate(BitMatrix(falses(3, 3)); radius = -1)
    end
end
