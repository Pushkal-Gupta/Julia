using Test
using ImageLab.Synth, ImageLab.Pyramids

@testset "Pyramids" begin
    @testset "reduce halves the size (rounded up)" begin
        img = rand(64, 96)
        r = reduce_image(img)
        @test size(r) == (32, 48)
    end

    @testset "odd-dimension reduce" begin
        img = rand(31, 41)
        r = reduce_image(img)
        # ceil(31/2)=16, ceil(41/2)=21 since indexing 1:2:end keeps elements 1,3,...,31
        @test size(r) == (16, 21)
    end

    @testset "expand matches the requested target size exactly" begin
        small = rand(8, 12)
        big = expand_image(small, (16, 24))
        @test size(big) == (16, 24)
    end

    @testset "gaussian_pyramid produces levels+1 entries with shrinking sizes" begin
        img = rand(64, 64)
        G = gaussian_pyramid(img; levels = 4)
        @test length(G) == 5
        @test size(G[1]) == (64, 64)
        @test size(G[2]) == (32, 32)
        @test size(G[3]) == (16, 16)
        @test size(G[4]) == (8, 8)
        @test size(G[5]) == (4, 4)
    end

    @testset "laplacian_pyramid sizes match gaussian_pyramid sizes" begin
        img = rand(48, 64)
        G = gaussian_pyramid(img; levels = 3)
        L = laplacian_pyramid(img; levels = 3)
        @test length(L) == length(G)
        for k in eachindex(L)
            @test size(L[k]) == size(G[k])
        end
    end

    @testset "Laplacian pyramid reconstructs the original" begin
        img = Synth.circle(64, 64; radius = 18) .+ 0.1 .* rand(64, 64)
        L = laplacian_pyramid(img; levels = 4)
        reconstructed = reconstruct_laplacian_pyramid(L)
        @test size(reconstructed) == size(img)
        @test maximum(abs.(reconstructed .- img)) < 1e-10
    end

    @testset "constant image stays constant in the interior" begin
        # The Burt-Adelson pyramid doesn't round-trip exactly at the border —
        # the zero-insertion in expand creates a pattern that any padding mode
        # treats slightly wrong at the edge. The full reconstruction does work
        # exactly (see the next test) because reduce and expand have the same
        # border behavior and the errors cancel.
        c = fill(0.7, 32, 32)
        @test maximum(abs.(reduce_image(c) .- 0.7)) < 1e-12
        # Interior of expand(reduce(c)) is exact; skip a 4-pixel border.
        roundtrip = expand_image(reduce_image(c), (32, 32))
        @test maximum(abs.(roundtrip[5:end-4, 5:end-4] .- 0.7)) < 1e-12

        # Round-trip through the Laplacian pyramid is exact everywhere,
        # border included.
        L = laplacian_pyramid(c; levels = 3)
        @test maximum(abs.(reconstruct_laplacian_pyramid(L) .- c)) < 1e-12
    end

    @testset "levels = 0 returns the image alone" begin
        img = rand(8, 8)
        G = gaussian_pyramid(img; levels = 0)
        @test length(G) == 1
        @test G[1] ≈ img
        L = laplacian_pyramid(img; levels = 0)
        @test length(L) == 1
        @test L[1] ≈ img
    end

    @testset "negative levels rejected" begin
        @test_throws ArgumentError gaussian_pyramid(rand(8, 8); levels = -1)
        @test_throws ArgumentError expand_image(rand(4, 4), (0, 8))
    end

    @testset "laplacian_blend with all-ones mask returns A" begin
        A = rand(64, 64)
        B = rand(64, 64)
        mask = ones(64, 64)
        out = laplacian_blend(A, B, mask; levels = 3)
        @test maximum(abs.(out .- A)) < 1e-10
    end

    @testset "laplacian_blend with all-zeros mask returns B" begin
        A = rand(64, 64)
        B = rand(64, 64)
        mask = zeros(64, 64)
        out = laplacian_blend(A, B, mask; levels = 3)
        @test maximum(abs.(out .- B)) < 1e-10
    end

    @testset "laplacian_blend gives the input far from the seam" begin
        # Mask = 1 on the left half, 0 on the right. Pixels far from the
        # boundary should match A on the left and B on the right exactly
        # (in the interior of the pyramid's smoothing radius).
        A = fill(0.8, 64, 64)
        B = fill(0.2, 64, 64)
        mask = zeros(64, 64); mask[:, 1:32] .= 1.0
        out = laplacian_blend(A, B, mask; levels = 3)
        # The left edge far from the seam should be close to A's value;
        # the right edge far from the seam close to B's. Both within a
        # small tolerance reflecting the pyramid's interior bandwidth.
        @test maximum(abs.(out[:, 1:10]  .- 0.8)) < 0.05
        @test maximum(abs.(out[:, 55:64] .- 0.2)) < 0.05
    end

    @testset "laplacian_blend smooths a sharp mask edge" begin
        # The naive `A * mask + B * (1-mask)` would jump from 0.8 to 0.2
        # in one pixel at the seam. The pyramid blend should produce a
        # gradual ramp over many pixels.
        A = fill(0.8, 64, 64)
        B = fill(0.2, 64, 64)
        mask = zeros(64, 64); mask[:, 1:32] .= 1.0
        out = laplacian_blend(A, B, mask; levels = 3)
        # Walk across the middle row and check we don't jump in a single pixel.
        deltas = diff(out[32, :])
        @test maximum(abs.(deltas)) < 0.15   # no single-pixel jump greater than 0.15
    end

    @testset "laplacian_blend rejects mismatched sizes" begin
        @test_throws DimensionMismatch laplacian_blend(rand(8, 8), rand(8, 10), rand(8, 8))
    end
end
