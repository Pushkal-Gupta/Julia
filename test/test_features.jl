using Test
using ImageLab.Synth, ImageLab.Edges, ImageLab.Features

@testset "Harris" begin
    @testset "constant image gives zero response" begin
        flat = fill(0.5, 24, 24)
        R = harris_response(flat; sigma = 1.0)
        @test maximum(abs.(R)) < 1e-12
        @test isempty(harris_corners(flat; sigma = 1.0))
    end

    @testset "straight edges score along the edge, not corners" begin
        # A vertical step. The Harris response should be *negative* along
        # the step (edge-like) and zero elsewhere — definitely no positive
        # corner peaks anywhere except possibly near the image boundary
        # where my pre-smooth interacts with padding.
        img = zeros(32, 32); img[:, 17:end] .= 1.0
        R = harris_response(img; sigma = 1.0)
        # Interior of the edge column should be ≤ 0 (Harris hates pure edges).
        @test maximum(R[8:24, 14:20]) ≤ 1e-8
    end

    @testset "rectangle has corners at its four corners" begin
        # Filled rectangle from row 10 to 30, col 12 to 28.
        img = zeros(48, 48)
        img[10:30, 12:28] .= 1.0
        corners = harris_corners(img; sigma = 1.2, k = 0.04,
                                  threshold = 0.01, min_distance = 4)
        @test 4 ≤ length(corners) ≤ 8   # the four corners, possibly some satellites
        # Each of the four geometric corners should have *some* detected
        # corner within ~3 pixels of it.
        expected = [(10, 12), (10, 28), (30, 12), (30, 28)]
        for (ei, ej) in expected
            @test any(max(abs(i - ei), abs(j - ej)) ≤ 3 for (i, j) in corners)
        end
    end

    @testset "rejects bad sigma" begin
        @test_throws ArgumentError harris_response(rand(10, 10); sigma = -1.0)
    end

    @testset "multiscale finds the rectangle's corners at level 0" begin
        # Rectangle from (10, 12) to (30, 28) on a 48×48 canvas.
        img = zeros(48, 48); img[10:30, 12:28] .= 1.0
        results = multiscale_harris_corners(img; levels = 2,
                                             sigma = 1.2,
                                             threshold = 0.02,
                                             min_distance = 4)
        # Should find at least the four corners at level 0.
        level0 = [(r.row, r.col) for r in results if r.level == 0]
        @test length(level0) ≥ 4
        for (ei, ej) in [(10, 12), (10, 28), (30, 12), (30, 28)]
            @test any(max(abs(i - ei), abs(j - ej)) ≤ 3 for (i, j) in level0)
        end
        # Every result is tagged with a level in [0, levels].
        @test all(0 ≤ r.level ≤ 2 for r in results)
    end

    @testset "multiscale levels map back to original-image coordinates" begin
        img = zeros(64, 64); img[16:48, 16:48] .= 1.0
        results = multiscale_harris_corners(img; levels = 2)
        # Coordinates should stay inside the original image bounds.
        for r in results
            @test 1 ≤ r.row ≤ 64
            @test 1 ≤ r.col ≤ 64
        end
    end
end

@testset "Hough lines" begin
    @testset "a single horizontal line produces a single dominant peak" begin
        edges = falses(40, 40); edges[20, :] .= true
        acc = hough_lines(edges; n_theta = 90)
        peaks = hough_peaks(acc; threshold = 0.9, max_peaks = 5)
        @test length(peaks) ≥ 1
        # A horizontal line at row 20 has θ ≈ π/2 (vertical normal) and
        # ρ ≈ 20. Check the strongest peak.
        θ, ρ = peaks[1]
        @test abs(θ - π/2) < 0.05
        @test abs(ρ - 20) < 2.0
    end

    @testset "a single vertical line peaks at θ ≈ 0 (or equivalently θ ≈ π)" begin
        edges = falses(40, 40); edges[:, 15] .= true
        acc = hough_lines(edges; n_theta = 90)
        peaks = hough_peaks(acc; threshold = 0.9, max_peaks = 5)
        @test length(peaks) ≥ 1
        θ, ρ = peaks[1]
        # Tie-break in hough_peaks favors ascending θ, so this should come
        # out as θ ≈ 0 / ρ ≈ 15. But (θ ≈ π, ρ ≈ -15) is the same line, so
        # I accept either.
        if θ < π / 2
            @test θ < 0.1
            @test abs(ρ - 15) < 2.0
        else
            @test abs(θ - π) < 0.1
            @test abs(ρ + 15) < 2.0
        end
    end

    @testset "two distinct lines give two distinct peaks" begin
        edges = falses(50, 50)
        edges[15, :] .= true       # horizontal at row 15
        edges[:, 30] .= true       # vertical at col 30
        acc = hough_lines(edges; n_theta = 180)
        peaks = hough_peaks(acc; threshold = 0.6, min_distance_theta = 20,
                            min_distance_rho = 5, max_peaks = 4)
        @test length(peaks) ≥ 2
        # The two top peaks should be near the expected (θ, ρ) for each line.
        θs = first.(peaks)
        @test any(abs(θ - π/2) < 0.05 for θ in θs)   # horizontal line
        @test any(abs(θ - 0) < 0.05  for θ in θs)    # vertical line
    end
end

@testset "Connected components" begin
    @testset "single blob → one component" begin
        mask = BitMatrix(falses(8, 8))
        mask[3:5, 3:5] .= true
        labels, n = connected_components(mask; connectivity = 8)
        @test n == 1
        @test sum(labels .== 1) == 9
        @test all(labels[3:5, 3:5] .== 1)
        @test all(labels[1:2, :] .== 0)
    end

    @testset "two disjoint blobs → two components with distinct labels" begin
        mask = BitMatrix(falses(10, 10))
        mask[2:3, 2:3] .= true
        mask[7:9, 7:9] .= true
        labels, n = connected_components(mask)
        @test n == 2
        # Two labels, used once each.
        @test sort(unique(labels[mask])) == [1, 2]
    end

    @testset "8-connectivity merges diagonal touches; 4-connectivity does not" begin
        # Two pixels touching only diagonally.
        mask = BitMatrix(falses(4, 4))
        mask[2, 2] = true
        mask[3, 3] = true
        _, n8 = connected_components(mask; connectivity = 8)
        _, n4 = connected_components(mask; connectivity = 4)
        @test n8 == 1
        @test n4 == 2
    end

    @testset "component_sizes counts correctly" begin
        mask = BitMatrix(falses(6, 6))
        mask[1, 1:3] .= true       # 3-pixel L on the top
        mask[5:6, 5:6] .= true     # 4-pixel block at the bottom-right
        labels, n = connected_components(mask)
        sizes = component_sizes(labels, n)
        @test sort(sizes) == [3, 4]
    end
end

@testset "Normalized cross-correlation" begin
    @testset "a template matched against itself scores 1.0" begin
        img = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12]
        T = Float64[6 7; 10 11]    # the (2, 2) window of img
        ncc = normalized_cross_correlation(img, T)
        @test ncc[2, 2] ≈ 1.0  atol = 1e-10
    end

    @testset "constant patch scores 0 against a varying template" begin
        img = Float64[5 5 5; 5 5 5; 5 5 5]
        T   = Float64[1 2; 3 4]
        ncc = normalized_cross_correlation(img, T)
        @test all(abs.(ncc) .< 1e-10)
    end

    @testset "brightness change doesn't affect NCC" begin
        img = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12]
        T   = Float64[6 7; 10 11]
        ncc_a = normalized_cross_correlation(img, T)
        # Shift the entire image up by 100 — every window's mean shifts too,
        # so NCC should be unchanged at every output position.
        ncc_b = normalized_cross_correlation(img .+ 100, T)
        @test ncc_a ≈ ncc_b  atol = 1e-10
    end

    @testset "ncc_peaks finds local maxima above threshold" begin
        ncc = zeros(10, 10)
        ncc[3, 3] = 0.9
        ncc[7, 8] = 0.85
        ncc[3, 4] = 0.88  # near (3, 3) — should be suppressed by min_distance
        peaks = ncc_peaks(ncc; threshold = 0.8, min_distance = 3)
        @test (3, 3) in peaks
        @test (7, 8) in peaks
        @test !((3, 4) in peaks)
    end

    @testset "rejects oversized template" begin
        @test_throws ArgumentError normalized_cross_correlation(
            rand(3, 3), rand(4, 4))
    end
end
