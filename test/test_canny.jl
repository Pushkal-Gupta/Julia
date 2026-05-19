using Test
using ImageLab.Synth, ImageLab.Edges

@testset "Canny stages" begin
    @testset "non-maximum suppression keeps a clean peak" begin
        # Magnitude image with a single horizontal ridge that's fat (3 pixels wide).
        # Direction = π/2 means the gradient is vertical, so NMS looks up/down.
        mag = zeros(5, 5)
        mag[2:4, 3] .= 1.0       # vertical bar of 1s, three pixels tall
        mag[2:4, 3] = [0.5, 1.0, 0.5]  # peak in the middle of the bar
        θ = fill(π/2, 5, 5)
        out = nonmaximum_suppression(mag, θ)
        # Only the middle peak survives — the 0.5 cells were smaller than the peak above/below.
        @test out[3, 3] == 1.0
        @test out[2, 3] == 0.0
        @test out[4, 3] == 0.0
    end

    @testset "NMS keeps a ridge that's a strict local max along its direction" begin
        # A horizontal ridge of all 1s. The gradient direction is vertical (π/2),
        # so each cell is compared to the one above and below. Both are zero
        # outside the ridge, so the whole ridge survives.
        mag = zeros(5, 7)
        mag[3, :] .= 1.0
        θ = fill(π/2, 5, 7)
        out = nonmaximum_suppression(mag, θ)
        @test out[3, 2:6] == ones(5)   # interior of the ridge survives
        @test all(out[2, :] .== 0)     # nothing above
        @test all(out[4, :] .== 0)     # nothing below
    end

    @testset "double_threshold partitions correctly" begin
        mag = [0.0  0.05  0.20;
               0.10 0.50  0.95;
               0.00 1.00  0.40]
        s, w = double_threshold(mag; low = 0.1, high = 0.5)   # 10% and 50% of max=1.0
        @test s == [false false false;
                    false true  true;
                    false true  false]
        @test w == [false false true;
                    true  false false;
                    false false true]
    end

    @testset "double_threshold absolute mode" begin
        mag = [0.0 0.1; 0.3 0.6]
        s, w = double_threshold(mag; low = 0.2, high = 0.5, relative = false)
        @test s == [false false; false true]
        @test w == [false false; true  false]
    end

    @testset "double_threshold rejects low > high" begin
        @test_throws ArgumentError double_threshold(rand(3, 3); low = 0.5, high = 0.2)
    end

    @testset "hysteresis connects a chain of weak pixels to a strong one" begin
        strong = falses(3, 5); strong[2, 1] = true
        weak   = falses(3, 5); weak[2, 2:4]  .= true
        out = hysteresis(strong, weak)
        # The weak chain all the way across is recruited because it touches the strong.
        @test out[2, :] == [true, true, true, true, false]
    end

    @testset "hysteresis ignores weak pixels that don't touch a strong one" begin
        strong = falses(5, 5); strong[1, 1] = true
        weak = falses(5, 5);   weak[5, 5] = true
        out = hysteresis(strong, weak)
        @test out[1, 1] == true
        @test out[5, 5] == false   # isolated weak → suppressed
    end

    @testset "canny on a flat image produces no edges" begin
        flat = fill(0.4, 32, 32)
        @test sum(canny(flat; sigma = 1.4, low = 0.05, high = 0.15)) == 0
    end

    @testset "canny on a vertical step gives a single thin column of edges" begin
        img = zeros(32, 32)
        img[:, 17:end] .= 1.0
        edges = canny(img; sigma = 1.0, low = 0.05, high = 0.15)
        # Most edges should sit on a single column near the step. Sum across
        # rows should be ~1 (one-pixel-wide ridge), exactly what NMS gives us.
        col_sums = vec(sum(edges; dims = 1))
        # The maximum column total should be high (most of the height), and most
        # other columns should be zero.
        @test maximum(col_sums) ≥ 24                  # at least 24 of 32 rows lit
        # Total edge pixels close to image height — confirms one-pixel-wide.
        @test 24 ≤ sum(edges) ≤ 40
    end

    @testset "canny_stages returns coherent intermediates" begin
        img = Synth.circle(48, 48; radius = 14)
        s = canny_stages(img; sigma = 1.0, low = 0.05, high = 0.15)
        @test size(s.blurred)   == size(img)
        @test size(s.magnitude) == size(img)
        @test size(s.edges)     == size(img)
        # Number of strong + weak pixels in NMS should match the threshold split.
        @test sum(s.strong) + sum(s.weak) == sum(s.nms .> 0.05 * maximum(s.nms))
        # Edges count is between strong and strong+weak inclusive.
        @test sum(s.strong) ≤ sum(s.edges) ≤ sum(s.strong) + sum(s.weak)
    end

    @testset "negative sigma is rejected" begin
        @test_throws ArgumentError canny(rand(8, 8); sigma = -0.5)
    end
end
