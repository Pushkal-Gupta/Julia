using Test
using ImageLab
using ImageLab.Flow
using ImageLab.Viz: flow_to_rgb, hsv_to_rgb

# Sample a continuous sinusoid pattern at two offsets — gives a true
# subpixel motion without any bilinear-resampling artifacts.
function sinusoid_frame(H, W; freq = 0.05, shift_u = 0.0, shift_v = 0.0)
    img = zeros(H, W)
    for j in 1:W, i in 1:H
        img[i, j] = 0.5 + 0.4 *
                    sin(2π * freq * (j - shift_u)) *
                    cos(2π * freq * (i - shift_v))
    end
    return img
end

@testset "Flow (Lucas-Kanade)" begin

    @testset "Recovers known continuous shifts" begin
        H, W = 80, 80
        for (du, dv) in ((0.5, 0.0), (0.0, 0.5), (0.3, 0.7),
                         (-0.5, 0.0), (1.0, 0.0))
            img1 = sinusoid_frame(H, W)
            img2 = sinusoid_frame(H, W; shift_u = du, shift_v = dv)
            f = lucas_kanade(img1, img2;
                             window_size = 15, sigma = 2.0,
                             det_threshold = 1e-12)
            inner_u = f.u[25:55, 25:55]
            inner_v = f.v[25:55, 25:55]
            mu = sum(inner_u) / length(inner_u)
            mv = sum(inner_v) / length(inner_v)
            # Plain (non-iterative) LK has a small OFC-linearization bias.
            # Five percent of the shift magnitude is comfortably inside it.
            @test abs(mu - du) < 0.06
            @test abs(mv - dv) < 0.06
        end
    end

    @testset "Identical frames give zero flow" begin
        img = sinusoid_frame(64, 64)
        f = lucas_kanade(img, img; window_size = 11, sigma = 2.0)
        @test maximum(abs, f.u) < 1e-10
        @test maximum(abs, f.v) < 1e-10
    end

    @testset "Confidence is high in textured regions, low in flat" begin
        # Mostly-flat image with a textured patch in the middle
        H, W = 64, 64
        img1 = fill(0.5, H, W)
        img1[20:44, 20:44] .= sinusoid_frame(25, 25)
        img2 = copy(img1)
        # Shift just the textured patch by one column
        img2[20:44, 21:45] .= img1[20:44, 20:44]
        f = lucas_kanade(img1, img2; window_size = 9, sigma = 1.5,
                         det_threshold = 1e-12)
        textured_conf = sum(f.confidence[28:36, 28:36]) / 81
        flat_conf     = sum(f.confidence[1:8,    1:8])  / 64
        @test textured_conf > flat_conf * 100
    end

    @testset "FlowField type" begin
        img = sinusoid_frame(32, 32)
        f = lucas_kanade(img, img)
        @test f isa FlowField
        @test size(f) == (32, 32)
        @test size(f.u) == (32, 32) && size(f.v) == (32, 32)
        @test size(f.confidence) == (32, 32)
    end

    @testset "flow_magnitude and flow_angle" begin
        f = FlowField([3.0 0.0; 0.0 -3.0],
                      [0.0 4.0; -4.0 0.0],
                      ones(2, 2))
        m = flow_magnitude(f)
        @test m ≈ [3.0 4.0; 4.0 3.0]
        a = flow_angle(f)
        @test a[1, 1] ≈ 0.0           # (u, v) = (3, 0) → right
        @test a[1, 2] ≈ π / 2          # (0, 4) → down
        @test a[2, 1] ≈ -π / 2         # (0, -4) → up
        @test abs(a[2, 2] - π) ≈ 0.0   # (-3, 0) → left
    end

    @testset "Errors on mismatched sizes / even window" begin
        a = zeros(16, 16); b = zeros(16, 17)
        @test_throws DimensionMismatch lucas_kanade(a, b)
        @test_throws ArgumentError lucas_kanade(zeros(16, 16), zeros(16, 16);
                                                 window_size = 10)
    end

    @testset "warp_bilinear basic identity" begin
        img = sinusoid_frame(32, 32)
        # Zero flow → output equals input
        warped = warp_bilinear(img, zeros(32, 32), zeros(32, 32))
        @test maximum(abs, warped .- img) < 1e-12
        # A constant 1-pixel right shift on the flow means we're sampling
        # one column to the right, so warp(img, 1, 0)[i, j] = img[i, j+1].
        u_const = fill(1.0, 32, 32)
        v_const = zeros(32, 32)
        warped2 = warp_bilinear(img, u_const, v_const)
        for j in 1:31, i in 1:32
            @test isapprox(warped2[i, j], img[i, j + 1]; atol = 1e-12)
        end
        # Size mismatch errors
        @test_throws DimensionMismatch warp_bilinear(img, zeros(32, 31), zeros(32, 32))
    end

    @testset "Pyramidal LK recovers large translations" begin
        H, W = 128, 128
        # 3-pixel shift — well outside plain LK's small-motion regime
        img1 = sinusoid_frame(H, W; freq = 0.04)
        img2 = sinusoid_frame(H, W; freq = 0.04, shift_u = 3.0, shift_v = -2.0)
        fp = lucas_kanade_pyramid(img1, img2;
                                  levels = 4, window_size = 15, sigma = 2.0,
                                  iters_per_level = 3)
        inner_u = fp.u[40:H-40, 40:W-40]
        inner_v = fp.v[40:H-40, 40:W-40]
        mu = sum(inner_u) / length(inner_u)
        mv = sum(inner_v) / length(inner_v)
        # Pyramid LK should get within 5% of a 3-pixel shift; plain LK
        # would be off by ~20%+.
        @test abs(mu - 3.0) < 0.15
        @test abs(mv - (-2.0)) < 0.15
    end

    @testset "Pyramidal LK matches plain LK on small motions" begin
        # On a sub-pixel shift the pyramid version should be at least as
        # accurate as plain LK, since the only "extra work" it does is
        # iterative refinement on a residual that's already small.
        H, W = 96, 96
        img1 = sinusoid_frame(H, W)
        img2 = sinusoid_frame(H, W; shift_u = 0.3, shift_v = 0.2)
        f_plain   = lucas_kanade(img1, img2)
        f_pyramid = lucas_kanade_pyramid(img1, img2;
                                         levels = 3, iters_per_level = 3)
        plain_err   = abs(sum(f_plain.u[20:76, 20:76])   / (57 * 57) - 0.3) +
                      abs(sum(f_plain.v[20:76, 20:76])   / (57 * 57) - 0.2)
        pyramid_err = abs(sum(f_pyramid.u[20:76, 20:76]) / (57 * 57) - 0.3) +
                      abs(sum(f_pyramid.v[20:76, 20:76]) / (57 * 57) - 0.2)
        @test pyramid_err ≤ plain_err + 0.01
    end

    @testset "Pyramidal LK on identical frames is zero" begin
        img = sinusoid_frame(64, 64)
        f = lucas_kanade_pyramid(img, img; levels = 3)
        @test maximum(abs, f.u) < 1e-10
        @test maximum(abs, f.v) < 1e-10
    end

    @testset "Pyramidal LK errors on bad input" begin
        @test_throws DimensionMismatch lucas_kanade_pyramid(zeros(16, 16), zeros(16, 17))
        @test_throws ArgumentError lucas_kanade_pyramid(zeros(16, 16), zeros(16, 16);
                                                        levels = -1)
        @test_throws ArgumentError lucas_kanade_pyramid(zeros(16, 16), zeros(16, 16);
                                                        iters_per_level = 0)
    end

    @testset "Horn-Schunck recovers small motions" begin
        H, W = 80, 80
        for (du, dv) in ((0.3, 0.0), (0.0, 0.4), (0.3, 0.4))
            img1 = sinusoid_frame(H, W; freq = 0.04)
            img2 = sinusoid_frame(H, W; freq = 0.04,
                                  shift_u = du, shift_v = dv)
            hs = horn_schunck(img1, img2;
                              alpha = 0.1, iterations = 200)
            inner_u = hs.u[20:60, 20:60]
            inner_v = hs.v[20:60, 20:60]
            mu = sum(inner_u) / length(inner_u)
            mv = sum(inner_v) / length(inner_v)
            # Same ~5% OFC-linearization bias as plain LK.
            @test abs(mu - du) < 0.07
            @test abs(mv - dv) < 0.07
        end
    end

    @testset "Horn-Schunck on identical frames is zero" begin
        img = sinusoid_frame(48, 48)
        hs = horn_schunck(img, img; alpha = 0.1, iterations = 100)
        @test maximum(abs, hs.u) < 1e-10
        @test maximum(abs, hs.v) < 1e-10
    end

    @testset "Horn-Schunck propagates flow into flat regions" begin
        # Textured patch in a flat field with motion only on the patch.
        H, W = 96, 96
        img1 = fill(0.5, H, W)
        img1[30:66, 30:66] .= sinusoid_frame(37, 37; freq = 0.04)
        img2 = fill(0.5, H, W)
        img2[30:66, 30:66] .= sinusoid_frame(37, 37; freq = 0.04, shift_u = 0.4)

        lk = lucas_kanade(img1, img2; window_size = 15, sigma = 2.0)
        hs = horn_schunck(img1, img2; alpha = 0.5, iterations = 3000)

        # Just outside the textured patch, LK returns zero (no constraint)
        # but HS has diffused some flow in from the moving patch.
        outside_lk = sum(abs, lk.u[5:20, 40:55]) / (16 * 16)
        outside_hs = sum(abs, hs.u[5:20, 40:55]) / (16 * 16)
        @test outside_lk < 1e-6
        @test outside_hs > 0.05
    end

    @testset "Horn-Schunck argument validation" begin
        @test_throws DimensionMismatch horn_schunck(zeros(16, 16), zeros(16, 17))
        @test_throws ArgumentError horn_schunck(zeros(16, 16), zeros(16, 16);
                                                alpha = 0.0)
        @test_throws ArgumentError horn_schunck(zeros(16, 16), zeros(16, 16);
                                                alpha = -0.5)
        @test_throws ArgumentError horn_schunck(zeros(16, 16), zeros(16, 16);
                                                iterations = 0)
    end

    @testset "hsv_to_rgb sanity" begin
        # Red, green, blue at saturation 1, value 1
        @test all(hsv_to_rgb(0.0, 1.0, 1.0)    .≈ (1.0, 0.0, 0.0))
        @test all(hsv_to_rgb(1/3,  1.0, 1.0)  .≈ (0.0, 1.0, 0.0))
        @test all(hsv_to_rgb(2/3,  1.0, 1.0)  .≈ (0.0, 0.0, 1.0))
        # Zero saturation → grayscale
        r, g, b = hsv_to_rgb(0.5, 0.0, 0.7)
        @test r ≈ g ≈ b ≈ 0.7
        # Hue wraps modulo 1
        @test all(hsv_to_rgb(0.0, 1.0, 1.0) .≈ hsv_to_rgb(1.0, 1.0, 1.0))
        @test all(hsv_to_rgb(0.0, 1.0, 1.0) .≈ hsv_to_rgb(-1.0, 1.0, 1.0))
    end

    @testset "flow_to_rgb basics" begin
        u = [1.0  0.0; -1.0  0.0]
        v = [0.0  1.0;  0.0 -1.0]
        R, G, B = flow_to_rgb(u, v)
        @test size(R) == size(G) == size(B) == (2, 2)
        @test all(0.0 .≤ R .≤ 1.0)
        @test all(0.0 .≤ G .≤ 1.0)
        @test all(0.0 .≤ B .≤ 1.0)
        # All-zero flow returns all white (S = 0 everywhere, V = 1)
        Z = zeros(4, 4)
        Rz, Gz, Bz = flow_to_rgb(Z, Z)
        @test all(Rz .≈ 1.0)
        @test all(Gz .≈ 1.0)
        @test all(Bz .≈ 1.0)
    end
end
