using Test
using ImageLab.Synth

@testset "Synth" begin
    @testset "checkerboard" begin
        img = checkerboard(4, 4; tile = 2)
        @test size(img) == (4, 4)
        @test extrema(img) == (0.0, 1.0)
        # Top-left tile should be one constant value.
        @test img[1, 1] == img[1, 2] == img[2, 1] == img[2, 2]
        # The neighboring tile should be the opposite.
        @test img[1, 3] != img[1, 2]
    end

    @testset "square / circle / impulse" begin
        sq = Synth.square(10, 10; side = 4, value = 1.0)
        @test sum(sq) ≈ 16.0  # 4×4 filled square
        @test sq[1, 1] == 0.0

        ci = Synth.circle(20, 20; radius = 5)
        @test 0.0 ≤ minimum(ci) && maximum(ci) ≤ 1.0
        # Roughly π r² pixels lit.
        @test 50 < sum(ci) < 110

        im = Synth.impulse(7, 7)
        @test sum(im) == 1.0
        @test im[4, 4] == 1.0  # exact center
    end

    @testset "ramp" begin
        rx = Synth.ramp(5, 5; axis = :x)
        @test rx[1, 1] == 0.0
        @test rx[1, 5] == 1.0
        @test all(rx[1, :] .== rx[3, :])  # constant along y

        ry = Synth.ramp(5, 5; axis = :y)
        @test ry[1, 1] == 0.0
        @test ry[5, 1] == 1.0
    end

    @testset "noise generators are reproducible with seed" begin
        base = Synth.square(16, 16; side = 8)
        a = Synth.gaussian_noise(base; sigma = 0.1, seed = 42)
        b = Synth.gaussian_noise(base; sigma = 0.1, seed = 42)
        @test a == b

        sp1 = Synth.salt_pepper(base; p = 0.2, seed = 1)
        sp2 = Synth.salt_pepper(base; p = 0.2, seed = 1)
        @test sp1 == sp2
    end
end
