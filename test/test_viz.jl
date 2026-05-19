using Test
using ImageLab.Viz

@testset "Viz" begin
    @testset "normalize01" begin
        x = [-2.0 0.0; 2.0 4.0]
        n = normalize01(x)
        @test extrema(n) == (0.0, 1.0)
        @test n[1, 1] == 0.0     # was the min
        @test n[2, 2] == 1.0     # was the max
        @test normalize01(fill(7.0, 3, 3)) == zeros(3, 3)
    end

    @testset "signed_to_gray pins zero to 0.5" begin
        x = [-1.0 0.0; 0.5 1.0]
        g = signed_to_gray(x)
        @test g[1, 1] == 0.0
        @test g[1, 2] == 0.5
        @test g[2, 2] == 1.0
        @test signed_to_gray(zeros(2, 2)) == fill(0.5, 2, 2)
    end

    @testset "montage assembles a grid" begin
        a = fill(0.1, 4, 5)
        b = fill(0.9, 4, 5)
        c = fill(0.5, 4, 5)
        m = montage([a, b, c]; cols = 3, gap = 1, background = 0.0)
        # rows = 1, out_h = 4, out_w = 5*3 + 2*1 = 17
        @test size(m) == (4, 17)
        @test m[:, 1:5] == a
        @test m[:, 7:11] == b
        @test m[:, 13:17] == c
        @test all(m[:, 6] .== 0.0)  # gap column
    end

    @testset "montage rejects mismatched sizes" begin
        @test_throws DimensionMismatch montage([rand(3, 3), rand(3, 4)])
    end
end
