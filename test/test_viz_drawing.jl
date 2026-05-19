using Test
using ImageLab.Viz

@testset "Viz drawing helpers" begin
    @testset "draw_line! is symmetric and stays in bounds" begin
        img = zeros(5, 5)
        draw_line!(img, 1, 1, 5, 5; value = 1.0)
        # Main diagonal should be lit.
        for k in 1:5
            @test img[k, k] == 1.0
        end
    end

    @testset "draw_line! handles a horizontal line" begin
        img = zeros(3, 5)
        draw_line!(img, 2, 1, 2, 5; value = 0.7)
        @test img[2, :] == fill(0.7, 5)
        @test all(img[1, :] .== 0)
        @test all(img[3, :] .== 0)
    end

    @testset "draw_line! out-of-bounds doesn't error" begin
        img = zeros(3, 3)
        draw_line!(img, -5, -5, 10, 10; value = 1.0)
        # Whatever lands inside is fine, the rest is silently skipped.
        @test img[1, 1] == 1.0
        @test img[3, 3] == 1.0
    end

    @testset "mark_points! stamps 3×3 markers at default size" begin
        img = zeros(7, 7)
        mark_points!(img, [(4, 4)]; size = 1, value = 1.0)
        @test all(img[3:5, 3:5] .== 1.0)
        @test img[2, 2] == 0.0
    end

    @testset "label_to_gray gives distinct values to distinct labels" begin
        labels = [0 1 2; 1 0 3; 2 3 0]
        g = label_to_gray(labels)
        @test g[1, 1] == 0.0   # background
        # Label values should all be in [0.2, 1.0] and distinct for distinct labels.
        unique_vals = unique(g[labels .> 0])
        @test length(unique_vals) == 3
        @test all(0.2 .≤ unique_vals .≤ 1.0)
    end
end
