using Test
using ImageLab.Kernels

@testset "Kernels" begin
    @testset "shape and sums" begin
        @test size(box(3)) == (3, 3)
        @test sum(box(5)) ≈ 1.0
        @test sum(gaussian(7; sigma = 1.0)) ≈ 1.0
        @test sum(identity_kernel(5)) == 1.0
    end

    @testset "Gaussian is centered and positive" begin
        K = gaussian(5; sigma = 1.0)
        c = 3
        @test K[c, c] == maximum(K)
        @test all(K .> 0)
        # Radially symmetric within floating point tolerance.
        @test K[c, c+1] ≈ K[c+1, c] ≈ K[c, c-1] ≈ K[c-1, c]
    end

    @testset "gradient kernels are antisymmetric and zero-sum" begin
        for K in (sobel_x(), sobel_y(), prewitt_x(), prewitt_y(),
                  scharr_x(), scharr_y(), roberts_x(), roberts_y())
            @test sum(K) == 0
        end
        # x-kernels are mirror-symmetric across their vertical axis (sign-flipped).
        Sx = sobel_x()
        @test Sx == -reverse(Sx; dims = 2)
    end

    @testset "Laplacians are zero-sum" begin
        @test sum(laplacian4()) == 0
        @test sum(laplacian8()) == 0
        @test abs(sum(laplacian_of_gaussian(7; sigma = 1.0))) < 1e-12
    end

    @testset "odd-size invariant" begin
        @test_throws ArgumentError box(4)
        @test_throws ArgumentError gaussian(6)
        @test_throws ArgumentError identity_kernel(2)
    end
end
