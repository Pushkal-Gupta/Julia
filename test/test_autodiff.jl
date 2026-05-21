using Test
using ImageLab
using ImageLab.AutoDiff, ImageLab.Kernels, ImageLab.Synth, ImageLab.Convolution
using Random

@testset "AutoDiff (differentiable filters)" begin

    # Uniform noise so every local 3×3 patch is distinct — that makes the
    # 9-element kernel fully observable from the input/target pair.
    # A periodic image like a checkerboard underdetermines the kernel and
    # the optimizer converges to a non-unique scaled / smoothed copy.
    Random.seed!(42)
    img    = rand(48, 48)
    target = correlate2d(img, sobel_x(); pad = :replicate)
    k_true = vec(sobel_x())

    @testset "Loss is zero at the true kernel" begin
        @test AutoDiff.kernel_loss(k_true, img, target) == 0.0
    end

    @testset "Gradient is zero at the true kernel" begin
        g = AutoDiff.kernel_gradient(k_true, img, target)
        @test length(g) == 9
        @test maximum(abs, g) < 1e-12
    end

    @testset "Loss is positive on a perturbed kernel" begin
        k_off = k_true .+ 0.05 .* randn(9)
        @test AutoDiff.kernel_loss(k_off, img, target) > 0.0
    end

    @testset "Gradient direction is descent" begin
        # If I step *against* the gradient by a small ε the loss should drop.
        k_off = k_true .+ 0.05 .* randn(9)
        L0 = AutoDiff.kernel_loss(k_off, img, target)
        g  = AutoDiff.kernel_gradient(k_off, img, target)
        ε  = 0.01
        L1 = AutoDiff.kernel_loss(k_off .- ε .* g, img, target)
        @test L1 < L0
    end

    @testset "Errors on bad input shapes" begin
        @test_throws ArgumentError AutoDiff.kernel_loss(zeros(8), img, target)
        @test_throws ArgumentError AutoDiff.kernel_loss(zeros(4), img, target)  # even side length
        small_target = zeros(16, 16)
        @test_throws DimensionMismatch AutoDiff.kernel_loss(zeros(9), img, small_target)
    end

    @testset "fit_kernel converges to the true Sobel kernel" begin
        Random.seed!(0)
        learned, history = AutoDiff.fit_kernel(img, target;
                                               iterations = 800, lr = 0.1)
        # Loss has dropped by many orders of magnitude.
        @test history[end] < history[1] * 1e-8
        # Learned kernel matches Sobel to ~1e-4.
        @test maximum(abs, learned .- sobel_x()) < 1e-3
    end

    @testset "fit_kernel respects init" begin
        Random.seed!(0)
        # Starting AT the answer should keep the kernel at the answer.
        learned, history = AutoDiff.fit_kernel(img, target;
                                               iterations = 50, lr = 0.1,
                                               init = sobel_x())
        @test all(history .== 0.0)
        @test learned ≈ sobel_x()
    end

    @testset "fit_kernel can learn a Laplacian too" begin
        # Same machinery, different target operator — proves the demo wasn't
        # a Sobel-specific fluke.
        Random.seed!(0)
        target_lap = correlate2d(img, laplacian4(); pad = :replicate)
        learned, hist = AutoDiff.fit_kernel(img, target_lap;
                                            iterations = 800, lr = 0.1)
        @test hist[end] < hist[1] * 1e-7
        @test maximum(abs, learned .- laplacian4()) < 1e-3
    end
end
