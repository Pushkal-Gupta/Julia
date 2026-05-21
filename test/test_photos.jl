using Test
using ImageLab.Synth, ImageLab.Photos

@testset "Photos (PNG round-trip)" begin
    @testset "grayscale round-trip preserves the image to 8-bit precision" begin
        img = Synth.checkerboard(32, 48; tile = 4)
        path = tempname() * ".png"
        try
            Photos.save_grayscale(path, img)
            @test isfile(path)
            reread = Photos.load_grayscale(path)
            @test size(reread) == size(img)
            # 8-bit quantization → max error is 1/255.
            @test maximum(abs.(reread .- img)) ≤ 1 / 255 + 1e-9
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "out-of-range values clamp instead of wrapping" begin
        img = [-0.5 0.0; 0.5 1.5]   # contains values outside [0, 1]
        path = tempname() * ".png"
        try
            Photos.save_grayscale(path, img)
            reread = Photos.load_grayscale(path)
            # Both -0.5 and 1.5 should clamp to the bounds, not wrap.
            @test reread[1, 1] == 0.0
            @test reread[2, 2] == 1.0
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "RGB round-trip on three matched planes" begin
        R = Synth.circle(16, 16; radius = 5)
        G = Synth.square(16, 16; side = 8) .* 0.5
        B = Synth.ramp(16, 16; axis = :x)
        path = tempname() * ".png"
        try
            Photos.save_rgb_planes(path, R, G, B)
            R2, G2, B2 = Photos.load_rgb_planes(path)
            @test size(R2) == size(R)
            tol = 1 / 255 + 1e-9
            @test maximum(abs.(R2 .- R)) ≤ tol
            @test maximum(abs.(G2 .- G)) ≤ tol
            @test maximum(abs.(B2 .- B)) ≤ tol
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "mismatched plane sizes are rejected" begin
        @test_throws DimensionMismatch Photos.save_rgb_planes(
            tempname() * ".png", rand(4, 4), rand(4, 5), rand(4, 4))
    end

    @testset "save_grayscale also accepts .pgm via Netpbm backend" begin
        img = Synth.checkerboard(16, 16; tile = 4)
        path = tempname() * ".pgm"
        try
            Photos.save_grayscale(path, img)
            @test isfile(path)
            reread = Photos.load_grayscale(path)
            @test size(reread) == size(img)
            @test maximum(abs.(reread .- img)) ≤ 1 / 255 + 1e-9
        finally
            isfile(path) && rm(path)
        end
    end
end
