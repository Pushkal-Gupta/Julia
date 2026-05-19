using Test
using ImageLab.PNM
using ImageLab.Synth

@testset "PNM (PGM round-trip)" begin
    img = Synth.checkerboard(8, 12; tile = 2)
    path = tempname() * ".pgm"
    try
        PNM.save_pgm(path, img)
        @test isfile(path)
        reread = PNM.load_pgm(path)
        @test size(reread) == size(img)
        # 8-bit quantization introduces at most 1/255 error per pixel.
        @test maximum(abs.(reread .- img)) ≤ 1 / 255 + 1e-9
    finally
        isfile(path) && rm(path)
    end
end
