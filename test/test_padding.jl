using Test
using ImageLab.Padding

@testset "Padding" begin
    A = reshape(1:12, 3, 4) |> Matrix  # 3×4

    @testset ":valid is a no-op" begin
        @test pad_array(A, (2, 2, 2, 2); mode = :valid) == A
    end

    @testset ":zero" begin
        P = pad_array(A, (1, 1, 1, 1); mode = :zero)
        @test size(P) == (5, 6)
        @test P[2:4, 2:5] == A
        @test all(P[1, :] .== 0)
        @test all(P[:, 1] .== 0)
        @test all(P[end, :] .== 0)
        @test all(P[:, end] .== 0)
    end

    @testset ":replicate clamps to edge" begin
        P = pad_array(A, (1, 1, 1, 1); mode = :replicate)
        @test P[1, 2:5] == A[1, :]            # top row = first source row
        @test P[end, 2:5] == A[end, :]        # bottom row = last source row
        @test P[2:4, 1] == A[:, 1]            # left column = first source column
        @test P[1, 1] == A[1, 1]              # corner clamps both ways
        @test P[1, end] == A[1, end]
    end

    @testset ":reflect mirrors without repeating the edge" begin
        v = collect(1:5)
        M = reshape(v, 1, 5) |> Matrix
        P = pad_array(M, (0, 0, 2, 2); mode = :reflect)
        @test vec(P) == [3, 2, 1, 2, 3, 4, 5, 4, 3]
    end

    @testset ":symmetric mirrors with the edge repeated" begin
        M = reshape(collect(1:5), 1, 5) |> Matrix
        P = pad_array(M, (0, 0, 2, 2); mode = :symmetric)
        @test vec(P) == [2, 1, 1, 2, 3, 4, 5, 5, 4]
    end

    @testset ":circular wraps around" begin
        M = reshape(collect(1:5), 1, 5) |> Matrix
        P = pad_array(M, (0, 0, 2, 2); mode = :circular)
        @test vec(P) == [4, 5, 1, 2, 3, 4, 5, 1, 2]
    end

    @testset "unknown mode rejected" begin
        @test_throws ArgumentError pad_array(A, (1, 1, 1, 1); mode = :gibberish)
    end

    @testset "pad_vector mirrors pad_array on 1D" begin
        v = collect(1:5)
        @test pad_vector(v, 2, 2; mode = :reflect) == [3, 2, 1, 2, 3, 4, 5, 4, 3]
        @test pad_vector(v, 2, 2; mode = :symmetric) == [2, 1, 1, 2, 3, 4, 5, 5, 4]
        @test pad_vector(v, 2, 2; mode = :circular) == [4, 5, 1, 2, 3, 4, 5, 1, 2]
        @test pad_vector(v, 1, 1; mode = :replicate) == [1, 1, 2, 3, 4, 5, 5]
        @test pad_vector(v, 1, 1; mode = :zero) == [0, 1, 2, 3, 4, 5, 0]
        @test pad_vector(v, 5, 5; mode = :valid) == v
    end
end
