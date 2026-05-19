using Test
using ImageLab.Metrics

@testset "edge_match_stats" begin
    @testset "identical masks score perfectly" begin
        m = falses(5, 5); m[3, :] .= true
        p, r, f = edge_match_stats(m, m; tolerance = 0)
        @test p == 1.0 && r == 1.0 && f == 1.0
    end

    @testset "tolerance lets near-matches score as correct" begin
        gt = falses(7, 7); gt[4, :] .= true
        # Prediction shifted down by 1.
        pred = falses(7, 7); pred[5, :] .= true
        p0, r0, _ = edge_match_stats(pred, gt; tolerance = 0)
        @test p0 == 0.0 && r0 == 0.0
        p1, r1, f1 = edge_match_stats(pred, gt; tolerance = 1)
        @test p1 == 1.0 && r1 == 1.0 && f1 == 1.0
    end

    @testset "disjoint and far-apart scores zero" begin
        gt   = falses(10, 10); gt[1, 1]    = true
        pred = falses(10, 10); pred[10, 10] = true
        p, r, f = edge_match_stats(pred, gt; tolerance = 1)
        @test p == 0.0 && r == 0.0 && f == 0.0
    end

    @testset "both empty is treated as perfect" begin
        e = falses(4, 4)
        @test edge_match_stats(e, e) == (1.0, 1.0, 1.0)
    end

    @testset "one empty is zero" begin
        gt   = falses(4, 4); gt[2, 2] = true
        pred = falses(4, 4)
        @test edge_match_stats(pred, gt) == (0.0, 0.0, 0.0)
    end

    @testset "rejects mismatched sizes" begin
        @test_throws DimensionMismatch edge_match_stats(
            BitMatrix(falses(3, 3)), BitMatrix(falses(3, 4)))
    end
end

@testset "iou_score" begin
    a = falses(3, 3); a[1, 1] = true; a[2, 2] = true
    b = falses(3, 3);                  b[2, 2] = true; b[3, 3] = true
    @test iou_score(a, b) ≈ 1 / 3           # 1 intersection, 3 union
    @test iou_score(a, a) == 1.0
    @test iou_score(falses(2, 2), falses(2, 2)) == 1.0    # both empty
    @test iou_score(falses(2, 2), trues(2, 2)) == 0.0
end
