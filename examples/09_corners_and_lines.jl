#!/usr/bin/env julia
# 09_corners_and_lines.jl
#
# I want to see Harris corners and Hough lines on the same image, with
# the detections overlaid back onto the original. The input is a small
# scene of rectangles and a triangle — geometry with known corner
# positions, so I can eyeball whether the detector is finding what it
# should.
#
#   julia --project=. examples/09_corners_and_lines.jl
#
# What I'm expecting:
#   - Harris finds the four corners of each rectangle and the three
#     corners of the triangle. Some "satellite" detections near
#     corners are normal; min_distance suppression keeps them rare.
#   - Hough finds straight lines through every long edge: the four
#     sides of each rectangle and the three sides of the triangle.
#   - Curved edges (none in this image) wouldn't get strong Hough
#     responses — the votes spread across many (θ, ρ) bins.

using ImageLab
using ImageLab.Synth, ImageLab.Edges, ImageLab.Features, ImageLab.Viz, ImageLab.PNM

function studio_image(h, w)
    img = zeros(Float64, h, w)
    # Rectangle 1, top-left
    img[10:35, 12:50] .= 0.85
    # Rectangle 2, mid-right
    img[40:70, 70:110] .= 0.6
    # Triangle, bottom-left (filled). Vertices: (75, 8), (115, 8), (95, 55).
    # Iterate rows; for each row figure out the inclusive [j0, j1] range.
    for i in 75:115
        # Both sides slope from the apex (95, 55) down to (75 or 115, 8).
        t = (i - 75) / (115 - 75)            # 0 at top vertex, 1 at base
        # Top vertex is at (75, 8); bottom vertices are (115, 8) and (95, 55).
        # I want the triangle pointing right with apex on the right edge.
        # Recompute: vertices (75, 8), (115, 8), (95, 55).
        # Left edge: (75, 8) → (95, 55). Right edge: (115, 8) → (95, 55).
        # Hmm this is getting tangled — let me use a clearer triangle.
        break
    end
    # Replace the loop above with a simpler equilateral-ish triangle.
    # Vertices: (75, 10), (115, 10), (95, 55).
    v1 = (75, 10); v2 = (115, 10); v3 = (95, 55)
    for i in v1[1]:v2[1], j in v1[2]:v3[2]
        # Inside-triangle test using barycentric signs.
        s1 = (v2[1] - v1[1]) * (j - v1[2]) - (v2[2] - v1[2]) * (i - v1[1])
        s2 = (v3[1] - v2[1]) * (j - v2[2]) - (v3[2] - v2[2]) * (i - v2[1])
        s3 = (v1[1] - v3[1]) * (j - v3[2]) - (v1[2] - v3[2]) * (i - v3[1])
        if (s1 ≥ 0 && s2 ≥ 0 && s3 ≥ 0) || (s1 ≤ 0 && s2 ≤ 0 && s3 ≤ 0)
            img[i, j] = 0.7
        end
    end
    return img
end

input = studio_image(128, 128)

# Corners
R = harris_response(input; sigma = 1.2, k = 0.04)
corners = harris_corners(input; sigma = 1.2, k = 0.04,
                         threshold = 0.02, min_distance = 5)

# Lines via Canny → Hough → peaks.
edges = canny(input; sigma = 1.0, low = 0.06, high = 0.18)
acc = hough_lines(edges; n_theta = 180)
lines = hough_peaks(acc; threshold = 0.4,
                    min_distance_theta = 6,
                    min_distance_rho = 6,
                    max_peaks = 10)

# Overlay corners and lines on a copy of the input for display.
H, W = size(input)

# Lines: for each (θ, ρ), find image-boundary intersections and draw between them.
function endpoints(θ, ρ, H, W)
    cosθ, sinθ = cos(θ), sin(θ)
    pts = Tuple{Int, Int}[]   # (row, col)
    # x = 1 → y = (ρ - cosθ) / sinθ
    if abs(sinθ) > 1e-9
        y = (ρ - cosθ) / sinθ
        1 ≤ y ≤ H && push!(pts, (round(Int, y), 1))
        y = (ρ - W * cosθ) / sinθ
        1 ≤ y ≤ H && push!(pts, (round(Int, y), W))
    end
    if abs(cosθ) > 1e-9
        x = (ρ - sinθ) / cosθ
        1 ≤ x ≤ W && push!(pts, (1, round(Int, x)))
        x = (ρ - H * sinθ) / cosθ
        1 ≤ x ≤ W && push!(pts, (H, round(Int, x)))
    end
    length(pts) < 2 ? nothing : (pts[1], pts[2])
end

lines_overlay = copy(input)
for (θ, ρ) in lines
    ends = endpoints(θ, ρ, H, W)
    ends === nothing && continue
    (y0, x0), (y1, x1) = ends
    Viz.draw_line!(lines_overlay, y0, x0, y1, x1; value = 1.0)
end

corners_overlay = copy(input) .* 0.5    # darken input so corner dots stand out
Viz.mark_points!(corners_overlay, corners; size = 1, value = 1.0)

# A "lines on black" tile: just the detected lines, no underlying input.
# Useful for sanity-checking that the line set as a whole reconstructs
# the geometry.
lines_only = zeros(Float64, H, W)
for (θ, ρ) in lines
    ends = endpoints(θ, ρ, H, W)
    ends === nothing && continue
    (y0, x0), (y1, x1) = ends
    Viz.draw_line!(lines_only, y0, x0, y1, x1; value = 1.0)
end

outdir = joinpath(@__DIR__, "..", "artifacts", "09_corners_and_lines")
mkpath(outdir)

PNM.save_pgm(joinpath(outdir, "00_input.pgm"),           input)
PNM.save_pgm(joinpath(outdir, "01_harris_response.pgm"), Viz.normalize01(R))
PNM.save_pgm(joinpath(outdir, "02_corners_overlay.pgm"), corners_overlay)
PNM.save_pgm(joinpath(outdir, "03_canny_edges.pgm"),     Float64.(edges))
# The Hough accumulator is its own size (180 × n_rho); saved separately
# so it doesn't have to fit the montage grid.
PNM.save_pgm(joinpath(outdir, "04_hough_accumulator.pgm"),
             Viz.normalize01(Float64.(acc.counts)))
PNM.save_pgm(joinpath(outdir, "05_lines_only.pgm"),      lines_only)
PNM.save_pgm(joinpath(outdir, "06_lines_overlay.pgm"),   lines_overlay)

tiles = [
    input,             Viz.normalize01(R),  corners_overlay,
    Float64.(edges),   lines_only,          lines_overlay,
]
grid = Viz.montage(tiles; cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage.pgm"), grid)

println("Corners and lines:")
println("  Harris corners detected: $(length(corners))")
println("  Hough peaks (≥40% of max): $(length(lines))")
println()
println("  tile (row-major):")
println("    input          | Harris response   | corners overlay")
println("    Canny edges    | detected lines    | lines overlaid")
println("  (The Hough accumulator is a separate PGM; it isn't 128×128.)")
println("→ $outdir")
