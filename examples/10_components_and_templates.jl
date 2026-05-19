#!/usr/bin/env julia
# 10_components_and_templates.jl
#
# Two questions on the same image:
#
#   1. If I run Canny and ask "how many separate objects are there?",
#      what do I get? Connected-component labeling on the Canny output
#      tells me — and the component sizes are how I'd filter out
#      spurious tiny segments in a real pipeline.
#
#   2. If I have a template (a small image patch I want to find) and
#      an image full of similar things, can normalized cross-correlation
#      pick them out? NCC normalizes per-window brightness and contrast,
#      so it handles uneven lighting in a way plain correlation can't.
#
#   julia --project=. examples/10_components_and_templates.jl
#
# What I'm expecting:
#   - The components plot colors each disconnected edge region with a
#     different gray value. Small specks should be obvious — that's
#     what `component_sizes` is for.
#   - NCC should peak strongly (>0.85) at every disk in the image and
#     much lower elsewhere. Even when I darken half the image, the
#     same disks still match.

using ImageLab
using ImageLab.Synth, ImageLab.Edges, ImageLab.Filters, ImageLab.Features
using ImageLab.Viz, ImageLab.PNM
using Printf

# ── Part 1: connected components on a Canny output ──────────────────────────

function components_image(h, w)
    # Several distinct shapes, including a couple of small ones that I
    # expect to show up as tiny components.
    img = zeros(Float64, h, w)
    img[15:35, 15:35]  .= 0.8                    # square
    cy, cx = 70, 30
    for j in 1:w, i in 1:h
        ((i - cy)^2 + (j - cx)^2 ≤ 12^2) && (img[i, j] = 0.9)   # disk 1
    end
    img[25:30, 70:95]  .= 0.7                    # short bar
    cy, cx = 80, 80
    for j in 1:w, i in 1:h
        ((i - cy)^2 + (j - cx)^2 ≤ 6^2) && (img[i, j] = 0.85)   # small disk
    end
    img[100:103, 100:103] .= 0.5                 # tiny speck (4×4)
    return Synth.gaussian_noise(img; sigma = 0.04, seed = 19)
end

cc_input = components_image(128, 128)
cc_edges = canny(cc_input; sigma = 1.0, low = 0.06, high = 0.18)
labels, n_components = connected_components(cc_edges; connectivity = 8)
sizes = component_sizes(labels, n_components)

outdir = joinpath(@__DIR__, "..", "artifacts", "10_components_and_templates")
mkpath(outdir)

PNM.save_pgm(joinpath(outdir, "00_components_input.pgm"),   cc_input)
PNM.save_pgm(joinpath(outdir, "01_components_edges.pgm"),   Float64.(cc_edges))
PNM.save_pgm(joinpath(outdir, "02_components_labels.pgm"),  Viz.label_to_gray(labels))

println("Connected components on the Canny output:")
println("  total components: $n_components")
println("  component sizes (sorted desc): $(sort(sizes; rev = true))")
println()

# ── Part 2: template matching via NCC ─────────────────────────────────────────

# Build an image with multiple instances of a target disk, scattered
# at varying positions and intensities. Then extract the disk pattern
# itself as the template.
function templates_image(h, w)
    img = fill(0.15, h, w)
    disk_centers = [(25, 25), (25, 70), (60, 40), (90, 80), (100, 25), (40, 105)]
    radius = 7
    for (k, (cy, cx)) in enumerate(disk_centers)
        # Vary the intensity per disk so NCC's brightness invariance has
        # something to prove against plain correlation.
        intensity = 0.6 + 0.05 * k
        for j in 1:w, i in 1:h
            if (i - cy)^2 + (j - cx)^2 ≤ radius^2
                img[i, j] = intensity
            end
        end
    end
    # Darken the right half: a brightness-invariant matcher should still
    # find the disks there.
    img[:, w÷2:end] .*= 0.5
    return img
end

template_img = templates_image(128, 128)

# Build the template by hand: a 17×17 disk on a dark background. I want
# the template's "average" brightness to make sense vs the target
# disks — the actual values don't matter once NCC normalizes.
template = fill(0.0, 17, 17)
for j in 1:17, i in 1:17
    ((i - 9)^2 + (j - 9)^2 ≤ 7^2) && (template[i, j] = 1.0)
end

ncc = normalized_cross_correlation(template_img, template)
matches = ncc_peaks(ncc; threshold = 0.75, min_distance = 10)

# Overlay match positions on the original image. NCC returns top-left
# corner coordinates; shift to the template center for the marker.
match_overlay = copy(template_img)
center_offsets = [(i + 8, j + 8) for (i, j) in matches]
Viz.mark_points!(match_overlay, center_offsets; size = 1, value = 1.0)

PNM.save_pgm(joinpath(outdir, "10_template_input.pgm"),  template_img)
PNM.save_pgm(joinpath(outdir, "11_template_pattern.pgm"), template)
PNM.save_pgm(joinpath(outdir, "12_ncc_response.pgm"),     Viz.normalize01(ncc))
PNM.save_pgm(joinpath(outdir, "13_matches_overlay.pgm"),  match_overlay)

println("Template matching via NCC:")
println("  template size: $(size(template))")
println("  NCC range: [$(round(minimum(ncc), digits=3)), $(round(maximum(ncc), digits=3))]")
println("  matches above 0.75: $(length(matches))")
for (k, (i, j)) in enumerate(matches)
    score = ncc[i, j]
    @printf("    #%d  top-left (%d, %d)  score=%.3f\n", k, i, j, score)
end
println()

# Two montages — one per part. Pad/crop so the cells match in size.
cc_tiles = [cc_input, Float64.(cc_edges), Viz.label_to_gray(labels)]
cc_grid = Viz.montage(cc_tiles; cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage_components.pgm"), cc_grid)

# Template-matching tiles: input, NCC response (sized H-th+1 × W-tw+1),
# and the matches overlay. NCC is smaller, so I pad it with the
# montage background instead of resizing.
function pad_to(img, h, w; bg = 0.5)
    H0, W0 = size(img)
    (H0 == h && W0 == w) && return img
    out = fill(bg, h, w)
    i0 = (h - H0) ÷ 2 + 1
    j0 = (w - W0) ÷ 2 + 1
    out[i0:i0 + H0 - 1, j0:j0 + W0 - 1] .= img
    return out
end
H, W = size(template_img)
ncc_padded = pad_to(Viz.normalize01(ncc), H, W)
tm_tiles = [template_img, ncc_padded, match_overlay]
tm_grid = Viz.montage(tm_tiles; cols = 3, gap = 4, background = 0.5)
PNM.save_pgm(joinpath(outdir, "montage_templates.pgm"), tm_grid)

println("→ $outdir")
println("Open montage_components.pgm and montage_templates.pgm for the visuals.")
