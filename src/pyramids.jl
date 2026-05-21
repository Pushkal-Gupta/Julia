"""
    Pyramids

Image pyramids — Gaussian and Laplacian, both built from the same two
primitives (`reduce` and `expand`). A few things this is good for:

- Scale-space reasoning: looking at the same scene at multiple
  resolutions. Useful for detectors that need to be scale-invariant.
- Compression: a Laplacian pyramid stores the difference between
  each Gaussian level and an upsampled version of the next, which
  concentrates most of the signal in the small levels.
- Multi-scale features: blob detection across scales, SIFT-style
  keypoint detection, image blending.

I follow Burt and Adelson's original 1983 construction: smooth with a
separable binomial filter `[1, 4, 6, 4, 1] / 16`, downsample by
taking every other pixel, and reverse the operation by inserting
zeros at odd positions and smoothing with the same filter scaled by
4 (the energy-compensation factor).
"""
module Pyramids

using ..Convolution: separable_correlate2d

export reduce_image, expand_image,
       gaussian_pyramid, laplacian_pyramid, reconstruct_laplacian_pyramid,
       laplacian_blend

# The standard Burt-Adelson binomial filter. Approximates a Gaussian
# with σ ≈ 1.0 over a 5-tap window. Sums to 1; outer product with
# itself is the 2D filter actually applied.
const _BURT_ADELSON_FILTER = [1.0, 4.0, 6.0, 4.0, 1.0] ./ 16

"""
    reduce_image(img; pad=:replicate) -> Matrix{Float64}

REDUCE: smooth `img` with the Burt-Adelson 5-tap binomial filter,
then downsample by 2 (keep every other pixel). Output is
roughly `(⌈H/2⌉, ⌈W/2⌉)`.
"""
function reduce_image(img::AbstractMatrix{<:Real}; pad::Symbol = :replicate)
    smoothed = separable_correlate2d(Float64.(img),
                                     _BURT_ADELSON_FILTER, _BURT_ADELSON_FILTER;
                                     pad = pad)
    return smoothed[1:2:end, 1:2:end]
end

"""
    expand_image(img, target_size; pad=:replicate) -> Matrix{Float64}

EXPAND: insert zeros at every other position to roughly double the
size, then smooth with the Burt-Adelson filter scaled by 4. The
factor 4 compensates for the energy loss of zero-insertion (only
1 in 4 output pixels are non-zero before smoothing).

`target_size` is the exact `(H, W)` to produce. This matters for
pyramid reconstruction: when we expand level k+1 to compare with
level k, we need the result to match level k's dimensions exactly,
which is `2·size(level_{k+1})` only when the lower level's dimensions
were even.
"""
function expand_image(img::AbstractMatrix{<:Real},
                      target_size::NTuple{2, Int};
                      pad::Symbol = :replicate)
    th, tw = target_size
    H, W = size(img)
    th ≥ 1 && tw ≥ 1 || throw(ArgumentError("target_size must be ≥ (1, 1)"))

    # Zero-insert into a `target_size` canvas. The original pixels go to
    # rows/cols 1, 3, 5, ... — keeping the (1, 1) corner aligned.
    big = zeros(Float64, th, tw)
    H_used = min(H, (th + 1) ÷ 2)
    W_used = min(W, (tw + 1) ÷ 2)
    big[1:2:2H_used - 1, 1:2:2W_used - 1] .= Float64.(img[1:H_used, 1:W_used])

    # Smooth with the 5-tap filter. Energy compensation is 4× total (only
    # 1 in 4 input pixels was non-zero), so each separable 1D pass scales
    # by 2 — the outer product is then 4 · (w · w^T).
    w = 2 .* _BURT_ADELSON_FILTER
    return separable_correlate2d(big, w, w; pad = pad)
end

"""
    gaussian_pyramid(img; levels=4, pad=:replicate) -> Vector{Matrix{Float64}}

Build a Gaussian pyramid: `[G_0, G_1, ..., G_levels]` where `G_0` is
the input and each subsequent level is `reduce(previous)`. The output
vector has `levels + 1` entries.

The smallest level can degenerate if `levels` is too large for the
input size; choose `levels` such that `min(H, W) ≥ 2^levels` for
sensible output.
"""
function gaussian_pyramid(img::AbstractMatrix{<:Real};
                          levels::Integer = 4,
                          pad::Symbol = :replicate)
    levels ≥ 0 || throw(ArgumentError("levels must be ≥ 0, got $levels"))
    pyramid = Vector{Matrix{Float64}}(undef, levels + 1)
    pyramid[1] = Float64.(img)
    for k in 2:(levels + 1)
        pyramid[k] = reduce_image(pyramid[k - 1]; pad = pad)
    end
    return pyramid
end

"""
    laplacian_pyramid(img; levels=4, pad=:replicate) -> Vector{Matrix{Float64}}

Build a Laplacian pyramid: `[L_0, L_1, ..., L_{levels-1}, G_levels]`
where `L_k = G_k − expand(G_{k+1})` and the last entry is the
smallest Gaussian (the "residual"). Total entries: `levels + 1`,
same as the Gaussian pyramid.

The Laplacian levels carry the high-frequency detail at each scale;
the smallest Gaussian carries the low-frequency residual that the
detail levels can't represent.
"""
function laplacian_pyramid(img::AbstractMatrix{<:Real};
                           levels::Integer = 4,
                           pad::Symbol = :replicate)
    G = gaussian_pyramid(img; levels = levels, pad = pad)
    L = Vector{Matrix{Float64}}(undef, levels + 1)
    for k in 1:levels
        expanded = expand_image(G[k + 1], size(G[k]); pad = pad)
        L[k] = G[k] .- expanded
    end
    L[end] = G[end]   # residual
    return L
end

"""
    reconstruct_laplacian_pyramid(L; pad=:replicate) -> Matrix{Float64}

Invert a Laplacian pyramid. Starting from the smallest residual,
repeatedly expand and add the next-larger Laplacian level. For a
pyramid built with `laplacian_pyramid`, the reconstruction matches
the original input to floating-point precision (the only error is
the rounding inside expand/reduce).
"""
function reconstruct_laplacian_pyramid(L::AbstractVector{<:AbstractMatrix{<:Real}};
                                       pad::Symbol = :replicate)
    isempty(L) && throw(ArgumentError("empty pyramid"))
    current = Float64.(L[end])
    for k in (length(L) - 1):-1:1
        current = expand_image(current, size(L[k]); pad = pad) .+ Float64.(L[k])
    end
    return current
end

"""
    laplacian_blend(A, B, mask; levels=4, pad=:replicate) -> Matrix{Float64}

Multi-band image blending via the Laplacian pyramid — the classical
application that motivated the pyramid algorithm in 1983.

Direct linear blending `A · mask + B · (1 − mask)` produces a visible
seam wherever the mask transitions sharply, because high-frequency
content from one image meets high-frequency content from the other at
the same pixel. Multi-band blending fixes this by blending each
*frequency band* at its own *spatial scale*: high frequencies get
blended with a sharp mask (so detail isn't smoothed out), low
frequencies get blended with a heavily-smoothed mask (so color shifts
fade in over many pixels).

The trick is that a Laplacian pyramid is precisely a frequency-band
decomposition, and a Gaussian pyramid of the mask gives the same
mask at every scale. Blending pyramids level-by-level with their
matched mask scales, then reconstructing, gives a seam-free composite.

Arguments:

- `A`, `B`     — the two images, same size.
- `mask`       — same size as A/B, values in `[0, 1]`. `1` means
                  "take from A", `0` means "take from B".
- `levels`     — how many pyramid levels to use. More levels = smoother
                  blend over a wider transition region.
- `pad`        — forwarded to the pyramid construction.
"""
function laplacian_blend(A::AbstractMatrix{<:Real},
                         B::AbstractMatrix{<:Real},
                         mask::AbstractMatrix{<:Real};
                         levels::Integer = 4,
                         pad::Symbol = :replicate)
    size(A) == size(B) == size(mask) || throw(DimensionMismatch(
        "A, B, and mask must share size; got $(size(A)), $(size(B)), $(size(mask))"))

    LA = laplacian_pyramid(A;    levels = levels, pad = pad)
    LB = laplacian_pyramid(B;    levels = levels, pad = pad)
    GM = gaussian_pyramid(mask;  levels = levels, pad = pad)

    blended = Vector{Matrix{Float64}}(undef, length(LA))
    for k in eachindex(blended)
        m = Float64.(GM[k])
        blended[k] = m .* Float64.(LA[k]) .+ (1 .- m) .* Float64.(LB[k])
    end
    return reconstruct_laplacian_pyramid(blended; pad = pad)
end

end # module Pyramids
