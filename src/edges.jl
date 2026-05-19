"""
    Edges

Classical edge detection — first- and second-order operators built on
top of the convolution engine. Functions here return *responses*, not
decisions: gradient images, magnitudes, directions, Laplacian / LoG /
DoG outputs, zero-crossing maps, threshold masks. The decision-making
(non-maximum suppression and hysteresis, i.e. Canny) lives elsewhere.

Operator dispatch is by `Symbol`:

| Symbol     | Kernel    | Notes                                            |
|------------|-----------|--------------------------------------------------|
| `:sobel`   | 3×3       | separable, smoothing weight `[1,2,1]`            |
| `:prewitt` | 3×3       | separable, uniform smoothing weight `[1,1,1]`    |
| `:scharr`  | 3×3       | separable, weight `[3,10,3]` — best rotational symmetry |
| `:roberts` | 2×2       | non-separable, direct-loop implementation       |

The first three use the separable factorizations from M2; Roberts gets
a hand-written 2×2 inner loop because forcing a 2×2 through the odd-only
2D engine would obscure what's happening.
"""
module Edges

using ..Padding
using ..Kernels
using ..Convolution

export gradient, gradient_magnitude, gradient_direction,
       quantize_direction,
       log_filter, dog_filter,
       zero_crossings,
       threshold_mask, percentile_threshold

const OPERATORS = (:sobel, :prewitt, :scharr, :roberts)

"""
    gradient(img, op=:sobel; pad=:replicate) -> (gx, gy)

Compute the x- and y-gradient images using the requested operator. `gx`
responds to vertical edges (intensity changes along x); `gy` responds to
horizontal edges. Returned matrices match `img`'s size.
"""
function gradient(img::AbstractMatrix{<:Real}, op::Symbol = :sobel;
                  pad::Symbol = :replicate)
    if op === :sobel
        kxx, kyx = sobel_x_separable()
        kxy, kyy = sobel_y_separable()
        return (separable_correlate2d(img, kxx, kyx; pad = pad),
                separable_correlate2d(img, kxy, kyy; pad = pad))
    elseif op === :prewitt
        kxx, kyx = prewitt_x_separable()
        kxy, kyy = prewitt_y_separable()
        return (separable_correlate2d(img, kxx, kyx; pad = pad),
                separable_correlate2d(img, kxy, kyy; pad = pad))
    elseif op === :scharr
        kxx, kyx = scharr_x_separable()
        kxy, kyy = scharr_y_separable()
        return (separable_correlate2d(img, kxx, kyx; pad = pad),
                separable_correlate2d(img, kxy, kyy; pad = pad))
    elseif op === :roberts
        return _roberts_gradient(img, pad)
    else
        throw(ArgumentError("unknown operator :$op; expected one of $OPERATORS"))
    end
end

# Direct-loop Roberts. The 2×2 kernels [1 0; 0 -1] and [0 1; -1 0] anchor
# the output at the top-left of the 2×2 footprint. We pad the bottom and
# right by 1 so the output keeps `size(img)`.
function _roberts_gradient(img::AbstractMatrix{<:Real}, pad::Symbol)
    A = Padding.pad_array(img, (0, 1, 0, 1); mode = pad)
    H, W = size(img)
    g1 = Matrix{Float64}(undef, H, W)
    g2 = Matrix{Float64}(undef, H, W)
    @inbounds for j in 1:W, i in 1:H
        a = A[i, j]; b = A[i, j + 1]; c = A[i + 1, j]; d = A[i + 1, j + 1]
        g1[i, j] = a - d   # ↘ diagonal
        g2[i, j] = b - c   # ↙ diagonal
    end
    return (g1, g2)
end

"""
    gradient_magnitude(gx, gy) -> Matrix{Float64}

Euclidean magnitude `√(gx² + gy²)`. Sign-blind — both edges of a step
respond equally.
"""
function gradient_magnitude(gx::AbstractMatrix{<:Real}, gy::AbstractMatrix{<:Real})
    size(gx) == size(gy) || throw(DimensionMismatch(
        "gx and gy must share size; got $(size(gx)) and $(size(gy))"))
    return sqrt.(Float64.(gx) .^ 2 .+ Float64.(gy) .^ 2)
end

"""
    gradient_direction(gx, gy) -> Matrix{Float64}

Gradient angle in radians, `atan(gy, gx)`, range `(-π, π]`. Direction
points along the steepest ascent of intensity (perpendicular to the
edge itself).
"""
function gradient_direction(gx::AbstractMatrix{<:Real}, gy::AbstractMatrix{<:Real})
    size(gx) == size(gy) || throw(DimensionMismatch(
        "gx and gy must share size; got $(size(gx)) and $(size(gy))"))
    return atan.(Float64.(gy), Float64.(gx))
end

"""
    quantize_direction(θ) -> Int

Quantize a gradient angle to one of four sectors used by non-maximum
suppression:

| Sector | Edge orientation | Angle range (mod π)              |
|--------|------------------|-----------------------------------|
| 0      | horizontal       | `[-π/8, π/8) ∪ [7π/8, π)`         |
| 1      | NE-diagonal (↗)  | `[π/8, 3π/8)`                     |
| 2      | vertical         | `[3π/8, 5π/8)`                    |
| 3      | NW-diagonal (↖)  | `[5π/8, 7π/8)`                    |

The edge itself is perpendicular to the gradient, so sector 2 (gradient
vertical) corresponds to a horizontal edge — and that's exactly which
neighbor pairs NMS will sample.
"""
function quantize_direction(θ::Real)
    t = mod(θ, π)   # collapse to [0, π); edge direction is modulo π
    if t < π/8 || t ≥ 7π/8
        return 0
    elseif t < 3π/8
        return 1
    elseif t < 5π/8
        return 2
    else
        return 3
    end
end

"""
    quantize_direction(θ::AbstractMatrix) -> Matrix{Int}

Element-wise version for a whole direction map.
"""
quantize_direction(θ::AbstractMatrix{<:Real}) = quantize_direction.(θ)

# ── Second-order operators ────────────────────────────────────────────────────

"""
    log_filter(img, n=9; sigma=n/6, pad=:replicate) -> Matrix{Float64}

Apply the Laplacian-of-Gaussian (Marr–Hildreth) operator. Equivalent to
Gaussian-blurring `img` with σ then taking the Laplacian, but expressed
as a single combined kernel.

Output is signed: positive on the dark side of an edge, negative on the
bright side, with a zero crossing along the edge itself. Visualize with
`Viz.signed_to_gray`.
"""
function log_filter(img::AbstractMatrix{<:Real}, n::Integer = 9;
                    sigma::Real = n / 6, pad::Symbol = :replicate)
    return correlate2d(img, laplacian_of_gaussian(n; sigma = sigma); pad = pad)
end

"""
    dog_filter(img; sigma1=1.0, sigma2=1.6, n=auto, pad=:replicate) -> Matrix{Float64}

Difference of Gaussians: `G(σ₁) ∗ img  −  G(σ₂) ∗ img`. A bandpass filter
— it picks out structure between the two scales. The classic ratio
`σ₂ / σ₁ ≈ 1.6` makes DoG a close approximation to LoG at a fraction of
the cost (two separable Gaussians instead of a 2D LoG kernel).

If `n` (kernel side) is not provided, it's set to `2·⌈3σ₂⌉ + 1`.
"""
function dog_filter(img::AbstractMatrix{<:Real};
                    sigma1::Real = 1.0, sigma2::Real = 1.6,
                    n::Integer = 2 * ceil(Int, 3 * max(sigma1, sigma2)) + 1,
                    pad::Symbol = :replicate)
    sigma1 > 0 && sigma2 > 0 || throw(ArgumentError(
        "both sigmas must be positive, got ($sigma1, $sigma2)"))
    isodd(n) || throw(ArgumentError("kernel size must be odd, got $n"))
    g1 = gaussian1d(n; sigma = sigma1)
    g2 = gaussian1d(n; sigma = sigma2)
    return separable_correlate2d(img, g1, g1; pad = pad) .-
           separable_correlate2d(img, g2, g2; pad = pad)
end

"""
    zero_crossings(img; min_diff=0.0) -> BitMatrix

Mark pixels where the 4-connected neighborhood contains a sign change
larger than `min_diff` in magnitude. The intended input is a Laplacian /
LoG / DoG response: zero crossings of these operators sit *exactly* on
edges.

`min_diff` suppresses zero crossings caused by float-noise on flat
regions. Tune it relative to the magnitude of the response — typically
a few percent of `maximum(abs, img)` works well.
"""
function zero_crossings(img::AbstractMatrix{<:Real}; min_diff::Real = 0.0)
    H, W = size(img)
    out = falses(H, W)
    @inbounds for j in 2:W - 1, i in 2:H - 1
        c = img[i, j]
        n = img[i - 1, j]; s = img[i + 1, j]
        w = img[i, j - 1]; e = img[i, j + 1]
        # Sign change against any neighbor + minimum local contrast.
        crosses = (c * n < 0 && abs(c - n) > min_diff) ||
                  (c * s < 0 && abs(c - s) > min_diff) ||
                  (c * w < 0 && abs(c - w) > min_diff) ||
                  (c * e < 0 && abs(c - e) > min_diff)
        out[i, j] = crosses
    end
    return out
end

# ── Threshold helpers ─────────────────────────────────────────────────────────

"""
    threshold_mask(img, t) -> BitMatrix

Simplest possible edge selector: `img .> t`. Useful on a gradient
magnitude image after you've eyeballed a good threshold.
"""
threshold_mask(img::AbstractMatrix{<:Real}, t::Real) = img .> t

"""
    percentile_threshold(img, pct) -> Float64

Return the value at the `pct`-th quantile of `img` (e.g. `pct=0.9` →
the value below which 90% of pixels fall). Combined with
`threshold_mask`, this gives a robust "keep the top 10% of gradient
responses" selector that doesn't depend on absolute pixel brightness.
"""
function percentile_threshold(img::AbstractMatrix{<:Real}, pct::Real)
    0 ≤ pct ≤ 1 || throw(ArgumentError("pct must be in [0,1], got $pct"))
    n = length(img)
    n == 0 && throw(ArgumentError("empty image"))
    k = clamp(round(Int, pct * n), 1, n)
    return Float64(partialsort(vec(img), k))
end

end # module Edges
