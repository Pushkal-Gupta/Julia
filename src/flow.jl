"""
    Flow

Optical flow — given two consecutive frames, estimate the per-pixel
2D motion vector `(u, v)` that maps frame 1 onto frame 2. `u` is the
horizontal displacement (positive = right), `v` is the vertical
displacement (positive = down).

The algorithm here is Lucas-Kanade (1981). It rests on two
assumptions:

1. *Brightness constancy* — a pixel keeps its intensity as it moves:
   `I(x, y, t) = I(x + u, y + v, t + 1)`. Taylor-expand and you get the
   per-pixel constraint `Iₓ · u + I_y · v + I_t = 0`. One equation,
   two unknowns — underdetermined.
2. *Spatial coherence* — flow is roughly constant in a small window
   around each pixel. So the constraint above holds for every pixel
   in the window with the *same* `(u, v)`, giving an overdetermined
   linear system to solve in the least-squares sense.

The solution to that least-squares system is the 2×2 normal equations:

```
[ Σ Iₓ²    Σ Iₓ·I_y ] [u]   [-Σ Iₓ·I_t]
[ Σ Iₓ·I_y Σ I_y²   ] [v] = [-Σ I_y·I_t]
```

Each `Σ` is a window sum, computed here with a separable Gaussian
window (so the algorithm shares its inner loop with the rest of the
repo). The 2×2 matrix on the left is the structure tensor — the same
one Harris uses for corners, which is not a coincidence: LK works
*best* on textured regions where Harris fires too, and fails on flat
regions where there's no constraint to satisfy.

Output is a `FlowField` with `u`, `v`, and a `confidence` matrix
(the structure-tensor determinant) so callers can mask out pixels
where the solution wasn't well-conditioned.

What's deliberately not in this submodule:

- *Hierarchical / pyramidal LK*. Plain LK only sees small
  displacements (≤ ~1 pixel per frame for a 15×15 window). Real
  motion needs the Gaussian-pyramid coarse-to-fine variant. The
  `Pyramids` module is already there; the pyramidal LK fits as a
  follow-up chunk.
- *Iterative refinement* (warping frame 2 by the current estimate and
  re-solving). One linear solve per pixel is enough for the
  small-displacement case I'm demonstrating.
- *Robust losses* (Charbonnier, Lorentzian). MSE / Gaussian-weighted
  is the textbook starting point.
"""
module Flow

using ..Padding
using ..Kernels: gaussian1d, sobel_x, sobel_y
using ..Convolution: correlate2d, separable_correlate2d
using ..Pyramids: gaussian_pyramid

export FlowField, lucas_kanade, lucas_kanade_pyramid,
       horn_schunck, horn_schunck_pyramid,
       warp_bilinear, flow_magnitude, flow_angle

"""
    FlowField

The output of `lucas_kanade`. `u` and `v` are the per-pixel
displacement components in image coordinates (`u` = column shift,
`v` = row shift); `confidence` is the determinant of the per-pixel
structure tensor, which doubles as a no-texture mask (low values
mean the linear system was nearly singular).
"""
struct FlowField{T<:Real}
    u::Matrix{T}
    v::Matrix{T}
    confidence::Matrix{T}
end

Base.size(f::FlowField) = size(f.u)

"""
    lucas_kanade(img1, img2; window_size = 15, sigma = 2.0,
                 pad = :replicate, det_threshold = 1e-6) -> FlowField

Estimate the per-pixel optical flow from `img1` to `img2`. Both
images must be the same size and indexed `(row, col)` with values
in `[0, 1]`.

`window_size` and `sigma` set the Gaussian window used to aggregate
the structure tensor. A larger window is more robust to noise but
blurs across motion boundaries. `det_threshold` is the minimum
determinant of the structure tensor for a pixel to be considered
solvable — below it, `u` and `v` are zeroed out and `confidence`
records the actual determinant.
"""
function lucas_kanade(img1::AbstractMatrix{<:Real},
                      img2::AbstractMatrix{<:Real};
                      window_size::Integer = 15,
                      sigma::Real = 2.0,
                      pad::Symbol = :replicate,
                      det_threshold::Real = 1e-6)
    size(img1) == size(img2) || throw(DimensionMismatch(
        "frames must be the same size; got $(size(img1)) and $(size(img2))"))
    isodd(window_size) || throw(ArgumentError(
        "window_size must be odd, got $window_size"))

    F1 = Float64.(img1)
    F2 = Float64.(img2)

    # Use the average frame for spatial gradients — symmetric in time,
    # noticeably more stable than picking one or the other. The factor of
    # 1/8 normalizes Sobel into a true gradient estimate: the kernel sums
    # to 8 across the positive entries, and that scale matters here
    # because `Ix` and `It` show up at different powers in the LK normal
    # equations (Σ Ix² vs Σ Ix·It), so they don't cancel out — without
    # it, LK recovers `u ≈ 0.125` for a 1-pixel shift instead of `1`.
    Iavg = 0.5 .* (F1 .+ F2)
    Ix = correlate2d(Iavg, sobel_x(); pad = pad) ./ 8
    Iy = correlate2d(Iavg, sobel_y(); pad = pad) ./ 8
    It = F2 .- F1

    # Pointwise products go into the structure-tensor entries.
    Ixx = Ix .* Ix
    Iyy = Iy .* Iy
    Ixy = Ix .* Iy
    Ixt = Ix .* It
    Iyt = Iy .* It

    # Gaussian-windowed sums (separable so the cost is linear in window_size).
    g = gaussian1d(window_size; sigma = sigma)
    Sxx = separable_correlate2d(Ixx, g, g; pad = pad)
    Syy = separable_correlate2d(Iyy, g, g; pad = pad)
    Sxy = separable_correlate2d(Ixy, g, g; pad = pad)
    Sxt = separable_correlate2d(Ixt, g, g; pad = pad)
    Syt = separable_correlate2d(Iyt, g, g; pad = pad)

    H, W = size(F1)
    u = zeros(Float64, H, W)
    v = zeros(Float64, H, W)
    conf = zeros(Float64, H, W)

    @inbounds for j in 1:W, i in 1:H
        a = Sxx[i, j]
        b = Sxy[i, j]
        d = Syy[i, j]
        det = a * d - b * b
        conf[i, j] = det
        if det > det_threshold
            # Cramer's rule on the 2×2 system, with the right-hand side
            # already negated to follow the LK sign convention.
            rx = -Sxt[i, j]
            ry = -Syt[i, j]
            u[i, j] = (d * rx - b * ry) / det
            v[i, j] = (a * ry - b * rx) / det
        end
    end

    return FlowField(u, v, conf)
end

"""
    flow_magnitude(flow::FlowField) -> Matrix{Float64}

The per-pixel Euclidean norm `√(u² + v²)`. Useful as the "value"
channel in a flow visualization or as a textureless-region mask
threshold.
"""
flow_magnitude(f::FlowField) = sqrt.(f.u .^ 2 .+ f.v .^ 2)

"""
    flow_angle(flow::FlowField) -> Matrix{Float64}

Per-pixel flow direction in radians, `atan(v, u)` ∈ `(-π, π]`. Maps
naturally to the hue channel of a color visualization.
"""
flow_angle(f::FlowField) = atan.(f.v, f.u)

# ── Bilinear warping ─────────────────────────────────────────────────────────

"""
    warp_bilinear(img, u, v) -> Matrix{Float64}

Inverse-warp `img` by the per-pixel flow field `(u, v)`. For each
output pixel `(i, j)`, sample the source at `(i + v[i,j], j + u[i,j])`
with bilinear interpolation. Out-of-bounds samples are clamped to the
image boundary (replicate padding).

The convention matches `lucas_kanade`: if `lucas_kanade(I₁, I₂)`
returns flow `(u, v)`, then `warp_bilinear(I₂, u, v)` should look
approximately like `I₁`. That's the property the pyramidal LK
iteration leans on — each pyramid level warps the next frame back
toward the current frame so the residual motion is small enough for
the OFC linearization to be valid.
"""
function warp_bilinear(img::AbstractMatrix{<:Real},
                       u::AbstractMatrix{<:Real},
                       v::AbstractMatrix{<:Real})
    size(u) == size(v) == size(img) || throw(DimensionMismatch(
        "img/u/v sizes differ: $(size(img)), $(size(u)), $(size(v))"))
    H, W = size(img)
    F = Float64.(img)
    out = zeros(Float64, H, W)
    @inbounds for j in 1:W, i in 1:H
        si = i + v[i, j]
        sj = j + u[i, j]
        i0 = floor(Int, si); j0 = floor(Int, sj)
        ai = si - i0;        aj = sj - j0
        i0c = clamp(i0,     1, H); i1c = clamp(i0 + 1, 1, H)
        j0c = clamp(j0,     1, W); j1c = clamp(j0 + 1, 1, W)
        v00 = F[i0c, j0c]; v01 = F[i0c, j1c]
        v10 = F[i1c, j0c]; v11 = F[i1c, j1c]
        out[i, j] = (1 - ai) * (1 - aj) * v00 + (1 - ai) * aj * v01 +
                    ai       * (1 - aj) * v10 + ai       * aj * v11
    end
    return out
end

# Internal: bilinear 2× upsample of a flow field to a given target size.
# The (u, v) magnitudes are NOT doubled here — that's the caller's job
# (1 pixel at a coarse level is 2 pixels at the next finer level).
function _upsample_field(field::AbstractMatrix{<:Real},
                         target_H::Integer, target_W::Integer)
    H, W = size(field)
    out = zeros(Float64, target_H, target_W)
    # Map output (i, j) back to source coordinates such that the
    # corners line up (i=1 ↔ si=1, i=target_H ↔ si=H).
    @inbounds for j in 1:target_W, i in 1:target_H
        si = target_H > 1 ? 1 + (i - 1) * (H - 1) / (target_H - 1) : 1.0
        sj = target_W > 1 ? 1 + (j - 1) * (W - 1) / (target_W - 1) : 1.0
        i0 = floor(Int, si); j0 = floor(Int, sj)
        ai = si - i0;        aj = sj - j0
        i0c = clamp(i0,     1, H); i1c = clamp(i0 + 1, 1, H)
        j0c = clamp(j0,     1, W); j1c = clamp(j0 + 1, 1, W)
        v00 = field[i0c, j0c]; v01 = field[i0c, j1c]
        v10 = field[i1c, j0c]; v11 = field[i1c, j1c]
        out[i, j] = (1 - ai) * (1 - aj) * v00 + (1 - ai) * aj * v01 +
                    ai       * (1 - aj) * v10 + ai       * aj * v11
    end
    return out
end

# ── Pyramidal Lucas-Kanade ───────────────────────────────────────────────────

"""
    lucas_kanade_pyramid(img1, img2;
                         levels = 4, window_size = 15, sigma = 2.0,
                         iters_per_level = 3, pad = :replicate,
                         det_threshold = 1e-6) -> FlowField

Coarse-to-fine Lucas-Kanade. The fix for the small-motion limit of
plain `lucas_kanade`: build Gaussian pyramids of both frames, run LK
at the coarsest level (where a big motion looks small), upsample the
estimate to the next level, warp the second frame by the current
estimate, run LK on the residual, repeat. At each level the
algorithm only needs to recover sub-pixel residual motion, so the
OFC linearization stays valid even when the underlying motion is
several pixels.

`levels` is the number of additional pyramid levels above the original
(so the coarsest level has roughly `H / 2^levels × W / 2^levels`
pixels). `iters_per_level` is the number of Newton-style refinement
sweeps within each level — each one warps `img2` by the current
estimate and adds the residual flow.

Returns a `FlowField` at the original image resolution.
"""
function lucas_kanade_pyramid(img1::AbstractMatrix{<:Real},
                              img2::AbstractMatrix{<:Real};
                              levels::Integer = 4,
                              window_size::Integer = 15,
                              sigma::Real = 2.0,
                              iters_per_level::Integer = 3,
                              pad::Symbol = :replicate,
                              det_threshold::Real = 1e-6)
    size(img1) == size(img2) || throw(DimensionMismatch(
        "frames must be the same size; got $(size(img1)) and $(size(img2))"))
    levels ≥ 0 || throw(ArgumentError("levels must be non-negative"))
    iters_per_level ≥ 1 ||
        throw(ArgumentError("iters_per_level must be ≥ 1"))

    F1 = Float64.(img1)
    F2 = Float64.(img2)

    pyr1 = gaussian_pyramid(F1; levels = levels)
    pyr2 = gaussian_pyramid(F2; levels = levels)
    L = length(pyr1)   # = levels + 1

    # Start with zero flow at the coarsest level.
    coarse_H, coarse_W = size(pyr1[L])
    u = zeros(Float64, coarse_H, coarse_W)
    v = zeros(Float64, coarse_H, coarse_W)
    conf = zeros(Float64, coarse_H, coarse_W)

    for k in L:-1:1
        I1 = pyr1[k]
        I2 = pyr2[k]
        # Iterate within this level: warp by current estimate, solve for residual.
        for _ in 1:iters_per_level
            warped = warp_bilinear(I2, u, v)
            residual = lucas_kanade(I1, warped;
                                    window_size = window_size,
                                    sigma = sigma,
                                    pad = pad,
                                    det_threshold = det_threshold)
            u .+= residual.u
            v .+= residual.v
            conf = residual.confidence
        end
        # Upsample to the next finer level, with the magnitude doubled
        # because 1 pixel of motion at level k is 2 pixels at level k-1.
        if k > 1
            next_H, next_W = size(pyr1[k - 1])
            u = 2 .* _upsample_field(u, next_H, next_W)
            v = 2 .* _upsample_field(v, next_H, next_W)
            conf = _upsample_field(conf, next_H, next_W)
        end
    end

    return FlowField(u, v, conf)
end

# ── Horn-Schunck (dense, global-smoothness flow) ────────────────────────────

# Horn & Schunck's weighted-Laplacian 3×3 kernel. Centre weight is zero
# (the formula uses the *average of the neighbours* as its estimate of the
# pixel's local mean), 4-neighbours are 1/6, diagonals are 1/12. Sums to 1.
const _HS_AVG_KERNEL = [1/12  1/6  1/12;
                        1/6   0.0  1/6;
                        1/12  1/6  1/12]

"""
    horn_schunck(img1, img2;
                 alpha = 0.1, iterations = 200, pad = :replicate)
        -> FlowField

Dense optical flow via Horn-Schunck's 1981 variational formulation.
Minimizes

```
∫∫ (Iₓ·u + I_y·v + I_t)² + α² · (|∇u|² + |∇v|²)  dA
```

— the OFC squared (data term) plus a smoothness prior that penalises
gradients of `u` and `v`. The Euler-Lagrange equations are a pair of
coupled PDEs; the standard solver is Jacobi iteration on

```
u ← ū − Iₓ · (Iₓ·ū + I_y·v̄ + I_t) / (α² + Iₓ² + I_y²)
v ← v̄ − I_y · (Iₓ·ū + I_y·v̄ + I_t) / (α² + Iₓ² + I_y²)
```

where `ū` and `v̄` are the Horn-Schunck weighted-Laplacian local
averages (4-neighbours at 1/6, diagonals at 1/12). `α` is the
smoothness weight: small `α` lets the data term dominate and the
result looks like noisy LK; large `α` smooths the flow heavily and
fills in textureless regions by diffusion. The right `α` depends
on the image intensity scale — for inputs in `[0, 1]` something in
`0.05 ≤ α ≤ 0.5` is usually about right; for inputs in `[0, 255]`
the classical recipe uses `5 ≤ α ≤ 15`. The default `0.1` matches
the `[0, 1]` convention the rest of the repo uses.

The big practical difference from LK: HS produces a *dense* flow
field, with sensible estimates even in regions that have no
texture. LK in the same region returns zero because its 2×2 system
is rank-deficient.

The `confidence` field on the returned `FlowField` is the squared
OFC residual `(Iₓ·u + I_y·v + I_t)²` — small where the flow fits
the data, large where it doesn't. That's a different quantity than
LK's structure-tensor determinant, but it serves the same role of
flagging unreliable pixels.
"""
function horn_schunck(img1::AbstractMatrix{<:Real},
                      img2::AbstractMatrix{<:Real};
                      alpha::Real = 0.1,
                      iterations::Integer = 200,
                      pad::Symbol = :replicate)
    size(img1) == size(img2) || throw(DimensionMismatch(
        "frames must be the same size; got $(size(img1)) and $(size(img2))"))
    iterations ≥ 1 || throw(ArgumentError("iterations must be ≥ 1"))
    alpha > 0    || throw(ArgumentError("alpha must be positive"))

    F1 = Float64.(img1)
    F2 = Float64.(img2)

    # Same gradient recipe as Lucas-Kanade: Sobel/8 on the frame average
    # for the spatial derivatives, plain forward difference for the
    # temporal one. Sharing the recipe keeps the two algorithms
    # comparable on the same inputs.
    Iavg = 0.5 .* (F1 .+ F2)
    Ix = correlate2d(Iavg, sobel_x(); pad = pad) ./ 8
    Iy = correlate2d(Iavg, sobel_y(); pad = pad) ./ 8
    It = F2 .- F1

    # Per-pixel denominator is constant across iterations.
    denom = alpha^2 .+ Ix .^ 2 .+ Iy .^ 2

    H, W = size(F1)
    u = zeros(Float64, H, W)
    v = zeros(Float64, H, W)

    # Jacobi sweep. The literature also discusses Gauss-Seidel (in-place
    # updates, converges ~2× faster) — I'm keeping it Jacobi for clarity.
    for _ in 1:iterations
        ū = correlate2d(u, _HS_AVG_KERNEL; pad = pad)
        v̄ = correlate2d(v, _HS_AVG_KERNEL; pad = pad)
        # Common numerator across u, v.
        num = Ix .* ū .+ Iy .* v̄ .+ It
        correction = num ./ denom
        u = ū .- Ix .* correction
        v = v̄ .- Iy .* correction
    end

    # OFC residual per pixel — small where the flow agrees with the data.
    residual = (Ix .* u .+ Iy .* v .+ It) .^ 2
    return FlowField(u, v, residual)
end

"""
    horn_schunck_pyramid(img1, img2;
                         levels = 4, alpha = 0.1,
                         iters_per_level = 100, pad = :replicate)
        -> FlowField

The coarse-to-fine driver from `lucas_kanade_pyramid`, wrapped around
`horn_schunck` instead of `lucas_kanade`. The driver itself is
algorithm-agnostic — for each pyramid level it warps frame 2 by the
current estimate and asks the inner flow algorithm to solve for the
*residual*, then accumulates. The only thing that changes from the
LK pyramid is the inner solver and the cost per level.

This buys HS the same big-motion robustness pyramidal LK has, at the
same multiplicative cost (≈ `(levels + 1) × iters_per_level` Jacobi
sweeps total). It also lets HS run with fewer iterations *per* level
than the single-shot version needs — the residual at each level is
small, so the Jacobi sweep converges faster.

`iters_per_level` defaults to 100 here, less than `horn_schunck`'s
default of 200, because each level only has to handle the residual
motion not the full motion.
"""
function horn_schunck_pyramid(img1::AbstractMatrix{<:Real},
                              img2::AbstractMatrix{<:Real};
                              levels::Integer = 4,
                              alpha::Real = 0.1,
                              iters_per_level::Integer = 100,
                              pad::Symbol = :replicate)
    size(img1) == size(img2) || throw(DimensionMismatch(
        "frames must be the same size; got $(size(img1)) and $(size(img2))"))
    levels ≥ 0           || throw(ArgumentError("levels must be non-negative"))
    iters_per_level ≥ 1  || throw(ArgumentError("iters_per_level must be ≥ 1"))
    alpha > 0            || throw(ArgumentError("alpha must be positive"))

    F1 = Float64.(img1)
    F2 = Float64.(img2)

    pyr1 = gaussian_pyramid(F1; levels = levels)
    pyr2 = gaussian_pyramid(F2; levels = levels)
    L = length(pyr1)

    coarse_H, coarse_W = size(pyr1[L])
    u = zeros(Float64, coarse_H, coarse_W)
    v = zeros(Float64, coarse_H, coarse_W)
    conf = zeros(Float64, coarse_H, coarse_W)

    for k in L:-1:1
        I1 = pyr1[k]
        I2 = pyr2[k]
        warped = warp_bilinear(I2, u, v)
        residual = horn_schunck(I1, warped;
                                alpha = alpha,
                                iterations = iters_per_level,
                                pad = pad)
        u .+= residual.u
        v .+= residual.v
        conf = residual.confidence
        if k > 1
            next_H, next_W = size(pyr1[k - 1])
            u = 2 .* _upsample_field(u, next_H, next_W)
            v = 2 .* _upsample_field(v, next_H, next_W)
            conf = _upsample_field(conf, next_H, next_W)
        end
    end

    return FlowField(u, v, conf)
end

end # module Flow
