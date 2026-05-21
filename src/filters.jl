"""
    Filters

Non-linear and edge-preserving spatial filters. The linear ones (box,
Gaussian) live in `Kernels` because they're just kernels you convolve
with. The ones here can't be expressed as a single convolution:

- `median_filter` is order-statistic-based (rank filter).
- `bilateral_filter` is data-dependent — weights change per pixel.
- `binary_dilate` is a morphological op on a `BitMatrix`.

I'm grouping them together because they all preprocess an image
without being linear convolutions, and that's a useful conceptual
distinction.
"""
module Filters

using ..Padding

export median_filter, bilateral_filter, binary_dilate, perona_malik

"""
    median_filter(img; window=3, pad=:replicate) -> Matrix{Float64}

Replace each pixel with the median of its `window × window`
neighborhood. Non-linear; the magic property is that a single outlier
pixel can't pull the output past the median, so salt-and-pepper noise
gets crushed without smoothing real edges the way a Gaussian would.

`window` must be odd. The internal buffer is small (window²) and
allocated once.
"""
function median_filter(img::AbstractMatrix{<:Real}; window::Integer = 3,
                       pad::Symbol = :replicate)
    isodd(window) || throw(ArgumentError("window must be odd, got $window"))
    p = window ÷ 2
    padded = Padding.pad_array(Float64.(img), (p, p, p, p); mode = pad)
    H, W = size(img)
    out = Matrix{Float64}(undef, H, W)
    buf = Vector{Float64}(undef, window * window)
    mid = (window * window + 1) ÷ 2   # exact median index for odd n
    @inbounds for j in 1:W
        for i in 1:H
            k = 0
            for jj in 1:window, ii in 1:window
                k += 1
                buf[k] = padded[i + ii - 1, j + jj - 1]
            end
            partialsort!(buf, mid)
            out[i, j] = buf[mid]
        end
    end
    return out
end

"""
    bilateral_filter(img; window=5, sigma_spatial=1.5, sigma_intensity=0.1, pad=:replicate)
        -> Matrix{Float64}

Edge-preserving smoothing. Output is a weighted average over the
window where the weight for a neighbor at offset `(di, dj)` with
intensity `v` is

    exp(-(di² + dj²) / 2σ_spatial²) · exp(-(v - center)² / 2σ_intensity²)

Two Gaussians: one on position, one on value. Neighbors that are
close *and* similar in intensity get high weight; neighbors across an
edge get suppressed by the intensity Gaussian even if they're spatially
close.

Tuning:
- Large `sigma_intensity` → behaves like a normal Gaussian blur.
- Small `sigma_intensity` → edges survive but smoothing is weak.
- I usually leave `sigma_spatial` around the window's half-width and
  pick `sigma_intensity` by what counts as a "real" intensity change
  in the image (≈ 5–10% of the dynamic range, for normalized inputs).
"""
function bilateral_filter(img::AbstractMatrix{<:Real};
                          window::Integer = 5,
                          sigma_spatial::Real = 1.5,
                          sigma_intensity::Real = 0.1,
                          pad::Symbol = :replicate)
    isodd(window) || throw(ArgumentError("window must be odd, got $window"))
    sigma_spatial > 0 && sigma_intensity > 0 || throw(ArgumentError(
        "both sigmas must be positive"))
    p = window ÷ 2
    padded = Padding.pad_array(Float64.(img), (p, p, p, p); mode = pad)
    H, W = size(img)
    out = Matrix{Float64}(undef, H, W)

    # Precompute the spatial weights — they don't depend on pixel data.
    sw = Matrix{Float64}(undef, window, window)
    s2_sp = 2 * sigma_spatial^2
    for jj in 1:window, ii in 1:window
        di = ii - p - 1
        dj = jj - p - 1
        sw[ii, jj] = exp(-(di^2 + dj^2) / s2_sp)
    end
    s2_int = 2 * sigma_intensity^2

    @inbounds for j in 1:W
        for i in 1:H
            center = padded[i + p, j + p]
            acc = 0.0
            wsum = 0.0
            for jj in 1:window, ii in 1:window
                v = padded[i + ii - 1, j + jj - 1]
                iw = exp(-(v - center)^2 / s2_int)
                w = sw[ii, jj] * iw
                acc += w * v
                wsum += w
            end
            out[i, j] = acc / wsum
        end
    end
    return out
end

"""
    binary_dilate(mask::BitMatrix; radius=1) -> BitMatrix

8-connected morphological dilation. Each iteration expands the mask
outward by one pixel; `radius` iterations expand it by `radius`.

Useful for the tolerance-based edge-matching in `Metrics.edge_match_stats`:
"a predicted edge counts as correct if a ground-truth pixel is within
`radius` pixels of it" is the same as "the predicted mask intersects
the dilated ground truth".
"""
function binary_dilate(mask::BitMatrix; radius::Integer = 1)
    radius ≥ 0 || throw(ArgumentError("radius must be ≥ 0, got $radius"))
    radius == 0 && return copy(mask)
    H, W = size(mask)
    cur = copy(mask)
    nxt = similar(cur)
    for _ in 1:radius
        copyto!(nxt, cur)
        @inbounds for j in 1:W, i in 1:H
            cur[i, j] && continue
            for dj in -1:1, di in -1:1
                (di == 0 && dj == 0) && continue
                ni, nj = i + di, j + dj
                (1 ≤ ni ≤ H && 1 ≤ nj ≤ W) || continue
                if cur[ni, nj]
                    nxt[i, j] = true
                    break
                end
            end
        end
        cur, nxt = nxt, cur
    end
    return cur
end

"""
    perona_malik(img; iterations=20, K=0.10, lambda=0.20, mode=:exponential)
        -> Matrix{Float64}

Anisotropic diffusion — a non-linear PDE-based edge-preserving
smoother. Linear heat diffusion `∂I/∂t = ∇²I` is equivalent to
Gaussian smoothing (the solution at time `t` is the image convolved
with a Gaussian of σ = √(2t)). The trouble with linear diffusion is
that the Laplacian is largest exactly at edges, so edges blur
*fastest*. Perona and Malik fixed this by making the conductivity
depend on the local gradient:

    ∂I/∂t = div( c(|∇I|) · ∇I )

where `c(·)` is small where `|∇I|` is large. So smooth regions
diffuse fast; edges almost don't.

Discretization (explicit Euler on a 4-neighbor stencil):

    I_{t+1}[i, j] = I_t[i, j] + λ · Σ_{N, S, E, W} c(|Δ_d|) · Δ_d

where `Δ_d` is the directional difference (neighbor minus center).
Stability of the explicit scheme requires `λ ≤ 1/4` (von Neumann
analysis on the 4-neighbor discrete Laplacian).

Parameters:

- `iterations` — number of time steps. More = stronger smoothing.
- `K` — the conduction "scale" in the diffusivity. Gradients much
  smaller than `K` diffuse fully; gradients much larger than `K`
  diffuse barely at all. Tune to the noise level (a bit above) vs.
  the signal contrast (well below). For images in [0, 1] try
  `K ≈ 0.05–0.20`.
- `lambda` — the explicit time step. Stays ≤ 0.25 for stability.
- `mode` — `:exponential` for `c(s) = exp(-(s/K)²)` (Perona & Malik's
  first proposal — slightly favors strong edges) or `:rational` for
  `c(s) = 1 / (1 + (s/K)²)` (their second — slightly favors wide
  smooth regions).

Boundary conditions: Neumann (zero-flux). Each pixel's missing
out-of-image neighbor contributes zero to the update. This is the
natural BC for diffusion, and corresponds to assuming the image
extends with constant values past its edges.
"""
function perona_malik(img::AbstractMatrix{<:Real};
                      iterations::Integer = 20,
                      K::Real = 0.10,
                      lambda::Real = 0.20,
                      mode::Symbol = :exponential)
    iterations ≥ 0 || throw(ArgumentError("iterations must be ≥ 0, got $iterations"))
    K > 0 || throw(ArgumentError("K must be positive, got $K"))
    lambda ≤ 0.25 || throw(ArgumentError(
        "lambda must be ≤ 0.25 for explicit-scheme stability, got $lambda"))
    mode in (:exponential, :rational) || throw(ArgumentError(
        "mode must be :exponential or :rational, got :$mode"))

    out = Float64.(img)
    H, W = size(out)
    nxt = similar(out)
    K2 = K^2

    @inbounds for _ in 1:iterations
        for j in 1:W, i in 1:H
            # Directional differences (Neumann BC: 0 at the image edge).
            gN = (i > 1) ? out[i - 1, j] - out[i, j] : 0.0
            gS = (i < H) ? out[i + 1, j] - out[i, j] : 0.0
            gW = (j > 1) ? out[i, j - 1] - out[i, j] : 0.0
            gE = (j < W) ? out[i, j + 1] - out[i, j] : 0.0
            # Conductivity per direction.
            if mode === :exponential
                cN = exp(-gN^2 / K2); cS = exp(-gS^2 / K2)
                cW = exp(-gW^2 / K2); cE = exp(-gE^2 / K2)
            else  # :rational
                cN = 1.0 / (1.0 + gN^2 / K2); cS = 1.0 / (1.0 + gS^2 / K2)
                cW = 1.0 / (1.0 + gW^2 / K2); cE = 1.0 / (1.0 + gE^2 / K2)
            end
            nxt[i, j] = out[i, j] + lambda * (cN * gN + cS * gS + cW * gW + cE * gE)
        end
        out, nxt = nxt, out
    end
    return out
end

end # module Filters
