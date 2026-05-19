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

export median_filter, bilateral_filter, binary_dilate

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

end # module Filters
