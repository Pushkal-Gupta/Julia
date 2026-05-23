"""
    Color

Color image processing. Everything else in the repo runs on a single
grayscale `Matrix{Float64}` — this submodule is what's needed to
work on the three-channel `(R, G, B)` tuples that `Photos` already
loads and the `RayTracer` already returns. Two kinds of operations
live here:

1. **Colour-space conversions.** RGB ↔ HSV (cylindrical, useful for
   tasks like hue-based segmentation), RGB ↔ YCbCr (linear,
   separates luminance from chrominance), plain RGB → luminance
   (the Rec. 709 weighted sum) for handing colour images back to
   the grayscale pipeline.
2. **Channel-wise operators.** `apply_per_channel(f, R, G, B)` runs
   a grayscale function `f` on each colour plane independently and
   reassembles. `color_gradient_magnitude(R, G, B)` computes Di
   Zenzo's vector-valued gradient magnitude — the right way to
   detect edges in a colour image, which is *not* the same as
   averaging the per-channel gradient magnitudes.

Conventions match the rest of the repo: every channel is a
`Matrix{Float64}` in `[0, 1]`, indexed `(row, col)`.
"""
module Color

using ..Kernels: sobel_x, sobel_y
using ..Convolution: correlate2d

export rgb_to_hsv, hsv_to_rgb,
       rgb_to_ycbcr, ycbcr_to_rgb,
       rgb_to_luminance,
       apply_per_channel,
       color_gradient_magnitude

# ── Helpers ──────────────────────────────────────────────────────────────────

# The Viz module also defines a single-pixel `hsv_to_rgb` — I deliberately
# re-implement the matrix version here so the conversion stays inside
# `Color` and the dependency direction is one-way (`Viz` doesn't have to
# know about colour-space conversions, and `Color` doesn't pull in `Viz`).

_clamp01(x::Real) = clamp(Float64(x), 0.0, 1.0)

function _check_same_size(R, G, B)
    size(R) == size(G) == size(B) || throw(DimensionMismatch(
        "channels have different sizes: R=$(size(R)), G=$(size(G)), B=$(size(B))"))
end

# ── RGB ↔ HSV ────────────────────────────────────────────────────────────────

"""
    rgb_to_hsv(R, G, B) -> (H, S, V)

Convert three RGB planes (each in `[0, 1]`) to HSV. `H ∈ [0, 1)`
with the wrap point at red, `S ∈ [0, 1]`, `V ∈ [0, 1]`. Pixels
with zero saturation get `H = 0` (the conventional choice — hue is
undefined for greys).
"""
function rgb_to_hsv(R::AbstractMatrix{<:Real},
                    G::AbstractMatrix{<:Real},
                    B::AbstractMatrix{<:Real})
    _check_same_size(R, G, B)
    H = zeros(Float64, size(R))
    S = zeros(Float64, size(R))
    V = zeros(Float64, size(R))
    @inbounds for k in eachindex(R)
        r = _clamp01(R[k]); g = _clamp01(G[k]); b = _clamp01(B[k])
        cmax = max(r, max(g, b))
        cmin = min(r, min(g, b))
        Δ    = cmax - cmin
        V[k] = cmax
        S[k] = cmax == 0 ? 0.0 : Δ / cmax
        if Δ == 0
            H[k] = 0.0
        elseif cmax == r
            H[k] = mod((g - b) / Δ, 6.0) / 6.0
        elseif cmax == g
            H[k] = ((b - r) / Δ + 2.0) / 6.0
        else                                     # cmax == b
            H[k] = ((r - g) / Δ + 4.0) / 6.0
        end
    end
    return (H, S, V)
end

"""
    hsv_to_rgb(H, S, V) -> (R, G, B)

Convert three HSV planes back to RGB. `H` is wrapped modulo 1, `S`
and `V` are clamped to `[0, 1]`. The output channels are in
`[0, 1]`.
"""
function hsv_to_rgb(H::AbstractMatrix{<:Real},
                    S::AbstractMatrix{<:Real},
                    V::AbstractMatrix{<:Real})
    _check_same_size(H, S, V)
    R = zeros(Float64, size(H))
    G = zeros(Float64, size(H))
    B = zeros(Float64, size(H))
    @inbounds for k in eachindex(H)
        h = mod(Float64(H[k]), 1.0)
        s = _clamp01(S[k])
        v = _clamp01(V[k])
        c = v * s
        x = c * (1 - abs(mod(h * 6, 2) - 1))
        m = v - c
        r, g, b = if h < 1/6
            (c, x, 0.0)
        elseif h < 2/6
            (x, c, 0.0)
        elseif h < 3/6
            (0.0, c, x)
        elseif h < 4/6
            (0.0, x, c)
        elseif h < 5/6
            (x, 0.0, c)
        else
            (c, 0.0, x)
        end
        R[k] = r + m
        G[k] = g + m
        B[k] = b + m
    end
    return (R, G, B)
end

# ── RGB ↔ YCbCr (ITU-R BT.601) ───────────────────────────────────────────────

# These are the BT.601 coefficients — same ones JPEG uses. The Y plane
# is the luminance the rest of the repo's grayscale code expects; Cb, Cr
# are chrominance differences shifted to `[0, 1]` (so 0.5 is the neutral
# "no colour" point).

"""
    rgb_to_ycbcr(R, G, B) -> (Y, Cb, Cr)

Linear RGB → YCbCr conversion using the ITU-R BT.601 coefficients.
`Y` is luminance in `[0, 1]`; `Cb` and `Cr` are chrominance
differences re-centred to `[0, 1]` so that `0.5` is "no colour".
The transform is exact (no gamma) for matched inputs and outputs in
`[0, 1]`.
"""
function rgb_to_ycbcr(R::AbstractMatrix{<:Real},
                      G::AbstractMatrix{<:Real},
                      B::AbstractMatrix{<:Real})
    _check_same_size(R, G, B)
    Y  = 0.299 .* Float64.(R) .+ 0.587 .* Float64.(G) .+ 0.114 .* Float64.(B)
    Cb = 0.5 .+ (-0.168736 .* Float64.(R) .- 0.331264 .* Float64.(G) .+ 0.5 .* Float64.(B))
    Cr = 0.5 .+ ( 0.5      .* Float64.(R) .- 0.418688 .* Float64.(G) .- 0.081312 .* Float64.(B))
    return (Y, Cb, Cr)
end

"""
    ycbcr_to_rgb(Y, Cb, Cr) -> (R, G, B)

Inverse of `rgb_to_ycbcr`. Output channels are clamped to `[0, 1]`.
"""
function ycbcr_to_rgb(Y::AbstractMatrix{<:Real},
                      Cb::AbstractMatrix{<:Real},
                      Cr::AbstractMatrix{<:Real})
    _check_same_size(Y, Cb, Cr)
    Yf  = Float64.(Y)
    Cbf = Float64.(Cb) .- 0.5
    Crf = Float64.(Cr) .- 0.5
    R = clamp.(Yf .+ 1.402    .* Crf,                          0.0, 1.0)
    G = clamp.(Yf .- 0.344136 .* Cbf .- 0.714136 .* Crf,       0.0, 1.0)
    B = clamp.(Yf .+ 1.772    .* Cbf,                          0.0, 1.0)
    return (R, G, B)
end

# ── Luminance ────────────────────────────────────────────────────────────────

"""
    rgb_to_luminance(R, G, B) -> Matrix{Float64}

Rec. 709 weighted luminance: `0.2126·R + 0.7152·G + 0.0722·B`. This
is the "right" way to collapse a colour image to grayscale when the
RGB values are linear; BT.601's `0.299/0.587/0.114` weights are
correct for gamma-encoded video. Both show up in the literature —
I default to Rec. 709 because the rest of the repo treats values as
linear `[0, 1]`.
"""
function rgb_to_luminance(R::AbstractMatrix{<:Real},
                          G::AbstractMatrix{<:Real},
                          B::AbstractMatrix{<:Real})
    _check_same_size(R, G, B)
    return 0.2126 .* Float64.(R) .+ 0.7152 .* Float64.(G) .+ 0.0722 .* Float64.(B)
end

# ── Channel-wise apply ───────────────────────────────────────────────────────

"""
    apply_per_channel(f, R, G, B; kwargs...) -> (R', G', B')

Apply `f` independently to each colour plane. `kwargs` are forwarded
to `f`. This is the boring-but-useful glue that lets `f` be
`correlate2d(_, gaussian(7); ...)` or `Filters.median_filter` or any
of the grayscale operators in the rest of the repo. Each channel
goes through the same function with the same parameters; the
caller decides whether that's the right thing (it usually is for
linear filters; usually isn't for things like histogram
equalization where per-channel processing tints the result).
"""
function apply_per_channel(f, R::AbstractMatrix{<:Real},
                              G::AbstractMatrix{<:Real},
                              B::AbstractMatrix{<:Real}; kwargs...)
    _check_same_size(R, G, B)
    Rp = f(R; kwargs...)
    Gp = f(G; kwargs...)
    Bp = f(B; kwargs...)
    return (Rp, Gp, Bp)
end

# ── Di Zenzo's vector-valued colour gradient ─────────────────────────────────

"""
    color_gradient_magnitude(R, G, B; pad = :replicate) -> Matrix{Float64}

The right way to detect edges in a colour image. Naïvely averaging
the three per-channel gradient magnitudes loses sign information —
two channels with opposite-sign edges would cancel in a vector
average, but their magnitudes both reinforce the colour edge. Di
Zenzo's 1986 formulation builds the 2×2 structure tensor of the
*colour gradient vectors* and takes the square root of its largest
eigenvalue:

```
J = [ Σ_c (∂R_c/∂x)²        Σ_c (∂R_c/∂x)·(∂R_c/∂y) ]
    [ Σ_c (∂R_c/∂x)·(∂R_c/∂y)  Σ_c (∂R_c/∂y)²       ]

|∇I|_color = √(λ_max(J))
```

`Σ_c` is the sum across colour channels. For greyscale input
(`R == G == B`) this reduces to the standard scalar gradient
magnitude.
"""
function color_gradient_magnitude(R::AbstractMatrix{<:Real},
                                  G::AbstractMatrix{<:Real},
                                  B::AbstractMatrix{<:Real};
                                  pad::Symbol = :replicate)
    _check_same_size(R, G, B)
    sx = sobel_x()
    sy = sobel_y()
    # Divide by 8 to normalize Sobel into a true gradient estimate, same
    # convention as Flow.lucas_kanade. The 1/8 cancels inside the
    # eigenvalue formula but keeping it makes the output magnitudes
    # comparable to single-channel gradient magnitudes.
    Rx = correlate2d(R, sx; pad = pad) ./ 8
    Ry = correlate2d(R, sy; pad = pad) ./ 8
    Gx = correlate2d(G, sx; pad = pad) ./ 8
    Gy = correlate2d(G, sy; pad = pad) ./ 8
    Bx = correlate2d(B, sx; pad = pad) ./ 8
    By = correlate2d(B, sy; pad = pad) ./ 8

    Jxx = Rx .^ 2 .+ Gx .^ 2 .+ Bx .^ 2
    Jyy = Ry .^ 2 .+ Gy .^ 2 .+ By .^ 2
    Jxy = Rx .* Ry .+ Gx .* Gy .+ Bx .* By

    # Largest eigenvalue of a symmetric 2×2 matrix [a b; b d]:
    # (a + d)/2 + √(((a − d)/2)² + b²)
    tr_half  = 0.5 .* (Jxx .+ Jyy)
    delta    = sqrt.(((Jxx .- Jyy) ./ 2) .^ 2 .+ Jxy .^ 2)
    lambda_max = tr_half .+ delta
    return sqrt.(max.(lambda_max, 0.0))
end

end # module Color
