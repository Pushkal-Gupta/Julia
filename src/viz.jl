"""
    Viz

Small visualization helpers. Nothing fancy — just the operations the
example scripts kept reinventing inline. Interactive tools and heatmaps
will land here later.
"""
module Viz

export normalize01, montage, signed_to_gray,
       draw_line!, mark_points!, label_to_gray,
       flow_to_rgb
# The single-pixel HSV → RGB helper used by `flow_to_rgb` is renamed
# `_hsv_pixel_to_rgb` and not exported. The matrix-level version that
# users want is `Color.hsv_to_rgb`.

"""
    normalize01(x) -> Matrix{Float64}

Affine-rescale `x` into `[0, 1]` based on its actual min/max. Returns an
all-zero matrix if the input is constant. Used to make signed gradient
images (which can go negative) inspectable as PGMs.
"""
function normalize01(x::AbstractMatrix{<:Real})
    lo, hi = extrema(x)
    hi == lo && return zeros(Float64, size(x))
    return (Float64.(x) .- lo) ./ (hi - lo)
end

"""
    signed_to_gray(x) -> Matrix{Float64}

Map a signed image into `[0, 1]` with zero pinned to mid-gray (0.5). Better
than `normalize01` for displaying derivative outputs where the *sign* of the
response is meaningful — bright = positive, dark = negative, mid-gray = no
edge.
"""
function signed_to_gray(x::AbstractMatrix{<:Real})
    m = maximum(abs, x)
    m == 0 && return fill(0.5, size(x))
    return 0.5 .+ 0.5 .* Float64.(x) ./ m
end

"""
    montage(images; cols=3, gap=2, background=0.5) -> Matrix

Tile a vector of equally-sized 2D images into a grid for one-glance
comparison. `gap` pixels of `background` fill separate the cells.
"""
function montage(images::AbstractVector{<:AbstractMatrix{<:Real}};
                 cols::Integer = 3,
                 gap::Integer = 2,
                 background::Real = 0.5)
    isempty(images) && throw(ArgumentError("nothing to tile"))
    h, w = size(first(images))
    all(size(img) == (h, w) for img in images) ||
        throw(DimensionMismatch("all images must share size $((h, w))"))
    cols ≥ 1 || throw(ArgumentError("cols must be ≥ 1"))

    n = length(images)
    rows = cld(n, cols)
    out_h = rows * h + (rows - 1) * gap
    out_w = cols * w + (cols - 1) * gap
    out = fill(Float64(background), out_h, out_w)

    for (k, img) in enumerate(images)
        r = (k - 1) ÷ cols
        c = (k - 1) % cols
        i0 = r * (h + gap) + 1
        j0 = c * (w + gap) + 1
        out[i0:i0+h-1, j0:j0+w-1] .= Float64.(img)
    end
    return out
end

"""
    draw_line!(img, y0, x0, y1, x1; value=1.0)

Rasterize a straight line from `(y0, x0)` to `(y1, x1)` into `img`
using Bresenham's algorithm. Pixels outside `img` are skipped
silently. Mutates and returns `img`.

I'm using `(row, col)` indexing to match the rest of the package
(`img[i, j]` means row `i`, column `j`), so the `y*` arguments are
row indices and `x*` are column indices.
"""
function draw_line!(img::AbstractMatrix, y0::Integer, x0::Integer,
                    y1::Integer, x1::Integer; value::Real = 1.0)
    H, W = size(img)
    x0, x1, y0, y1 = Int(x0), Int(x1), Int(y0), Int(y1)
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = x0 < x1 ? 1 : -1
    sy = y0 < y1 ? 1 : -1
    err = dx - dy
    x, y = x0, y0
    while true
        if 1 ≤ y ≤ H && 1 ≤ x ≤ W
            img[y, x] = value
        end
        (x == x1 && y == y1) && break
        e2 = 2 * err
        if e2 > -dy
            err -= dy
            x += sx
        end
        if e2 < dx
            err += dx
            y += sy
        end
    end
    return img
end

"""
    mark_points!(img, points; size=1, value=1.0)

Stamp a small filled square of half-width `size` at each `(row, col)`
position in `points`. `size=1` gives a 3×3 marker. Useful for overlaying
detected feature points (corners, template matches) onto an image.
"""
function mark_points!(img::AbstractMatrix, points;
                      size::Integer = 1, value::Real = 1.0)
    H, W = Base.size(img)
    @inbounds for (i, j) in points
        for dj in -size:size, di in -size:size
            ni, nj = i + di, j + dj
            (1 ≤ ni ≤ H && 1 ≤ nj ≤ W) && (img[ni, nj] = value)
        end
    end
    return img
end

"""
    label_to_gray(labels::AbstractMatrix{Int}; n::Integer = maximum(labels)) -> Matrix{Float64}

Map a connected-components label image to grayscale so each component
gets a different intensity. Background (label 0) stays 0.0. Labels
are interleaved across the `[0.2, 1.0]` range so adjacent integers
don't end up next to each other in brightness — useful when neighboring
components have consecutive IDs.
"""
function label_to_gray(labels::AbstractMatrix{Int};
                       n::Integer = maximum(labels))
    out = zeros(Float64, Base.size(labels))
    n ≤ 0 && return out
    @inbounds for k in eachindex(labels)
        v = labels[k]
        v == 0 && continue
        # Bit-reversal-ish interleaving so consecutive IDs are far apart in gray.
        frac = ((v * 0.6180339887) % 1.0)   # golden-ratio scramble
        out[k] = 0.2 + 0.8 * frac
    end
    return out
end

"""
    _hsv_pixel_to_rgb(h, s, v) -> (r, g, b)

Single-pixel HSV → RGB conversion. `h ∈ [0, 1)` is hue (wraps), `s`
and `v` are saturation and value in `[0, 1]`. The standard piecewise
formula — I wrote it out instead of pulling in `ColorTypes` so the
function works on plain `Float64` numbers without going through a
typed colour space. Not exported; the matrix-level version lives in
`Color.hsv_to_rgb`.
"""
function _hsv_pixel_to_rgb(h::Real, s::Real, v::Real)
    h = mod(Float64(h), 1.0)
    s = clamp(Float64(s), 0.0, 1.0)
    v = clamp(Float64(v), 0.0, 1.0)
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
    return (r + m, g + m, b + m)
end

"""
    flow_to_rgb(u, v; max_mag = nothing) -> (R, G, B)

Encode a 2D vector field as a colour image using the standard
optical-flow convention: hue = direction, saturation = magnitude,
value = 1. Zero flow maps to white; large flow maps to saturated
colour. If `max_mag` is left as `nothing`, magnitudes are normalized
to the maximum present in the field; pass an explicit value to keep
the colour scale stable across frames.

Returns three `Matrix{Float64}` channels in `[0, 1]`, matching the
shape `Photos.save_rgb_planes` and `PNM.save_ppm` already expect.
"""
function flow_to_rgb(u::AbstractMatrix{<:Real}, v::AbstractMatrix{<:Real};
                     max_mag::Union{Nothing, Real} = nothing)
    size(u) == size(v) || throw(DimensionMismatch(
        "u and v have different sizes: $(size(u)) vs $(size(v))"))
    mag = sqrt.(Float64.(u) .^ 2 .+ Float64.(v) .^ 2)
    mmax = isnothing(max_mag) ? maximum(mag) : Float64(max_mag)
    mmax = max(mmax, 1e-12)   # avoid divide-by-zero on an all-zero field
    R = zeros(Float64, Base.size(u))
    G = zeros(Float64, Base.size(u))
    B = zeros(Float64, Base.size(u))
    @inbounds for k in eachindex(u)
        # atan2 in (-π, π], shift so hue wraps at 0
        h = (atan(v[k], u[k]) / (2π)) + 0.5
        s = clamp(mag[k] / mmax, 0.0, 1.0)
        r, g, b = _hsv_pixel_to_rgb(h, s, 1.0)
        R[k] = r
        G[k] = g
        B[k] = b
    end
    return (R, G, B)
end

end # module Viz
