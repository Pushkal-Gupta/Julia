"""
    Viz

Small visualization helpers. Nothing fancy — just the operations every
example script kept reinventing inline. As we add interactive tools and
heatmaps in later milestones, those go here too.
"""
module Viz

export normalize01, montage, signed_to_gray

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

end # module Viz
