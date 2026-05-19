"""
    Synth

Synthetic grayscale image generators. Useful because:

1. You can hand-check the output of a filter — a Sobel applied to a square
   should produce a specific edge map you can predict.
2. No external dependencies, no network, no licensing on test images.
3. You can dial difficulty (noise, contrast, shape size) to stress-test
   operators.

All generators return `Matrix{Float64}` with values in `[0, 1]`.
"""
module Synth

using Random

export checkerboard, square, circle, impulse, ramp, lines, gaussian_noise, salt_pepper

"""
    checkerboard(h, w; tile=8) -> Matrix{Float64}

A binary checkerboard with `tile`-pixel squares. Classic for testing
high-frequency response — every operator that "detects edges" should
light up at every tile boundary.
"""
function checkerboard(h::Integer, w::Integer; tile::Integer = 8)
    img = Matrix{Float64}(undef, h, w)
    @inbounds for j in 1:w, i in 1:h
        img[i, j] = iseven((i - 1) ÷ tile + (j - 1) ÷ tile) ? 0.0 : 1.0
    end
    return img
end

"""
    square(h, w; side=min(h,w)÷2, value=1.0, bg=0.0) -> Matrix{Float64}

A filled axis-aligned square centered in an `h × w` canvas.
"""
function square(h::Integer, w::Integer;
                side::Integer = min(h, w) ÷ 2,
                value::Real = 1.0,
                bg::Real = 0.0)
    img = fill(Float64(bg), h, w)
    i0 = (h - side) ÷ 2 + 1
    j0 = (w - side) ÷ 2 + 1
    img[i0:i0+side-1, j0:j0+side-1] .= Float64(value)
    return img
end

"""
    circle(h, w; radius=min(h,w)÷3, value=1.0, bg=0.0) -> Matrix{Float64}

A filled disk. Useful for showing directional bias in gradient operators —
a perfect circle should produce a perfect ring under gradient magnitude.
"""
function circle(h::Integer, w::Integer;
                radius::Integer = min(h, w) ÷ 3,
                value::Real = 1.0,
                bg::Real = 0.0)
    img = fill(Float64(bg), h, w)
    cy, cx = (h + 1) / 2, (w + 1) / 2
    r2 = radius^2
    @inbounds for j in 1:w, i in 1:h
        if (i - cy)^2 + (j - cx)^2 ≤ r2
            img[i, j] = Float64(value)
        end
    end
    return img
end

"""
    impulse(h, w; value=1.0) -> Matrix{Float64}

A single bright pixel at the center. Convolving any kernel with an impulse
reproduces the (flipped) kernel itself — a great sanity check.
"""
function impulse(h::Integer, w::Integer; value::Real = 1.0)
    img = zeros(Float64, h, w)
    img[h÷2 + 1, w÷2 + 1] = Float64(value)
    return img
end

"""
    ramp(h, w; axis=:x) -> Matrix{Float64}

Linear intensity gradient from 0 to 1 along the chosen axis (`:x` or `:y`).
The gradient operator applied to a ramp should be (approximately) constant.
"""
function ramp(h::Integer, w::Integer; axis::Symbol = :x)
    img = Matrix{Float64}(undef, h, w)
    if axis === :x
        @inbounds for j in 1:w, i in 1:h
            img[i, j] = (j - 1) / max(w - 1, 1)
        end
    elseif axis === :y
        @inbounds for j in 1:w, i in 1:h
            img[i, j] = (i - 1) / max(h - 1, 1)
        end
    else
        throw(ArgumentError("axis must be :x or :y, got $axis"))
    end
    return img
end

"""
    lines(h, w; orientation=:horizontal, spacing=8, thickness=1) -> Matrix{Float64}

Periodic bright lines on a dark background. Good for testing directional
sensitivity: a horizontal-line image should be invisible to a horizontal
gradient (∂/∂x) and bright to a vertical one (∂/∂y).
"""
function lines(h::Integer, w::Integer;
               orientation::Symbol = :horizontal,
               spacing::Integer = 8,
               thickness::Integer = 1)
    img = zeros(Float64, h, w)
    if orientation === :horizontal
        for i in 1:spacing:h, t in 0:thickness-1
            row = i + t
            row ≤ h && (img[row, :] .= 1.0)
        end
    elseif orientation === :vertical
        for j in 1:spacing:w, t in 0:thickness-1
            col = j + t
            col ≤ w && (img[:, col] .= 1.0)
        end
    else
        throw(ArgumentError("orientation must be :horizontal or :vertical"))
    end
    return img
end

"""
    gaussian_noise(img; sigma=0.05, seed=nothing) -> Matrix{Float64}

Add zero-mean Gaussian noise with standard deviation `sigma`. Result is
clipped to `[0, 1]`. Pass `seed` for reproducibility.
"""
function gaussian_noise(img::AbstractMatrix{<:Real}; sigma::Real = 0.05, seed = nothing)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    out = similar(img, Float64)
    @inbounds for i in eachindex(img)
        out[i] = clamp(img[i] + sigma * randn(rng), 0.0, 1.0)
    end
    return out
end

"""
    salt_pepper(img; p=0.02, seed=nothing) -> Matrix{Float64}

Impulse (salt-and-pepper) noise: each pixel is independently flipped to 0 or
1 with probability `p` (split evenly). Brutal for naive smoothing, tame for
median filters — a useful contrast we'll explore in the noise lab.
"""
function salt_pepper(img::AbstractMatrix{<:Real}; p::Real = 0.02, seed = nothing)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    out = Matrix{Float64}(img)
    half = p / 2
    @inbounds for i in eachindex(out)
        r = rand(rng)
        if r < half
            out[i] = 0.0
        elseif r < p
            out[i] = 1.0
        end
    end
    return out
end

end # module Synth
