"""
    Kernels

A zoo of canonical 2D kernels for convolution / correlation experiments.

Design note: at this stage a kernel is just a `Matrix{Float64}`. Once we
need to track an origin offset (non-symmetric kernels), a separable
factorization, or a name for logging, we'll introduce a `Kernel` struct.
Premature abstractions hide what's happening on day one — and on day one
we want to *see* the numbers.

Mathematical convention used here: every function returns the kernel as
laid out on paper. Cross-correlation (`Convolution.correlate2d`) slides
this matrix directly. True convolution (`Convolution.convolve2d`) flips it
first. Symmetric kernels (box, Gaussian, Laplacian) give identical results
either way — asymmetric ones (Sobel, Prewitt) do not, and that's a teaching
moment we lean into in the Edges layer.
"""
module Kernels

export box, gaussian, sobel_x, sobel_y, prewitt_x, prewitt_y,
       scharr_x, scharr_y, roberts_x, roberts_y,
       laplacian4, laplacian8, laplacian_of_gaussian, sharpen, identity_kernel

"""
    identity_kernel(n=3) -> Matrix{Float64}

The `n × n` identity kernel (a single 1 at the center). Convolving with this
returns the image unchanged. Sanity check.
"""
function identity_kernel(n::Integer = 3)
    isodd(n) || throw(ArgumentError("kernel size must be odd, got $n"))
    K = zeros(Float64, n, n)
    K[n÷2 + 1, n÷2 + 1] = 1.0
    return K
end

"""
    box(n=3) -> Matrix{Float64}

The `n × n` normalized box filter — uniform average over a neighborhood.
The simplest possible smoother. Cheap, separable, and a useful baseline
against which to judge Gaussian's gentler frequency response.
"""
function box(n::Integer = 3)
    isodd(n) || throw(ArgumentError("kernel size must be odd, got $n"))
    return fill(1.0 / (n * n), n, n)
end

"""
    gaussian(n; sigma=n/6) -> Matrix{Float64}

An `n × n` discretized 2D Gaussian, normalized to sum to 1. Default
`sigma = n/6` puts roughly ±3σ inside the kernel window, which is the
usual rule of thumb.
"""
function gaussian(n::Integer; sigma::Real = n / 6)
    isodd(n) || throw(ArgumentError("kernel size must be odd, got $n"))
    sigma > 0 || throw(ArgumentError("sigma must be positive, got $sigma"))
    K = Matrix{Float64}(undef, n, n)
    c = n ÷ 2 + 1
    σ2 = 2 * sigma^2
    @inbounds for j in 1:n, i in 1:n
        K[i, j] = exp(-((i - c)^2 + (j - c)^2) / σ2)
    end
    K ./= sum(K)
    return K
end

# ── First-order gradient operators ────────────────────────────────────────────
# Convention: *_x detects vertical edges (it differentiates along x → horizontal),
#             *_y detects horizontal edges (differentiates along y → vertical).
# All operators return the matrix as you'd write it on paper. When fed to
# `correlate2d` the result is the gradient as conventionally defined.

"""    sobel_x() -> 3×3 Sobel x-gradient kernel."""
sobel_x() = Float64[-1 0 1
                    -2 0 2
                    -1 0 1]

"""    sobel_y() -> 3×3 Sobel y-gradient kernel."""
sobel_y() = Float64[-1 -2 -1
                     0  0  0
                     1  2  1]

"""    prewitt_x() -> 3×3 Prewitt x-gradient kernel."""
prewitt_x() = Float64[-1 0 1
                      -1 0 1
                      -1 0 1]

"""    prewitt_y() -> 3×3 Prewitt y-gradient kernel."""
prewitt_y() = Float64[-1 -1 -1
                       0  0  0
                       1  1  1]

"""    scharr_x() -> 3×3 Scharr x-gradient kernel (better rotational symmetry)."""
scharr_x() = Float64[-3   0   3
                     -10  0  10
                     -3   0   3]

"""    scharr_y() -> 3×3 Scharr y-gradient kernel."""
scharr_y() = Float64[-3 -10 -3
                      0   0  0
                      3  10  3]

"""    roberts_x() -> 2×2 Roberts cross-gradient (diagonal)."""
roberts_x() = Float64[ 1  0
                       0 -1]

"""    roberts_y() -> 2×2 Roberts cross-gradient (anti-diagonal)."""
roberts_y() = Float64[ 0  1
                      -1  0]

# ── Second-order operators ────────────────────────────────────────────────────

"""
    laplacian4() -> 3×3 four-connected Laplacian.

`∂²/∂x² + ∂²/∂y²` with 4-neighbor stencil. Edges show up as zero-crossings.
"""
laplacian4() = Float64[ 0 -1  0
                       -1  4 -1
                        0 -1  0]

"""    laplacian8() -> 3×3 eight-connected Laplacian (includes diagonals)."""
laplacian8() = Float64[-1 -1 -1
                       -1  8 -1
                       -1 -1 -1]

"""
    laplacian_of_gaussian(n; sigma=n/6) -> Matrix{Float64}

The Marr–Hildreth LoG operator. Equivalent to Gaussian-blurring then taking
the Laplacian, but combined into a single kernel for efficiency.
Mean-subtracted so it sums (approximately) to zero — important for a
"derivative" operator: it must give zero response on a flat region.
"""
function laplacian_of_gaussian(n::Integer; sigma::Real = n / 6)
    isodd(n) || throw(ArgumentError("kernel size must be odd, got $n"))
    sigma > 0 || throw(ArgumentError("sigma must be positive, got $sigma"))
    K = Matrix{Float64}(undef, n, n)
    c = n ÷ 2 + 1
    σ2 = sigma^2
    @inbounds for j in 1:n, i in 1:n
        x2 = (i - c)^2 + (j - c)^2
        K[i, j] = (x2 / σ2 - 2) * exp(-x2 / (2σ2)) / (π * σ2^2)
    end
    K .-= sum(K) / length(K)
    return K
end

"""
    sharpen(strength=1.0) -> 3×3 unsharp-mask-style sharpening kernel.

`identity + strength * laplacian4`. Boosts high frequencies. Crank `strength`
past ~2 and you'll see haloing — a perfect demonstration of why real
unsharp masking blends with a blurred image instead.
"""
function sharpen(strength::Real = 1.0)
    return identity_kernel(3) .+ strength .* laplacian4()
end

end # module Kernels
