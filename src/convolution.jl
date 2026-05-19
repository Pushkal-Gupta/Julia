"""
    Convolution

Naive, from-scratch 2D correlation and convolution. "Naive" here means
direct nested loops — not "wrong". The point at this layer is:

1. Get the math right and prove it with tests.
2. Make the algorithm transparent — every memory access visible.
3. Establish a baseline against which optimized versions (separable,
   in-place, FFT-based) will be measured later.

The distinction we lean into here:

- **Correlation** slides the kernel directly over the image. This is what
  CNN frameworks and most image libraries (mis-)call "convolution".
- **Convolution** (the mathematical one) flips the kernel both ways before
  sliding. It's commutative and associative; correlation isn't. For
  symmetric kernels (Gaussian, box, Laplacian) the two produce identical
  results. For Sobel and friends they differ by a sign.

Public API:

- `correlate2d(image, kernel; pad=:replicate)` — slide unflipped kernel
- `convolve2d(image, kernel; pad=:replicate)` — slide flipped kernel
- `correlate2d!(out, image, kernel; pad=:replicate)` — in-place
"""
module Convolution

using LinearAlgebra: svd
using ..Padding

export correlate2d, correlate2d!, convolve2d, convolve2d!,
       correlate1d, convolve1d,
       separable_correlate2d, separable_convolve2d,
       factor_separable

"""
    correlate2d(image, kernel; pad=:replicate) -> Matrix

2D cross-correlation. The output has the same size as `image` unless
`pad === :valid`, in which case it shrinks by `(kh-1, kw-1)`.

`pad` is forwarded to [`Padding.pad_array`](@ref); see that module for the
supported modes.
"""
function correlate2d(image::AbstractMatrix{T}, kernel::AbstractMatrix{K};
                     pad::Symbol = :replicate) where {T<:Real, K<:Real}
    R = promote_type(T, K, Float64)
    kh, kw = size(kernel)
    isodd(kh) && isodd(kw) || throw(ArgumentError(
        "kernel dimensions must be odd, got $((kh, kw))"))

    if pad === :valid
        H, W = size(image)
        out_h, out_w = H - kh + 1, W - kw + 1
        (out_h > 0 && out_w > 0) || throw(ArgumentError(
            "kernel $((kh,kw)) is larger than image $((H,W)) under :valid padding"))
        out = Matrix{R}(undef, out_h, out_w)
        _correlate_into!(out, image, kernel)
        return out
    end

    ph, pw = kh ÷ 2, kw ÷ 2
    padded = Padding.pad_array(image, (ph, ph, pw, pw); mode = pad)
    out = Matrix{R}(undef, size(image))
    _correlate_into!(out, padded, kernel)
    return out
end

"""
    correlate2d!(out, image, kernel; pad=:replicate)

In-place variant. `out` must have size `size(image)` for any padded mode,
or `size(image) .- size(kernel) .+ 1` for `:valid`. Same allocation
otherwise; included as a stepping stone toward fully allocation-free
pipelines in the performance lab.
"""
function correlate2d!(out::AbstractMatrix, image::AbstractMatrix, kernel::AbstractMatrix;
                      pad::Symbol = :replicate)
    kh, kw = size(kernel)
    isodd(kh) && isodd(kw) || throw(ArgumentError(
        "kernel dimensions must be odd, got $((kh, kw))"))

    if pad === :valid
        H, W = size(image)
        expected = (H - kh + 1, W - kw + 1)
        size(out) == expected || throw(DimensionMismatch(
            "out has size $(size(out)); expected $expected for :valid"))
        _correlate_into!(out, image, kernel)
        return out
    end

    size(out) == size(image) || throw(DimensionMismatch(
        "out has size $(size(out)); expected $(size(image))"))
    ph, pw = kh ÷ 2, kw ÷ 2
    padded = Padding.pad_array(image, (ph, ph, pw, pw); mode = pad)
    _correlate_into!(out, padded, kernel)
    return out
end

"""
    convolve2d(image, kernel; pad=:replicate) -> Matrix

Mathematical 2D convolution. Equivalent to correlating with the kernel
flipped along both axes.
"""
convolve2d(image::AbstractMatrix, kernel::AbstractMatrix; pad::Symbol = :replicate) =
    correlate2d(image, _flip(kernel); pad = pad)

convolve2d!(out::AbstractMatrix, image::AbstractMatrix, kernel::AbstractMatrix;
            pad::Symbol = :replicate) =
    correlate2d!(out, image, _flip(kernel); pad = pad)

# ── Internal kernels ──────────────────────────────────────────────────────────

"""
    _correlate_into!(out, A, K)

The naive inner loop. Assumes `A` is already padded such that
`size(out)` is the valid output size for `A` and `K`. No bounds checks
beyond the explicit `@inbounds` invariants documented above.
"""
function _correlate_into!(out::AbstractMatrix, A::AbstractMatrix, K::AbstractMatrix)
    kh, kw = size(K)
    oh, ow = size(out)
    R = eltype(out)
    @inbounds for j in 1:ow
        for i in 1:oh
            s = zero(R)
            for jk in 1:kw
                for ik in 1:kh
                    s += A[i + ik - 1, j + jk - 1] * K[ik, jk]
                end
            end
            out[i, j] = s
        end
    end
    return out
end

"""
    _flip(K)

Flip a 2D kernel along both axes — what turns correlation into convolution.
"""
_flip(K::AbstractMatrix) = K[end:-1:1, end:-1:1]

# ── 1D operators ──────────────────────────────────────────────────────────────

"""
    correlate1d(v::AbstractVector, k::AbstractVector; pad=:replicate) -> Vector

1D cross-correlation. Identical math to `correlate2d` restricted to a single
axis. Output has the same length as `v` unless `pad === :valid`, in which
case it shrinks by `length(k) - 1`.
"""
function correlate1d(v::AbstractVector{T}, k::AbstractVector{Tk};
                     pad::Symbol = :replicate) where {T<:Real, Tk<:Real}
    klen = length(k)
    isodd(klen) || throw(ArgumentError("kernel length must be odd, got $klen"))
    R = promote_type(T, Tk, Float64)
    n = length(v)

    if pad === :valid
        out_n = n - klen + 1
        out_n > 0 || throw(ArgumentError(
            "kernel length $klen exceeds vector length $n under :valid"))
        out = Vector{R}(undef, out_n)
        @inbounds for i in 1:out_n
            s = zero(R)
            for ik in 1:klen
                s += v[i + ik - 1] * k[ik]
            end
            out[i] = s
        end
        return out
    end

    p = klen ÷ 2
    padded = Padding.pad_vector(v, p, p; mode = pad)
    out = Vector{R}(undef, n)
    @inbounds for i in 1:n
        s = zero(R)
        for ik in 1:klen
            s += padded[i + ik - 1] * k[ik]
        end
        out[i] = s
    end
    return out
end

"""
    correlate1d(A::AbstractMatrix, k::AbstractVector; axis, pad=:replicate) -> Matrix

Apply a 1D kernel along a single axis of a 2D image. `axis` must be one of
`:horizontal` / `:x` (slides left-right within each row) or `:vertical` /
`:y` (slides top-bottom within each column).

Internally this reshapes the kernel as `1×k` or `k×1` and routes through
`correlate2d`. The 2D loop with one degenerate dimension only does `k`
multiplications per output pixel (not `k²`), which is exactly what we want.
"""
function correlate1d(A::AbstractMatrix, k::AbstractVector;
                     axis::Symbol, pad::Symbol = :replicate)
    klen = length(k)
    isodd(klen) || throw(ArgumentError("kernel length must be odd, got $klen"))
    if axis === :horizontal || axis === :x
        return correlate2d(A, reshape(collect(Float64, k), 1, klen); pad = pad)
    elseif axis === :vertical || axis === :y
        return correlate2d(A, reshape(collect(Float64, k), klen, 1); pad = pad)
    else
        throw(ArgumentError(
            "axis must be :horizontal/:x or :vertical/:y, got :$axis"))
    end
end

convolve1d(v::AbstractVector, k::AbstractVector; pad::Symbol = :replicate) =
    correlate1d(v, reverse(k); pad = pad)

convolve1d(A::AbstractMatrix, k::AbstractVector; axis::Symbol, pad::Symbol = :replicate) =
    correlate1d(A, reverse(k); axis = axis, pad = pad)

# ── Separable 2D ──────────────────────────────────────────────────────────────

"""
    separable_correlate2d(A, kx, ky; pad=:replicate) -> Matrix

Two-pass correlation: first apply 1D kernel `kx` horizontally (along x),
then 1D kernel `ky` vertically (along y). This is equivalent to
`correlate2d(A, ky * kx'; pad=pad)` whenever the 2D kernel is rank-1, but
with `O(h·w·(kh+kw))` work instead of `O(h·w·kh·kw)` — the central reason
Gaussian and box blurs are cheap.

Convention: `kx` matches the column index (length `kw`), `ky` matches the
row index (length `kh`). Their outer product `ky * kx'` reconstructs the
2D kernel.
"""
function separable_correlate2d(A::AbstractMatrix,
                               kx::AbstractVector, ky::AbstractVector;
                               pad::Symbol = :replicate)
    intermediate = correlate1d(A, kx; axis = :horizontal, pad = pad)
    return correlate1d(intermediate, ky; axis = :vertical, pad = pad)
end

"""
    separable_convolve2d(A, kx, ky; pad=:replicate) -> Matrix

Convolution-flavored counterpart: each 1D factor is reversed before being
applied, matching the mathematical definition.
"""
separable_convolve2d(A::AbstractMatrix, kx::AbstractVector, ky::AbstractVector;
                     pad::Symbol = :replicate) =
    separable_correlate2d(A, reverse(kx), reverse(ky); pad = pad)

"""
    factor_separable(K; tol=1e-10) -> Union{Tuple{Vector,Vector}, Nothing}

Attempt to factor a 2D kernel `K` as a rank-1 outer product `ky * kx'`.
Returns `(kx, ky)` on success; `nothing` if `K` is genuinely 2D
(higher-rank).

The test is `σ₂ / σ₁ ≤ tol` on the singular values. Lots of practical
kernels — box, Gaussian, Sobel, Prewitt, Scharr — are exactly rank-1.
The Laplacian is not.
"""
function factor_separable(K::AbstractMatrix{<:Real}; tol::Real = 1e-10)
    F = svd(Float64.(K))
    s = F.S
    (isempty(s) || s[1] == 0) && return nothing
    if length(s) ≥ 2 && s[2] / s[1] > tol
        return nothing
    end
    σ = sqrt(s[1])
    ky = F.U[:, 1] .* σ
    kx = F.Vt[1, :] .* σ
    return (kx, ky)
end

end # module Convolution
