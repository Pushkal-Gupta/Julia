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

using ..Padding

export correlate2d, correlate2d!, convolve2d, convolve2d!

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

end # module Convolution
