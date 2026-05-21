"""
    LinAlgView

The linear-algebra view of convolution. A 1D convolution is matrix-
vector multiplication with a structured matrix — Toeplitz for `:zero`
padding, circulant for `:circular`. Circulant matrices are
diagonalized by the DFT, which is exactly the result that makes
FFT-based convolution work.

I built this as a teaching submodule, not a performance one. Materializing
the convolution matrix costs `O(N²)` memory for an `N`-element signal,
which is fine for `N ≤ a few thousand` but wasteful otherwise. The
direct path (`Convolution.correlate1d`) and the FFT path
(`Convolution.fft_correlate2d`) are what you'd use in practice; this
submodule is for *seeing* why those two paths give the same answer.
"""
module LinAlgView

using FFTW: fft
using ..Padding

export toeplitz_conv_matrix, circulant_conv_matrix, circulant_eigenvalues

"""
    toeplitz_conv_matrix(kernel, n) -> Matrix{Float64}

The `n × n` matrix `M` such that `M * v == correlate1d(v, kernel; pad = :zero)`
for any vector `v` of length `n`. The result is a banded matrix that's
*constant along each diagonal* — that's the definition of Toeplitz —
because shifting the kernel by one position in the output corresponds
to shifting the row by one position in `M`.
"""
function toeplitz_conv_matrix(kernel::AbstractVector{<:Real}, n::Integer)
    klen = length(kernel)
    isodd(klen) || throw(ArgumentError(
        "kernel length must be odd, got $klen"))
    M = zeros(Float64, n, n)
    p = klen ÷ 2
    @inbounds for i in 1:n
        for ik in 1:klen
            src = i + ik - 1 - p
            (1 ≤ src ≤ n) && (M[i, src] = kernel[ik])
        end
    end
    return M
end

"""
    circulant_conv_matrix(kernel, n) -> Matrix{Float64}

The `n × n` matrix `M` such that `M * v == correlate1d(v, kernel; pad = :circular)`.
Every row is the previous row shifted by one position, with the rightmost
element wrapping around — that's the definition of circulant. A circulant
matrix is fully described by its first row.
"""
function circulant_conv_matrix(kernel::AbstractVector{<:Real}, n::Integer)
    klen = length(kernel)
    isodd(klen) || throw(ArgumentError(
        "kernel length must be odd, got $klen"))
    M = zeros(Float64, n, n)
    p = klen ÷ 2
    @inbounds for i in 1:n
        for ik in 1:klen
            src = mod(i + ik - 2 - p, n) + 1
            M[i, src] += kernel[ik]
        end
    end
    return M
end

"""
    circulant_eigenvalues(kernel, n) -> Vector{ComplexF64}

The eigenvalues of the circulant convolution matrix of size `n × n`.
Every circulant matrix `C` factors as `C = F⁻¹ · Λ · F` where `F` is
the DFT matrix and `Λ` is diagonal with entries equal to the DFT of
the matrix's first row. So *computing the FFT of the kernel* is
*computing the eigenvalues of the convolution operator*.

This is the linear-algebra heart of FFT-based convolution:
spatial-domain convolution is matrix-vector multiplication, the
matrix is circulant (for circular padding), the DFT diagonalizes it,
so multiplying by the matrix becomes "FFT → pointwise multiply → IFFT".
"""
function circulant_eigenvalues(kernel::AbstractVector{<:Real}, n::Integer)
    klen = length(kernel)
    isodd(klen) || throw(ArgumentError(
        "kernel length must be odd, got $klen"))
    n ≥ klen || throw(ArgumentError(
        "n ($n) must be ≥ kernel length ($klen)"))
    # The first row of the circulant matrix encodes the kernel with its
    # center aligned to index 1 (column 1 of the matrix). Going leftward
    # from the center, the kernel wraps to the end.
    first_row = zeros(Float64, n)
    p = klen ÷ 2
    for ik in 1:klen
        idx = mod(ik - 1 - p, n) + 1
        first_row[idx] += kernel[ik]
    end
    return fft(first_row)
end

end # module LinAlgView
