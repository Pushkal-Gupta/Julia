# Convolution as linear algebra

The view that connects spatial filtering to the FFT, the frequency
response of a filter to its eigenvalues, and image processing in
general to a single line of linear algebra. This is the result I
wanted to make concrete before moving on from the imaging core.

## 1D convolution is matrix-vector multiplication

For a vector `v` of length `n` and a kernel `K` of length `k`, the
"same"-size convolution is

```
out[i] = Σ_{ik=1..k} K[ik] · v[i + ik − 1 − p]    where p = k ÷ 2
```

That's a linear function of `v`, so it can be written as `M · v`
for some `n × n` matrix `M`. The structure of `M`:

- Entry `M[i, j]` is `K[ik]` whenever `j = i + ik − 1 − p` for some
  valid `ik`, and zero otherwise.
- Equivalently, row `i` of `M` is the kernel placed so its center
  sits at column `i`.
- The same relative-offset pattern repeats in every row — only the
  position shifts. That makes `M` **Toeplitz**: constant along each
  diagonal.

I built `toeplitz_conv_matrix(K, n)` for the `:zero`-padded case.
Here's what comes out for `K = [1, 2, 1] / 4` and `n = 8`:

```
0.50  0.25  0.00  0.00  0.00  0.00  0.00  0.00
0.25  0.50  0.25  0.00  0.00  0.00  0.00  0.00
0.00  0.25  0.50  0.25  0.00  0.00  0.00  0.00
0.00  0.00  0.25  0.50  0.25  0.00  0.00  0.00
0.00  0.00  0.00  0.25  0.50  0.25  0.00  0.00
0.00  0.00  0.00  0.00  0.25  0.50  0.25  0.00
0.00  0.00  0.00  0.00  0.00  0.25  0.50  0.25
0.00  0.00  0.00  0.00  0.00  0.00  0.25  0.50
```

The band of `0.25, 0.50, 0.25` repeats down each row. At the corners
the band gets clipped because the kernel would sample past the
image's edge and `:zero` padding contributes nothing.

The test `M * v ≈ correlate1d(v, K; pad = :zero)` codifies this
equivalence — and passes to within `1e-12`.

## Circulant matrices and the FFT

The `:circular`-padding case wraps. The matrix is still made of the
same band, but instead of being clipped at the corners, the band
*continues* into the opposite corner:

```
0.50  0.25  0.00  0.00  0.00  0.00  0.00  0.25
0.25  0.50  0.25  0.00  0.00  0.00  0.00  0.00
0.00  0.25  0.50  0.25  0.00  0.00  0.00  0.00
                ... (interior unchanged) ...
0.00  0.00  0.00  0.00  0.00  0.25  0.50  0.25
0.25  0.00  0.00  0.00  0.00  0.00  0.25  0.50
```

A matrix where every row is the previous one shifted by one with
wraparound is called **circulant**, and circulant matrices have a
spectacular property: they're diagonalized by the DFT.

Concretely, for any circulant `C` of size `n × n`:

```
C = F⁻¹ · Λ · F
```

where `F` is the DFT matrix and `Λ` is diagonal. The entries of `Λ`
(the eigenvalues of `C`) are exactly the DFT of the first row of `C`.

I verified this with `circulant_eigenvalues(K, n)` (which just FFTs
the first row) against `eigvals(C)` (which factors the matrix the
hard way). For an 8×8 case the maximum absolute difference is
`2.22e-16` — a single ULP of `Float64`. They're identical.

## Why this is the result that makes FFT-convolution work

Multiplying by a circulant matrix `C v` is the same as `F⁻¹ · Λ · F · v`,
which spelled out is:

1. `F · v` — take the DFT of `v`.
2. `Λ · (F v)` — multiply pointwise by the eigenvalues.
3. `F⁻¹ · (...)` — inverse DFT.

That's the FFT-convolution algorithm. `fft_correlate2d`, the
function I wrote for the performance lab, is doing exactly this —
the eigenvalues `Λ` are the DFT of the (zero-padded, center-aligned)
kernel, and the multiply-then-inverse-FFT is the only spatial work.

So when I asked "why does FFT-convolution give the same answer as
the direct loop?", the answer is: because both compute `C · v`, just
in different bases — the standard basis (looking at the matrix
directly) or the Fourier basis (where the matrix is diagonal).

The reason FFT-convolution is fast: matrix-vector multiplication
with an `n × n` dense matrix is `O(n²)`; the FFT path is
`O(n log n)`.

## The eigenvalues *are* the frequency response

A connection I didn't fully appreciate until I wrote this script:
the eigenvalues of the circulant convolution matrix *are* the
frequency response of the filter.

For an 8×8 [1, 2, 1]/4 smoother:

```
|λ| via eigvals(C):  [1.000, 0.854, 0.854, 0.500, 0.500, 0.146, 0.146, 0.000]
```

These are sorted by magnitude, and they correspond to DFT bin
positions (`k = 0` for DC, `k = n/2` for Nyquist, etc.). Specifically:

- `|λ|_0 = 1.000` — DC component passes unchanged.
- `|λ|_{n/2} = 0.000` — Nyquist (highest representable frequency)
  is completely blocked.
- Intermediate frequencies attenuated by intermediate amounts.

This *is* what people mean by "Gaussian is a low-pass filter" — the
operator's eigenvalues form a low-pass profile. The mathematical
content of the statement is the spectrum of a specific structured
matrix.

I confirmed the same for a more realistic Gaussian (11-tap,
σ = 1.5, `n = 128`): DC eigenvalue is 1.0, Nyquist eigenvalue is
0.0001. Almost-perfect DC preservation, near-total Nyquist
attenuation. That's a strong low-pass filter, and we can read it
straight off the matrix.

## 2D, briefly

The 2D version is *doubly block circulant*. For an `H × W` image
flattened to a vector of length `HW`, the 2D circular convolution
matrix is `HW × HW`. It has block structure: `H × H` blocks, each of
size `W × W`, where each block is a circulant matrix on `W` and the
arrangement of blocks is itself circulant on `H`. So "circulant of
circulants" — doubly block circulant.

The doubly-block-circulant matrix is diagonalized by the 2D DFT,
which is just the 1D DFT applied along rows then columns. That's
the foundation of the 2D FFT-convolution algorithm.

I didn't build the full 2D version in `LinAlgView` because the
matrix would be `(H · W) × (H · W)` — for a 32×32 image that's
1024 × 1024, already 8 MB of `Float64`. The 1D case is enough to
make the point and the 2D generalization is conceptually
straightforward.

## Why this isn't the implementation path

It's worth being explicit: this submodule is *for understanding*,
not for use. Materializing the matrix costs `O(N²)` memory; the
direct loop is `O(N · k)` memory; the FFT path is `O(N)` memory.
For real workloads I'd never call `toeplitz_conv_matrix` and then
multiply. The matrix view is a teaching device — once you know that
spatial convolution is diagonalized by the Fourier transform, all
the other algorithms make sense as different ways of computing the
same matrix-vector product.

## References

- Strang, G. (1986). *Introduction to Applied Mathematics* — the
  circulant-diagonalization result is the cleanest in chapter 5 or
  6 of any edition.
- For the doubly-block-circulant 2D version: Jain, A. K. (1989),
  *Fundamentals of Digital Image Processing*, chapter 5.
