# Separable convolution

The single biggest performance lever in classical image processing, and
a nice applied-linear-algebra moment as a bonus.

## The idea

A 2D kernel `K` of size `kh × kw` is **separable** if it can be written
as an outer product of two 1D vectors:

```
K[i, j] = ky[i] · kx[j]      ⇔      K = ky · kx'   (column · row)
```

When that's true, convolving with `K` is identical to first convolving
each row with `kx`, then convolving each column of the result with `ky`:

```
A * K  ≡  (A * kx_row) * ky_col
```

Why it matters comes down to arithmetic:

| Approach         | Mults / output pixel |
|------------------|----------------------|
| Naive 2D         | `kh · kw`            |
| Separable 2-pass | `kh + kw`            |

For a symmetric `k × k` kernel the speedup is `k² / (2k) = k/2`. At
`k=15` that's already an order of magnitude; at `k=31` it's 15×. I
verified this in `examples/03_separable_vs_naive.jl` — at `k=21`,
σ=3.5, 384×384 image:

```
naive: 38.3 ms   separable: 3.6 ms   speedup: 10.8×
```

Theory says `k/2 = 10.5×`. Measured: 10.8×. Numerical outputs agree to
within `~1e-15`, which is just the order in which floats accumulate.

## Which kernels are separable

| Kernel              | Separable? | Factorization                       |
|---------------------|------------|-------------------------------------|
| Box `n×n`           | yes        | `(1/n)·1_n × (1/n)·1_n`             |
| 2D Gaussian         | yes        | `g(σ) × g(σ)` (the 1D Gaussian)     |
| Sobel x             | yes        | `[1, 2, 1] × [-1, 0, 1]`            |
| Prewitt x           | yes        | `[1, 1, 1] × [-1, 0, 1]`            |
| Scharr x            | yes        | `[3, 10, 3] × [-1, 0, 1]`           |
| Laplacian (4 or 8)  | no         | rank 2                              |
| LoG                 | no         | rank > 1                            |
| Median (not linear) | n/a        | (not a convolution at all)          |

The reason classical derivative-of-Gaussian filters are everywhere
isn't just that they're mathematically convenient — they all happen to
be rank-1.

## Detecting separability automatically

Any matrix `K` has an SVD: `K = U Σ V'`. The kernel is rank-1 iff
exactly one singular value is non-zero. Numerically I check
`σ₂ / σ₁ < tol`. If true, `K = σ₁ · u₁ · v₁'`, and the scalar absorbs
symmetrically:

```julia
kx = v₁ · √σ₁
ky = u₁ · √σ₁
```

That's what `Convolution.factor_separable` does. It returns `(kx, ky)`
on a rank-1 input, or `nothing` if the kernel is genuinely 2D. The
tests verify that factoring Sobel / Prewitt / Scharr / a 2D Gaussian
and running the recovered factors through `separable_correlate2d`
reproduces the naive 2D result to within `~1e-10`. The Laplacian
returns `nothing`, as it should.

## Why `ky · kx'` and not `kx · ky'`

Indexing convention. Julia matrices are `K[row, col]`. The row index
varies along `y` (vertical), so the "row factor" is `ky`. The column
index varies along `x` (horizontal), so the "column factor" is `kx`.
The outer product

```
(ky · kx')[i, j] = ky[i] · kx[j]
```

matches `K[i, j] = ky[i] · kx[j]`. Hence `ky · kx'` — column vector
times row vector, in that order.

When you call `separable_correlate2d(A, kx, ky)`, the horizontal pass
applies `kx` along each row first; the vertical pass applies `ky`
along each column. Order doesn't matter for linear correlation, but
the naming convention does — if you swap them, you've transposed your
output (i.e. exchanged `*_x` and `*_y`).

## Edge effects

Padding is applied during each 1D pass, not once before both. So
whichever mode you pick at the API level affects two passes — and
for `:reflect` / `:replicate` / `:symmetric` / `:zero` you get an
output identical to the 2D version padded once. The only mode where
naive and separable can drift apart is `:circular` for non-symmetric
kernels, and even there only by float rounding, because the operation
is associative.
