# 02 — Separable convolution

The single biggest performance lever in classical image processing, and a
beautiful applied linear-algebra moment to boot.

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

The reason this matters is arithmetic:

| Approach        | Mults / output pixel |
|-----------------|----------------------|
| Naive 2D        | `kh · kw`            |
| Separable 2-pass | `kh + kw`            |

For symmetric `k × k` kernels the speedup is `k² / (2k) = k/2`. At `k=15`
that's already an order of magnitude; at `k=31` it's 15×. We measure this
in `examples/03_separable_vs_naive.jl`:

```
k=21  σ=3.5   naive: 38.3 ms   separable: 3.6 ms   speedup: 10.8×
```

Theory says `k/2 = 10.5×`; we hit `10.8×`. The numerical outputs agree
to within `~1e-15`, which is just the order in which we accumulate
floats.

## Which kernels are separable?

| Kernel              | Separable? | Factorization                        |
|---------------------|------------|--------------------------------------|
| Box `n×n`           | yes        | `(1/n)·1_n` × `(1/n)·1_n`           |
| 2D Gaussian         | yes        | `g(σ) × g(σ)` (the 1D Gaussian)     |
| Sobel x             | yes        | `[1,2,1]` × `[-1,0,1]`              |
| Prewitt x           | yes        | `[1,1,1]` × `[-1,0,1]`              |
| Scharr x            | yes        | `[3,10,3]` × `[-1,0,1]`             |
| Laplacian (4 or 8)  | **no**     | rank 2                              |
| LoG                 | no         | rank > 1                            |
| Median (not linear) | n/a        | (not a convolution at all)          |

The reason classical derivative-of-Gaussian filters are everywhere isn't
just that they're mathematically convenient — it's that they all happen to
be rank-1. Roberts, Sobel, Prewitt, Scharr, box, Gaussian, integral image
edges: all separable.

## Detecting separability automatically

Any matrix `K` has an SVD: `K = U Σ V'`. The kernel is rank-1 iff
exactly one singular value is non-zero. Numerically we check
`σ₂ / σ₁ < tol`. If true, `K = σ₁ · u₁ · v₁'`, and we can absorb the
scalar `σ₁` symmetrically:

```julia
kx = v₁ · √σ₁
ky = u₁ · √σ₁
```

That's what `Convolution.factor_separable` does. It returns `(kx, ky)`
or `nothing`. The test suite verifies that for Sobel, Prewitt, and a 2D
Gaussian, factoring then running through `separable_correlate2d`
reproduces the naive 2D result to within `~1e-10`.

## Why `ky · kx'` and not `kx · ky'`?

Indexing convention. Julia matrices are `K[row, col]`. The row index
varies along `y` (vertical), so the "row factor" is `ky`. The column
index varies along `x` (horizontal), so the "column factor" is `kx`. The
outer product:

```
(ky · kx')[i, j] = ky[i] · kx[j]
```

matches `K[i, j] = ky[i] · kx[j]`. Hence `ky · kx'` — column vector times
row vector, in that order.

When you call `separable_correlate2d(A, kx, ky)`, the horizontal pass
applies `kx` along each row first; the vertical pass then applies `ky`
along each column. Order doesn't matter for linear correlation — but the
naming convention does, and getting it wrong silently corresponds to
transposing your output (i.e. swapping `*_x` and `*_y` operators).

## Edge effect

Padding is applied during each 1D pass, not once before both. So the
padding mode you pick at the API level affects two passes — and for
`:reflect` / `:replicate` / `:symmetric` this gives an output identical
to the 2D version padded once. For `:zero` they also agree exactly. The
only mode where naive and separable can drift apart is `:circular` for
non-symmetric kernels, and even there only by float rounding, because the
operation is associative.
