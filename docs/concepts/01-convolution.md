# 01 — Correlation vs. convolution

Two operations, almost the same, with opposite cultural histories.

## The math

For a 2D image `A` and a 2D kernel `K`:

```
correlation: (A ⋆ K)[i, j] = Σ Σ A[i+m, j+n] · K[m, n]
              m  n

convolution: (A ∗ K)[i, j] = Σ Σ A[i-m, j-n] · K[m, n]
              m  n
```

Correlation slides the kernel as-is. Convolution slides the kernel
*flipped* in both directions.

## Why two names exist

- **Image processing libraries** (OpenCV, scikit-image, PyTorch's "Conv2d")
  almost universally implement correlation but call it convolution. For
  symmetric kernels (Gaussian, box, Laplacian) the distinction vanishes,
  so the abuse of language rarely bites anyone.
- **Signal processing and pure math** keep the distinction because
  convolution is what makes the Fourier transform pleasant
  (`F(A ∗ K) = F(A) · F(K)`) and what makes the operation commutative and
  associative.

We keep both in `ImageLab` to make the distinction crisp. When in doubt,
check the impulse response: correlating an impulse with `K` returns `K`
rotated 180°; convolving an impulse with `K` returns `K`.

## Sanity check (also a test)

```julia
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution

im = Synth.impulse(7, 7)
K  = sobel_x()

correlate2d(im, K; pad = :zero)[3:5, 3:5] ==
    reverse(reverse(K; dims = 1); dims = 2)   # K rotated 180°

convolve2d(im, K; pad = :zero)[3:5, 3:5] == K   # K, as-is
```

Both invariants are codified in `test/test_convolution.jl`.

## A note on indexing

Julia is 1-based and column-major. We use `img[row, col]`. The kernel's
"center" is taken to be `(kh ÷ 2 + 1, kw ÷ 2 + 1)`, which for a 3×3 kernel
is `(2, 2)` — the integer middle. Even-sized kernels don't have an integer
center, which is one reason we currently require odd dimensions. Once we
need 2×2 Roberts cross *correlation*, we'll generalize by tracking an
explicit `origin` field on a `Kernel` struct.

## When does it actually matter?

| Operator              | Symmetric? | corr == conv? |
|-----------------------|------------|---------------|
| box                   | yes        | yes           |
| Gaussian              | yes        | yes           |
| Laplacian (4 or 8)    | yes        | yes           |
| Laplacian of Gaussian | yes        | yes           |
| Sobel x / y           | **no**     | **no**        |
| Prewitt x / y         | **no**     | **no**        |
| Roberts               | **no**     | **no**        |

So: it only matters for gradient operators, and there it matters only as a
sign. If you're building Canny and your edges look "right" but with the
gradient direction flipped, this is your bug.
