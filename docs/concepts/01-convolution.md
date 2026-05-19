# Correlation vs. convolution

Two operations, almost the same, with opposite cultural histories — and
exactly the kind of small distinction that quietly burns a few hours
the first time you hit it.

## The math

For a 2D image `A` and a 2D kernel `K`:

```
correlation:  (A ⋆ K)[i, j] = Σ Σ A[i+m, j+n] · K[m, n]
                              m  n

convolution:  (A ∗ K)[i, j] = Σ Σ A[i-m, j-n] · K[m, n]
                              m  n
```

Correlation slides the kernel as-is over the image. Convolution slides
the kernel *flipped* in both directions.

## Why the two names both exist

Image-processing libraries (OpenCV, scikit-image, PyTorch's `Conv2d`)
almost universally implement correlation but call it convolution. For
symmetric kernels — Gaussian, box, Laplacian — the distinction
vanishes, so the abuse of language rarely bites anyone in practice.

Signal processing and applied math keep the distinction because
convolution is what makes the Fourier transform pleasant
(`F(A ∗ K) = F(A) · F(K)`) and what makes the operation commutative and
associative. Correlation isn't either of those.

I kept both in the package because the distinction is a great test
case. Once you've coded both, the difference is concrete: convolving
an impulse with a kernel returns the kernel; correlating with the same
kernel returns the kernel rotated 180°. If you ever build Canny and
your gradient direction is "off by a sign", this is your bug.

## A sanity check that's also a test

```julia
using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution

im = Synth.impulse(7, 7)
K  = sobel_x()

# Correlating with K reproduces K rotated 180°:
correlate2d(im, K; pad = :zero)[3:5, 3:5] ==
    reverse(reverse(K; dims = 1); dims = 2)

# Convolving with K reproduces K as-is:
convolve2d(im, K; pad = :zero)[3:5, 3:5] == K
```

Both invariants are codified in `test/test_convolution.jl`.

## A note on indexing

Julia is 1-based and column-major. I use `img[row, col]` everywhere.
The kernel's "center" is taken to be `(kh ÷ 2 + 1, kw ÷ 2 + 1)`, which
for a 3×3 kernel is `(2, 2)` — the integer middle.

Even-sized kernels don't have an integer center, which is one reason I
currently require odd dimensions. Once I need 2×2 Roberts correlation
directly through `correlate2d`, I'll generalize by tracking an explicit
`origin` field on a `Kernel` struct. For now Roberts goes through a
direct loop in `Edges._roberts_gradient`.

## When does the distinction actually matter?

| Operator              | Symmetric? | corr == conv? |
|-----------------------|------------|---------------|
| box                   | yes        | yes           |
| Gaussian              | yes        | yes           |
| Laplacian (4 or 8)    | yes        | yes           |
| Laplacian of Gaussian | yes        | yes           |
| Sobel x / y           | no         | no            |
| Prewitt x / y         | no         | no            |
| Roberts               | no         | no            |

So in practice it only matters for gradient operators, and there it
shows up as a sign. Worth knowing the next time my edges look right but
the direction map is upside down.
