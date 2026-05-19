# Image pyramids

A pyramid is the same image at multiple resolutions, stored as a
stack. It's the canonical move when I need to reason about an image
at multiple *scales* — what counts as "a corner" or "an edge" depends
on how zoomed in you are, and a pyramid makes that explicit.

I built two related variants (Gaussian and Laplacian) on top of two
primitives. The whole thing is in `src/pyramids.jl` as the `Pyramids`
submodule.

## REDUCE and EXPAND

Burt and Adelson's 1983 construction uses two operators:

- **REDUCE**: smooth the image with a 5-tap separable binomial filter
  `[1, 4, 6, 4, 1] / 16`, then take every other pixel. Output size is
  about half along each axis.
- **EXPAND**: insert zeros between every pair of pixels (so 1 in 4
  output pixels is "real", the rest are zero), then smooth with the
  same filter scaled by `4` to compensate for the energy loss from
  zero-insertion. Result size is about double along each axis.

REDUCE and EXPAND are *approximately* inverse — `EXPAND(REDUCE(I))`
loses some high-frequency content but preserves the low-frequency
shape of `I`. The exact equality I care about is at the pyramid
level, not at this primitive level.

A subtle implementation detail: the `4×` factor in EXPAND applies to
the 2D filter `w · w^T`. Since I apply it as two separable 1D passes,
each 1D pass gets a factor of `2` (so the outer product is
`(2w) · (2w)^T = 4 · w · w^T`). Confused this the first time I wrote
it — got a result 4× too big at the borders and 16× too big in the
corners.

## The Gaussian pyramid

`gaussian_pyramid(img; levels=4)` returns a vector of `levels + 1`
matrices: the original `G_0`, then `G_1 = REDUCE(G_0)`,
`G_2 = REDUCE(G_1)`, and so on.

For a 128×128 input with 4 levels, that gives sizes
`(128, 64, 32, 16, 8)`. Each level is a low-pass-filtered version of
the previous, downsampled. The pyramid takes about `4/3` the memory
of the original (geometric series of halving sizes).

What it's good for:

- A "where might a big-scale feature live?" search — run a detector
  on the small levels first, refine on the bigger ones.
- A starting point for the Laplacian pyramid (next section).
- Multi-scale Harris or DoG keypoint detection.

## The Laplacian pyramid

`laplacian_pyramid(img; levels=4)` builds the *band-pass* version of
the pyramid: each level holds the detail that's in `G_k` but not in
`EXPAND(G_{k+1})`. Concretely:

```
L_k = G_k − EXPAND(G_{k+1})    for k = 0, 1, ..., levels-1
L_levels = G_levels             (the residual)
```

The detail levels `L_0 .. L_{levels-1}` are signed — they're
differences. Visualize with `Viz.signed_to_gray`.

What makes the Laplacian pyramid special: it's an *exact* invertible
multi-scale decomposition. The reconstruction:

```
G_levels = L_levels                                   (start from the bottom)
G_{k}   = L_k + EXPAND(G_{k+1})                        (work upward)
```

reproduces the original within floating-point precision. My
`examples/12_pyramid_decomposition.jl` measures this — for a
128×128 input the max abs error after reconstruction is `5.55e-17`,
which is a single ULP of `Float64`. That's not "close enough", that's
*exact*. The way this works is that any non-zero border behavior in
EXPAND gets recorded into the corresponding `L_k`, and the
reconstruction inverts it perfectly because it uses the same EXPAND.

Practical uses:

- **Compression** (the original 1983 motivation): the smallest level
  is fine to store at full precision; the detail levels are mostly
  near-zero except along edges, so they compress well.
- **Multi-band image blending**: blend each Laplacian level
  separately, then reconstruct. Avoids the visible seam you'd get
  blending the originals directly.
- **Multi-scale feature detection**: SIFT, for example, builds a
  difference-of-Gaussians pyramid that's essentially this with a
  different kernel choice.

## Border behavior

The pyramid is exact in the interior. At the border the EXPAND
operation amplifies the zero-insertion pattern slightly under any
padding mode — there's no clean way to extend "I just inserted zeros
in a sparse pattern, what's outside?" past the image edge. I tested
this directly: `expand(reduce(constant_image))` returns the constant
value exactly in the interior, but at the corners it can be off by
~25%. (My test `constant image stays constant in the interior`
checks the interior and skips a 4-pixel border for exactly this
reason.)

The reason the *full* reconstruction is still exact is that REDUCE
introduces the same border bias, and EXPAND undoes it. The pyramid
isn't intended for "preserve the constant value at the border" —
it's intended for "decompose into bands that recompose to the
original". The latter is what I verified.

## Sizing rule

Each level is roughly half of the previous along each axis, so for
`levels = L` I should have `min(H, W) ≥ 2^L`. At `levels = 4` that
needs at least 16 pixels on the short side, which is comfortable for
my 128×128 test images. Pushing `levels` past that gives degenerate
1×1 levels at the bottom that aren't useful.

## What's next

Pyramids are the foundation for a few more things I want to build:

- A **multi-scale Harris** that runs corner detection on every level
  of the Gaussian pyramid and merges the results — Harris is
  scale-dependent, so a 3×3 window picks up different features at
  different resolutions.
- **DoG keypoint detection** — exactly the same as my `dog_filter`
  but applied across pyramid levels instead of as a single in-place
  filter. This is the "scale-space" version of blob detection that
  underlies SIFT.
- **Image blending**, eventually, as the showcase application that
  motivated the algorithm in the first place.

## References

- Burt, P. J., & Adelson, E. H. (1983). *The Laplacian pyramid as a
  compact image code*. IEEE Transactions on Communications, 31(4),
  532–540.
