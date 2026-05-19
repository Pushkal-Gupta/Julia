# Second-order edges: Laplacian, LoG, DoG, zero-crossings

A different way to find edges: instead of looking for *peaks* of the
first derivative, look for *zero crossings* of the second derivative.

Where the gradient magnitude `|∇I|` is biggest, the Laplacian `∇²I`
crosses zero. That's not an approximation — it's an exact relationship
for smooth signals.

Why bother with this when the gradient already works? A few reasons:

1. Zero-crossings give a one-pixel-wide edge map without needing a
   magnitude threshold to thin them. That's why second-order methods
   were popular before non-maximum suppression became standard.
2. The Laplacian is rotationally symmetric, so it doesn't have a
   "preferred orientation" the way a single-axis Sobel does.
3. Combined with a Gaussian (→ LoG), it gives a scale-tunable edge
   detector — change σ and you change which size of edge you're
   detecting.

## The Laplacian

The discrete Laplacian (`laplacian4()` in `Kernels`) is:

```
 0 -1  0
-1  4 -1
 0 -1  0
```

That's `∂²/∂x² + ∂²/∂y²` discretized with a 4-neighbor stencil. The
8-connected version (`laplacian8()`) adds the diagonals. Both sum to
zero — every "derivative" operator must, because a flat region has to
give zero response.

Run it directly on a noisy image and you get... a noisier image. The
Laplacian amplifies high frequencies, and noise *is* high frequency.
That's why Laplacian-of-Gaussian exists.

## Laplacian of Gaussian (LoG)

The Marr–Hildreth idea: smooth first with a Gaussian, *then* take the
Laplacian. Both operations are linear, so you can combine them into a
single kernel:

```
LoG_σ(x, y) = (1 / πσ⁴) · (1 − (x² + y²) / 2σ²) · exp(−(x² + y²) / 2σ²)
```

I implement this directly as `Kernels.laplacian_of_gaussian(n; sigma)`
and apply it through `Edges.log_filter`.

A few things worth noting about LoG:

- It's signed. Positive on the dark side of an edge, negative on the
  bright side. Visualize with `Viz.signed_to_gray` (zero → mid-gray).
- σ controls the scale. Small σ picks up fine detail; large σ smooths
  past fine detail and only large-scale structure survives.
- The kernel is rotationally symmetric, so it isn't separable —
  unlike a 2D Gaussian (which is) or Sobel (which is). That's a
  performance cost.

## Difference of Gaussians (DoG)

DoG is the cheap approximation to LoG:

```
DoG(σ₁, σ₂) = G_σ₁ ∗ I  −  G_σ₂ ∗ I       (with σ₁ < σ₂)
```

Each Gaussian is separable, so the whole thing costs `O(n)` instead
of LoG's `O(n²)` per pixel. The classical ratio `σ₂ / σ₁ ≈ 1.6` (Marr's
choice) makes the two operators almost visually identical. I confirm
this in `examples/05_log_dog_zero_crossings.jl` by computing
`LoG(σ=2) − DoG(σ₁=1, σ₂=1.6)` — the difference image is mostly
mid-gray, which means the operators are nearly the same up to a
constant scaling.

## Zero-crossings

Once I have a signed response from LoG or DoG, the edges are wherever
the sign changes between adjacent pixels. `Edges.zero_crossings` scans
the 4-neighborhood of each interior pixel and marks `true` wherever a
neighbor has the opposite sign *and* the local difference exceeds a
threshold (`min_diff`).

The threshold matters. Without it, every speck of noise that wiggles
the LoG response across zero would register. With it set to a few
percent of `maximum(abs, response)`, only "real" zero crossings
survive. In the example I use 2% of the response range, which gives
clean thin edges.

## Where second-order methods fall short

Two issues that pushed the field toward Canny:

1. They're noise-sensitive even with Gaussian pre-smoothing. The
   second derivative amplifies any high-frequency content that
   survives the smoothing.
2. Zero-crossings can wander on weak edges, producing closed contours
   where none exist in the image. (Marr called this the "constant
   contour" property — flattering until you realize it's also a
   failure mode.)

Canny solves the first by using a gradient (only one derivative) and
the second by using NMS + hysteresis to make the edge decision
binary and connected.

## References

- Marr, D., & Hildreth, E. (1980). *Theory of edge detection*.
  Proceedings of the Royal Society of London B, 207, 187–217.
- For the σ₂/σ₁ ≈ 1.6 derivation, the original Marr–Hildreth paper is
  still the cleanest reference.
