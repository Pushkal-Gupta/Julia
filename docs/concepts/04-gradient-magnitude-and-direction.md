# Gradient magnitude and direction

The gradient is the most direct edge detector there is: it asks "how
fast is intensity changing here, and which way?" That's it. Everything
else in first-order edge detection — Sobel, Prewitt, Scharr, Roberts —
is just a different way of estimating the local partial derivatives
∂I/∂x and ∂I/∂y from discrete pixels.

## What each piece is

`gradient(img, op)` returns `(gx, gy)`. Both are signed.

- `gx[i, j]` is large where `img` changes fast along x (i.e. across a
  *vertical* edge — counter-intuitive at first).
- `gy[i, j]` is large where `img` changes fast along y (across a
  *horizontal* edge).

`gradient_magnitude(gx, gy)` is `√(gx² + gy²)`. Sign-blind. Big
wherever *any* direction has a strong derivative.

`gradient_direction(gx, gy)` is `atan(gy, gx)` in `(-π, π]`. Points
toward the brighter side of the edge — perpendicular to the edge
itself. The angle is what NMS will use to pick the right pair of
neighbors to compare against.

## Why each operator looks the way it does

All three of Sobel, Prewitt, and Scharr have the same structure:
they're a 1D derivative `[-1, 0, 1]` smoothed perpendicularly with a
1D blur. The blur weights are the only difference:

| Operator | Smoothing weights | Notes                                |
|----------|-------------------|--------------------------------------|
| Prewitt  | `[1, 1, 1]`       | uniform — simplest                   |
| Sobel    | `[1, 2, 1]`       | binomial — closer to Gaussian        |
| Scharr   | `[3, 10, 3]`      | optimized for rotational symmetry    |

Scharr is the winner for "responds equally to edges at all
orientations". Sobel is what almost everyone uses by default. Prewitt
is a fine simple choice; the uniform smoothing makes it slightly more
sensitive to noise than the others.

Roberts is the odd one out. It samples a 2×2 patch instead of 3×3,
which means:

- No smoothing — it's pure differencing of diagonal neighbors.
- Half-pixel offset in the output (the response sits "between" pixels).
- Twitchier on noisy inputs because there's no implicit low-pass.

I keep Roberts mostly for historical interest. In real pipelines I'd
pick Sobel or Scharr.

## Visualizing direction

PGM is grayscale, and direction is an angle — so I need a mapping
that's both informative and not actively misleading. A few things
worth knowing:

- Edge *direction* is modulo π. A horizontal edge has the same
  orientation whether you call the gradient `+y` or `−y`.
- Pixels in flat regions have meaningless direction (the angle is
  random — it's the atan of two tiny noisy numbers).

So in `examples/04_edge_operator_studio.jl` I:

1. Mask the direction map by `gradient_magnitude > some_floor`. Below
   the floor, the tile shows mid-gray (no information).
2. Map the remaining angles via `mod(θ, π) / π` to `[0, 1)`. This
   collapses the up/down ambiguity and avoids a wraparound seam.

For a color visualization (later, when I have a real plotting setup)
the natural thing is HSV: hue = direction, value = magnitude. Same
idea, but with hue you don't have to throw away half the angle range.

## Sanity checks (also tests)

A few invariants I can hand-check:

- Constant image → both `gx` and `gy` are zero everywhere.
- Black/white vertical step → `gx` peaks at the step, `gy` is zero.
- Perfect disk → `gradient_magnitude` is rotationally symmetric (the
  test compares it to its 90°-rotated version).

These are in `test/test_edges.jl`. If I ever break the gradient code
they'll catch it.
