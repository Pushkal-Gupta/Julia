# Pyramidal Lucas-Kanade

Plain LK from the previous chunk fails on big motions, by design.
The optical-flow constraint `Iₓ · u + I_y · v + I_t = 0` comes from a
first-order Taylor expansion of brightness constancy, so it's only
accurate when `|u|` and `|v|` are small relative to the local feature
size. On a sinusoid with one cycle per 25 pixels, "small" means
roughly sub-pixel; past about one pixel of motion the recovered flow
starts overshooting the truth.

Pyramidal LK (Bouguet, 2001) is the standard fix, and it composes
two pieces I already had on the shelf: `Pyramids.gaussian_pyramid`
from the multi-scale chunk, and `Flow.lucas_kanade` from the previous
chunk. The new piece is bilinear image warping and a small driver
that ties them together.

## The idea

The OFC linearization holds for sub-pixel motion. So if I have a
large motion that breaks plain LK at the original resolution, I run
LK at a *coarser* resolution where the same physical motion covers
fewer pixels. Specifically: a 4-pixel motion at level 0 is a 2-pixel
motion at level 1, a 1-pixel motion at level 2, half a pixel at
level 3. Eventually I get to a level where the linearization is
valid, and there LK actually works.

But the flow at level 3 only tells me about 8× downsampled
displacement. I need the flow at level 0. So I upsample, then I
*refine* — warp frame 2 at level 2 by the upsampled estimate, run LK
on the residual (which is now small again at this level), accumulate.
Repeat all the way to level 0.

The whole loop:

```
build Gaussian pyramids of I₁ and I₂   (L = levels + 1 images each)
(u, v) ← zeros, at the coarsest size
for k = L, L-1, ..., 1:
    for _ in 1:iters_per_level:
        warped ← warp(pyr2[k], u, v)         # warp frame 2 back
        residual ← lucas_kanade(pyr1[k], warped)  # solve for what's left
        u ← u + residual.u
        v ← v + residual.v
    if k > 1:
        u ← 2 · bilinear_upsample(u, size of pyr1[k-1])
        v ← 2 · bilinear_upsample(v, size of pyr1[k-1])
return (u, v)
```

A few subtleties worth calling out:

- **Why the factor of 2 in the upsample.** A 1-pixel motion at the
  coarse level is a 2-pixel motion at the next-finer level (the
  pyramid downsamples by 2). So when I move the flow estimate up
  one level, I have to double its magnitude *and* spatially
  upsample it. Doing one without the other gives a flow that's
  geometrically right but numerically half what it should be (or
  vice versa).
- **Why warp at every level, not just once.** If I only warped at
  the original resolution, the linearization at each coarser level
  would still see big motions and overshoot. Warping at each level
  means LK at that level only ever solves for sub-pixel residuals.
- **Why iterate within a level.** Even after warping by the
  upsampled estimate from the previous level, the residual might
  still be ~0.5 pixels. That's inside LK's regime but still has
  some linearization bias. One or two extra Newton-style sweeps
  (warp → solve → accumulate) drive the residual down to well
  under 0.1 pixels.

## Bilinear warping

`warp_bilinear(img, u, v)` does the inverse-warp:

```
output[i, j] = bilinear_sample(img, i + v[i, j], j + u[i, j])
```

The four-tap bilinear is a one-liner: clamp the four corner indices
to the image boundary, compute the two fractional offsets, blend.
Out-of-bounds samples replicate the edge so the warp doesn't punch
holes near the borders.

Sanity check encoded as a test: if `lucas_kanade(I₁, I₂)` returns
flow `(u, v)`, then `warp_bilinear(I₂, u, v)` should look like `I₁`.
On the demo input with `lucas_kanade_pyramid`, the mean per-pixel
residual `|warp(I₂, flow) − I₁|` comes out to **0.0004** — the warp
reconstructs `I₁` to within four parts per ten thousand on a `[0, 1]`
image.

## The headline numbers

`examples/21_pyramidal_optical_flow.jl` runs both algorithms on a
4-pixel-by-2.5-pixel translation of the same sinusoid pattern from
the previous chunk:

```
plain LK:    recovered (u, v) = (+3.995, -2.829)
pyramid LK:  recovered (u, v) = (+4.001, -2.495)
```

Plain LK overshoots `v` by 13% (the OFC linearization breaking
down). Pyramidal LK is accurate to about a part in a thousand on
both components. And the level-by-level estimates trace out a clean
coarse-to-fine convergence:

```
level 4 (  8 ×   8): mean (u, v) = (+0.000, +0.000)   # pattern too smoothed
level 3 ( 16 ×  16): mean (u, v) = (+0.018, -0.014)   # still featureless
level 2 ( 32 ×  32): mean (u, v) = (+1.508, -0.893)   # texture appears
level 1 ( 64 ×  64): mean (u, v) = (+2.635, -1.308)   # refining
level 0 (128 × 128): mean (u, v) = (+5.010, -2.570)   # final, interior is +4.0, -2.5
```

The "level 0 mean" is over the full image including the borders
where the warp's edge-replication contaminates things. The mean
over the interior (the number reported as "pyramid LK" above) is
within 0.01 of truth on both axes.

## Cost

Building two Gaussian pyramids at 4 levels adds about 33% to the
single-level cost (geometric series `1 + 1/4 + 1/16 + 1/64 + 1/256
≈ 4/3`). Running LK at each level adds the same constant work.
Each warp at level k is `O(H_k · W_k)`. Three iterations per level
× 5 levels = 15 LK + warp passes total instead of 1 — so the
pyramid version is roughly 20× the cost of plain LK, but it works
on motions plain LK can't touch at all. Worth it.

## What this doesn't fix

- **Motion boundaries.** If two objects move differently and meet,
  the Gaussian-windowed LK averages over the boundary and gets
  something in between. The classical fix is robust losses
  (Lorentzian, Charbonnier, …) that downweight residuals from the
  wrong-motion side of the window.
- **Aperture problem.** A straight-edge feature inside a window
  still only constrains motion in one direction, no matter how many
  pyramid levels I have. The confidence map (structure-tensor
  determinant) flags this as a near-zero eigenvalue.
- **Brightness constancy** itself. LK assumes a moving pixel keeps
  its intensity, which is wrong across occlusions, specular
  highlights, and lighting changes. Pyramid LK propagates whatever
  bias brightness-non-constancy introduces through every level.

The interesting follow-up beyond all of these is *dense* optical
flow — Horn & Schunck's global-smoothness formulation that solves
one big system instead of a per-pixel small one. That's a chunk on
its own; pyramidal LK is where I stop for now.

## References

- Bouguet, J.-Y. (2001). *Pyramidal implementation of the affine
  Lucas-Kanade feature tracker*. Intel Technical Report. The
  canonical reference for this coarse-to-fine recipe.
- Bergen, J. R., Anandan, P., Hanna, K. J., & Hingorani, R.
  (1992). *Hierarchical model-based motion estimation*. ECCV. The
  earlier pyramid-based motion-estimation paper that Bouguet's note
  formalizes for the LK case.
- Baker, S., & Matthews, I. (2004). *Lucas-Kanade 20 years on: a
  unifying framework*. IJCV. Connects the warp-and-refine loop here
  to the inverse-compositional family and the general
  Gauss-Newton-on-images literature.
