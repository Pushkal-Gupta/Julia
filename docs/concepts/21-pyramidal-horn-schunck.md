# Pyramidal Horn-Schunck

A short chunk — the coarse-to-fine driver I wrote for LK is
*algorithm-agnostic*. Swap the inner solver and you get the same
big-motion fix for whatever flow algorithm you plug in. So this is
mostly an exercise in noticing that the same code applies twice,
and writing down what the inner solver swap does and doesn't
change.

## What's new

One function: `horn_schunck_pyramid(img1, img2; levels, alpha,
iters_per_level, pad)`. Implementation is structurally identical to
`lucas_kanade_pyramid` — build Gaussian pyramids of both frames,
walk coarse-to-fine, at each level warp frame 2 by the current
estimate and call the inner flow solver on the residual, accumulate,
upsample (with magnitude doubling) for the next finer level. The
only line that changes is

```diff
- residual = lucas_kanade(I1, warped; window_size, sigma, ...)
+ residual = horn_schunck(I1, warped; alpha, iterations, ...)
```

`iters_per_level` defaults to 100 instead of `horn_schunck`'s
standalone default of 200. The residual at each level is small (the
warp has already cancelled out the bulk of the motion), so each
level converges faster than a from-scratch HS solve.

## Headline numbers

`examples/23_pyramidal_horn_schunck.jl` on a (4, -2)-pixel
translation:

```
plain HS    : recovered (u, v) = (+4.279, -2.246)
pyramid HS  : recovered (u, v) = (+3.997, -1.956)
```

Plain HS overshoots by ~7% on both axes (OFC linearization
breaking down at this motion magnitude). Pyramid HS is within 2.2%
on both. The motion sweep shows the algorithm holding up across the
range:

```
shift = 1.0  →  recovered u = +1.023
shift = 2.0  →  recovered u = +2.005
shift = 4.0  →  recovered u = +4.003
shift = 6.0  →  recovered u = +5.972
```

— a 6-pixel motion recovered to 0.5% accuracy. The warp residual
`|warp(I₂, recovered_flow) − I₁|` averages `0.0008` on a `[0, 1]`
image, comparable to what pyramidal LK achieved in the previous
chunk.

## Aliasing — the surprise I caught in this chunk

My first version of the pyramidal-HS demo used the same 0.04
cycles/pixel sinusoid pattern I'd been using for everything in the
flow thread. With 4 pyramid levels and a 128×128 input, pyramid HS
diverged — recovering ~20 pixels for a true 2.5-pixel motion. LK
pyramid on the same input is fine.

The culprit: the Gaussian pyramid's 5-tap `[1, 4, 6, 4, 1] / 16`
filter isn't a brick-wall low-pass. A 0.04-cycle/pixel sinusoid
becomes 0.32 cycles/pixel after three downsamplings — that's past
Nyquist for a 5-tap filter, so the coarsest-level image has
aliasing artifacts. LK is local — its 2×2 solve in an aliased
region returns zero or near-zero because the gradient structure is
incoherent. HS is global — its smoothness term propagates whatever
spurious flow appears in the aliased region across the whole field,
and the bogus estimate gets doubled at each upsampling, so the
final flow is wildly off.

Two ways to fix:

1. Pick test patterns whose spatial frequency stays inside the
   pyramid filter's safe band. 0.025 cycles/pixel at the original
   becomes 0.2 at level 3 — comfortably below 0.25 where the 5-tap
   filter starts attenuating. This is what the studio script does.
2. Use fewer pyramid levels. At `levels = 2`, even the 0.04-cycle
   pattern stays inside the safe band at all levels.

Neither is a bug in the HS pyramid driver; it's a general property
of pyramidal flow methods on high-frequency content. The LK
pyramid has the same vulnerability in principle but its local solve
masks the symptom. Worth knowing because real video content rarely
has the kind of monochromatic high-frequency content that
synthetic test patterns do.

The default `levels = 4` works for natural images and the 0.025
test pattern. For 0.04-cycle synthetic content, use `levels = 2` or
`levels = 3`.

## What this completes and what's left

This closes out the optical-flow thread for now. The four functions
(`lucas_kanade`, `lucas_kanade_pyramid`, `horn_schunck`,
`horn_schunck_pyramid`) plus `warp_bilinear` and `flow_to_rgb`
cover the standard classical-flow toolkit: local vs global
regularization, small motions vs big motions, sparse vs dense
output. Logical follow-ups not built here:

- *Combined local + global formulation* (Bruhn et al., 2005) — one
  algorithm that's LK in textured regions and HS in flat ones.
  Standard modern variational baseline.
- *Robust losses* — Charbonnier or Lorentzian in place of squared
  residuals, for occlusion-robust flow.
- *Median filtering between iterations* — the "Sun et al." trick
  that improves practical accuracy on real video at almost no
  algorithmic cost.

The first of those would be a satisfying capstone but isn't on the
critical path.

## References

- Bouguet, J.-Y. (2001). *Pyramidal implementation of the affine
  Lucas-Kanade feature tracker*. Intel Technical Report. The
  coarse-to-fine driver applied to LK; the same driver applies to
  HS here.
- Horn, B. K. P., & Schunck, B. G. (1981). *Determining optical
  flow*. Artificial Intelligence (journal), 17(1-3), 185-203. The
  inner solver.
- Burt, P. J., & Adelson, E. H. (1983). *The Laplacian pyramid as a
  compact image code*. IEEE TOC. The pyramid used here, with the
  same 5-tap filter that explains the aliasing observation above.
