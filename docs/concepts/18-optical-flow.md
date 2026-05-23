# Optical flow (Lucas-Kanade)

Given two consecutive frames of a sequence, optical flow asks: for
every pixel, how much did the *content under it* move between
frames? The output is a 2D vector field — one `(u, v)` per pixel,
where `u` is horizontal displacement and `v` is vertical. This is
the first time in the repo where the output isn't a single
`Matrix{Float64}` of intensities; it's a vector field, and so it
needs a new kind of visualization too.

Lucas-Kanade (1981) is the simplest algorithm that gives a useful
answer, and it slots straight on top of the gradient code I built
for edge detection. The same structure tensor that Harris uses for
corners shows up here in the LK normal equations, which makes sense
once you think about it: LK wants pixels that constrain motion in
two independent directions, which is exactly the definition of a
corner.

## The setup: brightness constancy

The starting assumption is that a small bit of image content keeps
its intensity as it moves:

```
I(x, y, t) = I(x + u, y + v, t + 1)
```

Taylor-expand the right-hand side to first order:

```
I(x, y, t+1) + Iₓ · u + I_y · v ≈ I(x, y, t)
```

Rearrange:

```
Iₓ · u + I_y · v + I_t = 0           (the optical-flow constraint)
```

where `I_t = I(x, y, t+1) − I(x, y, t)`. One equation, two
unknowns. Underdetermined.

This linearization is the source of LK's biggest limitation: it's
only accurate when the actual motion is small enough that the
first-order Taylor approximation holds. For displacements
significantly less than a pixel the equation is fine; for
displacements past about one pixel the linearization breaks down
and the recovered flow becomes biased. The classical fix is
*pyramidal LK*: run LK at a coarse pyramid level (where large
motions look small), then refine at successively finer levels. I
haven't implemented that yet; it's the natural next chunk.

## Spatial coherence to the rescue

The second LK assumption: the motion is roughly constant in a
small window around each pixel. So instead of one equation per
pixel I have *N* equations per pixel — one for every neighbor in
the window — but still only two unknowns:

```
[ Iₓ(p₁)   I_y(p₁) ]        [-I_t(p₁)]
[ Iₓ(p₂)   I_y(p₂) ] [u]    [-I_t(p₂)]
[   ...      ...   ] [v]  = [   ...  ]
[ Iₓ(p_N)  I_y(p_N)]        [-I_t(p_N)]
```

That's overdetermined. The least-squares solution is the normal
equations:

```
[ Σ Iₓ²    Σ Iₓ·I_y ] [u]   [-Σ Iₓ·I_t]
[ Σ Iₓ·I_y Σ I_y²   ] [v] = [-Σ I_y·I_t]
```

The 2×2 matrix on the left is the *structure tensor*. The same
matrix Harris evaluates per pixel to find corners. That's not a
coincidence: Harris fires where the structure tensor has two large
eigenvalues, which is exactly the condition for LK to be
well-conditioned. Flat regions (one or both eigenvalues small) are
no-texture failures for both algorithms.

## The implementation

In `Flow.lucas_kanade`:

1. Compute the spatial gradients `Ix`, `Iy` from the *average* of
   the two frames (more symmetric and a touch less noisy than
   picking one). Divide by 8 to normalize Sobel into a true
   gradient estimate — without that factor, `Σ Iₓ²` and `Σ Iₓ·I_t`
   pick up different powers of 8 and the recovered `u` is off by a
   constant (~1/8 instead of 1).
2. Compute the temporal gradient `It = frame2 − frame1`.
3. Build the five pointwise products `Iₓ²`, `I_y²`, `Iₓ·I_y`,
   `Iₓ·I_t`, `I_y·I_t`.
4. Aggregate each one with a separable Gaussian window — same
   smoothing trick from the very first chunk of this repo. Window
   `σ = 2`, size 15, by default.
5. Per pixel, solve the 2×2 system via Cramer's rule. If the
   determinant is below a threshold (`1e-6` by default), set `u`,
   `v` to zero — that pixel has no texture to constrain the motion.

The whole solve runs in one pass through the image: O(H·W·k) for
window size `k`, which is the same complexity as Harris.

## Why the Sobel-by-8 detail matters

The first version of this submodule used raw Sobel and recovered
`u = 0.125` for a one-pixel shift. Took me a while to track down.
The reason: Sobel's positive coefficients sum to 8, so applied to a
ramp `α·x` it produces `8α` instead of `α`. That factor cancels
between `Σ Iₓ²` and `Σ Iₓ·I_t` if both sides have the same number
of Sobels — but `I_t` is a plain subtraction (no Sobel), so the
left-hand `Σ Iₓ²` picks up a factor of 64 while the right-hand
`Σ Iₓ·I_t` only gets 8. The ratio leaves a stray `1/8` in `u`.

Lessons I keep paying for: when a kernel says "Sobel" without a
divisor, it's *responses*, not gradients. Dividing by the
absolute-coefficient sum gives me gradients.

## Visualizing a vector field

The convention I went with — and that almost every paper uses — is
the colour-wheel encoding:

- **Hue** = flow direction, mapped from `atan2(v, u)` (in
  `(-π, π]`) onto the `[0, 1)` hue range with the wrap point shifted
  to the zero-flow direction.
- **Saturation** = flow magnitude, normalized to whatever maximum
  is appropriate for the scene.
- **Value** = 1.

So no flow → white, large flow east → red, large flow south → green
(in my colour wheel), etc. The colour wheel image in
`artifacts/20_optical_flow/06_color_wheel.ppm` is the legend for
reading the flow images.

I added `Viz.hsv_to_rgb(h, s, v)` and `Viz.flow_to_rgb(u, v)` for
this. They work on plain `Float64` numbers, so no `ColorTypes`
dependency creeps into the visualization path.

## Results on a synthetic test

`examples/20_optical_flow.jl` demonstrates four things:

1. **Pure translation.** Frame B is frame A shifted by `(0.7, 0.3)`
   pixels (continuous shift, sampled by re-evaluating the analytic
   pattern at the offset position). LK recovers
   `(0.719, 0.310)` — about 3% error.
2. **Pure rotation.** Frame B is frame A rotated 1° around the
   image centre. At the four edge midpoints the true flows are
   roughly `(±0.78, 0)` and `(0, ±0.76)`; LK recovers them to within
   5%. The rotational structure shows up clearly in the colour-wheel
   visualization as a swirl.
3. **Confidence map.** A textured patch is sitting in a sea of
   uniform grey, and only the patch is moving. The structure-tensor
   determinant inside the patch is ~5 orders of magnitude larger
   than outside it. In a real application I'd use that to mask out
   the unreliable flow estimates in flat regions.
4. **The colour wheel** as a legend.

## Limitations I haven't fixed

- **The OFC linearization breaks down past ~1 pixel.** A 2-pixel
  shift recovers `u ≈ 2.3` on the same pattern — visible bias.
  Pyramidal LK is the standard fix: run LK at a coarse pyramid
  level (where 2 pixels of motion look like 0.25 pixels), upsample
  the result, refine. `Pyramids` is already there; the
  coarse-to-fine driver is a follow-up chunk.
- **No iterative refinement (warping).** Once I have an estimate I
  could warp frame 2 by `(-u, -v)`, re-run LK on the warped pair,
  and accumulate the residual. Each iteration peels off another
  order of approximation error. Cheap to add but not in this chunk.
- **Robust losses.** Real video has occlusion edges where
  brightness constancy fails completely. Robust losses (Huber,
  Charbonnier, Lorentzian) reduce the influence of those outliers
  on the per-pixel solve.
- **Aperture problem.** A pure straight edge inside the window
  gives only one independent equation — the component of motion
  along the edge is unrecoverable from local information alone.
  This is fundamental, not a bug, but the confidence map catches it
  (one eigenvalue of the structure tensor is zero, so the
  determinant is zero too).

## References

- Lucas, B. D., & Kanade, T. (1981). *An iterative image
  registration technique with an application to stereo vision*.
  IJCAI. The original LK paper.
- Horn, B. K. P., & Schunck, B. G. (1981). *Determining optical
  flow*. Artificial Intelligence (journal), 17(1-3), 185-203. The
  contemporary global-smoothness alternative to LK; the 2×2×2
  finite-difference scheme it introduces is still the standard
  discrete-derivative recipe in this area.
- Bouguet, J.-Y. (2001). *Pyramidal implementation of the affine
  Lucas-Kanade feature tracker*. Intel Tech. Report. The standard
  reference for the coarse-to-fine variant I haven't built yet.
- Baker, S., & Matthews, I. (2004). *Lucas-Kanade 20 years on: a
  unifying framework*. IJCV. A modern survey that covers
  iterative warping, robust losses, and the inverse-compositional
  form.
