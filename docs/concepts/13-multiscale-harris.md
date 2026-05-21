# Multi-scale Harris

The single-scale Harris detector from earlier has a built-in
assumption: a "corner" is something that fits in its 3×3 (or
slightly bigger) window. That's fine for sharp small geometry, but
useless for a soft curved corner that's 12 pixels across. The
structure tensor inside a 3×3 box on a slowly-curving boundary
looks more like an edge than a corner — both eigenvalues come out
small, the Harris response stays low, no detection.

The fix is the obvious one: run Harris on multiple resolutions.

## The pyramid trick

A Gaussian pyramid (from the previous chunk) gives me the same
image at progressively halved resolution. A 3×3 window at level 2
corresponds to a 12-pixel window in the original image — same
detector, different scale.

So `multiscale_harris_corners` runs Harris on every level of a
Gaussian pyramid, and re-maps each detection back into the original
image's coordinates by multiplying by `2^level`.

```julia
multiscale_harris_corners(img; levels=3, sigma, k, threshold, min_distance)
    -> Vector{(row=Int, col=Int, level=Int)}
```

Each detection is tagged with the level it fired on. `level=0` is
the original image; `level=2` is a corner detected at quarter
resolution (so its 3×3 detection corresponds to a ~4-pixel feature
in the original).

## Measured on a mixed-scale test image

I built a test image with two competing features:

1. A sharp 20×20 rectangle — every corner fits in a 3×3 window.
2. A larger rectangle softened by repeated local averaging — the
   corners are 4–6 pixels wide.

Single-scale Harris on this image found 5 detections — the four
sharp corners plus a tiny dot I added as noise. Multi-scale found
**20 detections across 3 levels**:

```
level 0 (≈1px features): 5
level 1 (≈2px features): 8
level 2 (≈4px features): 7
```

The level-1 and level-2 detections fall along the soft rectangle's
corners — features that simply don't trigger at the original
resolution. The single-scale picture would say "no corner there";
the multi-scale picture correctly identifies them but with a coarse
spatial resolution (off by 2–4 pixels because that's the
quantization at the higher levels).

## Why I use the *Gaussian* pyramid (not the Laplacian)

The Gaussian pyramid carries low-frequency content at every level —
it's a smoothed-and-resampled version of the original. That's
exactly what Harris wants: a clean local intensity field whose
structure tensor is meaningful.

The Laplacian pyramid carries *band-pass* detail, which is sparse
and signed. Running Harris on a Laplacian level would give a
detector that fires on contrast peaks at that band — useful for
some applications (a flavor of DoG keypoint detection), but
different from what classical Harris does.

## What this connects to

A multi-scale corner detector is the corner-flavored sibling of
DoG-based keypoint detection (the heart of SIFT). The general
pattern in both:

1. Build a pyramid (or scale-space).
2. Run a per-pixel response at each scale.
3. Pick local maxima in `(row, col, scale)` — a 3D NMS.
4. Optionally refine to sub-pixel / sub-octave precision.

My implementation does the first three (with the 3D NMS only
implicit — I rely on Harris's own NMS within each level). Adding
explicit cross-scale NMS (don't fire at level 1 if level 0 already
fired at this position) would be a one-screen change; I'll do it
when I actually need scale-merged keypoints rather than the union
of all detections.

## Performance budget

Each level is `O(H · W / 4^level)` pixels, and Harris is `O(1)` per
pixel modulo the smoothing convolutions. So the total cost across
`L` levels is bounded by:

```
H · W · (1 + 1/4 + 1/16 + ... + 1/4^L)  ≈  4/3 · H · W
```

i.e. it's only ~33% more work than a single-scale pass on the
original. Pyramids are cheap, multi-scale detection is cheap, and
the additional detections are essentially free.

## What's still ad hoc

- The `threshold` parameter is fraction-of-max at each level
  independently. That's robust to absolute response strength
  drifting across levels, but doesn't enforce a global "this is
  a stronger corner than that one across scales" ordering. For
  proper scale-space NMS I'd need a normalized response (the
  Lindeberg `σ²` trick).
- No sub-pixel refinement. The level-2 detections are quantized to
  multiples of 4 pixels in the original image.
- The level tag is the coarsest information you can have about
  scale. SIFT's scale-space carries a continuous σ value per
  keypoint. That'd be the next thing to add if I want real
  keypoints I can match across views.
