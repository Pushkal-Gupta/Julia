# Mid-level features: corners, lines, components, matches

After Canny I have a binary edge map, which is already useful but
still just pixels. The next jump is to *semantic* primitives — corner
points, line parameters, segmented regions, matched template
positions. Each of these answers a specific question about the image
that "list of edge pixels" can't.

I built four:

- Harris — "where are the corners?"
- Hough — "are there straight lines, and where do they live in
  parameter space?"
- Connected components — "how many separate things are in this edge
  map, and how big is each?"
- Normalized cross-correlation — "where does this template appear in
  the image, regardless of lighting?"

All four live in `src/features.jl` as the `Features` submodule.

## Harris

The Harris corner detector is built on top of the gradient I already
have. The trick is a 2×2 matrix called the structure tensor:

```
M = [ Σ Ix²   Σ Ix·Iy ]
    [ Σ Ix·Iy Σ Iy²   ]
```

where the sum is over a small window (Gaussian-weighted, in my
implementation). Two eigenvalues come out:

- Both small → flat region (no edge, no corner).
- One small, one big → edge (gradient is strong in one direction).
- Both big → corner (gradient is strong in *two* directions).

Computing eigenvalues for every pixel is expensive, so Harris
proposed a shortcut: `R = det(M) - k·trace(M)²`. This is large and
positive when both eigenvalues are big, large and negative when only
one is (i.e. it actively disprefers edges), and near zero on flat
regions. `k = 0.04` is the standard choice.

My implementation:

1. Sobel gradients (already there).
2. Pointwise `Ix²`, `Iy²`, `Ix·Iy`.
3. Gaussian-smooth each of those (separable, so cheap).
4. Combine: `R = Sxx·Syy - Sxy² - k·(Sxx + Syy)²`.
5. Pick local maxima above a fraction of `max(R)`, with
   Chebyshev-distance NMS so I don't get clusters of detections at
   the same corner.

On the test image (two rectangles and a triangle, 11 geometric
corners total) Harris finds 11 corners — exactly the right count.

## Hough lines

Hough is one of those algorithms that looks dense the first time and
obvious the second. The idea: every point `(x, y)` lies on infinitely
many lines, parameterized as `ρ = x·cos(θ) + y·sin(θ)`. So for each
edge pixel, sweep `θ` over `[0, π)` and "vote" in an accumulator
indexed by `(θ_bin, ρ_bin)`.

Lines in the image map to *peaks* in the accumulator, because every
edge pixel on the same image-line votes for the same `(θ, ρ)`
parameters. Find the peaks, get the lines.

My parameterization: rows are `y`, columns are `x`, so for an edge
pixel at `(i, j)`, the vote at `θ` is for `ρ = j·cos(θ) + i·sin(θ)`.
That puts the parameter-space origin at the image's `(1, 1)` corner.

Two subtle things worth knowing:

1. The accumulator has two "shapes". I bin `θ` into `n_theta`
   uniform bins over `[0, π)` (default 180), and `ρ` into
   `2·⌈ρ_max⌉ + 1` integer bins where `ρ_max = √(H² + W²)`. So the
   accumulator for a 128×128 image is `180 × 363`. That's not the
   same as the image size, which is why my studio script saves the
   accumulator separately rather than dropping it into a montage.

2. Vertical lines have *two* equivalent peaks in `[0, π)`: at
   `θ ≈ 0` (ρ = column) and at `θ ≈ π` (ρ = −column). The two
   describe the same line. I break ties in `hough_peaks` to prefer
   smaller θ, so the output is predictable.

On the test image (4 + 4 + 3 = 11 geometric sides), Hough finds 10
peaks. The miss is usually a short edge whose votes spread thinly
enough that it falls below my 40%-of-max threshold.

## Connected components

This is the simplest of the four — it just labels disjoint regions
of `true` pixels in a binary mask. The implementation is one pass of
stack-based DFS: for each unvisited true pixel, allocate a new label,
push it on a stack, and recursively recruit any unvisited true
neighbor (4- or 8-connected, depending on `connectivity`).

Total work is `O(H·W)` — every pixel is pushed onto the stack at
most once and popped at most once.

Two uses in a real pipeline:

- Filter spurious tiny edge segments. After Canny on the noise-lab
  image, I get ~19 components, but only 4 of them are bigger than 20
  pixels. The rest are noise that survived hysteresis. Drop them by
  filtering `component_sizes < min_size`.
- Count objects. If the input was a binary mask of "thing or not",
  connected_components gives you a labeled image and an object
  count for free.

## Normalized cross-correlation

NCC is the brightness-and-contrast-invariant version of template
matching. For each output position `(i, j)`:

```
NCC(i, j) = Σ (I[i+u, j+v] − μ_window) · (T[u, v] − μ_T)
            ────────────────────────────────────────────
                       σ_window · σ_T
```

The means and standard deviations are computed over the template
window. Subtracting the means removes any constant offset; dividing
by the standard deviations removes any contrast scaling. So a patch
that's "the same shape but half as bright" still scores 1.0.

Range is `[-1, 1]`. +1 is a perfect match, 0 is uncorrelated, -1
would be a perfect inverse (every bright pixel in the template lined
up with a dark pixel in the image).

I verified this on a test image with 6 disks at varying intensities,
where I additionally halved the brightness of the right half. NCC
found all 6:

```
matches above 0.75: 6
  #1  top-left (92, 17)  score=1.000
  #2  top-left (32, 97)  score=1.000
  #3  top-left (82, 72)  score=1.000
  #4  top-left (52, 32)  score=1.000
  #5  top-left (17, 17)  score=1.000
  #6  top-left (17, 62)  score=0.974
```

Five scored exactly 1.000. The sixth disk (the one at score 0.974)
sat right on the boundary of the brightness change — half its pixels
were at full intensity, half were darkened. The disk is no longer
uniform, which is what the template assumes, so the correlation
drops slightly. This is exactly what NCC should report: "still a
very strong match, but not perfect, because the shape isn't
identical anymore".

## Performance notes

Some rough complexity numbers, useful for sizing real-world inputs:

| Operation                    | Per-pixel work       | 128×128 wall-clock  |
|------------------------------|----------------------|---------------------|
| Harris response              | `O(k²)` smoothing    | a few ms            |
| Hough lines                  | `O(E · n_theta)`     | ~10 ms for E≈1000   |
| Connected components         | `O(1)`               | <1 ms               |
| NCC (`th × tw` template)     | `O(th · tw)`         | ~50 ms for 17×17    |

None of these are tuned. The NCC inner loop in particular could be
sped up substantially by using FFT-based correlation for the cross
term and box-filter accumulation for the mean/variance — that lands
in the performance lab later, not now.
