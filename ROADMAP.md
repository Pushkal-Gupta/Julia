# What I'm building

This is the rough plan I keep nearby while I work. It's a living file —
when something lands I update the section, and when I learn something
worth recording mid-build I jot it down here. Nothing is on a schedule.

## Where I'm at

Engine, depth, edges, Canny, noise study, and the features layer are
done. Next is the performance lab.

Roughly:

- Convolution engine — done
- Convolution depth (1D, separable, factor_separable) — done
- Edge operators (first- and second-order) — done
- Canny pipeline from scratch — done
- Noise + preprocessing study with precision/recall metrics — done
- Features (Harris, Hough, connected components, template match) — done
- Performance lab (BenchmarkTools, inline conv, FFT crossover) — done
- Image pyramids and scale-space basics — done
- Anisotropic diffusion (Perona–Malik) — done
- Real-image I/O via ImageIO + a natural-image pipeline demo — done
- Multi-scale Harris on the Gaussian pyramid — done
- Laplacian-pyramid image blending — done
- Linear-algebra view of convolution (Toeplitz / circulant matrices,
  DFT diagonalization) — done
- Next: tiny ray tracer for Julia practice outside imaging, then
  differentiable filters via ForwardDiff
- Performance lab (benchmarks, FFT crossover, `@code_warntype`) — folded
  in as needed; a real pass once Canny is solid
- Features (Harris, Hough, connected components, template matching) —
  later
- Interactive playground (Pluto or a small Makie app) — later
- Stretch: linear-algebra view of convolution, image diffusion, a tiny
  ray tracer, differentiable filters

## The convolution engine

What I have:

- A package at `src/ImageLab.jl` with submodules for synthetic images,
  kernels, padding, convolution, visualization, and Netpbm I/O.
- Naive `correlate2d` / `convolve2d` with in-place variants.
- All six padding modes (`:zero`, `:replicate`, `:reflect`, `:symmetric`,
  `:circular`, `:valid`) for both 2D arrays and 1D vectors.
- A kernel zoo: box, Gaussian, Sobel, Prewitt, Scharr, Roberts (2D
  and separable forms), Laplacian (4- and 8-connected), LoG, sharpen.
- Synthetic image generators: checkerboard, square, circle, impulse,
  ramp, lines, Gaussian and salt-and-pepper noise.
- A pure-Julia PGM/PPM reader and writer — no external dependencies.

This was the boring-but-important work. Most of the lessons came from
getting padding right — the "interior is copied, exterior is filled per
mode" pattern is a clean separation that lets the convolution inner
loop stay trivial.

## Convolution depth

What I added on top of the engine:

- `correlate1d` / `convolve1d` for both vectors and matrices, with an
  `axis = :horizontal | :vertical` keyword on the matrix variant.
- `separable_correlate2d` that composes two 1D passes.
- Pre-factored 1D kernels: `gaussian1d`, `box1d`, plus
  `sobel_*_separable`, `prewitt_*_separable`, `scharr_*_separable`.
- `factor_separable(K)` that uses SVD to detect rank-1 kernels and
  return their factors. Useful when I don't already know whether a
  kernel separates.
- A small `Viz` submodule for normalization and montage tiling.

The lesson I keep coming back to: separable convolution is a clean
applied-linear-algebra moment — a 2D kernel is "really" a 1D kernel
twice if and only if its SVD has one nonzero singular value. The
benchmark in `examples/03_separable_vs_naive.jl` shows the theoretical
`k/2` speedup ratio almost exactly at `k=21` (10.8× measured, 10.5×
predicted).

## Edge detection

What I built:

- First-order operators: `gradient(img, op)` dispatches on `:sobel`,
  `:prewitt`, `:scharr`, `:roberts`. The first three go through the
  separable engine; Roberts is a hand-written 2×2 inner loop because
  forcing a 2×2 through the odd-only 2D engine would be ugly.
- `gradient_magnitude(gx, gy)` (Euclidean) and `gradient_direction`
  (`atan(gy, gx)` in `(-π, π]`).
- `quantize_direction(θ)` maps angles to 4 sectors (horizontal,
  NE-diag, vertical, NW-diag) for use in non-maximum suppression.
- Second-order: `log_filter` and `dog_filter`. Verified empirically
  that DoG with `σ₂/σ₁ ≈ 1.6` looks visually almost identical to LoG
  at `σ ≈ σ₁` — that's the Marr approximation, and it's much cheaper
  because both Gaussians are separable.
- `zero_crossings(img; min_diff)` for second-derivative edges.
- Threshold helpers: `threshold_mask` (absolute) and
  `percentile_threshold` (robust to brightness changes).

Two studios produce visual artifacts:

- `examples/04_edge_operator_studio.jl` — first-order operators side
  by side on a single synthetic input. 3×3 montage with input, Sobel
  `gx` / `gy`, magnitudes for all four operators, masked direction
  map, and a thresholded edge mask.
- `examples/05_log_dog_zero_crossings.jl` — second-order operators
  with a σ sweep, DoG at two ratios, zero-crossings for both LoG and
  DoG, plus a `LoG − DoG` difference tile that shows how close the
  two operators are at the classical 1:1.6 ratio.

Concept notes:

- `docs/concepts/04-gradient-magnitude-and-direction.md` —
  comparing the four first-order operators, how to visualize
  direction on grayscale.
- `docs/concepts/05-second-order-edges.md` — Laplacian → LoG → DoG →
  zero-crossings, and why the field eventually moved on from
  second-order methods toward Canny.

## Canny from scratch

What I built:

- `nonmaximum_suppression(mag, θ)` — thins fat ridges into one-pixel
  edges by suppressing any pixel that isn't a strict local maximum
  along the gradient direction. Uses the 4-sector `quantize_direction`
  for neighbor lookup.
- `double_threshold(mag; low, high, relative=true)` — partitions
  pixels into strong / weak / dropped. Thresholds default to relative
  fractions of `maximum(mag)` so they survive brightness changes.
- `hysteresis(strong, weak)` — stack-based DFS that recruits any
  weak pixel 8-connected to a strong pixel (transitively). O(H·W).
- `canny_stages(img; sigma, low, high, pad)` — runs all four stages
  and returns a `CannyStages` struct with every intermediate.
- `canny(img; ...)` — convenience wrapper that returns just the final
  `BitMatrix` edge map.

Two studios:

- `examples/06_canny_pipeline.jl` — every intermediate on a single
  input, tiled into a 3×3 montage so I can read the pipeline flow
  top-left to bottom-right.
- `examples/07_canny_parameter_sweep.jl` — same input, nine different
  parameter settings (3 σ × 3 threshold pairs). The interesting
  observation: at large σ the thresholds barely matter, because by
  the time the noise is gone every NMS-surviving pixel is a "real"
  edge with a high magnitude. At small σ the thresholds do all the
  work.

Concept note: `docs/concepts/06-canny.md`.

## Noise + preprocessing

What I built:

- `Filters` module — `median_filter` (rank filter, kills salt-and-pepper),
  `bilateral_filter` (edge-preserving, spatial × intensity Gaussian
  weighting), and `binary_dilate` (8-connected morphological
  dilation, used to give edge matching a pixel-level tolerance).
- `Metrics` module — `edge_match_stats(predicted, gt; tolerance)`
  returns precision / recall / F1; `iou_score` for a single-number
  summary.
- `examples/08_noise_lab.jl` runs eight smoothers (none, box 3×3 /
  5×5, Gaussian σ=1 / σ=2, median 3×3 / 5×5, bilateral) against
  two noise types (Gaussian and salt-and-pepper), scoring each
  Canny output against the noise-free Canny reference.

Headline numbers (F1 at 1-pixel tolerance):

- Gaussian noise: every smoother lands above 0.98. Bilateral nudges
  out the field at 0.995.
- Salt-and-pepper: dramatic. F1 = 0.21 with no smoother, 0.989 with
  median 5×5. Bilateral fails completely (0.21) because it
  preserves the salt specks as edges.

Concept note: `docs/concepts/07-noise-and-preprocessing.md`.

## Performance lab

What I built:

- `correlate2d_inline` — no padding copy. Splits the output into an
  interior region (in-bounds, tight inner loop) and a border region
  (bounds-aware sampling via the padding rule).
- `fft_correlate2d` and `fft_convolve2d` — FFT-based, via FFTW.
  Supports `:zero` and `:circular` padding (other modes don't combine
  cleanly with the FFT). Uses `rfft` for real-valued speedups.
- `examples/11_performance_lab.jl` — `BenchmarkTools` measurements of
  all four implementations on a 384×384 image at kernel sizes 3..31.

Headline measurements (best-of-8, on a 384×384 image):

```
k    naive (ms)   inline (ms)  separable    fft (ms)
3        1.20        1.27        1.23        5.91
9        4.30        4.65        1.85        1.69
15      16.58       17.05        3.01        6.04
21      39.61       40.61        3.59        5.85
31      98.12       93.71        4.88        3.31
```

What surprised me: the inline version is essentially a wash with
naive. The padding copy isn't the bottleneck — the inner loop
arithmetic is. Separable matches the theoretical `k/2` speedup and
slightly *beats* it at k=31 (probably L2 cache effects from the
narrower working set per 1D pass). FFT crossover with separable is
around k=25 on this image size.

A `@code_warntype` audit on the hot paths (`correlate2d`,
`correlate2d_inline`, `_correlate_into!`, `gradient_magnitude`,
`nonmaximum_suppression`, `fft_correlate2d`) reports zero `::Any` and
zero `::Union` boxes. Every inner loop is specializable on the
concrete element type.

Concept note: `docs/concepts/09-performance.md`.

## Features

What I built:

- `harris_response(img; sigma, k)` — the 2×2 structure tensor smoothed
  with a separable Gaussian; response = `det(M) - k·trace(M)²`. The
  classic Harris.
- `harris_corners(img; threshold, min_distance)` — local-max picking
  with Chebyshev NMS so I don't get clusters of detections per corner.
  On the geometric test image (4 + 4 + 3 corners): finds 11 corners,
  exactly the right count.
- `hough_lines(edges; n_theta, rho_step)` — accumulator over `(θ, ρ)`
  with `ρ = j·cos(θ) + i·sin(θ)`. Returns a `HoughAccumulator` struct
  with the counts and bin centers.
- `hough_peaks(acc; threshold, min_distance_*, max_peaks)` — local max
  picking on the accumulator, tie-broken to prefer smaller θ so
  vertical-line output is predictable (without the tie-break, every
  vertical line has two equivalent peaks at θ ≈ 0 and θ ≈ π).
- `connected_components(mask; connectivity=4|8)` — stack-based DFS
  labeling. Pairs with `component_sizes(labels, n)` for size-based
  filtering of spurious tiny components.
- `normalized_cross_correlation(img, template)` — brightness- and
  contrast-invariant template matching. Output in `[-1, 1]`. Verified
  on a 6-disk test image where the right half is darkened — NCC still
  scores all 6 disks at 0.974 or higher.
- `ncc_peaks(ncc; threshold, min_distance)` — pick top matches.

Two studios:

- `examples/09_corners_and_lines.jl` — Harris + Hough on a
  rectangles-and-triangle scene. 3×2 montage with input, Harris
  response, corners overlay, Canny edges, detected lines on black,
  and lines overlaid on the input.
- `examples/10_components_and_templates.jl` — two halves on two
  different inputs. CC on a multi-shape Canny output (19 components,
  with a heavy tail of small ones — exactly the filtering target),
  plus NCC on a 6-disk varying-intensity scene.

Drawing helpers in `Viz`:

- `draw_line!(img, y0, x0, y1, x1; value)` — Bresenham line raster.
- `mark_points!(img, points; size, value)` — stamps for feature overlays.
- `label_to_gray(labels)` — golden-ratio scrambled grayscale so
  adjacent label IDs come out as distant intensities.

Concept note: `docs/concepts/08-features.md`.

## Interactive playground

Pluto notebook with sliders for σ, kernel size, thresholds, denoiser
choice, operator choice — one knob per question I keep asking the
static example scripts. Pluto is the path of least friction on Julia
1.11; if I find it slow I'll try a Makie window instead.

## Anisotropic diffusion

What I built:

- `perona_malik(img; iterations, K, lambda, mode)` in `Filters`.
  Non-linear PDE-based smoother. Each iteration does an explicit
  Euler step of `∂I/∂t = div(c(|∇I|) · ∇I)` on a 4-neighbor
  stencil, with Neumann boundary conditions (zero flux at the image
  edge). `lambda` must stay ≤ 1/4 for stability (von Neumann
  analysis on the discrete Laplacian).
- Two conduction functions in `mode`: `:exponential` for
  `c(s) = exp(-(s/K)²)` and `:rational` for `c(s) = 1/(1 + (s/K)²)`,
  both from the original 1990 paper.
- `examples/13_anisotropic_diffusion.jl` compares against Gaussian
  and bilateral on a noisy image, with F1 scores against the clean-
  image Canny.

The result table:

| smoother              | F1    | notes                               |
|-----------------------|-------|-------------------------------------|
| raw (no smooth)       | 0.845 | recall 0.995, precision 0.734 — noisy edges everywhere |
| Gaussian σ=2          | 0.955 |                                     |
| bilateral             | 0.962 |                                     |
| PM exp K=0.05         | 0.962 | matches bilateral                   |
| PM exp K=0.15         | 0.910 | precision 1.000, recall 0.835 — K too big |
| PM rational K=0.10    | 0.910 | same story                          |

The K parameter is the knob: too small under-smooths (noise edges
survive), too big over-smooths (real edges look like noise). With
the right K the algorithm matches bilateral exactly in F1.

Concept note: `docs/concepts/11-anisotropic-diffusion.md`.

## Linear-algebra view of convolution

What I built:

- `LinAlgView.toeplitz_conv_matrix(K, n)` — the `n × n` matrix that
  represents 1D `:zero`-padded correlation as `M · v`. Banded, Toeplitz
  (constant along each diagonal).
- `LinAlgView.circulant_conv_matrix(K, n)` — same for `:circular`
  padding. The band wraps around at the corners, making the matrix
  circulant.
- `LinAlgView.circulant_eigenvalues(K, n)` — the eigenvalues of the
  circulant matrix, computed as the DFT of its first row. They equal
  `eigvals(C)` to machine epsilon (`2.22e-16` measured). This is the
  DFT-diagonalizes-circulant theorem made executable.

Two payoff results from the studio script:

- For an 8×8 [1, 2, 1]/4 smoother, the eigenvalues are
  `[1.000, 0.854, 0.854, 0.500, 0.500, 0.146, 0.146, 0.000]` — a
  perfectly visible low-pass profile from DC to Nyquist.
- For an 11-tap Gaussian-smoothing matrix at `n = 128`, the DC
  eigenvalue is `1.000` and the Nyquist eigenvalue is `0.0001`. The
  filter's "frequency response" is its operator's spectrum.

For teaching, not performance — materializing the matrix is
`O(n²)`. Concept note: `docs/concepts/15-conv-as-linear-algebra.md`.

## Laplacian-pyramid image blending

What I built:

- `laplacian_blend(A, B, mask; levels, pad)` in `Pyramids`. Builds
  Laplacian pyramids of `A` and `B`, a Gaussian pyramid of the
  mask, blends each level pointwise as `G_M[k] · L_A[k] + (1 − G_M[k])
  · L_B[k]`, and reconstructs via the exact-inverse path. Different
  frequency bands get blended with different mask smoothness — high
  frequencies with a sharp mask (detail preserved), low frequencies
  with a smoothed mask (color transition over many pixels).
- `examples/16_laplacian_blending.jl` shows two demos: a vertical
  split (sharp seam) and a circular cutout. Naive linear blending
  produces visible seams; the pyramid blend hides them.

The transition width scales roughly as `3 × 2^levels` pixels — at
`levels = 5` on a 128×128 image, the seam fades over ~96 pixels.

Concept note: `docs/concepts/14-laplacian-blending.md`.

## Real-image I/O

- `Photos.load_grayscale(path)` → `Matrix{Float64}` in `[0, 1]`.
  Color inputs collapse via the standard luminance formula.
- `Photos.save_grayscale(path, img)` writes PNG / TIFF / PGM, chosen
  by extension. Values get clamped to `[0, 1]` and quantized to 8-bit.
- `Photos.load_rgb_planes` / `Photos.save_rgb_planes` for color
  workflows.
- `examples/14_real_image_pipeline.jl` runs Canny, Harris, and a
  Gaussian pyramid on a PNG (real or synthesized). With a path
  argument it processes whatever I throw at it.
- `PNM` (the pure-Julia Netpbm one) stays around as the pedagogical
  version — `Photos` is the practical one.
- Concept note: `docs/concepts/12-real-image-io.md`.

## Multi-scale Harris

- `multiscale_harris_corners(img; levels, sigma, k, threshold,
  min_distance)` in `Features`. Runs Harris on every level of a
  Gaussian pyramid and re-maps detections back to the original
  image's coordinates. Each detection comes tagged with the level
  it fired on (`level = 0` is the original).
- Same Harris parameters at every level; threshold is fraction-of-max
  per level, so the comparison stays meaningful.
- Total cost across L levels is bounded by `4/3 × H × W` — only ~33%
  more than single-scale.
- `examples/15_multiscale_harris.jl` compares single vs multi-scale
  on a mixed-scale test image. Single-scale finds 5 corners
  (the sharp ones); multi-scale finds 20 across 3 levels, with the
  extras correctly sitting on softer larger-scale features.
- Concept note: `docs/concepts/13-multiscale-harris.md`.

## Image pyramids

What I built:

- `reduce_image(img)` — smooth with the Burt-Adelson 5-tap binomial
  filter `[1, 4, 6, 4, 1] / 16`, then downsample by 2.
- `expand_image(img, target_size)` — zero-insert, then smooth with
  the same filter scaled for 4× energy compensation (split as `2×`
  per separable 1D pass).
- `gaussian_pyramid(img; levels)` — vector of `levels + 1` images at
  halving resolution.
- `laplacian_pyramid(img; levels)` — the band-pass decomposition
  `L_k = G_k − EXPAND(G_{k+1})` plus the residual.
- `reconstruct_laplacian_pyramid(L)` — exact inverse.

The reconstruction is *exact* to within float-64 ULP precision
(measured: `5.55e-17` max abs error on a 128×128 test image). The
trick: even though `expand(reduce(I))` isn't exact at the borders,
the same border bias appears in both directions and the Laplacian
pyramid records and undoes it perfectly.

Concept note: `docs/concepts/10-pyramids.md`.

## Stretch

Once the image core is mature, things I'd want to explore:

- The linear-algebra view: convolution as a doubly block circulant
  matrix; the DFT diagonalizes it; that's where FFT-based convolution
  comes from. A short notebook.
- Anisotropic diffusion (Perona–Malik) — convolution-like but
  non-linear. Beautiful smoothing.
- A tiny ray tracer to get me out of imaging and into rendering. The
  MIT course goes here.
- Differentiable filters with `Zygote` or `ForwardDiff` — preview of
  where classical CV meets autodiff.
- GPU acceleration of the naive kernel via `KernelAbstractions.jl`
  once the performance lab tells me it'd pay off.

## References

- MIT Computational Thinking (Fall 2020) — <https://computationalthinking.mit.edu/Fall20/>
- MIT 18.S191 course materials — <https://github.com/mitmath/18S191>
- MIT Computational Thinking repos — <https://github.com/mitmath/computational-thinking>
- JuliaImages docs — <https://github.com/JuliaImages/juliaimages.github.io>
