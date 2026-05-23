# ImageLab

I'm using this repo to learn Julia properly — by building image-processing
operators from scratch and seeing what they do. The trajectory so far:
naive convolution → six padding modes → 1D / separable convolution →
edge detection (first and second order) → Canny from scratch → noise
study → Harris / Hough / connected components / NCC → performance lab
with FFT-based convolution → Gaussian and Laplacian pyramids →
anisotropic diffusion → real-image I/O → multi-scale Harris →
multi-band image blending → convolution as Toeplitz / circulant matrices
→ a tiny Whitted-style ray tracer → differentiable filters via
ForwardDiff → Lucas-Kanade optical flow → pyramidal LK for big
motions → Horn-Schunck dense flow. See `ROADMAP.md` for the
chunk-by-chunk breakdown and `docs/concepts/` for the writeups.

The rule I keep coming back to: write the naive version, prove it works
with tests, look at the output as an actual image, *then* think about
optimizing or reaching for a library. The point isn't a thin wrapper
around `Images.jl`. I want to know why the operators work, why the math
is shaped the way it is, and what each line of Julia is buying me.

## What's in here right now

A small package at `src/ImageLab.jl` with these submodules:

- `Synth` — synthetic test images (checkerboard, square, circle, impulse,
  ramp, lines, noise). Hand-checkable, no downloaded data.
- `Kernels` — about a dozen canonical kernels plus their 1D / separable
  factorizations: box, Gaussian, Sobel / Prewitt / Scharr / Roberts,
  Laplacian (4- and 8-connected), LoG, sharpen.
- `Padding` — six border modes: `:zero`, `:replicate`, `:reflect`,
  `:symmetric`, `:circular`, `:valid`. Works for both 2D arrays and 1D
  vectors.
- `Convolution` — `correlate2d` and `convolve2d` (kernel-flipped),
  in-place versions, 1D operators, and `separable_correlate2d` which
  composes two 1D passes. There's also a `factor_separable(K)` that
  uses SVD to detect rank-1 kernels.
- `Edges` — first- and second-order edge operators (`gradient` for
  Sobel / Prewitt / Scharr / Roberts, `gradient_magnitude`,
  `gradient_direction`, `quantize_direction`, `log_filter`, `dog_filter`,
  `zero_crossings`, threshold helpers) plus the full Canny pipeline:
  `nonmaximum_suppression`, `double_threshold`, `hysteresis`, and a
  `canny` / `canny_stages` entry point that returns either the final
  edge map or every intermediate.
- `Filters` — non-linear and edge-preserving smoothers: `median_filter`,
  `bilateral_filter`, `perona_malik` (anisotropic diffusion), and
  `binary_dilate` for morphological tolerance on edge masks.
- `Features` — Harris corner detector (`harris_response`,
  `harris_corners`, plus `multiscale_harris_corners` that runs
  Harris on every level of a Gaussian pyramid), Hough line transform
  (`hough_lines`, `hough_peaks`, `HoughAccumulator`), connected
  component labeling (`connected_components`, `component_sizes`),
  and normalized cross-correlation template matching
  (`normalized_cross_correlation`, `ncc_peaks`).
- `Pyramids` — Burt-Adelson Gaussian and Laplacian pyramids
  (`reduce_image`, `expand_image`, `gaussian_pyramid`,
  `laplacian_pyramid`, `reconstruct_laplacian_pyramid`), and
  `laplacian_blend(A, B, mask; levels)` for multi-band image
  blending. Reconstruction matches the original to a single ULP.
- `LinAlgView` — the linear-algebra view of convolution. Build the
  Toeplitz / circulant convolution matrix (`toeplitz_conv_matrix`,
  `circulant_conv_matrix`); the FFT diagonalization makes the
  eigenvalues match the kernel's frequency response
  (`circulant_eigenvalues`). For teaching, not performance.
- `Metrics` — `edge_match_stats(predicted, gt; tolerance)` returns
  precision / recall / F1; `iou_score` for a single-number summary.
- `Viz` — `normalize01`, `signed_to_gray` (for signed gradient images),
  `montage` for assembling comparison grids, plus drawing helpers:
  `draw_line!` (Bresenham), `mark_points!` for feature overlays, and
  `label_to_gray` for colorizing connected-components output.
- `PNM` — pure-Julia Netpbm (PGM/PPM) reader and writer. No external
  deps; opens in Preview on macOS.
- `Photos` — PNG / TIFF / Netpbm I/O via `FileIO` + `ImageIO`.
  `load_grayscale(path)` returns a `Matrix{Float64}` in `[0, 1]`;
  `save_grayscale` and the RGB pair go the other way.
- `RayTracer` — a small Whitted-style ray tracer (spheres, planes,
  point lights, hard shadows, mirror reflection). Renders into
  three `Matrix{Float64}` channels so the rest of the pipeline
  (Photos, PNM, Viz) can take over from there.
- `AutoDiff` — differentiable filters via ForwardDiff. The naive
  `correlate2d` happens to be type-generic, so `Dual`-valued
  kernels just work; on top of that I built `kernel_loss`,
  `kernel_gradient`, and a tiny SGD loop in `fit_kernel`.
- `Flow` — Optical flow between two frames. `lucas_kanade` solves
  the 2×2 normal equations per pixel for sub-pixel motions;
  `lucas_kanade_pyramid` is the coarse-to-fine variant that
  handles motions of several pixels by walking a Gaussian pyramid
  and warping at each level. `horn_schunck` is the dense
  global-smoothness alternative — Jacobi iteration on the
  Horn-Schunck weighted Laplacian, fills in flow in textureless
  regions by diffusion. `warp_bilinear` does inverse-warping with
  edge-clamped four-tap bilinear interpolation. Visualization via
  `Viz.flow_to_rgb` using the standard hue-direction /
  saturation-magnitude colour-wheel convention.

Tests live in `test/` and currently pass 1485. Run them with
`julia --project=. test/runtests.jl`.

## Quickstart

```sh
# One-time setup:
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# Run tests:
julia --project=. test/runtests.jl

# Generate the example artifacts:
julia --project=. examples/01_first_convolutions.jl
julia --project=. examples/02_padding_modes_studio.jl
julia --project=. examples/03_separable_vs_naive.jl
julia --project=. examples/04_edge_operator_studio.jl
julia --project=. examples/05_log_dog_zero_crossings.jl
julia --project=. examples/06_canny_pipeline.jl
julia --project=. examples/07_canny_parameter_sweep.jl
julia --project=. examples/08_noise_lab.jl
julia --project=. examples/09_corners_and_lines.jl
julia --project=. examples/10_components_and_templates.jl
julia --project=. examples/11_performance_lab.jl
julia --project=. examples/12_pyramid_decomposition.jl
julia --project=. examples/13_anisotropic_diffusion.jl
julia --project=. examples/14_real_image_pipeline.jl   # or pass a PNG path
julia --project=. examples/15_multiscale_harris.jl
julia --project=. examples/16_laplacian_blending.jl
julia --project=. examples/17_convolution_as_linear_algebra.jl
julia --project=. examples/18_tiny_ray_tracer.jl
julia --project=. examples/19_differentiable_filters.jl
julia --project=. examples/20_optical_flow.jl
julia --project=. examples/21_pyramidal_optical_flow.jl
julia --project=. examples/22_horn_schunck.jl
open artifacts/
```

A 30-second REPL tour:

```julia
julia> using ImageLab

julia> using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution, ImageLab.Viz, ImageLab.PNM

julia> img = Synth.circle(64, 64; radius = 20);

julia> edges = correlate2d(img, sobel_x(); pad = :replicate);

julia> PNM.save_pgm("circle_edges.pgm", Viz.signed_to_gray(edges));
```

## Where I'm at

I started with the package skeleton — a Project.toml, the submodule
layout, naive `correlate2d` with all six padding modes, a kernel zoo,
synthetic image generators, the Netpbm writer. The first example script
(`examples/01_first_convolutions.jl`) blurs and gradients a synthetic
image and dumps eight PGMs to `artifacts/`.

After that I added 1D operators and separable convolution. The point of
separable is that a rank-1 2D kernel can be applied as two cheap 1D
passes — Gaussian, box, Sobel, Prewitt, Scharr are all rank-1. The
arithmetic drops from `kh·kw` mults per pixel to `kh + kw`, so at a
15×15 kernel the speedup is theoretically ~7.5×. On a 384×384 image
with `k=21` I measured **10.8× faster**, max numerical drift `~1e-15`
between the two implementations.

The padding studio (`examples/02_padding_modes_studio.jl`) tiles all
five non-`:valid` modes into a 2×3 montage so you can see what each one
does at the border. The most striking is `:circular` — bright content
from one corner leaks onto the opposite side.

Then I built the edge operators. First-order — Sobel / Prewitt / Scharr
/ Roberts gradients plus magnitude and direction — and second-order —
Laplacian, LoG, DoG, and a zero-crossing detector. Two new studios
(`04_edge_operator_studio.jl`, `05_log_dog_zero_crossings.jl`) tile the
operators side by side on the same synthetic input.

After that came the full Canny pipeline: Gaussian smoothing → Sobel
gradient → non-maximum suppression along the gradient direction →
double threshold → 8-connected hysteresis. The pipeline is split into
named stages (`nonmaximum_suppression`, `double_threshold`,
`hysteresis`) so I can inspect intermediates. `canny_stages` returns
all eight intermediates in one struct; `canny` returns just the final
edge map. `examples/06_canny_pipeline.jl` walks every stage on one
input; `examples/07_canny_parameter_sweep.jl` runs a 3×3 grid over
(σ, low, high) so I can see how each knob affects the output.

After that I wrote `Filters` (median, bilateral, binary dilation) and
`Metrics` (precision / recall / F1 with a pixel-tolerance, plus IoU),
then `examples/08_noise_lab.jl`, which sets Canny against eight
smoother choices on two noise types and scores each one against a
ground-truth edge map (the Canny output on the clean image).

The result was a much cleaner story than I expected:

- On Gaussian noise, every smoother gets F1 above 0.98 and the
  differences between them are fractions of a percent. Bilateral
  edges out the rest (0.995), but you don't really need it.
- On salt-and-pepper noise, the picture is brutal. With no smoother
  F1 is 0.21 — every speck becomes a tiny false edge. Median 5×5
  jumps it to 0.99. Bilateral fails completely (0.21, same as no
  smoother) because it treats each speck as an edge worth preserving
  — the same property that wins it the Gaussian-noise race.

Full numbers in `docs/concepts/07-noise-and-preprocessing.md`.

After the noise study came the features layer — Harris corners,
Hough lines, connected components, and normalized cross-correlation
template matching. Two studios:

- `examples/09_corners_and_lines.jl` runs Harris and Hough on a
  scene of two rectangles and a triangle. Harris finds 11 corners
  (matching the 4+4+3 geometric corners exactly). Hough finds 10
  peaks against the 11 actual sides — usually the missing one is a
  short edge whose votes spread thinly enough to fall under the
  threshold.
- `examples/10_components_and_templates.jl` does two things on
  separate inputs. First: Canny → connected components → component
  sizes on a multi-shape image. Then: NCC template matching against
  six disks of varying intensity, half of which sit in a darkened
  region of the image. NCC's brightness invariance picks all six up,
  scoring five at exactly 1.0 and one at 0.974 — the 0.974 one
  straddles the brightness boundary and so isn't uniform anymore,
  which is what NCC should report.

Concept doc: `docs/concepts/08-features.md`.

After that came the performance lab. I added two new convolution
implementations: `correlate2d_inline` (no padding copy, single bounds-aware
loop split into interior + border), and `fft_correlate2d` /
`fft_convolve2d` via FFTW. Then `examples/11_performance_lab.jl`
benchmarks all four (naive, inline, separable, FFT) across kernel
sizes 3..31 on a 384×384 image with `BenchmarkTools`.

A few results that surprised me:

- Inline is basically a wash with naive. The folk wisdom that "the
  padding copy is the bottleneck" is wrong here — the inner-loop
  arithmetic dominates. Best inline result was a 5% speedup at k=31.
- Separable matches theory (`k/2` speedup) and *beats* it at k=31
  (20× measured vs 15.5× theoretical), I'm pretty sure because the
  two passes fit cleaner in L2 cache.
- FFT crossover is around k≈25 on this image size. Below that
  separable wins; above it FFT pulls ahead (29.6× vs naive at k=31).
- A `@code_warntype` pass on the hot paths showed zero `::Any` and
  zero `::Union` boxes — the compiler can specialize every inner
  loop on the concrete element type.

Full numbers in `docs/concepts/09-performance.md`.

After the performance lab came image pyramids. `Pyramids.reduce_image`
and `Pyramids.expand_image` are the Burt-Adelson 1983 operators (the
5-tap binomial filter `[1, 4, 6, 4, 1] / 16` plus zero-insertion with
a 4× energy compensation). On top of those, `gaussian_pyramid` and
`laplacian_pyramid` give multi-scale decompositions, and
`reconstruct_laplacian_pyramid` inverts them.

The headline number for the Laplacian pyramid: on a 128×128 input
with 4 levels, the round-trip reconstruction error is
`5.55e-17` — a single ULP of `Float64`. Exact invertibility, not
approximate. The reason this works even though `expand(reduce(I))`
isn't exact at the borders is that the same border bias appears in
both directions and cancels.

`examples/12_pyramid_decomposition.jl` builds a 5-level Gaussian
and Laplacian pyramid, shows them in a 2-row montage, and prints
the reconstruction error.

After pyramids came Perona-Malik anisotropic diffusion. It's a
non-linear PDE-based smoother that solves the same problem as
bilateral (linear smoothing wrecks edges) with a different
intuition — replace the constant diffusivity in the heat equation
with one that depends on the local gradient magnitude, so flat
regions smooth fast and edges don't smooth at all. Discretized via
explicit Euler on a 4-neighbor stencil, with `λ ≤ 1/4` for stability.

`examples/13_anisotropic_diffusion.jl` runs it head-to-head against
Gaussian and bilateral on a noisy image. With `K = 0.05` and 40
iterations Perona-Malik ties bilateral at F1 = 0.962 (vs 0.955 for
Gaussian σ=2 and 0.845 for raw Canny on the noisy input). With
`K = 0.15` precision climbs to 1.000 but recall drops to 0.835 — the
filter starts treating real edges as noise and smooths them away.
The knob is explicit: above the right K you under-detect; below it
you over-detect.

Connecting to MIT-spirit territory: the line `linear diffusion =
Gaussian smoothing at time t = σ²/2` is one of those one-equation
results that ties spatial filtering to a time-evolving PDE. The
concept doc walks through it.

After diffusion came two pieces that connect the lab to the real
world. First, a `Photos` submodule built on `FileIO` + `ImageIO`
that loads PNG / TIFF / Netpbm into a `Matrix{Float64}` in `[0, 1]`
and writes results back. `examples/14_real_image_pipeline.jl` runs
the whole stack (Canny + Harris + Gaussian pyramid) on a real PNG —
the script generates a synthetic sample if none is provided, but
will happily take a real photo path on the command line.

Second, `multiscale_harris_corners` — run Harris on every level of
a Gaussian pyramid, tag each detection with the level it fired on,
re-map coordinates back to the original. On a test image with a
sharp small rectangle plus a larger soft-cornered one,
single-scale Harris finds 5 corners (the sharp ones plus a noise
dot); multi-scale finds 20 detections across 3 levels, the extras
all sitting on the soft-rectangle corners that a 3×3 window at the
original scale can't see.

`examples/15_multiscale_harris.jl` runs the comparison and writes
a 3-tile montage. The cost of multi-scale is only ~33% more than
single-scale (geometric series: `1 + 1/4 + 1/16 + ... ≈ 4/3`).

After multi-scale Harris I did the classical pyramid application:
multi-band image blending. `Pyramids.laplacian_blend(A, B, mask)`
takes two images and a mask, builds Laplacian pyramids of each
image and a Gaussian pyramid of the mask, blends each level
pointwise, and reconstructs. The result has no visible seam even
when the mask is sharp — high frequencies blend with a sharp mask
(detail survives), low frequencies blend with a heavily-smoothed
mask (color transitions fade in over many pixels).

`examples/16_laplacian_blending.jl` does the classic "left half
from A, right half from B" demo plus a circular cut-and-paste. Two
montage outputs make the difference between naive and pyramid
blending obvious.

After blending I did the linear-algebra view of convolution, which
ties the FFT-conv path I had to the underlying matrix structure.
`LinAlgView.toeplitz_conv_matrix(K, n)` and
`circulant_conv_matrix(K, n)` materialize the matrix that
convolution is implicitly multiplying by; both check out against
`correlate1d` to within `1e-12`.

The interesting part: `circulant_eigenvalues(K, n)` (which is just
the FFT of the first row) returns the same set of values as
`eigvals(C)` (the hard linear-algebra computation). They match to
`2.22e-16` — machine epsilon. That's the DFT-diagonalizes-circulant
theorem in code, which is also the result that makes
FFT-convolution work.

And the eigenvalues of an 11-tap Gaussian-smoothing circulant
matrix at `n = 128` come out to `|λ|_DC = 1.0` (DC passes
unchanged) and `|λ|_Nyquist = 0.0001` (highest frequency almost
completely blocked) — which is exactly the frequency response of a
strong low-pass filter, but read straight off the matrix's spectrum.

`examples/17_convolution_as_linear_algebra.jl` prints the
8×8 Toeplitz and circulant matrices, verifies the two
eigenvalue computations, and saves the matrices as images so the
band structure is visible.

Then I built the ray tracer, mostly to prove to myself that I can
write something in Julia that isn't an image filter. `RayTracer` is
a small Whitted-style renderer — `V3` arithmetic, ray-sphere and
ray-plane intersection, Phong shading with hard shadows, recursive
mirror reflection capped by a depth parameter. The basic scene
(red and blue spheres plus a chrome sphere over a checkerboard
floor) renders at 384×256 in 0.22 seconds, about 2.2 microseconds
per pixel.

The reason this slots into `ImageLab` and not its own repo:
`render(scene, camera)` returns `(R, G, B)` as three
`Matrix{Float64}` channels in `[0, 1]`. That's the same shape
`Photos.save_rgb_planes` and `PNM.save_ppm` expect, so a finished
render is a first-class image — I can take its luminance and run
Canny on it, build a Gaussian pyramid of it, fit a Toeplitz
matrix to a row of it. Synthesis (ray tracer) and analysis (every
other module) meet at the same data type.

`examples/18_tiny_ray_tracer.jl` renders the basic scene, a
depth-by-depth comparison showing what each bounce adds (max
luminance: 0.92 → 0.74 → 0.74 → 0.74 — two bounces is enough), a
three-angle camera orbit, and the luminance of the basic render so
I can feed it back to the rest of the pipeline.

Then I did the differentiable-filters chunk, which closes out the
classical → modern bridge. The whole thing rests on a single
observation: my naive `correlate2d` from the very first chunk of this
repo is generic over its element type, so I can pass a kernel made of
`ForwardDiff.Dual` numbers and the autodiff machinery just goes
through it. The submodule (`AutoDiff`) on top is ~100 lines:
`kernel_loss` (MSE between filter output and target),
`kernel_gradient` (one call to `ForwardDiff.gradient`), and
`fit_kernel` (vanilla SGD).

The demo learns four classical kernels — Sobel-x, Sobel-y,
Laplacian, sharpen — from input/target pairs starting at random
initializations. In 800 SGD steps at `lr = 0.1`, every kernel
converges to within `~1e-6` of the hand-written one. The learned
Sobel-x prints out as exactly the textbook `[-1 0 1; -2 0 2; -1 0 1]`
to 4 decimal places.

One thing the chunk made me confront explicitly: I tried training on
a checkerboard first and the optimizer fit the *loss* but not the
*kernel* — the recovered numbers were 0.12 off in places. The
checkerboard underdetermines the kernel, because only a few distinct
3×3 patches appear. Switching to a uniform-noise training image
makes every 3×3 patch unique and the system becomes well-posed. Same
lesson that haunts deep-learning data preparation.

That bridges classical CV (write the kernel) and modern CV (learn
it) through one ~100-line submodule and the exact same convolution
code I wrote on day one.

Then I picked up optical flow. The output isn't a single grayscale
matrix any more — it's a 2D vector field with one `(u, v)` per
pixel — so this is also the first chunk in the repo where I had to
think about how to visualize something that isn't a single
intensity. `Flow.lucas_kanade(frame1, frame2)` solves the classical
1981 LK setup: per-pixel brightness constancy gives the optical-flow
constraint `Iₓ·u + I_y·v + I_t = 0` (one equation, two unknowns,
underdetermined), and the LK assumption that motion is constant in a
small window turns that into an overdetermined least-squares system
solved via the 2×2 normal equations.

The structure tensor on the left of those normal equations is
*exactly* the same matrix Harris uses for corners. That's not a
coincidence — Harris fires where two independent motion directions
constrain the system, which is the exact condition for LK to be
well-conditioned. The `Flow.FlowField` return value carries the
structure-tensor determinant as a `confidence` channel for masking
the no-texture pixels.

For visualization I added `Viz.hsv_to_rgb` and `Viz.flow_to_rgb`,
which encode flow with the standard colour-wheel convention
(hue = direction, saturation = magnitude, value = 1). The colour
wheel image the example writes out is the legend.

Numbers from `examples/20_optical_flow.jl`: a known continuous
translation of (0.7, 0.3) pixels recovers as (0.719, 0.310) — 3%
error. A 1° rotation around the image centre recovers within ~5%.
The confidence in a textured patch is ~5 orders of magnitude
larger than in a flat background.

The biggest known limitation of plain LK: it linearizes the OFC,
so it's accurate only for sub-pixel motions. A 2-pixel shift
recovers `u ≈ 2.3` — visible bias. The classical fix is pyramidal
LK.

So the next chunk was exactly that. `Pyramids` was already there;
the new pieces were `warp_bilinear(img, u, v)` (inverse-warp with
bilinear interpolation, four-tap, edge-clamped) and
`lucas_kanade_pyramid` itself — coarse-to-fine driver that walks
the Gaussian pyramid from the coarsest level to the finest,
warping frame 2 by the current estimate at each level and adding
the residual flow LK finds on the warped pair.

The numbers are the satisfying part. On a (4, -2.5)-pixel
translation of the same sinusoid the plain LK demo used:

```
plain LK:    recovered (u, v) = (+3.995, -2.829)   # v overshoots by 13%
pyramid LK:  recovered (u, v) = (+4.001, -2.495)   # both within 0.2%
```

And the level-by-level convergence traces out exactly what the
algorithm is supposed to do — flow accumulates from 0 at the
coarsest level (8×8, pattern smoothed away) up to ~4, -2.5 at the
finest (128×128). The mean per-pixel residual `|warp(I₂, flow) −
I₁|` is `0.0004` on a `[0, 1]` image, which means the recovered
flow reconstructs frame 1 to within four parts per ten thousand.

`examples/21_pyramidal_optical_flow.jl` saves the per-level
snapshots, a plain-vs-pyramid side-by-side, and the warp-residual
image. Concept note: `docs/concepts/19-pyramidal-lucas-kanade.md`.

Then I went sideways and built Horn-Schunck — the contemporary
1981 alternative to LK and the standard worked example of "local
vs global formulation" in CV. Same `Iₓ`, `I_y`, `I_t` ingredients
as LK, completely different regularization: instead of windowing
the OFC into a local least-squares solve, HS minimizes a global
cost functional with a data term and a smoothness term. The
Euler-Lagrange equations give a coupled PDE; the standard solver
is Jacobi iteration on the Horn-Schunck weighted Laplacian
(4-neighbours at 1/6, diagonals at 1/12, centre 0).

The headline contrast — same input (a textured patch moving in a
flat field) given to both:

```
LK on patch (interior):  u = +0.514   (matches truth, ~3% bias)
LK in flat region:       u = +0.000   (no texture → no constraint → 0)
HS on patch (interior):  u = +0.507
HS in flat region:       u = +0.268   (diffused outward via smoothness)
```

LK returns zero in the flat region because the 2×2 structure
tensor there has zero determinant. HS returns a non-zero value
because the smoothness term makes "zero next to non-zero" expensive;
the iteration relaxes the field by propagating flow from the
textured boundary inward. The iteration trace shows it happening:
after 50 sweeps the flow has just appeared inside the patch; after
1000 sweeps it's filled in and started to spill outward; after
5000 sweeps it crosses the entire flat region.

The `α` knob sets the smoothness weight. For images in `[0, 1]`
(the convention I use everywhere else in the repo), `α ≈ 0.1` is
about right; smaller `α` matches LK, larger `α` pulls everything to
zero. The default I shipped is `0.1`.

Concept note: `docs/concepts/20-horn-schunck.md`.

## House rules

A few things I've decided up front so I don't drift:

- I write the naive version first and prove it works on tiny
  hand-checkable inputs. Optimization waits until a benchmark says it
  matters.
- Every functional change ships with a test. If a docstring says "sums
  to 1", there's a test that asserts it.
- Every chunk of new operator code writes at least one PGM into
  `artifacts/<topic>/` so I can actually see what it does. Type-checking
  isn't enough.
- I don't generalize a struct until I have three real callers asking
  for the same shape. Three similar lines beat a premature abstraction.
- Where I reach for a package, I justify it briefly. Current deps:
  - `LinearAlgebra`, `Random` — stdlib, no-brainer.
  - `FFTW` — the FFT path in `Convolution`. Performance and the
    DFT-diagonalization story would be much weaker without it.
  - `BenchmarkTools` — the performance lab.
  - `FileIO`, `ImageIO`, `ColorTypes`, `FixedPointNumbers` — real-world
    image formats in `Photos`. The pure-Julia `PNM` stays as the
    teaching version.
  - `ForwardDiff` — differentiable filters in `AutoDiff`.

## Repo layout

```
.
├── Project.toml
├── README.md
├── ROADMAP.md
├── src/
│   ├── ImageLab.jl
│   ├── synth.jl
│   ├── kernels.jl
│   ├── padding.jl
│   ├── convolution.jl
│   ├── edges.jl
│   ├── filters.jl
│   ├── features.jl
│   ├── pyramids.jl
│   ├── metrics.jl
│   ├── linalg_view.jl
│   ├── viz.jl
│   ├── io.jl
│   ├── photos.jl
│   ├── raytracer.jl
│   ├── autodiff.jl
│   └── flow.jl
├── test/
├── examples/
│   ├── 01_first_convolutions.jl
│   ├── 02_padding_modes_studio.jl
│   ├── 03_separable_vs_naive.jl
│   ├── 04_edge_operator_studio.jl
│   ├── 05_log_dog_zero_crossings.jl
│   ├── 06_canny_pipeline.jl
│   ├── 07_canny_parameter_sweep.jl
│   ├── 08_noise_lab.jl
│   ├── 09_corners_and_lines.jl
│   ├── 10_components_and_templates.jl
│   ├── 11_performance_lab.jl
│   ├── 12_pyramid_decomposition.jl
│   ├── 13_anisotropic_diffusion.jl
│   ├── 14_real_image_pipeline.jl
│   ├── 15_multiscale_harris.jl
│   ├── 16_laplacian_blending.jl
│   ├── 17_convolution_as_linear_algebra.jl
│   ├── 18_tiny_ray_tracer.jl
│   ├── 19_differentiable_filters.jl
│   ├── 20_optical_flow.jl
│   ├── 21_pyramidal_optical_flow.jl
│   └── 22_horn_schunck.jl
├── docs/concepts/
└── artifacts/      # generated PGMs, gitignored except .gitkeep
```

## Concept notes

One write-up per major chunk under `docs/concepts/`. They expand on
the "where I'm at" prose above with the actual derivations,
measurements, and "things I got wrong the first time" notes.

- `01-convolution.md` — naive 2D correlation, why the kernel is
  flipped for "real" convolution, how `:zero` / `:replicate` /
  `:reflect` change behavior at the border.
- `02-separable-convolution.md` — when a 2D kernel factors into two
  1D passes (SVD rank-1 test) and the `k/2` speedup with measurements.
- `03-padding-modes.md` — the six modes side by side.
- `04-gradient-magnitude-and-direction.md` — Sobel / Prewitt / Scharr
  / Roberts comparison, how to display direction on grayscale.
- `05-second-order-edges.md` — Laplacian → LoG → DoG → zero-crossings.
- `06-canny.md` — the full pipeline stage by stage.
- `07-noise-and-preprocessing.md` — Gaussian vs salt-and-pepper, why
  bilateral wins one and fails the other.
- `08-features.md` — Harris, Hough, connected components, NCC.
- `09-performance.md` — naive vs inline vs separable vs FFT, the
  surprising "padding copy isn't the bottleneck" result.
- `10-pyramids.md` — Burt-Adelson, why reconstruction is exact to ULP.
- `11-anisotropic-diffusion.md` — Perona-Malik PDE, how K controls the
  noise-vs-detail tradeoff.
- `12-real-image-io.md` — FileIO / ImageIO, gotchas around JpegTurbo
  and N0f8 quantization.
- `13-multiscale-harris.md` — Harris on every level of a Gaussian
  pyramid, the geometric-series cost.
- `14-laplacian-blending.md` — multi-band image blending, transition
  width scaling.
- `15-conv-as-linear-algebra.md` — Toeplitz / circulant matrices,
  DFT diagonalization, eigenvalues are the frequency response.
- `16-ray-tracing.md` — the Whitted-style ray tracer.
- `17-differentiable-filters.md` — autodiff on the convolution
  operators; learning Sobel and friends from input/target pairs.
- `18-optical-flow.md` — Lucas-Kanade, the optical-flow constraint,
  why the same 2×2 structure tensor shows up that Harris already
  used for corners.
- `19-pyramidal-lucas-kanade.md` — coarse-to-fine LK, bilinear
  image warping, why a 4-pixel motion needs five levels of pyramid
  to recover and a sub-pixel motion only needs one.
- `20-horn-schunck.md` — dense optical flow via Horn-Schunck's
  1981 variational formulation. Local-vs-global regularization,
  the Jacobi iteration, why HS gives non-zero flow in textureless
  regions where LK gives nothing.

## References

The framing here is heavily shaped by the MIT Computational Thinking
course — that's where I picked up the idea that Julia is a good
language to think computationally in, not just to use as a numpy
replacement.

- MIT 18.S191 / Computational Thinking (Fall 2020) — <https://computationalthinking.mit.edu/Fall20/>
- MIT 18.S191 course repository — <https://github.com/mitmath/18S191>
- MIT Computational Thinking repos — <https://github.com/mitmath/computational-thinking>
- JuliaImages docs (for ecosystem comparisons later on) — <https://github.com/JuliaImages/juliaimages.github.io>

## License

MIT — see `LICENSE`.
