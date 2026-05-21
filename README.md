# ImageLab

I'm using this repo to learn Julia properly — by building image-processing
operators from scratch and seeing what they do. Convolution, edge
detection, eventually a from-scratch Canny, and after that wherever this
takes me (image diffusion, a tiny ray tracer, maybe a webcam edge
detector — see `ROADMAP.md`).

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

Tests live in `test/` and currently pass 374. Run them with
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

Next: probably the ray tracer — a small standalone module that
makes Julia feel like a general-purpose tool, not just an
image-processing one. Differentiable filters via ForwardDiff is
also on the table; might combine both into one ambitious chunk
since each is small.

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
- Where I reach for a package, I justify it briefly. Right now the only
  deps are `LinearAlgebra` and `Random` (both stdlib).

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
│   └── photos.jl
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
│   └── 17_convolution_as_linear_algebra.jl
├── docs/concepts/
└── artifacts/      # generated PGMs, gitignored except .gitkeep
```

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
