# What I'm building

This is the rough plan I keep nearby while I work. It's a living file —
when something lands I update the section, and when I learn something
worth recording mid-build I jot it down here. Nothing is on a schedule.

## Where I'm at

The convolution engine, the 1D / separable layer, all the edge
operators, and the full Canny pipeline are done. I'm starting on the
noise / preprocessing experiments next.

Roughly:

- Convolution engine — done
- Convolution depth (1D, separable, factor_separable) — done
- Edge operators (first- and second-order) — done
- Canny pipeline from scratch — done
- Noise + preprocessing experiments — next
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

Once Canny is solid, I want to ask: how much does the denoiser matter?

- Box vs Gaussian vs median vs bilateral.
- Salt-and-pepper noise vs Gaussian noise inputs.
- Same Canny parameters, sweep the denoiser, compare against the
  noise-free ground-truth edge map.

The synthetic image generators already produce ground-truth edge
locations, so I can score this quantitatively (precision/recall on
edge pixels at a fixed dilation tolerance).

## Performance lab

The performance question I want to answer cleanly: where does FFT-based
convolution beat the naive nested loop, and where does the separable
two-pass approach beat both?

- Naive 2D loop (already have).
- Inline bounds-aware loop that skips the padding copy.
- Separable two-pass (already have).
- FFT-based convolution using `FFTW`. This is the one new dependency
  I'd justify here.

For each: kernel size sweep on a 1024×1024 image, log-log plot of
ms-per-megapixel, identify the crossover points. A real benchmark
suite using `BenchmarkTools` (not `@elapsed`).

While I'm at it I want to do a `@code_warntype` audit on the hot paths
and document anything I find. Type instability is the easiest Julia
performance bug to introduce by accident.

## Features

This gets us into proper computer vision territory:

- Harris corner detector — a Sobel-flavored window response built on
  what I already have.
- Hough transform for lines.
- Connected component labeling on the Canny output.
- Template matching via normalized cross-correlation.
- Image pyramids and a small scale-space exploration.

These don't need to be industrial-strength; they need to be teachable
and produce visible outputs.

## Interactive playground

Pluto notebook with sliders for σ, kernel size, thresholds, denoiser
choice, operator choice — one knob per question I keep asking the
static example scripts. Pluto is the path of least friction on Julia
1.11; if I find it slow I'll try a Makie window instead.

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
