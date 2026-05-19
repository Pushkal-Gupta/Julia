# ImageLab — Roadmap

A living plan. Each milestone is a self-contained deliverable: working code,
tests, one example script that writes artifacts to `artifacts/<milestone>/`,
and a short note in this file when it lands.

Philosophy: build the simplest correct version → make it testable → make it
visual → make it reusable → make it fast → compare to ecosystem. Don't
generalize until the third use case appears.

---

## Milestone 1 — Foundation (✓ done)

**Goal:** prove the package layout works, get a correct naive 2D convolution
running end-to-end, and produce the first visual artifacts.

Shipped:

- `Project.toml`, package skeleton at `src/ImageLab.jl` with four submodules:
  `Synth`, `Kernels`, `Padding`, `Convolution`, `IO`.
- Synthetic image generators: checkerboard, square, circle, impulse, ramp,
  lines, Gaussian and salt-and-pepper noise.
- A small kernel zoo: box, Gaussian, Sobel/Prewitt/Scharr/Roberts (x and y),
  Laplacian (4- and 8-connected), Laplacian of Gaussian, unsharp-style
  sharpen, identity.
- Padding module with `:zero`, `:replicate`, `:reflect`, `:symmetric`,
  `:circular`, `:valid`.
- Naive but correct `correlate2d` / `convolve2d` (+ in-place variants).
- Pure-Julia Netpbm (PGM/PPM) writer + reader — no external deps.
- Unit tests for every module (`julia --project=. test/runtests.jl`).
- `examples/01_first_convolutions.jl` writes 8 PGMs comparing blur, gradient,
  Laplacian, and sharpen on a synthetic image.

**What this teaches:** module structure, multiple dispatch on `AbstractMatrix`,
type promotion, padding as a separate concern from the inner loop, and why
"correlation" and "convolution" deserve separate functions.

---

## Milestone 2 — Convolution depth lab (✓ done)

**Goal:** turn the naive engine into a proper teaching reference, and start
building the comparison story that runs through the whole repo.

Shipped:

- 1D `correlate1d` / `convolve1d` for both vectors and matrices (with
  `axis=:horizontal/:x` or `:vertical/:y`).
- `Padding.pad_vector` for the 1D case under all six border modes.
- **Separable convolution**: `separable_correlate2d` / `separable_convolve2d`
  (two-pass implementation) plus pre-factored canonical 1D kernels
  (`gaussian1d`, `box1d`, `sobel_*_separable`, `prewitt_*_separable`,
  `scharr_*_separable`).
- **`factor_separable(K)`**: SVD-based rank-1 detection for arbitrary
  kernels. Recovers factors that reproduce the naive 2D result to ~1e-10.
- New `Viz` submodule with `normalize01`, `signed_to_gray`, and `montage`
  for one-glance comparison grids.
- **`examples/02_padding_modes_studio.jl`**: 96×96 image blurred with a
  σ=3 Gaussian under all five non-`:valid` modes, packed into a 2×3
  montage PGM.
- **`examples/03_separable_vs_naive.jl`**: timing table across kernel
  sizes 3..21. Measured speedup at k=21 is **10.8×** (theory says
  k/2 = 10.5×). Max numerical drift between naive and separable: ~1e-15.
- Concept docs: `docs/concepts/02-separable-convolution.md`,
  `docs/concepts/03-padding-modes.md`.
- 117 passing tests (was 74).

**Deferred to a later milestone:**

- **Stride and dilation** parameters — not yet needed for the edge-detection
  arc; will land alongside the feature-detection layer when we want to
  build image pyramids cheaply.
- **Kernel-walking visualization** (numbered PGMs for an animation) — fun
  but lower leverage than the Canny pipeline; revisit when M8 (interactive
  playground) starts.

---

## Milestone 3 — Edge detection lab v1

**Goal:** every classical edge operator, from scratch, with a comparison
studio.

Planned:

- Gradient magnitude + direction with proper handling of `atan2`.
- Roberts / Prewitt / Sobel / Scharr side-by-side on the same input.
- Laplacian-of-Gaussian and Difference-of-Gaussians, with σ sweeps.
- Zero-crossing detector for second-derivative methods.
- Threshold-based and percentile-based edge maps.
- Comparison studio script: one input → grid of every operator at the same
  scale.
- Per-operator concept docs explaining what it sees and what fools it.

---

## Milestone 4 — Canny from scratch

**Goal:** a textbook-faithful Canny pipeline, with every intermediate saved.

Planned stages:

1. Gaussian blur (parameterized σ).
2. Sobel gradient magnitude + direction.
3. Non-maximum suppression (proper interpolation across the gradient angle).
4. Double threshold with hysteresis (8-connectivity).

Each stage saves an intermediate PGM. A single `examples/canny_studio.jl`
sweeps `(σ, low_threshold, high_threshold)` to teach parameter sensitivity.

---

## Milestone 5 — Performance lab

**Goal:** measure everything we've built so we know where to spend effort
later. Establish discipline around `BenchmarkTools` and `@code_warntype`.

Planned:

- Add `BenchmarkTools` as a dev dependency.
- Compare:
  1. Naive nested loop (M1).
  2. Inline bounds-aware loop that skips the padding copy.
  3. Separable two-pass loop.
  4. FFT-based convolution (`FFTW`) for large kernels.
- Sweep kernel sizes from 3×3 to 31×31 on a 1024×1024 image; plot crossover
  curves.
- `@code_warntype` audit on hot paths; document any type instabilities found
  and how they were fixed.
- A `benchmarks/` directory with reproducible scripts and recorded outputs.

---

## Milestone 6 — Noise & preprocessing

- Box / Gaussian / median / bilateral exploration.
- Salt-and-pepper + Gaussian noise generators (already in `Synth`).
- Systematic experiment: fix the edge operator, vary the denoiser, score
  results against the ground-truth edge map of a synthetic image.

---

## Milestone 7 — Feature lab

- Harris corner detector.
- Hough transform for lines.
- Connected component labeling.
- Template matching via normalized cross-correlation.
- Image pyramids and scale-space basics.

---

## Milestone 8 — Interactive playground

Decision point at the time: Pluto notebook, GLMakie app, or a small Tk-style
widget. The user has Julia 1.11; Pluto is the path of least friction. Build
an "edge playground" where filter, σ, kernel size, and thresholds are live
sliders.

---

## Branching: MIT-style adjacencies

Once the image core is solid, optional spurs in the MIT Computational
Thinking spirit:

- **Linear-algebra view of convolution:** Toeplitz / circulant matrices,
  doubly-block-circulant for 2D. A short notebook.
- **Diffusion as convolution:** anisotropic diffusion, Perona-Malik.
- **A tiny ray tracer** to broaden Julia intuition outside imaging.
- **Differentiable filters:** auto-diff on a single Sobel filter via
  `Zygote` or `ForwardDiff` — preview of where deep learning meets classical
  vision.
- **GPU acceleration** for the naive kernel via `KernelAbstractions.jl`
  once the performance lab tells us it's worth it.

---

## References (canonical)

- MIT 18.S191 / Computational Thinking (Fall 2020) — <https://computationalthinking.mit.edu/Fall20/>
- MIT 18.S191 course materials — <https://github.com/mitmath/18S191>
- MIT Computational Thinking repo family — <https://github.com/mitmath/computational-thinking>
- JuliaImages docs — <https://github.com/JuliaImages/juliaimages.github.io>
