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
- `Edges` — first- and second-order edge operators: `gradient` (Sobel /
  Prewitt / Scharr / Roberts), `gradient_magnitude`, `gradient_direction`,
  `quantize_direction` (4-way for NMS), `log_filter`, `dog_filter`,
  `zero_crossings`, plus threshold helpers (`threshold_mask`,
  `percentile_threshold`).
- `Viz` — `normalize01`, `signed_to_gray` (for signed gradient images),
  and a `montage` helper for assembling comparison grids.
- `PNM` — pure-Julia Netpbm (PGM/PPM) reader and writer. No external
  deps; opens in Preview on macOS.

Tests live in `test/` and currently pass 146. Run them with
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

Canny from scratch is what I'm working on next.

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
│   ├── viz.jl
│   └── io.jl
├── test/
├── examples/
│   ├── 01_first_convolutions.jl
│   ├── 02_padding_modes_studio.jl
│   ├── 03_separable_vs_naive.jl
│   ├── 04_edge_operator_studio.jl
│   └── 05_log_dog_zero_crossings.jl
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
