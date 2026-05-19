# ImageLab

A first-principles image-processing lab in Julia. The goal is not to wrap
an existing library — it's to build the operators of classical computer
vision from scratch, see them work on images we can predict, and learn
Julia the way it deserves to be learned: by writing code that does
something visible.

The starting arc is convolution → edge detection → Canny → noise &
preprocessing → features. After that, the plan branches into the wider
MIT Computational Thinking spirit: linear-algebra views of convolution,
diffusion, simulation, a tiny ray tracer, differentiable filters. See
[ROADMAP.md](ROADMAP.md) for the full plan.

## Why this repo exists

Three things matter:

1. **Understanding through implementation.** Knowing that Sobel is
   `[-1 0 1; -2 0 2; -1 0 1]` is not the same as having coded a 2D
   correlation that produces the right edges on a synthetic square and
   wondering for half an hour why your gradient direction is flipped
   before realising you wanted convolution, not correlation. The
   confusions are the lesson.

2. **Julia as a thinking tool.** Multiple dispatch, type promotion,
   in-place mutation, separable algorithms, performance instrumentation
   — these become real when you can point at the code that needed them.
   We build the simplest correct version first, prove it with tests, then
   optimize against measurements.

3. **A body of work.** Notebooks, modules, comparison studios, benchmark
   reports, a Canny built from scratch, an edge playground. By the time
   the roadmap is done, this is a portfolio repo, not a tutorial folder.

References that shape the worldview:

- MIT 18.S191 / Computational Thinking — <https://computationalthinking.mit.edu/Fall20/>
- MIT Computational Thinking repos — <https://github.com/mitmath/computational-thinking>
- JuliaImages docs (for ecosystem comparisons later) — <https://github.com/JuliaImages/juliaimages.github.io>

## Layout

```
.
├── Project.toml              # package metadata, deps
├── src/
│   ├── ImageLab.jl           # top module
│   ├── synth.jl              # synthetic image generators
│   ├── kernels.jl            # box, Gaussian, Sobel, Prewitt, Scharr, Roberts,
│   │                         #   Laplacian, LoG, sharpen, identity
│   ├── padding.jl            # :zero / :replicate / :reflect / :symmetric /
│   │                         #   :circular / :valid
│   ├── convolution.jl        # naive 2D correlate2d / convolve2d (+ in-place)
│   └── io.jl                 # pure-Julia PGM/PPM (Netpbm) reader and writer
├── test/                     # one test_<module>.jl per src file
├── examples/
│   └── 01_first_convolutions.jl   # writes 8 PGMs to artifacts/
├── docs/concepts/            # short concept notes per topic
├── artifacts/                # generated PGMs (gitignored except .gitkeep)
└── ROADMAP.md
```

## Quickstart

```sh
# Once, to set up the manifest:
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# Run the tests:
julia --project=. test/runtests.jl

# Generate the Milestone 1 artifacts:
julia --project=. examples/01_first_convolutions.jl
open artifacts/01_first_convolutions/   # PGMs open in Preview on macOS
```

A 30-second tour from the REPL:

```julia
julia> using ImageLab

julia> using ImageLab.Synth, ImageLab.Kernels, ImageLab.Convolution, ImageLab.PNM

julia> img = Synth.circle(64, 64; radius = 20);

julia> edges = correlate2d(img, sobel_x(); pad = :replicate);

julia> PNM.save_pgm("circle_edges.pgm", (edges .- minimum(edges)) ./ (maximum(edges) - minimum(edges)));
```

## Design rules

- **First-principles before packages.** Every core operator gets a naive
  Julia implementation before we touch `Images.jl` or `DSP.jl`. The
  ecosystem comparison is a learning instrument; it isn't the answer.
- **One test per claim.** If the docstring says "sums to 1", there's a
  test for it. If the docstring says "anti-symmetric", there's a test
  for that too.
- **Visual output is part of correctness.** Every milestone produces
  PGMs into `artifacts/<milestone>/`. Numbers in a test file aren't
  enough — you need to see the edges.
- **Build for the third use case.** No abstractions until the third
  caller asks for the same structure. Three similar lines beat a
  premature struct.
- **Measure before optimizing.** Performance work waits for Milestone 5
  and the benchmark suite. Until then, "naive but correct" wins.

## Status

**Milestone 1: Foundation — done.**

- Package skeleton with five submodules.
- Naive 2D correlation and convolution with six border modes.
- Twelve canonical kernels.
- Seven synthetic image generators.
- 74 passing unit tests.
- 8 sample PGM artifacts demonstrating blur, gradient, magnitude,
  Laplacian, and sharpen on a synthetic image.

**Milestone 2: Convolution depth lab — next.** Separable filters, 1D
operators, stride and dilation, side-by-side padding-mode comparisons.
See [ROADMAP.md](ROADMAP.md).

## License

MIT — see [LICENSE](LICENSE).
