# Anisotropic diffusion (Perona-Malik)

This is one of those algorithms that gets me back to *why* I'm
learning Julia. It connects three things that look unrelated at
first: image smoothing, the heat equation, and edge detection. And
when you set them next to each other, the choice of conductivity in
a partial differential equation tells you why a Gaussian blur is
"wrong" near edges and what to do about it.

## The setup: heat equation = Gaussian smoothing

Linear diffusion in 2D is `∂I/∂t = ∇²I`. If I start with an
image `I_0(x, y)` and let it diffuse for time `t`, the solution is

```
I(x, y, t) = (G_σ * I_0)(x, y)    where σ = √(2t)
```

That is: linear diffusion *is* Gaussian smoothing, with the time
parameter playing the role of the squared standard deviation. This
is a clean result that connects an ODE-style time evolution to a
spatial filter, and it's the start of "image processing as PDE
solving".

The Laplacian is what does the work. `∇²I` is big and positive where
the image has a local maximum, big and negative at a minimum, and
near zero in flat regions. Integrating `∂I/∂t = ∇²I` over time
pushes each pixel toward its local average — that's exactly what
blurring does.

## Why linear diffusion is wrong near edges

The Laplacian is *also* big at edges. So linear diffusion smooths
edges fastest, which is the opposite of what I usually want. If
I'm denoising an image, I want the noise (small high-frequency
fluctuations) to go away while the edges (large-scale structural
discontinuities) stay sharp.

This is the same problem the bilateral filter solves with a
data-dependent weight. Perona and Malik (1990) solved it inside the
PDE itself.

## The Perona-Malik trick

Replace the constant diffusivity in the heat equation with one that
depends on the local gradient magnitude:

```
∂I/∂t = div( c(|∇I|) · ∇I )
```

where `c(s)` is large when `s` is small and small when `s` is
large. Two reasonable choices, both in Perona & Malik's original
paper:

```
c₁(s) = exp( -(s/K)² )           — exponential, slightly favors strong edges
c₂(s) = 1 / (1 + (s/K)²)         — rational, slightly favors wider regions
```

The parameter `K` is the "scale" — gradients much smaller than `K`
diffuse freely (conductivity ≈ 1); gradients much larger than `K`
diffuse barely at all (conductivity ≈ 0). Tune `K` somewhere above
the noise level and well below the real edges.

## Discretization

Explicit Euler on a 4-neighbor stencil. For each interior pixel,
compute the four directional differences and the four conductivities:

```
I_{t+1}[i, j] = I_t[i, j] + λ · ( c_N · Δ_N + c_S · Δ_S + c_W · Δ_W + c_E · Δ_E )

where Δ_N = I[i-1, j] - I[i, j],  etc.
      c_N = c(|Δ_N|),              etc.
```

The step size `λ` must satisfy `λ ≤ 1/4` for the scheme to stay
stable — that's the von Neumann stability condition for the
4-neighbor discrete Laplacian. I default to `λ = 0.20`, which is
comfortably inside the safe region.

Boundary conditions: Neumann (zero gradient at the edges). Each
pixel's missing out-of-image neighbor contributes zero. This is
the natural BC for diffusion (no flux through the boundary) and
corresponds to assuming the image extends as a constant past its
edges.

## Measured: where does it actually help?

I ran `examples/13_anisotropic_diffusion.jl` — same Canny
configuration, same ground truth (edges of the clean image), same
1-pixel tolerance for the F1 score:

| Smoother              | edge px | precision | recall | F1    |
|-----------------------|---------|-----------|--------|-------|
| raw (no smooth)       | 677     | 0.734     | 0.995  | 0.845 |
| Gaussian σ=2          | 513     | 0.936     | 0.974  | 0.955 |
| bilateral             | 469     | 0.979     | 0.947  | 0.962 |
| PM exp K=0.05         | 516     | 0.940     | 0.985  | **0.962** |
| PM exp K=0.15         | 393     | 1.000     | 0.835  | 0.910 |
| PM rational K=0.10    | 394     | 1.000     | 0.835  | 0.910 |

Three observations:

1. **Raw has perfect recall but terrible precision.** Canny on the
   noisy image picks up almost every real edge (0.995) but also a
   pile of noise edges (precision 0.734). This is the picture
   Canny's internal Gaussian was designed to clean up.

2. **Tuned right, Perona-Malik ties bilateral.** `K = 0.05` with
   the exponential conduction lands at F1 = 0.962, identical to
   bilateral. Recall is slightly higher (0.985 vs 0.947), precision
   slightly lower (0.940 vs 0.979). Bilateral is doing the same job
   from a different mathematical starting point.

3. **K too large pushes precision to 1.0 at recall's expense.** At
   `K = 0.15` the diffusivity stops treating real edges as edges,
   smooths them like noise, and Canny finds *only* the strongest
   surviving boundaries (precision 1.000) but misses ~17% of them
   (recall 0.835). This is the bias built into the algorithm: when
   in doubt, keep smoothing.

The story: Perona-Malik gives you a knob (`K`) that explicitly
trades recall for precision. Below the right `K`, you under-smooth
and get false positives; above it, you over-smooth and get false
negatives. Bilateral gives you the same knob but through a different
parameter (`sigma_intensity`).

## Comparison to bilateral

Both filters fix the same problem (linear smoothing wrecks edges)
with the same intuition (treat edges differently from flat regions).
The differences:

| Aspect                | Bilateral                            | Perona-Malik                       |
|-----------------------|--------------------------------------|------------------------------------|
| Per-pixel work        | `O(window²)` (look at a neighborhood) | `O(1)` per iteration               |
| Total work            | one pass                             | `iterations` passes                |
| Tunable scale         | `sigma_intensity`                    | `K`                                |
| Iterable to taste     | no, run-once                         | yes, more iterations = more smoothing |
| Math                  | weighted average                     | discretized PDE                    |

Practically: bilateral is faster per call but less tunable mid-flight.
Perona-Malik takes more total CPU (one per iteration) but lets you
stop early or keep going as a single knob.

## References

- Perona, P., & Malik, J. (1990). *Scale-space and edge detection
  using anisotropic diffusion*. IEEE TPAMI, 12(7), 629–639.
