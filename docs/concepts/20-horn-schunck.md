# Horn-Schunck dense optical flow

Lucas-Kanade and Horn-Schunck dropped within months of each other
in 1981, both as proposals for solving the same underdetermined
optical-flow constraint, and the two papers have been the standard
worked examples of "local vs global formulation" in computer vision
ever since. LK regularizes by *windowing* — assume motion is
constant in a small neighbourhood, solve the resulting
overdetermined least-squares per pixel. HS regularizes by
*smoothness penalty* — write down a global cost functional with a
data term and a smoothness term, minimize it across the whole
image at once.

The two have very different properties. LK is fast, local, and
returns zero in textureless regions because there's nothing to
constrain the answer there. HS is slow (iterative global solver),
expensive, and returns a *dense* flow field — it propagates flow
from textured regions into nearby textureless ones via the
smoothness term. Neither is universally better; they're answers to
the same question with different inductive biases. Worth building
both because the contrast makes the design space visible.

## The cost functional

Horn-Schunck minimizes

```
E(u, v) = ∫∫  (Iₓ · u + I_y · v + I_t)²     # data term
           +  α² · (|∇u|² + |∇v|²)            # smoothness term
       dA
```

The data term is the same OFC residual that LK uses, just squared
instead of solved exactly inside a window. The smoothness term is
a quadratic penalty on the spatial gradients of `u` and `v`. The
coefficient `α` sets the relative weight: small `α` lets the data
term dominate and the result looks like a noisy per-pixel LK; large
`α` heavily smooths the flow and fills in textureless regions by
diffusion.

## Deriving the Jacobi iteration

The Euler-Lagrange equations of `E(u, v)` are

```
Iₓ · (Iₓ · u + I_y · v + I_t) = α² · ∇²u
I_y · (Iₓ · u + I_y · v + I_t) = α² · ∇²v
```

If I approximate the Laplacian `∇²u` by `ū − u` where `ū` is the
local mean of `u`'s neighbours (a discrete trick that Horn &
Schunck used in the original paper, and that the literature has
mostly stuck with), the equations rearrange into

```
(α² + Iₓ²) · u + Iₓ · I_y · v = α² · ū − Iₓ · I_t
Iₓ · I_y · u + (α² + I_y²) · v = α² · v̄ − I_y · I_t
```

— a 2×2 linear system per pixel coupled through `ū`, `v̄` to the
flow at neighbouring pixels. The whole thing is one big sparse
linear system. The simplest way to solve it is Jacobi iteration —
plug the current `(u, v)` into the right-hand side, solve the
local 2×2 system at each pixel, repeat:

```
u^{k+1} = ū^k − Iₓ · (Iₓ · ū^k + I_y · v̄^k + I_t) / (α² + Iₓ² + I_y²)
v^{k+1} = v̄^k − I_y · (Iₓ · ū^k + I_y · v̄^k + I_t) / (α² + Iₓ² + I_y²)
```

That's the update rule in `horn_schunck`. Each Jacobi sweep is one
correlation (for the averaging) plus a handful of pointwise
operations on the result. The whole sweep is `O(H · W)`. The
denominator `α² + Iₓ² + I_y²` is constant across iterations, so I
compute it once.

The averaging operator `ū` is the special "Horn-Schunck weighted
Laplacian":

```
1/12  1/6  1/12
1/6    0   1/6
1/12  1/6  1/12
```

— 4-neighbours at 1/6, diagonals at 1/12, centre at 0. The kernel
sums to 1 and is rotationally symmetric to a higher order than
plain 4-neighbour averaging would be.

## Picking `α`

`α` is the only knob, and the right value depends on the intensity
scale of the inputs. For images in `[0, 255]` the classical recipe
sets `α ≈ 5`–`15`; for the `[0, 1]` convention I use everywhere
else in the repo, the equivalent is `α ≈ 0.02`–`0.6`. The chunk's
default is `0.1`, which sits a bit toward the data-term-dominant
end so HS reproduces LK on textured inputs while still doing some
visible smoothing.

The `α` sweep from `examples/22_horn_schunck.jl`:

```
α = 0.05  →  recovered (u, v) = (+0.515, +0.309)
α = 0.10  →  recovered (u, v) = (+0.515, +0.309)
α = 0.50  →  recovered (u, v) = (+0.496, +0.298)
α = 2.00  →  recovered (u, v) = (+0.108, +0.065)
```

True shift is `(0.5, 0.3)`. At `α = 0.05` and `α = 0.10` HS matches
LK's accuracy (~3% bias, identical to plain LK). At `α = 0.5` the
smoothness term has started softening the answer; at `α = 2.0` it
dominates completely and the data term can't pull `u` and `v` off
zero in the iteration budget.

## The dense-flow property

The headline LK-vs-HS contrast is what happens in a textureless
region. The demo puts a textured patch in a flat grey field, moves
*only* the patch, and asks both algorithms what flow they
recovered:

```
LK on patch (interior):  u = +0.514   (matches true 0.5, ~3% bias)
LK in flat region:       u = +0.000   (no texture → no constraint → zero)
HS on patch (interior):  u = +0.507
HS in flat region:       u = +0.268   (diffused outward via the smoothness term)
```

LK can't say anything about the flat region — the 2×2 structure
tensor there has zero determinant, so the per-pixel solve returns
nothing. HS *can* say something, because the smoothness penalty
makes a zero flow in the flat region "incompatible" with the
non-zero flow next to it inside the patch. The iteration
propagates the patch's flow outward, slowly:

```
iters =   10  →  inside = +0.036   outside = +0.000
iters =   50  →  inside = +0.168   outside = +0.000
iters =  200  →  inside = +0.406   outside = +0.006
iters = 1000  →  inside = +0.501   outside = +0.110
```

At 10 iterations the flow inside the patch hasn't even fully
converged; at 1000 it's settled and the outside region is starting
to fill in. With 5000 iterations the propagation reaches across the
entire flat region, with the magnitude attenuating away from the
patch.

## What this is and isn't

HS produces a **dense** flow — every pixel has a value — but
"dense" doesn't mean "correct everywhere". The values in flat
regions are interpolated from the textured boundary, not measured.
For a flat region that's moving differently than the textured one
next to it (say, a billboard sliding past a uniform sky), HS will
produce a smooth blend instead of a step change. The flow has the
*topology* of dense, not the *information content* of dense.

Compared to LK + pyramid for the same job:

- **HS gives dense flow without pyramids.** Single algorithm, one
  knob, fills in textureless regions. Pyramids only help with big
  motions, not with rank-deficient regions.
- **HS handles big motions poorly out of the box** — the OFC
  linearization breaks down the same way it does for plain LK. A
  pyramidal HS variant exists (same coarse-to-fine driver as
  pyramidal LK), but I haven't built it; the obvious extension is
  to wrap `horn_schunck` in the same kind of pyramid loop I wrote
  for LK.
- **HS is slow.** Jacobi iteration converges at the rate of the
  Laplacian's spectrum, which is `O((α / scale)²)` per iteration in
  the worst case. For dense propagation across a flat region, that
  means hundreds to thousands of iterations. Gauss-Seidel (in-place
  updates) converges roughly twice as fast for the same number of
  sweeps, but is harder to vectorize. The literature recommends
  multigrid for serious work.

## What I deliberately skipped

- **Pyramidal HS.** Same coarse-to-fine wrapper I wrote for LK,
  applied to HS. Trivial to add — the warp + residual loop is
  algorithm-agnostic. Held back to keep this chunk focused.
- **Gauss-Seidel updates.** ~2× faster convergence than Jacobi.
  Hard to vectorize cleanly in Julia broadcasting style; would
  need a hand-rolled inner loop.
- **Robust data terms.** Replace `(OFC)²` with a Charbonnier or
  Lorentzian penalty for occlusion-robustness.
- **Anisotropic smoothness.** Replace `α² · (|∇u|² + |∇v|²)` with
  something that respects image edges (no smoothing across
  intensity discontinuities). This is where modern variational
  flow algorithms live — Brox et al., the entire post-2004
  literature.

## References

- Horn, B. K. P., & Schunck, B. G. (1981). *Determining optical
  flow*. Artificial Intelligence (journal), 17(1-3), 185-203. The
  original. Section 4 (the iterative scheme) is exactly what
  `horn_schunck` implements.
- Sun, D., Roth, S., & Black, M. J. (2010). *Secrets of optical
  flow estimation and their principles*. CVPR. A modern look at
  what makes HS-flavoured methods work in practice; useful for
  understanding why the smoothness term and the data term need to
  be balanced carefully.
- Bruhn, A., Weickert, J., & Schnörr, C. (2005). *Lucas/Kanade
  meets Horn/Schunck: combining local and global optic flow
  methods*. IJCV. Explicit construction of a "combined" formulation
  that's LK in textured regions and HS in flat ones — a satisfying
  bridge between the two algorithms.
