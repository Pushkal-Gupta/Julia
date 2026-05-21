# Differentiable filters

The whole repo up to this point treats filter kernels as fixed
numbers I write down — Sobel, Prewitt, Scharr, Laplacian. The
operator is a `Matrix{Float64}` literal in `src/kernels.jl`. This
chunk takes the opposite stance: the kernel is a free parameter,
the user supplies an example of what they want the filter to do,
and a gradient-based optimizer figures out the numbers.

The technical content is short — it's basically *one observation*
that my naive convolution code from the first chunk of this repo is
already generic enough to support it. But the conceptual payoff is
large: it's the line that connects classical CV (write the kernel)
to modern deep CV (learn the kernel), through a single Julia
package and one ~100-line submodule.

## The one observation

`Convolution.correlate2d` is declared like this:

```julia
function correlate2d(image::AbstractMatrix{T}, kernel::AbstractMatrix{K};
                     pad::Symbol = :replicate) where {T<:Real, K<:Real}
    R = promote_type(T, K, Float64)
    ...
end
```

It's generic over `K<:Real`. `ForwardDiff.Dual` numbers are a
subtype of `Real`. So when I pass a kernel made of `Dual` numbers,
`promote_type` decides the output element type is `Dual`, the
output matrix gets allocated as `Matrix{Dual}`, and every `+`, `*`,
`sum` inside `_correlate_into!` automatically routes through the
overloaded `Dual` arithmetic.

I didn't write any of that. ForwardDiff did, by defining `Dual`
methods for the basic arithmetic operations. My naive convolution
code from the very first chunk works unchanged.

That's the punchline. The rest of this doc is what I built on top.

## Forward-mode AD in 60 seconds

A `Dual` number is a pair `(x, ẋ)` representing a value `x` and a
derivative `ẋ` w.r.t. some scalar input. The arithmetic rules just
implement the chain rule on the second component:

```
(x, ẋ) + (y, ẏ) = (x + y, ẋ + ẏ)
(x, ẋ) * (y, ẏ) = (x · y, x · ẏ + y · ẋ)
f((x, ẋ))       = (f(x), f'(x) · ẋ)
```

So if I pass `x = (3.0, 1.0)` through any function `f`, the result
is `(f(3.0), f'(3.0))`. The derivative computed *in the same pass*
as the value, hence "forward mode". The cost is a constant factor
over the original evaluation — for the chain `x → y → z`, computing
`(z, dz/dx)` costs about 2× computing `z` alone.

For a function with `n` inputs, I need `n` partial derivatives, so
the cost grows to `(n + 1) ×` the original. ForwardDiff batches them
using vector-valued duals (`Dual{T, V, N}` with `N` partials per
number) and "chunking", which keeps the constant manageable for
small `n`.

For a 3×3 kernel that's `n = 9` parameters. ForwardDiff's default
chunk size handles all 9 in one sweep through `correlate2d`.

## The loss function

I want a scalar that's *small* when the kernel does the right thing
and *large* when it doesn't. The standard choice for regression-flavoured
problems: mean-squared error between the filter output and a target.

```julia
function kernel_loss(kernel_flat, img, target; pad = :replicate)
    K   = reshape(kernel_flat, k, k)
    out = correlate2d(img, K; pad = pad)
    diff = out .- target
    return sum(abs2, diff) / length(diff)
end
```

A few design notes I had to think about:

- **Why flatten the kernel.** `ForwardDiff.gradient` takes a vector
  of parameters and returns a vector of partials. Keeping the
  kernel flat matches that shape. The reshape inside the loss is
  cheap and gets erased by the compiler.
- **Why MSE.** It's smooth, convex in the output, gives a
  well-conditioned gradient. L1 would also work but the gradient
  has a discontinuity at zero residual.
- **Why explicit `length(diff)` normalization.** So the loss is
  scale-invariant in image size; the learning rate transfers
  between 32×32 and 128×128.

`kernel_gradient` is then a one-liner:

```julia
kernel_gradient(k, img, target; pad) =
    gradient(k -> kernel_loss(k, img, target; pad = pad), k)
```

That `gradient` is `ForwardDiff.gradient`. Returns a `Vector{Float64}`
of length 9 for a 3×3 kernel — the partial of the loss w.r.t. each
kernel entry.

## SGD on top

The minimal optimization loop:

```julia
for _ in 1:iterations
    push!(history, kernel_loss(k, img, target; pad))
    grad = kernel_gradient(k, img, target; pad)
    @. k = k - lr * grad
end
```

No Adam, no momentum, no minibatches. For 9 parameters and a
well-conditioned MSE loss, vanilla SGD is fine. `lr = 0.1` and
`iterations = 800` converge well in my tests; `lr = 0.5` blows up
because the gradient magnitude on a high-contrast image
overshoots, and `lr = 0.01` is slow.

## The headline numbers

`examples/19_differentiable_filters.jl` runs the loop for four
target kernels — Sobel-x, Sobel-y, Laplacian-4, sharpen — starting
from random initialization:

```
sobel_x       loss[1]=1.169e+00 → loss[end]=1.910e-12   L∞ kernel err = 2.52e-06
sobel_y       loss[1]=1.024e+00 → loss[end]=1.125e-12   L∞ kernel err = 2.05e-06
laplacian4    loss[1]=1.733e+00 → loss[end]=5.385e-12   L∞ kernel err = 5.29e-06
sharpen       loss[1]=2.708e+00 → loss[end]=7.141e-12   L∞ kernel err = 6.29e-06
```

Loss drops by 12 orders of magnitude. The learned Sobel-x kernel
matches the hand-written one to 4 decimal places:

```
learned         |   hand-written
-1.0000  -0.00000  +1.0000   |   -1  +0  +1
-2.0000  -0.00000  +2.0000   |   -2  +0  +2
-1.0000  +0.00000  +1.0000   |   -1  +0  +1
```

That's a 1989 paper recoverable in 800 lines of gradient descent.

## Why I trained on noise, not on a real image

My first attempt trained on a checkerboard. The fit converged on
loss but the recovered kernel was wrong by ~0.12 in some entries.
The reason: a checkerboard has only a few distinct local 3×3
patches. The system "what kernel turns these patches into these
outputs?" is *underdetermined* — multiple kernels solve it.

A uniform-noise image has the opposite property: every 3×3 patch is
essentially unique. Now the system is well-posed and the optimizer
recovers the kernel uniquely. The lesson is the same one that
shows up in deep learning training data: if your inputs don't span
the function class you're trying to learn, the optimizer will find
*some* solution that fits the training set, but it won't be the
solution you wanted.

That's also the test I codified in `test/test_autodiff.jl` — I
deliberately use a noise image so the convergence assertion is
meaningful.

## What this opens up

Once a filter is differentiable, all the modern CV moves become
available:

- **Filter pre-training** — fit a filter to a known target operator
  (the Sobel demo). Useful when I want a fast separable approximation
  to a slow exact filter.
- **Joint kernel + threshold optimization** — make the whole Canny
  pipeline differentiable, then learn `σ`, `low`, `high`, *and* the
  smoothing kernel jointly to maximize F1 against an edge ground
  truth. Would need a soft (smooth) version of the discrete
  threshold operations.
- **Filter banks** — instead of one kernel, learn a stack and
  combine them with another learned linear layer. That's literally
  the first layer of a convolutional neural network.
- **Image-to-image regression** — learn a kernel that turns a noisy
  image into a clean one. The training pair is `(noisy, clean)`,
  not `(input, ground-truth-edge)`. A linear filter is a weak
  hypothesis class but you can already learn surprisingly good
  denoisers this way for additive Gaussian noise.

None of those are in this submodule — I'm keeping the scope to the
"differentiable correlation + SGD" core. The submodule is ~100
lines including doc strings; everything else above is what gets
unlocked.

## What I deliberately didn't do

- **Reverse-mode AD via Zygote** — for ≤ 100-ish parameters,
  forward mode is comparable or faster. Reverse mode shines for
  millions of parameters (= deep nets).
- **GPU acceleration** — the gradient already lands in
  ~0.5 seconds for 9 parameters and a 64×64 image. Once kernels
  get to 11×11 or larger, batching across many parallel autodiff
  calls would be worth a thought.
- **A proper optimizer (Adam / RMSProp)** — vanilla SGD already
  converges 12 orders of magnitude. Adam would converge faster but
  the demo isn't bottlenecked on training speed.

## References

- Bischof, C., Carle, A., Khademi, P., & Mauer, A. (1996).
  *ADIFOR 2.0: Automatic differentiation of Fortran 77 programs* —
  the original forward-mode-by-source-transformation paper.
- Revels, J., Lubin, M., & Papamarkou, T. (2016). *Forward-mode
  automatic differentiation in Julia* — the paper behind ForwardDiff
  itself. The trick is exactly the operator overloading I described
  above.
- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M.
  (2017). *Automatic differentiation in machine learning: a survey*
  — the standard reference for the forward / reverse distinction.
- ForwardDiff.jl docs — <https://juliadiff.org/ForwardDiff.jl/>
