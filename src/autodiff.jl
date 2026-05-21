"""
    AutoDiff

Differentiable filters. The whole submodule is one observation: the
naive `Convolution.correlate2d` I wrote in the very first chunk of this
repo is generic over its element type, so I can pass a kernel made of
`ForwardDiff.Dual` numbers, get back a `Matrix` of `Dual` numbers, and
read the partial derivatives out for free. No autodiff library needs
to know anything about convolution; it just needs the arithmetic.

What that gets us: any classical filter kernel becomes a *learnable*
parameter. Define a loss between a filtered image and a target, and
the gradient of that loss with respect to the kernel coefficients
drops out. The demo in `examples/19_differentiable_filters.jl` uses
this to *learn* the Sobel-x kernel from input/target pairs.

This is the bridge from classical CV (write the kernel by hand) to
modern CV (let an optimizer pick it). Same operator, different stance
on where the numbers in the kernel come from.
"""
module AutoDiff

using ForwardDiff: gradient
using ..Convolution: correlate2d

export kernel_loss, kernel_gradient, fit_kernel

"""
    kernel_loss(kernel_flat, img, target; pad = :replicate) -> Real

Mean-squared error between `correlate2d(img, K)` and `target`, where
`K = reshape(kernel_flat, k, k)` and `k = √length(kernel_flat)`. The
kernel must be square with odd side length. This is the scalar
function I differentiate.

Why a *flat* kernel parameter: `ForwardDiff.gradient` takes a vector
of parameters and a function of that vector. Keeping the kernel flat
matches that shape; reshaping inside the loss is cheap.
"""
function kernel_loss(kernel_flat::AbstractVector{<:Real},
                     img::AbstractMatrix{<:Real},
                     target::AbstractMatrix{<:Real};
                     pad::Symbol = :replicate)
    n = length(kernel_flat)
    k = isqrt(n)
    k * k == n || throw(ArgumentError(
        "kernel_flat length $n is not a perfect square"))
    isodd(k) || throw(ArgumentError(
        "kernel side length $k must be odd"))
    K = reshape(kernel_flat, k, k)
    out = correlate2d(img, K; pad = pad)
    size(out) == size(target) || throw(DimensionMismatch(
        "filtered output size $(size(out)) ≠ target size $(size(target))"))
    diff = out .- target
    return sum(abs2, diff) / length(diff)
end

"""
    kernel_gradient(kernel_flat, img, target; pad = :replicate) -> Vector

Forward-mode AD gradient of `kernel_loss` with respect to
`kernel_flat`. For a small kernel (9 entries for 3×3) ForwardDiff's
chunked Dual numbers usually compute the whole gradient in one
sweep of `correlate2d`.
"""
function kernel_gradient(kernel_flat::AbstractVector{<:Real},
                         img::AbstractMatrix{<:Real},
                         target::AbstractMatrix{<:Real};
                         pad::Symbol = :replicate)
    return gradient(k -> kernel_loss(k, img, target; pad = pad), kernel_flat)
end

"""
    fit_kernel(img, target; ksize = 3, iterations = 200, lr = 0.5,
               pad = :replicate, init = nothing)
        -> (kernel::Matrix{Float64}, history::Vector{Float64})

Vanilla gradient descent on `kernel_loss`. Returns the learned `k × k`
kernel and the loss at every iteration. `init`, if given, must be a
length-`ksize²` vector or a `ksize × ksize` matrix; otherwise the
optimizer starts from small random values.

No Adam, no momentum, no minibatching — this is the simplest thing
that demonstrates the idea. For the Sobel-learning demo, 500 SGD
steps at `lr = 0.1` converges to within `~1e-2` of the true Sobel
kernel; 2000 steps takes that to `~1e-4`.
"""
function fit_kernel(img::AbstractMatrix{<:Real},
                    target::AbstractMatrix{<:Real};
                    ksize::Integer = 3,
                    iterations::Integer = 500,
                    lr::Real = 0.1,
                    pad::Symbol = :replicate,
                    init = nothing)
    isodd(ksize) || throw(ArgumentError("ksize must be odd, got $ksize"))
    n = ksize * ksize
    k = if init === nothing
        0.1 .* randn(n)
    else
        v = collect(Float64, vec(init))
        length(v) == n || throw(DimensionMismatch(
            "init has $(length(v)) entries; expected $n"))
        v
    end
    history = Float64[]
    for _ in 1:iterations
        push!(history, kernel_loss(k, img, target; pad = pad))
        grad = kernel_gradient(k, img, target; pad = pad)
        @. k = k - lr * grad
    end
    return reshape(copy(k), ksize, ksize), history
end

end # module AutoDiff
