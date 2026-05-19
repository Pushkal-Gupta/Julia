"""
    Features

Mid-level computer vision primitives — things that consume the output
of edge detection and turn it into something semantically richer:
corner points, line parameters, connected segments, template matches.

Each one is small enough to read and modify, and each one builds on
machinery already in `Edges`, `Convolution`, or `Filters`. The point
of this module isn't to be production-grade — it's to make every step
inspectable.
"""
module Features

using ..Padding
using ..Kernels: gaussian1d
using ..Convolution: separable_correlate2d
using ..Edges: gradient

export harris_response, harris_corners,
       hough_lines, hough_peaks, HoughAccumulator,
       connected_components, component_sizes,
       normalized_cross_correlation, ncc_peaks

# ── Harris corners ────────────────────────────────────────────────────────────

"""
    harris_response(img; sigma=1.0, k=0.04, pad=:replicate) -> Matrix{Float64}

The Harris corner response `R = det(M) - k · trace(M)²`, where `M` is
the 2×2 structure tensor of the gradient: `M = [[Σ Ix², Σ Ix·Iy],
[Σ Ix·Iy, Σ Iy²]]`. The window is a separable Gaussian of standard
deviation `sigma`.

The response is large and positive at corners, large and negative
along edges, near zero in flat regions. The standard `k` is 0.04
(Harris's original suggestion); 0.06 is also common and slightly less
edge-responsive.
"""
function harris_response(img::AbstractMatrix{<:Real};
                         sigma::Real = 1.0, k::Real = 0.04,
                         pad::Symbol = :replicate)
    sigma > 0 || throw(ArgumentError("sigma must be positive, got $sigma"))
    Ix, Iy = gradient(Float64.(img), :sobel; pad = pad)
    Ixx = Ix .* Ix
    Iyy = Iy .* Iy
    Ixy = Ix .* Iy

    n = 2 * ceil(Int, 3 * sigma) + 1
    g = gaussian1d(n; sigma = sigma)
    Sxx = separable_correlate2d(Ixx, g, g; pad = pad)
    Syy = separable_correlate2d(Iyy, g, g; pad = pad)
    Sxy = separable_correlate2d(Ixy, g, g; pad = pad)

    det_M   = Sxx .* Syy .- Sxy .^ 2
    trace_M = Sxx .+ Syy
    return det_M .- k .* trace_M .^ 2
end

"""
    harris_corners(img; sigma, k, threshold, min_distance, pad) -> Vector{Tuple{Int,Int}}

Run Harris and pick local maxima above `threshold` (as a fraction of
the maximum response). `min_distance` enforces minimum spacing
between corners — at most one corner per `(2·min_distance + 1)` window
in Chebyshev distance.

Returns `(row, col)` index pairs. Sorted by descending response, so
the first entries are the strongest corners.
"""
function harris_corners(img::AbstractMatrix{<:Real};
                        sigma::Real = 1.0,
                        k::Real = 0.04,
                        threshold::Real = 0.01,
                        min_distance::Integer = 3,
                        pad::Symbol = :replicate)
    R = harris_response(img; sigma = sigma, k = k, pad = pad)
    rmax = maximum(R)
    rmax ≤ 0 && return Tuple{Int, Int}[]
    cutoff = threshold * rmax

    # Collect candidates with response above cutoff.
    cand = Tuple{Float64, Int, Int}[]
    H, W = size(R)
    @inbounds for j in 1:W, i in 1:H
        R[i, j] > cutoff && push!(cand, (R[i, j], i, j))
    end
    sort!(cand; by = x -> -x[1])

    # Greedy NMS with Chebyshev distance.
    keeps = Tuple{Int, Int}[]
    for (_, i, j) in cand
        ok = true
        for (ki, kj) in keeps
            if max(abs(i - ki), abs(j - kj)) ≤ min_distance
                ok = false
                break
            end
        end
        ok && push!(keeps, (i, j))
    end
    return keeps
end

# ── Hough transform for lines ─────────────────────────────────────────────────

"""
    HoughAccumulator

The accumulator from a line-Hough transform.

- `counts[t, r]` = number of edge pixels that voted for the `(θ[t],
  rho[r])` line.
- `thetas` are bin centers in `[0, π)`.
- `rhos` are bin centers in `[-rmax, rmax]` (signed distance from
  origin, where origin is the image's `(1, 1)` corner).
"""
struct HoughAccumulator
    counts::Matrix{Int}
    thetas::Vector{Float64}
    rhos::Vector{Float64}
end

"""
    hough_lines(edges::BitMatrix; n_theta=180, rho_step=1.0) -> HoughAccumulator

Build the (θ, ρ) accumulator for line detection. Each edge pixel at
`(row, col)` votes for every line that passes through it,
parameterized as `ρ = col · cos(θ) + row · sin(θ)`. We sweep `θ` over
`n_theta` bins in `[0, π)`.

This is the slow-but-clear implementation: a double loop over edge
pixels × θ-bins. For a typical Canny output (a few hundred to a few
thousand edge pixels) it runs in milliseconds.
"""
function hough_lines(edges::BitMatrix;
                     n_theta::Integer = 180, rho_step::Real = 1.0)
    n_theta ≥ 1 || throw(ArgumentError("n_theta must be ≥ 1, got $n_theta"))
    rho_step > 0 || throw(ArgumentError("rho_step must be > 0, got $rho_step"))
    H, W = size(edges)

    thetas = [(t - 0.5) * π / n_theta for t in 1:n_theta]   # bin centers in [0, π)
    cosθ = cos.(thetas)
    sinθ = sin.(thetas)

    rho_max = sqrt(H^2 + W^2)
    n_rho = 2 * ceil(Int, rho_max / rho_step) + 1
    rho_offset = (n_rho + 1) ÷ 2   # bin index for ρ = 0
    rhos = [(r - rho_offset) * rho_step for r in 1:n_rho]

    counts = zeros(Int, n_theta, n_rho)
    @inbounds for j in 1:W, i in 1:H
        edges[i, j] || continue
        for t in 1:n_theta
            ρ = j * cosθ[t] + i * sinθ[t]
            r = round(Int, ρ / rho_step) + rho_offset
            (1 ≤ r ≤ n_rho) && (counts[t, r] += 1)
        end
    end
    return HoughAccumulator(counts, thetas, rhos)
end

"""
    hough_peaks(acc; threshold=0.5, min_distance_theta=5, min_distance_rho=5, max_peaks=Inf)
        -> Vector{Tuple{Float64, Float64}}

Pick local maxima in the accumulator, returning `(θ, ρ)` line
parameters sorted by vote count descending. `threshold` is a fraction
of the accumulator max. `min_distance_*` are minimum separations in
bin units (each direction independent).
"""
function hough_peaks(acc::HoughAccumulator;
                     threshold::Real = 0.5,
                     min_distance_theta::Integer = 5,
                     min_distance_rho::Integer = 5,
                     max_peaks::Real = Inf)
    counts = acc.counts
    cmax = maximum(counts)
    cmax == 0 && return Tuple{Float64, Float64}[]
    cutoff = threshold * cmax

    n_theta, n_rho = size(counts)
    cand = Tuple{Int, Int, Int}[]   # (count, t, r)
    @inbounds for r in 1:n_rho, t in 1:n_theta
        counts[t, r] ≥ cutoff && push!(cand, (counts[t, r], t, r))
    end
    # Primary: descending count. Secondary: ascending θ so vertical lines
    # come out at θ ≈ 0 rather than θ ≈ π (the two are mathematically the
    # same line, but predictable output is friendlier).
    sort!(cand; by = x -> (-x[1], x[2]))

    keeps = Tuple{Int, Int}[]
    for (_, t, r) in cand
        length(keeps) ≥ max_peaks && break
        ok = true
        for (kt, kr) in keeps
            if abs(t - kt) ≤ min_distance_theta && abs(r - kr) ≤ min_distance_rho
                ok = false
                break
            end
        end
        ok && push!(keeps, (t, r))
    end
    return [(acc.thetas[t], acc.rhos[r]) for (t, r) in keeps]
end

# ── Connected components ──────────────────────────────────────────────────────

"""
    connected_components(mask::BitMatrix; connectivity=8) -> (labels, n_components)

Label each connected region of `true` pixels in `mask`. Background
stays zero; each component gets a unique positive integer.
`connectivity` is either `4` (axis neighbors) or `8` (also diagonals).

Stack-based DFS, single pass, `O(H · W)`.
"""
function connected_components(mask::BitMatrix; connectivity::Integer = 8)
    connectivity in (4, 8) || throw(ArgumentError(
        "connectivity must be 4 or 8, got $connectivity"))
    H, W = size(mask)
    labels = zeros(Int, H, W)
    n = 0

    offsets = connectivity == 4 ?
        ((-1, 0), (1, 0), (0, -1), (0, 1)) :
        ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    stack = Tuple{Int, Int}[]
    @inbounds for j in 1:W, i in 1:H
        (mask[i, j] && labels[i, j] == 0) || continue
        n += 1
        labels[i, j] = n
        push!(stack, (i, j))
        while !isempty(stack)
            ci, cj = pop!(stack)
            for (di, dj) in offsets
                ni, nj = ci + di, cj + dj
                (1 ≤ ni ≤ H && 1 ≤ nj ≤ W) || continue
                if mask[ni, nj] && labels[ni, nj] == 0
                    labels[ni, nj] = n
                    push!(stack, (ni, nj))
                end
            end
        end
    end
    return labels, n
end

"""
    component_sizes(labels::Matrix{Int}, n::Integer) -> Vector{Int}

Return the pixel count of each component (`sizes[k]` = number of
pixels with `labels .== k`). Background (label 0) is not counted.
"""
function component_sizes(labels::AbstractMatrix{Int}, n::Integer)
    sizes = zeros(Int, n)
    @inbounds for v in labels
        v > 0 && (sizes[v] += 1)
    end
    return sizes
end

# ── Normalized cross-correlation ──────────────────────────────────────────────

"""
    normalized_cross_correlation(img, template) -> Matrix{Float64}

Slide `template` over `img` and report the Pearson correlation
coefficient of each window with the template. Values in `[-1, 1]`:
+1 is a perfect match, -1 is a perfect inverse, 0 is uncorrelated.

The output is `:valid`-shaped: `(H - th + 1, W - tw + 1)`. Brightness
and contrast cancel out per window, so this matches templates against
unevenly-lit images in a way that plain correlation can't.
"""
function normalized_cross_correlation(img::AbstractMatrix{<:Real},
                                      template::AbstractMatrix{<:Real})
    th, tw = size(template)
    H, W = size(img)
    (H ≥ th && W ≥ tw) || throw(ArgumentError(
        "template $(size(template)) is larger than image $((H, W))"))
    A = Float64.(img)
    T = Float64.(template)
    nT = th * tw

    # Pre-compute template statistics once.
    μT = sum(T) / nT
    Tc = T .- μT
    σT = sqrt(sum(Tc .^ 2))
    σT == 0 && throw(ArgumentError("template is constant; NCC is undefined"))

    out_h, out_w = H - th + 1, W - tw + 1
    out = Matrix{Float64}(undef, out_h, out_w)
    @inbounds for j in 1:out_w, i in 1:out_h
        # Window mean
        s = 0.0
        for jj in 1:tw, ii in 1:th
            s += A[i + ii - 1, j + jj - 1]
        end
        μw = s / nT
        # Cross term and window variance
        num = 0.0
        sw² = 0.0
        for jj in 1:tw, ii in 1:th
            a = A[i + ii - 1, j + jj - 1] - μw
            num  += a * Tc[ii, jj]
            sw²  += a * a
        end
        denom = sqrt(sw²) * σT
        out[i, j] = denom == 0 ? 0.0 : num / denom
    end
    return out
end

"""
    ncc_peaks(ncc; threshold=0.7, min_distance=5) -> Vector{Tuple{Int,Int}}

Find local maxima of an NCC map above `threshold`, with a Chebyshev
`min_distance` between accepted matches. Returns top-left positions
sorted by descending score.
"""
function ncc_peaks(ncc::AbstractMatrix{<:Real};
                   threshold::Real = 0.7,
                   min_distance::Integer = 5)
    H, W = size(ncc)
    cand = Tuple{Float64, Int, Int}[]
    @inbounds for j in 1:W, i in 1:H
        ncc[i, j] ≥ threshold && push!(cand, (ncc[i, j], i, j))
    end
    sort!(cand; by = x -> -x[1])
    keeps = Tuple{Int, Int}[]
    for (_, i, j) in cand
        ok = true
        for (ki, kj) in keeps
            if max(abs(i - ki), abs(j - kj)) ≤ min_distance
                ok = false; break
            end
        end
        ok && push!(keeps, (i, j))
    end
    return keeps
end

end # module Features
