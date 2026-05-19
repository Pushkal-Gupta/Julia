# Canny from scratch

Canny is what you build when you've spent enough time staring at
gradient magnitude images to be irritated by three things:

1. The "edge" is two or three pixels wide and you can't tell which one
   is the *actual* edge.
2. Noise creates speckle that looks just like a real edge if you
   threshold naively.
3. Long edges break into dotted lines because the gradient dips below
   your threshold in a few places.

Canny addresses each one with a specific step. The whole pipeline is
just gradient + three post-processing ideas, but it's the combination
that makes it work.

## The four stages

1. **Pre-smooth with a Gaussian.** Sets the scale and kills the
   noise-vs-signal problem. σ is the one knob you really tune — it
   controls how big a feature has to be to count as an edge. For
   most synthetic test images, σ around 1.0–1.5 is fine.

2. **Compute gradient magnitude and direction** (Sobel works best;
   Scharr would be slightly better for rotational symmetry but the
   difference is marginal in practice).

3. **Non-maximum suppression along the gradient direction.** This is
   the trick that thins the edge. For each pixel, look at the two
   neighbors lying along the gradient direction. If you're not at
   least as bright as both of them, you're not at the *peak* of the
   edge — you're on the slope. Zero out.

4. **Double threshold + hysteresis.** Pick a high threshold for
   "definitely an edge" and a low threshold for "maybe an edge".
   Anything below low is gone. Anything above high is in. Pixels in
   between survive only if they connect (via 8-neighborhood) to
   something that's already in. This is the broken-edge fix.

The pipeline output is a clean one-pixel-wide binary edge map — the
line drawing you keep imagining when you look at a gradient
magnitude image.

## Non-maximum suppression in detail

The discrete gradient direction `atan(gy, gx)` falls into eight
possible sectors when you mod-π it (because edge orientation is
symmetric). The four NMS-relevant sectors are:

| Sector | Gradient direction | Neighbors I compare against |
|--------|--------------------|------------------------------|
| 0      | horizontal (east-west)   | `(i, j-1)` and `(i, j+1)` |
| 1      | NE diagonal (↗)          | `(i-1, j+1)` and `(i+1, j-1)` |
| 2      | vertical (north-south)   | `(i-1, j)` and `(i+1, j)` |
| 3      | NW diagonal (↖)          | `(i-1, j-1)` and `(i+1, j+1)` |

The edge runs perpendicular to the gradient. So in sector 2 — gradient
vertical — the edge is *horizontal*, and the relevant neighbors to
suppress against sit above and below the current pixel.

In `Edges.nonmaximum_suppression`, this is a single `@inbounds` loop
over interior pixels. Each pixel is either kept (with its magnitude)
or set to zero. The result is a magnitude image where edges are
exactly one pixel wide.

A subtle thing: this is a *strict* one-pixel-wide assumption. For
edges that aren't axis-aligned and aren't at 45° either, the "true"
edge passes through pixels at an angle and the sub-pixel-accurate NMS
would interpolate. Some implementations do this. I don't — the
quantized 4-direction version is good enough for almost everything,
and it's what makes the code readable.

## Double threshold + hysteresis

I take `low` and `high` as fractions of the max NMS magnitude by
default. So `low=0.05, high=0.15` means "weak threshold = 5% of the
brightest NMS pixel, strong threshold = 15%". Three buckets:

- `magnitude ≥ high` → strong
- `low ≤ magnitude < high` → weak
- `magnitude < low` → discarded

Hysteresis then runs a flood fill from every strong pixel, recruiting
any weak pixel that's 8-connected to the strong set (directly or via
other weak pixels). The implementation is a stack-based DFS:

```julia
function hysteresis(strong, weak)
    out = copy(strong)
    stack = [(i, j) for (i, j) in CartesianIndices(strong) if strong[i, j]]
    while !isempty(stack)
        i, j = pop!(stack)
        for di in -1:1, dj in -1:1
            ...
            if weak[ni, nj] && !out[ni, nj]
                out[ni, nj] = true
                push!(stack, (ni, nj))
            end
        end
    end
    return out
end
```

Each weak pixel is visited at most once, so the whole hysteresis pass
is `O(H · W)`.

## Tuning σ and the thresholds

I run `examples/07_canny_parameter_sweep.jl` whenever I'm building
intuition for a new kind of input. Three observations from that
study:

- At small σ, thresholds matter a lot — noise edges and real edges
  are both surviving NMS, and the threshold is what discriminates.
- At larger σ, thresholds matter much less — by the time the noise
  has been smoothed away, almost everything NMS keeps is a "real"
  edge.
- The high threshold sets which edges are unambiguous; the low
  threshold sets how aggressive the recruitment is. A useful default
  ratio is `high / low ≈ 2.5`–`3`.

For real images the sweet spot depends heavily on contrast and
noise level. For synthetic test images (high contrast, mild noise),
`σ ≈ 1.0–1.5` and `low ≈ 0.05, high ≈ 0.15` usually look right.

## What this code doesn't do

A few things a full-fat Canny would have that I'm deliberately not
building:

- Sub-pixel interpolation in NMS. (Useful when you need sub-pixel
  edge localization; usually overkill.)
- Adaptive thresholds (Otsu-style on the gradient magnitude). Easy
  to add later if I ever want it.
- Edge linking that traces edges into segments. The output here is a
  pixel mask, not a list of curves. Connected-component labeling
  (later in the features layer) can extract segments from this.

## References

- Canny, J. (1986). *A computational approach to edge detection*.
  IEEE TPAMI, PAMI-8(6), 679–698.
