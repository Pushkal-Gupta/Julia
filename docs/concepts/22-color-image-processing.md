# Colour image processing

Everything in the repo up to this chunk has been single-channel.
The convolution engine, the edge operators, the filters, optical
flow — they all work on a single `Matrix{Float64}`. But `Photos`
loads colour images, `RayTracer` emits colour images, and the
ecosystem comparison (JuliaImages and friends) revolves around
typed colour types. This chunk is the bridge: a `Color` submodule
that does colour-space conversions and the few colour-specific
operators that don't naturally factor as "do it per channel".

## What's here

Two families of operations:

1. **Conversions between the three colour spaces I reach for
   most often.**
   - `rgb_to_hsv` / `hsv_to_rgb` — cylindrical coordinates around
     the colour wheel. Useful when I want to threshold on hue
     (the "find all the red things in this image" kind of task),
     or when I want to manipulate brightness independently of
     colour.
   - `rgb_to_ycbcr` / `ycbcr_to_rgb` — linear separation of
     luminance (`Y`) from chrominance differences (`Cb`, `Cr`).
     This is what JPEG uses, and it lets me apply a more
     aggressive filter to chrominance than luminance without
     visible quality loss (the eye is less sensitive to chroma).
   - `rgb_to_luminance` — Rec. 709 weighted sum, single output
     plane. The "give me a grayscale version of this colour image"
     operation. The right weights for linear RGB are
     `0.2126 R + 0.7152 G + 0.0722 B`; the older BT.601 weights
     (`0.299 / 0.587 / 0.114`) are for gamma-encoded video and
     show up everywhere by historical accident. I default to Rec.
     709 because the rest of the repo treats values as linear.

2. **Channel-aware operations** that need to *know* they're
   processing a colour image, not three independent grayscales.
   - `apply_per_channel(f, R, G, B)` — the boring glue. Runs `f`
     on each plane independently. Useful for things like
     per-channel Gaussian blur, per-channel median filtering, etc.
     where treating channels independently is the right thing.
   - `color_gradient_magnitude(R, G, B)` — Di Zenzo's 1986
     vector-valued gradient magnitude. *Not* the same as
     averaging per-channel gradient magnitudes, and that
     difference is exactly the point of the function.

## Why HSV needs writing out

The conversion is purely algebraic and small enough to fit in a
`for` loop:

```
let cmax = max(R, G, B), cmin = min(R, G, B), Δ = cmax − cmin

  V = cmax
  S = (cmax == 0) ? 0 : Δ / cmax
  H = depending on which channel is max:
        cmax == R: ((G − B) / Δ) mod 6
        cmax == G:  (B − R) / Δ + 2
        cmax == B:  (R − G) / Δ + 4
      then divide by 6 to land in [0, 1)
```

The `mod 6` on the red branch and the additive offset on the green
and blue branches is the trick that makes hue wrap around the
colour wheel continuously instead of jumping at the corners.

The round-trip is exact at machine precision (measured `1.94e-16`
max error on the demo input) because there's no float matrix
multiplication in either direction — just min/max/divide/mod.

## Why YCbCr isn't quite invertible

The BT.601 conversion matrix and its standard published inverse
are *both* truncated to six decimal places. They're not the exact
inverse of each other. The round-trip error on the demo is around
`5 × 10⁻⁷`, which is fine for any practical use but well above
machine precision. If I needed exact invertibility I'd compute the
true inverse from the forward matrix and not use the textbook
constants.

The Y plane this returns is the same `0.299 R + 0.587 G + 0.114 B`
that "luminance" means in the BT.601 / video tradition — *not* the
Rec. 709 value `rgb_to_luminance` returns. Both functions exist on
purpose; `Color.rgb_to_luminance` is the right one for linear
imagery, `Color.rgb_to_ycbcr`'s Y plane is the right one if you're
in a workflow that already deals with YCbCr chrominance.

## Di Zenzo: the right way to differentiate a colour image

Here's the trap that motivated this chunk:

```
"colour edge magnitude" ≠ mean(per-channel edge magnitudes)
```

If two channels have edges of *opposite* sign at the same pixel —
say the red intensity drops as you cross the boundary and the
green intensity rises by the same amount — the per-channel
gradient *magnitudes* are both large, but they describe edges going
in opposite directions in colour space. Averaging the magnitudes
gives a single number that's not wrong but also not the "true"
colour edge strength.

Di Zenzo's 1986 trick: think of each pixel as having a *colour
gradient*, which is a 6-vector `(R_x, G_x, B_x, R_y, G_y, B_y)`.
Stack the per-channel gradients into a 3×2 matrix and form the
2×2 *structure tensor* `J = MᵀM`:

```
J = [ Σ_c (∂c/∂x)²        Σ_c (∂c/∂x) · (∂c/∂y) ]
    [ Σ_c (∂c/∂x)(∂c/∂y)  Σ_c (∂c/∂y)²          ]
```

`Σ_c` sums over the three channels. The eigenvectors of `J` point
along the directions of fastest colour change; the eigenvalues
measure how much colour changes when you move along them. Di Zenzo's
*colour gradient magnitude* is the square root of the largest
eigenvalue:

```
|∇I|_color = √(λ_max(J))
```

For a greyscale image (R = G = B), this reduces (up to a `√3`
factor that I keep for consistency with the per-channel
normalization) to the standard scalar gradient magnitude.

The 2×2 eigenvalue is closed-form for a symmetric matrix
`[a b; b d]`:

```
λ_max = (a + d) / 2 + √(((a − d) / 2)² + b²)
```

So `color_gradient_magnitude` is one Sobel-x and one Sobel-y per
channel (six convolutions), six pointwise multiplies, three
additions, then the closed-form `λ_max` and a square root per
pixel. Cheap.

### Measured comparison

`examples/24_color_processing.jl` includes a region designed
specifically to expose the difference: a horizontal strip with
bright red on the left half and bright green on the right half,
zero blue throughout. The transition is a hard step:

- The naïve "average of per-channel gradient magnitudes" registers
  the red drop and the green rise at the boundary but averages
  with the zero-gradient blue channel, attenuating the response.
- Di Zenzo's structure-tensor formulation sees the colour as
  moving from `(0.9, 0.1, 0)` to `(0.1, 0.9, 0)` — a strong
  colour-space displacement — and registers it as a strong edge.

Measured:

```
red→green strip max:  naive average = 0.267   Di Zenzo = 0.566   (ratio 2.12×)
```

The Di Zenzo edge is over twice as strong as the naïve one. On the
greyscale-ish parts of the image (the red square, the green
circle, the yellow rectangle, the soft background) the two
methods give very similar answers — it's only at *opposite-sign
colour edges* that they diverge meaningfully. But that's the kind
of edge you usually want to detect, so the difference matters.

## What I deliberately didn't do

- **CIE Lab.** Perceptually-uniform colour space, the gold standard
  for "what looks like the same colour difference to a human?". The
  conversion goes through XYZ with a gamma-style nonlinearity, and
  the right gamma depends on whether the input is linear RGB or
  sRGB. Skipped to keep the chunk focused on conversions that
  don't need to know about gamma. The trip RGB → linear → XYZ →
  Lab and back is a clean follow-up.
- **Per-channel histogram equalization.** Tints the output because
  each channel gets stretched independently. The right way is to
  equalize the Y of YCbCr (or the V of HSV) and leave the chroma
  alone. Easy on top of what's here.
- **Demosaicing.** Bayer-pattern reconstruction. Different problem
  entirely — input is single-channel raw data, not three RGB
  planes.

## References

- Di Zenzo, S. (1986). *A note on the gradient of a multi-image*.
  CVGIP, 33(1), 116-125. The original two-page note that
  introduced the structure-tensor formulation for colour
  gradients.
- ITU-R Recommendation BT.601 (1982, repeatedly revised). *Studio
  encoding parameters of digital television for standard 4:3 and
  wide-screen 16:9 aspect ratios*. The source of the `0.299 / 0.587
  / 0.114` luminance weights and the YCbCr conversion matrix.
- ITU-R Recommendation BT.709 (1990, revised). *Parameter values
  for the HDTV standards for production and international programme
  exchange*. The source of the `0.2126 / 0.7152 / 0.0722` linear
  luminance weights.
- Smith, A. R. (1978). *Color gamut transform pairs*. SIGGRAPH.
  Where the HSV cylindrical model comes from.
