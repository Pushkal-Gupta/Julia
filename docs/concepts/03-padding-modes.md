# 03 — Padding modes

When a kernel slides past the edge of an image, those out-of-bounds
positions need values. The choice is silent — there's no warning, no
broken output — but it changes every pixel within `kernel_radius` of the
border. For a 21×21 Gaussian on a 96×96 image, that's the outermost
~40% of pixels.

`ImageLab.Padding` exposes six options. The visual difference is
captured in `examples/02_padding_modes_studio.jl`, which blurs an image
with an off-center bright object under each mode and assembles a 2×3
montage.

## The modes

```
Source row:       1 2 3 4 5
:zero      → 0 0 [1 2 3 4 5] 0 0
:replicate → 1 1 [1 2 3 4 5] 5 5
:reflect   → 3 2 [1 2 3 4 5] 4 3
:symmetric → 2 1 [1 2 3 4 5] 5 4
:circular  → 4 5 [1 2 3 4 5] 1 2
:valid     → just  [1 2 3 4 5]   (output shrinks)
```

- **`:zero`** — assumes the world outside is black. The simplest, and the
  default in many DSP texts. *Cost:* introduces a fake dark edge that
  contaminates anything brighter than `0` near the border. You'll see
  a dark halo around bright content in the studio output.

- **`:replicate`** — clamps the index to the image bounds. The edge color
  extends outward. *Cost:* assumes the world is locally constant. Good
  when there's no strong gradient at the border, bad when there is (the
  edge becomes "flat", which dampens the gradient response).

- **`:reflect`** — mirrors across the border *without repeating the
  border pixel*. This is the textbook choice for smoothing because it
  preserves local smoothness: a horizontal ramp continues smoothly past
  the border in mirror form, so a Gaussian blur sees no discontinuity.
  Used by `scipy.ndimage`'s default.

- **`:symmetric`** — mirrors *with* the border pixel repeated. Subtly
  different from `:reflect`; differs only at exactly the border. Common
  in wavelets.

- **`:circular`** — wraps around (torus). This is what an FFT-based
  convolution implicitly does, because the DFT treats the signal as
  periodic. Useful when you know the signal really is periodic (e.g.
  textures); destructive otherwise — bright pixels in one corner can
  bleed into the opposite corner.

- **`:valid`** — no padding. The output shrinks by `(kh - 1, kw - 1)`.
  Often the right answer when you only care about "real" values and
  intend to crop anyway, especially for stacked convolutions in deep
  learning.

## Decision table

| Goal                              | Recommended mode       |
|-----------------------------------|------------------------|
| Smoothing for noise reduction     | `:reflect`             |
| Smoothing before gradient compute | `:reflect`             |
| You implement FFT-based conv      | `:circular`            |
| You want to crop after anyway     | `:valid`               |
| The image really is on a torus    | `:circular`            |
| You don't trust the border at all | `:replicate` and crop  |
| Quick-and-dirty / textbook DSP    | `:zero`                |

## How to see it for yourself

```sh
julia --project=. examples/02_padding_modes_studio.jl
open artifacts/02_padding_modes_studio/montage.pgm
```

In the montage, look at:

- The top-left corner across `:zero` (halo) vs `:replicate` (no halo).
- The bottom edge under `:reflect` vs `:symmetric` — almost identical.
- Anywhere on the border for `:circular` — content from the opposite side
  is now leaking in.
