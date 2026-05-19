# Padding modes

When a kernel slides past the edge of an image, those out-of-bounds
positions need values from somewhere. The choice is silent — there's no
warning, no broken output — but it changes every pixel within
`kernel_radius` of the border. For a 21×21 Gaussian on a 96×96 image
that's the outermost ~40% of pixels.

`ImageLab.Padding` exposes six options. The visual difference is in
`examples/02_padding_modes_studio.jl`, which blurs an image with an
off-center bright object under each mode and tiles them into a 2×3
montage.

## The modes

```
Source row:        1 2 3 4 5
:zero      →  0 0 [1 2 3 4 5] 0 0
:replicate →  1 1 [1 2 3 4 5] 5 5
:reflect   →  3 2 [1 2 3 4 5] 4 3
:symmetric →  2 1 [1 2 3 4 5] 5 4
:circular  →  4 5 [1 2 3 4 5] 1 2
:valid     →    just [1 2 3 4 5]   (output shrinks)
```

- **`:zero`** — assumes the world outside is black. The default in many
  DSP texts. Downside: introduces a fake dark edge that contaminates
  anything brighter than 0 near the border. In the studio output you
  can see a dark halo around bright content.

- **`:replicate`** — clamps the index. The edge color extends outward.
  Assumes the world is locally constant. Good when there's no strong
  gradient at the border; bad when there is (the gradient response
  gets flattened).

- **`:reflect`** — mirrors across the border *without repeating the
  border pixel*. This is the textbook choice for smoothing because it
  preserves local smoothness: a horizontal ramp continues smoothly past
  the border in mirror form, so a Gaussian blur sees no discontinuity.
  `scipy.ndimage` uses this as the default.

- **`:symmetric`** — mirrors *with* the border pixel repeated. Subtly
  different from `:reflect`; differs only at exactly the border.
  Common in wavelets.

- **`:circular`** — wraps around (torus). This is what an FFT-based
  convolution does implicitly, because the DFT treats the signal as
  periodic. Useful when the signal really is periodic (textures);
  destructive otherwise — bright pixels in one corner bleed onto the
  opposite corner.

- **`:valid`** — no padding. Output shrinks by `(kh - 1, kw - 1)`.
  Often the right call when you only care about "real" values and
  intend to crop anyway — especially for stacked convolutions in deep
  learning.

## Decision table

| Goal                              | Mode I'd pick          |
|-----------------------------------|------------------------|
| Smoothing for noise reduction     | `:reflect`             |
| Smoothing before gradient compute | `:reflect`             |
| Implementing FFT-based conv       | `:circular`            |
| Want to crop afterwards anyway    | `:valid`               |
| The image really is on a torus    | `:circular`            |
| Don't trust the border at all     | `:replicate` + crop    |
| Quick-and-dirty / textbook DSP    | `:zero`                |

## See it for yourself

```sh
julia --project=. examples/02_padding_modes_studio.jl
open artifacts/02_padding_modes_studio/montage.pgm
```

In the montage I look at three things:

- Top-left corner across `:zero` (halo) vs `:replicate` (no halo).
- Bottom edge under `:reflect` vs `:symmetric` — almost identical.
- Anywhere on the border for `:circular` — content from the opposite
  side is leaking in.
