# Laplacian-pyramid image blending

The 1983 application that motivated the pyramid algorithm in the
first place. Two images, one mask, one composite — but blended in a
way that hides the seam.

## What goes wrong with naive blending

If I have two images `A` and `B` and a mask `M` (1 means "take A",
0 means "take B"), the obvious recipe is:

```
out = M · A + (1 − M) · B
```

When `M` transitions sharply (the "left half from A, right half
from B" case), this produces a visible seam everywhere `M` jumps
from 1 to 0. The pixel at the seam smashes high-frequency content
from `A` next to high-frequency content from `B`, and our eyes are
extremely good at spotting that.

The fix isn't to smooth the mask. That blurs the transition,
yes — but it also smears `A`'s content into `B`'s region (and vice
versa) over a wide band. Detail gets averaged away.

What I want is: smooth the *mask*, but only smooth it as much as
the *frequency band* I'm currently blending requires.

## The pyramid view

A Laplacian pyramid is exactly the decomposition I need. Each level
`L_k` holds the band-pass detail at scale `2^k`. Lower levels =
high frequencies (sharp edges, fine textures); higher levels = low
frequencies (overall brightness, broad color gradients).

The algorithm:

1. Build `L_A`, `L_B`: Laplacian pyramids of `A` and `B`.
2. Build `G_M`: a Gaussian pyramid of the mask. Each level of `G_M`
   is the mask smoothed and downsampled, so the mask at coarse
   levels is a heavily-smoothed version of the original.
3. Blend each level pointwise: `L_blend[k] = G_M[k] · L_A[k] +
   (1 − G_M[k]) · L_B[k]`.
4. Reconstruct from `L_blend` via the exact inverse of step 1.

The key insight: at level 0 (high-frequency detail) the mask is
sharp, so fine edges from one side don't bleed into the other. At
level 4 (low-frequency intensity) the mask has been smoothed over a
32-pixel-wide region, so the overall brightness change fades in
gradually. Different bands, different mask smoothness, all blended
consistently because they all flow through the same pyramid
machinery.

## In code

```julia
laplacian_blend(A, B, mask; levels=4, pad=:replicate) -> Matrix{Float64}
```

Lives in `Pyramids`. The implementation is three lines of pyramid
construction plus one loop:

```julia
LA = laplacian_pyramid(A;    levels)
LB = laplacian_pyramid(B;    levels)
GM = gaussian_pyramid(mask;  levels)

blended = [GM[k] .* LA[k] .+ (1 .- GM[k]) .* LB[k] for k in eachindex(LA)]
return reconstruct_laplacian_pyramid(blended)
```

The reason it works at all is that `laplacian_pyramid` and
`reconstruct_laplacian_pyramid` are exact inverses (the residual
test I did earlier shows reconstruction error of `5.55e-17`). So if
I had `blended = LA` directly, I'd get `A` back exactly. The blend
just interpolates each level's content.

## What the tests pin down

In `test/test_pyramids.jl`:

- All-ones mask returns `A` to within `1e-10`.
- All-zeros mask returns `B` to within `1e-10`.
- For a sharp half-and-half mask: far-from-seam pixels are within
  `0.05` of their source image (so the blend doesn't drift the
  interior).
- The blended output has no one-pixel jump bigger than `0.15` across
  the entire image (the seam region gets smeared over many pixels).

The 0.15 jump bound is the quantitative version of "no visible
seam" — naive blending on this test image has a jump of `0.6` at
the seam (the difference between A and B), so the pyramid version
is at least 4× better.

## How many levels?

`levels` controls how wide the transition region is in the output.
Rough rule: the transition width is `~3 · 2^levels` pixels, because
that's how far the Gaussian pyramid spreads the mask edge by level
`L`. So:

- `levels = 2`: transition ~12 pixels. Sharp.
- `levels = 4`: transition ~48 pixels. Gentle.
- `levels = 6`: transition ~192 pixels. Practically the whole
  image. (Only useful for large images.)

For the studio image (128×128) I used `levels = 5`, which puts the
transition at ~96 pixels — enough to hide the seam without making
the whole composition look like a cross-fade.

## Where this connects

Laplacian-pyramid blending is one of those "small algorithm,
big effect" classics that I'd been hearing about since I read
Burt & Adelson. Now that I've coded the pyramid from scratch and
seen the reconstruction match to ULP precision, the blending step
feels almost free — it's just one more way to use a pyramid that
already exists.

The same idea generalizes to other multi-scale composites:

- **Tone mapping** — blend a low-frequency tone curve with the
  original high-frequency content.
- **Texture transfer** — copy fine-scale detail from one image into
  the coarse-scale structure of another.
- **Detail enhancement** — boost the lower-level pyramid bands
  before reconstruction.

I'm not building all of those, but they're all the same recipe:
decompose → operate per level → reconstruct.

## References

- Burt, P. J., & Adelson, E. H. (1983). *A multiresolution spline
  with application to image mosaics*. ACM Transactions on Graphics,
  2(4), 217–236.
