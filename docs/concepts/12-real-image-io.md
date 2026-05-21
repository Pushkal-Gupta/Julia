# Real-image I/O

Up to this point everything in the lab ran on synthetic images that
I constructed inside the script — checkerboards, disks, rectangles
with controlled noise. Useful for testing and visualization, but
also a bit suspicious: would the noise-lab story still hold up on a
real photograph? Does Perona-Malik still beat bilateral on a
textured natural image?

To find out I need to load real images. That's what `Photos` is for.

## The submodule

`Photos` is a small wrapper around `FileIO` + `ImageIO`:

```julia
load_grayscale(path)                  -> Matrix{Float64}
save_grayscale(path, img)             -> path
load_rgb_planes(path)                 -> (R, G, B) of Matrix{Float64}
save_rgb_planes(path, R, G, B)        -> path
```

PNG, TIFF, and Netpbm work out of the box because their ImageIO
backends were precompiled when I added the dependency. JPEG is
trickier (the `JpegTurbo` package writes via byte buffers and
sometimes wants explicit configuration), so I dropped it from the
test matrix until I need it on a real photo workflow.

## Why the conversion ladder

A real image file lives on disk as bytes. By the time it reaches my
pipeline I want it as a `Matrix{Float64}` in `[0, 1]`. The path:

```
bytes  →  Matrix{Gray{N0f8}}  →  Matrix{Float64}
       (FileIO + ImageIO)        (Float64.(...))
```

Two type details that took me a moment:

- **`N0f8`** is a fixed-point type for the byte range `0..255`
  rescaled to `[0, 1]`. So a saved PNG with raw byte `128` arrives
  as `Gray(0.5019...N0f8)`. Doing `Float64.(::Matrix{Gray{N0f8}})`
  gives the matrix in `[0, 1]` directly — no `/ 255` needed.
- **`Gray`** is a single-channel wrapper from `ColorTypes`. It can
  be constructed from any `Colorant`, and color-to-luminance
  conversion happens automatically (via the standard
  `Rec. 601`-style weights). So `Gray.(rgb_image)` is the
  ergonomic way to get from RGB to grayscale.

For saving, the inverse: clamp to `[0, 1]`, wrap in `Gray{N0f8}`
(which quantizes to 8 bits), let FileIO figure out the on-disk
encoding from the extension.

## Why I kept `PNM` around

The pure-Julia PGM/PPM module from earlier (`PNM`) didn't get
deleted when I added `Photos`. It's a teaching artifact — the
Netpbm format is small enough to implement in 30 lines, and
implementing it makes the "what's actually in an image file" idea
concrete. `Photos` is the *practical* I/O module; `PNM` is the
*pedagogical* one. Different audiences for the same kind of thing.

## What it unlocks

The 14th example script is a self-contained pipeline demo:

```sh
julia --project=. examples/14_real_image_pipeline.jl path/to/photo.png
```

…or run with no arguments and the script generates a synthetic
sample PNG, loads it back, and runs everything on that. Either way
the output is a directory of PNGs: blurred, Canny edges, edge
overlay, Harris overlay, and the Gaussian pyramid as a grid.

The moment that matters: this is the first time the whole pipeline
takes a real-world image file as input. Now I can point Canny at a
photo, or run Perona-Malik on a textured natural scene, or do
multi-scale Harris on a building facade.

## References

- JuliaImages documentation — <https://juliaimages.org>
- `FileIO` / `ImageIO` ecosystem README — <https://github.com/JuliaIO/ImageIO.jl>
