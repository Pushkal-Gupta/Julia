"""
    Photos

Real-world image I/O built on top of `FileIO` + `ImageIO`. Loads PNG,
JPEG, TIFF, and a few others into the `Matrix{Float64}` shape that
the rest of `ImageLab` works with, and writes results back.

The `PNM` submodule still exists as the from-scratch teaching
version — pure-Julia PGM/PPM with no external dependencies. `Photos`
is what I reach for when I want to run the pipeline on a real photo.

The conversion path: file → `Gray{N0f8}` matrix (via `FileIO.load`)
→ `Float64` in `[0, 1]`. Going the other way: `Float64` → `clamp` →
`Gray{N0f8}` → on-disk encoding chosen by the file extension.
"""
module Photos

using FileIO: load as fileio_load, save as fileio_save
using ColorTypes: Gray, RGB, red, green, blue
using FixedPointNumbers: N0f8

export load_grayscale, save_grayscale, load_rgb_planes, save_rgb_planes

"""
    load_grayscale(path) -> Matrix{Float64}

Read an image from disk and return it as a grayscale matrix of
`Float64` values in `[0, 1]`. Color inputs are converted to luminance
via the standard `Gray()` coercion in `ColorTypes`.
"""
function load_grayscale(path::AbstractString)
    raw = fileio_load(path)
    return Float64.(Gray.(raw))
end

"""
    save_grayscale(path, img)

Write a grayscale matrix (real values, expected in `[0, 1]`) to
disk. Values outside the range are clamped silently. The output
format is inferred from the file extension — `.png`, `.jpg`, `.tif`,
`.pgm` all work.
"""
function save_grayscale(path::AbstractString, img::AbstractMatrix{<:Real})
    clamped = clamp.(Float64.(img), 0.0, 1.0)
    fileio_save(path, Gray{N0f8}.(clamped))
    return path
end

"""
    load_rgb_planes(path) -> (R, G, B) of Matrix{Float64}

Load an image and return its three color channels as separate
`Float64` matrices in `[0, 1]`. For grayscale inputs the same plane
is returned three times.
"""
function load_rgb_planes(path::AbstractString)
    raw = fileio_load(path)
    rgb = RGB.(raw)
    R = Float64.(red.(rgb))
    G = Float64.(green.(rgb))
    B = Float64.(blue.(rgb))
    return (R, G, B)
end

"""
    save_rgb_planes(path, R, G, B)

Save three matched grayscale planes as a single RGB image. Each
plane is clamped to `[0, 1]` and quantized to 8-bit.
"""
function save_rgb_planes(path::AbstractString,
                         R::AbstractMatrix{<:Real},
                         G::AbstractMatrix{<:Real},
                         B::AbstractMatrix{<:Real})
    size(R) == size(G) == size(B) || throw(DimensionMismatch(
        "channel sizes differ: $(size(R)), $(size(G)), $(size(B))"))
    Rc = clamp.(Float64.(R), 0.0, 1.0)
    Gc = clamp.(Float64.(G), 0.0, 1.0)
    Bc = clamp.(Float64.(B), 0.0, 1.0)
    img = RGB{N0f8}.(Rc, Gc, Bc)
    fileio_save(path, img)
    return path
end

end # module Photos
