"""
    PNM

Minimal Netpbm (PGM / PPM) readers and writers. No external dependencies —
the entire grayscale image format fits in ~30 lines of Julia, and writing
it ourselves is much more instructive than `using SomeImageLib`.

We use the binary variants (`P5` for grayscale, `P6` for RGB) because they
write to disk fast and Preview.app on macOS opens them natively.

When we need JPEG / PNG / TIFF (real-world images), we'll add `ImageIO` as
a proper dependency. For now: keep it pure, keep it inspectable.
"""
module PNM

export save_pgm, load_pgm, save_ppm

"""
    save_pgm(path, img; maxval=255)

Save a `Matrix{<:Real}` to a binary PGM (P5) file. Floats are assumed to
lie in `[0, 1]`; integers are clamped to `[0, maxval]`.
"""
function save_pgm(path::AbstractString, img::AbstractMatrix{T}; maxval::Integer = 255) where {T<:Real}
    1 ≤ maxval ≤ 65535 || throw(ArgumentError("maxval must be in 1:65535, got $maxval"))
    h, w = size(img)
    bytes = _to_byte_buffer(img, maxval)
    open(path, "w") do io
        write(io, "P5\n$w $h\n$maxval\n")
        write(io, bytes)
    end
    return path
end

"""
    save_ppm(path, r, g, b; maxval=255)

Save three matched grayscale planes as a single RGB PPM (P6).
"""
function save_ppm(path::AbstractString,
                  r::AbstractMatrix{<:Real},
                  g::AbstractMatrix{<:Real},
                  b::AbstractMatrix{<:Real};
                  maxval::Integer = 255)
    size(r) == size(g) == size(b) || throw(DimensionMismatch(
        "channel sizes differ: $(size(r)), $(size(g)), $(size(b))"))
    h, w = size(r)
    br = _to_byte_buffer(r, maxval)
    bg = _to_byte_buffer(g, maxval)
    bb = _to_byte_buffer(b, maxval)
    # Interleave R, G, B per pixel.
    buf = Vector{UInt8}(undef, 3 * h * w)
    @inbounds for k in 1:(h*w)
        buf[3k - 2] = br[k]
        buf[3k - 1] = bg[k]
        buf[3k]     = bb[k]
    end
    open(path, "w") do io
        write(io, "P6\n$w $h\n$maxval\n")
        write(io, buf)
    end
    return path
end

"""
    load_pgm(path) -> Matrix{Float64}

Read a binary PGM (P5) file back into a `Matrix{Float64}` normalized to
`[0, 1]`. Mostly for tests — it ensures we can round-trip what we wrote.
"""
function load_pgm(path::AbstractString)
    open(path, "r") do io
        magic = strip(readline(io))
        magic == "P5" || throw(ArgumentError("not a P5 PGM file: magic=$magic"))
        w, h = (parse(Int, t) for t in split(_read_nonblank(io)))
        maxval = parse(Int, strip(_read_nonblank(io)))
        bytes_per_sample = maxval ≤ 255 ? 1 : 2
        n = w * h
        raw = read(io, n * bytes_per_sample)
        length(raw) == n * bytes_per_sample || error("PGM truncated")
        bytes_per_sample == 1 || error("16-bit PGM read not implemented yet")
        # On-disk raw is row-major. `reshape(raw, w, h)` interprets it column-major,
        # so each Julia column k is image row k. permutedims gets us back to an
        # h×w matrix indexed as `out[row, col]`.
        out = permutedims(reshape(raw, w, h))
        return Float64.(out) ./ maxval
    end
end

# Skip blank / comment lines (Netpbm allows `#` comments and arbitrary whitespace).
function _read_nonblank(io::IO)
    while !eof(io)
        line = readline(io)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        return s
    end
    error("unexpected EOF in PGM header")
end

function _to_byte_buffer(img::AbstractMatrix{T}, maxval::Integer) where {T<:Real}
    h, w = size(img)
    buf = Vector{UInt8}(undef, h * w)
    # PGM is row-major; Julia is column-major. Walk rows then columns to write
    # in the correct on-disk order.
    @inbounds for i in 1:h
        for j in 1:w
            v = img[i, j]
            buf[(i - 1) * w + j] = _to_byte(v, maxval)
        end
    end
    return buf
end

@inline function _to_byte(v::Real, maxval::Integer)
    # Floats live in [0, 1]; integers in [0, maxval].
    x = v isa AbstractFloat ? v * maxval : Float64(v)
    return UInt8(round(clamp(x, 0, maxval)))
end

end # module PNM
