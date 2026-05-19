"""
    Padding

Border-handling for 2D image operations. When a kernel slides past the edge
of an image, the missing pixels have to come from *somewhere*. The choice
affects every output along the border, so it deserves its own module.

Modes supported:

| Mode          | What the missing pixel becomes                          |
|---------------|---------------------------------------------------------|
| `:zero`       | `0` (also called *constant* padding)                    |
| `:replicate`  | Nearest in-bounds pixel (clamps the index)              |
| `:reflect`    | Mirror, *excluding* the border pixel itself             |
| `:symmetric`  | Mirror, *including* the border pixel                    |
| `:circular`   | Wrap around (treat the image as a torus)                |
| `:valid`      | No padding — output shrinks instead                     |

Why so many? Different modes match different physical assumptions:

- `:replicate` says "the world outside the frame looks like the edge."
- `:reflect` says "the signal is locally symmetric" — usually best for
  smoothing because it doesn't introduce a fake edge.
- `:circular` is what an FFT-based convolution implicitly uses.

We allocate an explicit padded array here. It's the simplest correct
implementation and makes the inner convolution loop trivial. Later (M5)
we'll compare against an inline bounds-aware variant that skips the copy.
"""
module Padding

export pad_array, padded_size

const PAD_MODES = (:zero, :replicate, :reflect, :symmetric, :circular, :valid)

"""
    padded_size(input_size, pad)

Given a tuple of pad amounts `pad = (top, bottom, left, right)`, return the
size of the padded array.
"""
padded_size(sz::NTuple{2,Int}, pad::NTuple{4,Int}) =
    (sz[1] + pad[1] + pad[2], sz[2] + pad[3] + pad[4])

"""
    pad_array(A, pad; mode=:zero) -> Matrix

Return `A` padded by `pad = (top, bottom, left, right)` pixels under the
given `mode`. The original data sits in `A_padded[top+1:top+H, left+1:left+W]`.

`mode = :valid` returns `A` unchanged (ignored padding amount).
"""
function pad_array(A::AbstractMatrix{T}, pad::NTuple{4,Int};
                   mode::Symbol = :zero) where {T}
    mode in PAD_MODES || throw(ArgumentError(
        "unknown padding mode :$mode; expected one of $(PAD_MODES)"))
    mode === :valid && return Matrix{T}(A)

    pt, pb, pl, pr = pad
    H, W = size(A)
    Hp, Wp = H + pt + pb, W + pl + pr

    out = mode === :zero ? zeros(T, Hp, Wp) : Matrix{T}(undef, Hp, Wp)

    # Copy the interior. Same for every mode.
    @inbounds out[pt+1:pt+H, pl+1:pl+W] .= A

    mode === :zero && return out

    @inbounds for j in 1:Wp, i in 1:Hp
        # Skip the interior — already filled.
        (pt+1 ≤ i ≤ pt+H && pl+1 ≤ j ≤ pl+W) && continue
        # Map (i, j) in padded coords to (si, sj) in source coords.
        si = _reflect_index(i - pt, H, mode)
        sj = _reflect_index(j - pl, W, mode)
        out[i, j] = A[si, sj]
    end
    return out
end

"""
    _reflect_index(idx, n, mode)

Map a (possibly out-of-bounds) 1-based index `idx` into `1:n` according to
`mode`. The interior case (`1 ≤ idx ≤ n`) is handled by the caller; this
helper only needs to be correct for out-of-bounds inputs, but it's robust
to in-bounds inputs too.
"""
@inline function _reflect_index(idx::Int, n::Int, mode::Symbol)
    1 ≤ idx ≤ n && return idx
    if mode === :replicate
        return clamp(idx, 1, n)
    elseif mode === :reflect
        # Mirror without repeating the edge: 0 → 2, -1 → 3, n+1 → n-1, …
        # Equivalent to reflecting across positions 1 and n.
        period = 2 * (n - 1)
        period == 0 && return 1
        k = mod(idx - 1, period)
        return 1 + (k ≤ n - 1 ? k : period - k)
    elseif mode === :symmetric
        # Mirror *with* the edge repeated: 0 → 1, -1 → 2, n+1 → n, n+2 → n-1.
        period = 2 * n
        k = mod(idx - 1, period)
        return 1 + (k < n ? k : period - 1 - k)
    elseif mode === :circular
        return mod(idx - 1, n) + 1
    else
        throw(ArgumentError("unhandled padding mode :$mode in _reflect_index"))
    end
end

end # module Padding
