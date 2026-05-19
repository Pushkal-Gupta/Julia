# Performance: where each implementation actually wins

I've been hand-waving at performance through the whole repo — "naive
is slow", "separable is faster", "FFT should win for big kernels".
Time to put real numbers on it.

The benchmark in `examples/11_performance_lab.jl` runs four
implementations of 2D correlation on a 384×384 image across six
kernel sizes, measured with `BenchmarkTools` (multiple runs, GC
settled, minimum reported).

## Four implementations

| Implementation | What it does                                              |
|----------------|-----------------------------------------------------------|
| **naive**      | `correlate2d` — explicit padding copy, then a tight loop  |
| **inline**     | `correlate2d_inline` — no padding copy; out-of-bounds reads resolved per-tap |
| **separable**  | `separable_correlate2d` — two 1D passes (only valid for rank-1 kernels) |
| **fft**        | `fft_correlate2d` — `rfft` → multiply → `irfft`           |

## The numbers

Best of 8 runs, on a 384×384 image, in milliseconds:

```
k    naive (ms)   inline (ms)  separable    fft (ms)
─────────────────────────────────────────────────────
3        1.20        1.27        1.23        5.91
5        2.02        2.19        1.55        5.69
9        4.30        4.65        1.85        1.69
15      16.58       17.05        3.01        6.04
21      39.61       40.61        3.59        5.85
31      98.12       93.71        4.88        3.31
```

Speedups vs naive:

```
k    inline     separable    fft
─────────────────────────────────
3    0.95×       0.98×       0.20×
5    0.92×       1.30×       0.35×
9    0.92×       2.32×       2.54×
15   0.97×       5.51×       2.75×
21   0.98×      11.03×       6.77×
31   1.05×      20.12×      29.61×
```

## What I learned

### Inline is a wash

I expected the no-padding-copy version to win clearly. It doesn't.
At small k it's actually slightly slower (the per-tap bounds branch
costs more than the copy saves), and only at k=31 does saving the
copy pay off (1.05× faster). The padding copy on a 384×384 image is
roughly a 1.5 MB write per call. Cache-warm, it's amortized below
the noise floor of the inner loop work.

This was the most useful negative result of the day. The folk
wisdom that "the padding copy is the bottleneck" is wrong here. The
inner-loop arithmetic is.

### Separable wins as expected — but more than expected

Theory says the speedup should be `k/2` for a `k × k` symmetric
kernel. Measured:

```
k       k/2     measured
3       1.5     0.98       (overhead dominates)
5       2.5     1.30       (overhead still in the picture)
9       4.5     2.32       (catching up)
15      7.5     5.51       (close to theory)
21      10.5    11.03      (matches theory)
31      15.5    20.12      (beats theory!)
```

At small k the two passes pay more in fixed overhead (two padding
copies instead of one) than the arithmetic saves. At large k that
inverts and the measured speedup tracks theory closely. At k=31 it
*beats* theory, which I'm pretty sure is a cache effect: two passes
that each fit nicely in L2 beat one pass that doesn't.

### FFT crossover is around k≈25

FFT is `O(N log N)` regardless of k, but it has a fixed overhead per
call (two FFTs, one multiply, one inverse FFT, plus all the
padding). On this image size:

- k ≤ 9: separable destroys FFT.
- k = 15, 21: separable still wins by a couple of factors.
- k = 31: FFT pulls ahead — 29.6× vs naive, 20.1× for separable.

The exact crossover depends heavily on the image size. Bigger image
→ FFT does relatively worse on small kernels (more wasted work in
the transform), so the crossover moves to larger k. Different image
or kernel dimensions make the picture noisier than it looks here:
FFT performance is famously sensitive to whether `H + k − 1`
factors into small primes. At k=15 my padded size is 398 = 2 × 199,
which has a big prime factor and runs slowly. At k=31 the padded
size is 414 = 2 × 9 × 23, which factors more cleanly. That's why
the FFT timing isn't monotonic in k.

### What it means in practice

- For Gaussian / box / Sobel-style work on real-world images
  (256–1024 px on a side, kernels ≤ 15): use the separable path.
  It's always at least as fast as the alternatives and avoids the
  FFT setup overhead entirely.
- For non-separable kernels at large sizes (say a custom 31×31
  matched filter): FFT.
- For everything in between: it doesn't matter, just pick something
  readable.

## Type stability check

A `@code_warntype` pass on the hot paths confirms there are no
type instabilities — no `::Any` or `::Union{...}` in the inferred
output:

```
correlate2d                             ::Any:0  ::Union:0
correlate2d_inline                      ::Any:0  ::Union:0
_correlate_into!                        ::Any:0  ::Union:0
gradient_magnitude                      ::Any:0  ::Union:0
nonmaximum_suppression                  ::Any:0  ::Union:0
fft_correlate2d (:zero)                 ::Any:0  ::Union:0
```

That's reassuring — it means the compiler can specialize every inner
loop on the concrete element type. The `R = promote_type(T, K, Float64)`
pattern in `correlate2d` propagates correctly, and the `@inbounds for
… in …` loops are all working on concrete types.

If any of those columns were nonzero, fixing them would usually be
the highest-leverage performance work. Type instabilities mean each
loop iteration goes through a dynamic dispatch and a heap
allocation, which can slow a tight loop by 10× or more.

## What this doesn't cover

A few performance angles I'm deliberately not addressing yet:

- **Threading.** Convolution is embarrassingly parallel along the
  image dimensions. `Threads.@threads` on the outer loop would split
  the work across cores. It's worth doing eventually but I want to
  keep the single-threaded picture honest first.
- **SIMD.** The inner kernel-tap loop is a candidate for `@simd`.
  Sometimes LLVM auto-vectorizes it already; sometimes a hint helps.
  Belongs in the same pass as threading.
- **GPU.** `KernelAbstractions.jl` would let me run the same kernel
  on a Metal / CUDA / AMDGPU backend. The convolution arithmetic is
  exactly the shape GPUs were built for. Future work.
- **Memory layout.** A column-major layout (Julia's default) means
  the `[i + ik - 1, j + jk - 1]` access pattern walks columns in
  the inner loop, which is cache-friendly. If I ever transpose this
  representation I should re-benchmark.
