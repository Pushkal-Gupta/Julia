# A tiny ray tracer

A small detour out of pure image processing. Everything in `ImageLab`
up to this point has been *analysis* — take an image and extract
something from it. The ray tracer is the other direction: take a 3D
scene and *synthesize* an image. It's the same output type
(`Matrix{Float64}` in `[0, 1]`), so the rest of the pipeline still
applies, but the inside of `render` looks completely different from
everything else in the repo.

I built it for two reasons. One, because it makes Julia feel like a
general-purpose language and not just a place where I write image
filters. Two, because ray tracing is the textbook example of a
problem where the math is short, the implementation is short, and
the output is satisfying. The whole submodule fits in ~300 lines.

## The frame: forward vs inverse rendering

A pinhole camera projects every 3D point onto a 2D image plane.
"Forward rendering" means: I know the scene, I want the image.
Ray tracing is one of two main forward-rendering algorithms; the
other is rasterization (project geometry onto the image, then
shade). Ray tracing has the property that shading depends only on
the scene at the point a ray hits, which makes shadows, reflections,
and refractions easy to express recursively. Rasterization is
faster but reflections become a hack.

For each pixel I shoot one ray through the camera origin into the
scene, find what it hits, work out how much light is travelling
*back* along that ray, and put that value in the pixel.

## Ray-sphere intersection: just a quadratic

The ray is parameterized as `p(t) = origin + t · direction` with
`direction` unit. A sphere is `|p − center|² = radius²`. Substitute:

```
|origin + t · direction − center|² = radius²
```

Expanding with `oc = origin − center`:

```
t² (direction · direction) + 2t (oc · direction) + (oc · oc − radius²) = 0
```

Since `direction` is unit, `direction · direction = 1` and the
quadratic simplifies to `t² + 2bt + c = 0` with `b = oc · direction`
and `c = oc · oc − radius²`. The discriminant is `b² − c`:

- Negative → ray misses the sphere.
- Zero → ray is tangent.
- Positive → two roots, `t = −b ± √(b² − c)`.

I pick the smaller positive root (the near intersection). If that
root is below an epsilon (`tmin = 1e-4`), I fall back to the far
root — this handles the case where the ray's origin is *inside* the
sphere, which happens during shadow rays that originate on a
surface.

## Ray-plane intersection: even simpler

A plane with point `p₀` and unit normal `n` satisfies
`(p − p₀) · n = 0`. Plug in:

```
(origin + t · direction − p₀) · n = 0
t = ((p₀ − origin) · n) / (direction · n)
```

If `direction · n ≈ 0` the ray is parallel to the plane and there's
no intersection. Otherwise, accept `t ≥ tmin`.

One detail: a plane has two sides, but a surface normal points one
way. I flip the returned normal to face the ray. Without that,
shading on the back of the plane would have `n · l < 0` everywhere
and the back would always be black.

## Phong illumination

For every visible point I want a number between 0 and 1 per colour
channel that says "how bright is this pixel". The Phong model splits
that into three terms:

```
I = I_ambient + Σ_lights (I_diffuse + I_specular)
```

- **Ambient** — constant, models scattered light bouncing off
  everything. Mostly a hack so shadows aren't pitch black.
- **Diffuse (Lambertian)** — proportional to `max(0, n · l)` where
  `l` is the unit direction to the light. Models a matte surface
  that scatters incoming light equally in all directions.
- **Specular** — a sharp highlight where the reflection direction
  aligns with the view direction. I use the Blinn-Phong variant:
  compute the half-vector `h = normalize(l + v)` and use
  `max(0, n · h)^shininess`. It's cheaper than `(reflection · v)^k`
  and looks practically identical.

The constant out in front of each term is a material property —
`ambient`, `diffuse`, `specular` floats per `Material`.

## Shadows are just another ray cast

Before adding a light's contribution to a point, I ask: is anything
in the way? Fire a ray from the surface point toward the light, and
if any object intersects between distance `1e-3` and the light's
distance, the light is blocked. The `1e-3` epsilon is the same
self-intersection guard as before — without it, every surface
point would shadow itself.

This is *hard shadows*: a binary "is the light visible". A real
photograph has soft shadows because real lights have area. To get
soft shadows I'd shoot many shadow rays toward random points on the
light surface and average the results. I didn't build that —
introduces randomness, slows the render by N times, and the result
isn't conceptually new.

## Whitted recursion for reflections

This is the move that made Whitted's 1980 paper famous. When a ray
hits a reflective surface, instead of just shading it locally, I
also compute the *reflection ray* and recursively trace it:

```
reflection_direction = direction − 2 (direction · normal) normal
```

That formula comes from decomposing `direction` into components
parallel and perpendicular to `normal` and flipping the
perpendicular part. The recursive trace returns the colour
arriving from the reflected direction, and I blend it with the
local shading proportional to the material's `reflective` parameter.

The depth study in `examples/18_tiny_ray_tracer.jl` shows what each
extra bounce contributes:

```
depth = 0 : mean luminance = 0.358, max = 0.919  (mirror is black)
depth = 1 : mean luminance = 0.405, max = 0.737  (mirror reflects)
depth = 2 : mean luminance = 0.408, max = 0.737  (~converged)
depth = 3 : mean luminance = 0.408, max = 0.737  (no visible change)
```

Two bounces is enough for this scene. For a scene with two facing
mirrors I'd need many more, and there's no upper bound — the
algorithm only converges because each bounce attenuates by
`reflective` and eventually the contribution falls below visible.

## Why the output is `Matrix{Float64}`, not something fancier

Every other module in `ImageLab` works on `Matrix{Float64}` in
`[0, 1]`. The ray tracer returns three of them — `R`, `G`, `B` —
matching the signature of `Photos.save_rgb_planes` and
`PNM.save_ppm`. That means the moment a render finishes, the result
is a first-class citizen of the rest of the pipeline. Take the
luminance:

```julia
lum = 0.299 .* R .+ 0.587 .* G .+ 0.114 .* B
```

and now `lum` is just a grayscale image. I can run Canny on it, slap
a Gaussian pyramid on it, fit a Toeplitz convolution matrix on a row
of it. Synthesis and analysis meet at the same data type.

## What I left out, and why

- **Triangle meshes** — would need an acceleration structure (BVH)
  to be tolerable. The conceptual content is the same intersection
  test, just for a triangle instead of a sphere.
- **Refraction** — Snell's law plus a second recursive call. Same
  shape as reflection, just at a different angle and weighted by
  the transmission coefficient. Easy to add later.
- **Anti-aliasing** — supersampling at 2× width/height and
  downsampling with a box filter would smooth the jagged edges I
  see at low resolutions. Not done because it'd just be another
  outer loop.
- **Area lights / soft shadows** — N shadow rays per light. Adds
  randomness, adds runtime, doesn't change the algorithm.
- **A BVH** — Currently `cast_ray` is O(scene size). A bounding
  volume hierarchy makes it O(log N). For three spheres and a plane,
  not worth it. For a million triangles, mandatory.
- **Importance sampling / path tracing** — global illumination via
  Monte Carlo. That's a different algorithm (and a much bigger
  project).

The 1980 Whitted paper has all of the above except path tracing,
which came later. Mine is one notch simpler than Whitted's.

## Performance

The 384×256 reference render runs in **0.22 seconds** — roughly
**2.2 microseconds per pixel**. For each pixel: one primary ray
intersection test against four objects, one shadow ray
intersection test, one reflection bounce on the mirror sphere. So
on the order of ~10 floating-point intersection tests per pixel,
plus shading arithmetic. That's about 2,000 nanoseconds budget for
~50 floating-point operations and the V3 struct allocations — Julia
optimizes the immutable `V3` struct away as registers, which is why
it doesn't fall over from allocation pressure.

If I wanted real performance I'd vectorize across pixels with
`SIMD.jl` or thread the outer pixel loop. Neither is necessary at
this resolution.

## References

- Whitted, T. (1980). *An Improved Illumination Model for Shaded
  Display* — the original "trace rays into the scene, recursively
  follow reflections" paper.
- Phong, B. T. (1975). *Illumination for Computer Generated
  Pictures* — the original Phong model. Blinn (1977) introduced
  the half-vector variant I'm using.
- Shirley, P. (2020). *Ray Tracing in One Weekend* (the free
  online book). My ray tracer is roughly book-1 minus the
  diffuse-bounce Monte Carlo.
- Pharr, Jakob & Humphreys, *Physically Based Rendering* — the
  serious reference for everything I deliberately skipped.
