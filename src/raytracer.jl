"""
    RayTracer

A tiny Whitted-style ray tracer. The point isn't to compete with anything —
it's to make Julia feel like a general-purpose language and not just a place
where I write image filters. The output is still a `Matrix{Float64}` in
`[0, 1]` (three of them — `R`, `G`, `B`), so the rest of `ImageLab`
(`Photos.save_rgb_planes`, `PNM.save_ppm`, `Viz.montage`, …) keeps working
on the result.

What's here:

- A 3D vector type `V3` with the arithmetic I care about.
- `Ray`, `Sphere`, `Plane`, `PointLight`, `Camera`, `Material`, `Scene`.
- `intersect_object(ray, sphere)` and `intersect_object(ray, plane)`
  return either a `Hit` (closest forward intersection) or `nothing`.
- `trace(scene, ray; depth)` — recursive primary ray + one mirror bounce
  per `depth`. Ambient + Lambertian diffuse + Blinn-Phong specular +
  hard shadows.
- `render(scene, camera; width, height) -> (R, G, B)` — the only
  user-facing entry point most of the time.

What's deliberately not here: refraction, area lights, anti-aliasing,
acceleration structures (`BVH`), triangle meshes, textures, importance
sampling. Any of those would double the size of this submodule and
none of them change the conceptual story.
"""
module RayTracer

export V3, Ray, Material, Sphere, Plane, PointLight, Camera, Scene, Hit,
       intersect_object, cast_ray, trace, render, basic_scene

# ── Vector type ──────────────────────────────────────────────────────────────
# I'm rolling my own instead of `SVector{3, Float64}` to keep the dependency
# graph small. The performance difference at 256×256 is invisible.

struct V3
    x::Float64
    y::Float64
    z::Float64
end

V3(x::Real, y::Real, z::Real) = V3(Float64(x), Float64(y), Float64(z))

Base.:+(a::V3, b::V3) = V3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:-(a::V3, b::V3) = V3(a.x - b.x, a.y - b.y, a.z - b.z)
Base.:-(a::V3)         = V3(-a.x, -a.y, -a.z)
Base.:*(a::V3, s::Real) = V3(a.x * s, a.y * s, a.z * s)
Base.:*(s::Real, a::V3) = a * s
Base.:/(a::V3, s::Real) = V3(a.x / s, a.y / s, a.z / s)

# Component-wise product — I want it for "scene light × material albedo"
# which is a wavelength-by-wavelength multiplication. Naming it `hadamard`
# would be more correct but `⊙` reads fine inline.
⊙(a::V3, b::V3) = V3(a.x * b.x, a.y * b.y, a.z * b.z)

vdot(a::V3, b::V3) = a.x * b.x + a.y * b.y + a.z * b.z
vnorm(a::V3)       = sqrt(vdot(a, a))
vunit(a::V3)       = a / vnorm(a)
vcross(a::V3, b::V3) = V3(a.y * b.z - a.z * b.y,
                          a.z * b.x - a.x * b.z,
                          a.x * b.y - a.y * b.x)

# ── Geometry primitives ──────────────────────────────────────────────────────

struct Ray
    origin::V3
    direction::V3      # always stored as a unit vector
    function Ray(o::V3, d::V3)
        n = vnorm(d)
        n < 1e-12 && throw(ArgumentError("ray direction has zero length"))
        return new(o, d / n)
    end
end

ray_at(r::Ray, t::Real) = r.origin + r.direction * t

struct Material
    color::V3          # albedo, components in [0, 1]
    ambient::Float64   # ambient coefficient
    diffuse::Float64   # Lambertian coefficient
    specular::Float64  # Blinn-Phong coefficient
    shininess::Float64 # Blinn-Phong exponent
    reflective::Float64 # mirror fraction in [0, 1]
end

Material(color::V3) = Material(color, 0.1, 0.7, 0.4, 64.0, 0.0)
Material(color::V3, reflective::Real) =
    Material(color, 0.1, 0.7, 0.4, 64.0, Float64(reflective))

struct Hit
    t::Float64
    point::V3
    normal::V3         # outward-facing, unit length
    material::Material
end

struct Sphere
    center::V3
    radius::Float64
    material::Material
end

struct Plane
    point::V3
    normal::V3         # unit
    material::Material
    checker::Bool      # if true, alternate `material.color` and `check_color`
    check_size::Float64
    check_color::V3
end

Plane(point::V3, normal::V3, material::Material) =
    Plane(point, vunit(normal), material, false, 1.0, V3(0.0, 0.0, 0.0))

function Plane(point::V3, normal::V3, mat::Material,
               check_color::V3; check_size::Real = 1.0)
    return Plane(point, vunit(normal), mat, true, Float64(check_size), check_color)
end

struct PointLight
    position::V3
    intensity::V3      # RGB intensity, components ≥ 0
end

mutable struct Scene
    spheres::Vector{Sphere}
    planes::Vector{Plane}
    lights::Vector{PointLight}
    ambient::V3        # ambient light color
    background::V3     # sky color, returned when a ray hits nothing
end

Scene() = Scene(Sphere[], Plane[], PointLight[],
                V3(0.1, 0.1, 0.1), V3(0.55, 0.72, 0.92))

struct Camera
    origin::V3
    look_at::V3
    up::V3             # world-up, doesn't need to be perpendicular to forward
    fov_deg::Float64   # vertical field of view in degrees
end

# ── Intersection ─────────────────────────────────────────────────────────────

"""
    intersect_object(ray, sphere; tmin = 1e-4, tmax = Inf) -> Union{Hit, Nothing}

Closed-form ray-sphere intersection from the quadratic
`|origin + t · dir − center|² = radius²`. The leading 1e-4 `tmin` is the
standard "self-intersection ε" — without it, every reflection bounce would
re-hit the surface it just left.
"""
function intersect_object(r::Ray, s::Sphere; tmin::Real = 1e-4, tmax::Real = Inf)
    oc = r.origin - s.center
    b  = vdot(oc, r.direction)
    c  = vdot(oc, oc) - s.radius * s.radius
    disc = b * b - c
    disc < 0 && return nothing
    sq = sqrt(disc)
    t = -b - sq
    if t < tmin
        t = -b + sq
    end
    (t < tmin || t > tmax) && return nothing
    p = ray_at(r, t)
    n = vunit(p - s.center)
    return Hit(Float64(t), p, n, s.material)
end

"""
    intersect_object(ray, plane; tmin = 1e-4, tmax = Inf) -> Union{Hit, Nothing}

Plug `origin + t · dir` into `(p − plane.point) · normal = 0` and solve.
The plane is two-sided: the returned normal flips to face the camera so
shading is well-behaved no matter which side a ray comes from.
"""
function intersect_object(r::Ray, pl::Plane; tmin::Real = 1e-4, tmax::Real = Inf)
    denom = vdot(pl.normal, r.direction)
    abs(denom) < 1e-8 && return nothing
    t = vdot(pl.point - r.origin, pl.normal) / denom
    (t < tmin || t > tmax) && return nothing
    point = ray_at(r, t)
    n = denom < 0 ? pl.normal : -pl.normal

    mat = pl.material
    if pl.checker
        s = pl.check_size
        # Project onto the plane's two largest-magnitude axes. For a
        # y-up floor, that's x and z, which is the common case.
        ax = abs(pl.normal.x)
        ay = abs(pl.normal.y)
        az = abs(pl.normal.z)
        u, v = if ay ≥ ax && ay ≥ az
            point.x, point.z
        elseif ax ≥ az
            point.y, point.z
        else
            point.x, point.y
        end
        parity = mod(floor(Int, u / s) + floor(Int, v / s), 2)
        c = parity == 0 ? mat.color : pl.check_color
        mat = Material(c, mat.ambient, mat.diffuse,
                       mat.specular, mat.shininess, mat.reflective)
    end

    return Hit(Float64(t), point, n, mat)
end

# ── Camera / scene queries ───────────────────────────────────────────────────

"""
    camera_ray(camera, u, v, aspect) -> Ray

`u`, `v` are normalized image coordinates in `[-1, 1]`, with `v` pointing
up. `aspect = width / height`. The right vector is built from the supplied
`up` so a non-perpendicular `up` still produces a sensible orthonormal
frame (Gram-Schmidt style).
"""
function camera_ray(cam::Camera, u::Real, v::Real, aspect::Real)
    forward = vunit(cam.look_at - cam.origin)
    right   = vunit(vcross(forward, cam.up))
    true_up = vcross(right, forward)
    half_h  = tan(deg2rad(cam.fov_deg / 2))
    half_w  = aspect * half_h
    dir = vunit(forward + right * (u * half_w) + true_up * (v * half_h))
    return Ray(cam.origin, dir)
end

"""
    cast_ray(scene, ray; tmin, tmax) -> Union{Hit, Nothing}

Loop over every object in the scene, keep the closest forward hit. Linear
in scene size — fine for the handful of objects in my test scenes; a
serious renderer would use a BVH here.
"""
function cast_ray(scene::Scene, r::Ray; tmin::Real = 1e-4, tmax::Real = Inf)
    best::Union{Hit, Nothing} = nothing
    best_t = Float64(tmax)
    for s in scene.spheres
        h = intersect_object(r, s; tmin, tmax = best_t)
        h !== nothing && h.t < best_t && (best = h; best_t = h.t)
    end
    for pl in scene.planes
        h = intersect_object(r, pl; tmin, tmax = best_t)
        h !== nothing && h.t < best_t && (best = h; best_t = h.t)
    end
    return best
end

function in_shadow(scene::Scene, p::V3, ldir::V3, ldist::Real)
    sr = Ray(p, ldir)
    for s in scene.spheres
        h = intersect_object(sr, s; tmin = 1e-3, tmax = ldist - 1e-3)
        h !== nothing && return true
    end
    for pl in scene.planes
        h = intersect_object(sr, pl; tmin = 1e-3, tmax = ldist - 1e-3)
        h !== nothing && return true
    end
    return false
end

# ── Shading ──────────────────────────────────────────────────────────────────

"""
    shade(scene, ray, hit) -> V3

Phong illumination: ambient + Σ (visible-light) [diffuse + Blinn-Phong
specular]. Specular uses the half-vector formulation, which converges to
a Phong-like lobe but is cheaper and slightly more physically motivated.
"""
function shade(scene::Scene, r::Ray, h::Hit)
    mat = h.material
    col = (scene.ambient ⊙ mat.color) * mat.ambient
    view_dir = -r.direction
    for L in scene.lights
        to_light = L.position - h.point
        dist     = vnorm(to_light)
        ldir     = to_light / dist
        in_shadow(scene, h.point, ldir, dist) && continue
        ndotl = vdot(h.normal, ldir)
        ndotl ≤ 0 && continue
        # Lambertian diffuse term.
        col = col + (L.intensity ⊙ mat.color) * (ndotl * mat.diffuse)
        # Blinn-Phong specular term — white highlight regardless of albedo.
        halfv = vunit(ldir + view_dir)
        spec_dot = max(0.0, vdot(h.normal, halfv))
        col = col + L.intensity * (spec_dot ^ mat.shininess * mat.specular)
    end
    return col
end

"""
    trace(scene, ray; depth = 3) -> V3

Recursive ray tracing. Each call returns the radiance arriving at the
ray's origin. The recursion only happens for reflective materials, and
`depth` bounds how many mirror bounces to allow. Refraction would slot
in here as a second recursive call weighted by transmission.
"""
function trace(scene::Scene, r::Ray; depth::Int = 3)
    h = cast_ray(scene, r)
    h === nothing && return scene.background
    local_col = shade(scene, r, h)
    mat = h.material
    if mat.reflective > 0 && depth > 0
        # Mirror reflection: r' = r − 2 (r · n) n
        rdir = r.direction - h.normal * (2 * vdot(r.direction, h.normal))
        # Offset the new ray origin along the normal so it doesn't
        # immediately re-hit the surface it left.
        refl_ray = Ray(h.point + h.normal * 1e-4, vunit(rdir))
        refl_col = trace(scene, refl_ray; depth = depth - 1)
        return local_col * (1.0 - mat.reflective) + refl_col * mat.reflective
    end
    return local_col
end

# ── Render ───────────────────────────────────────────────────────────────────

"""
    render(scene, camera; width = 256, height = 256, depth = 3)
        -> (R, G, B) of Matrix{Float64}

Loop over pixels, cast a primary ray per pixel, return the three colour
channels. The channels are clamped to `[0, 1]` so `Photos.save_rgb_planes`
and `PNM.save_ppm` can write them directly. Row 1 is the top of the image
(standard image convention), column 1 is the left.
"""
function render(scene::Scene, cam::Camera;
                width::Integer = 256, height::Integer = 256,
                depth::Int = 3)
    R = zeros(Float64, height, width)
    G = zeros(Float64, height, width)
    B = zeros(Float64, height, width)
    aspect = width / height
    @inbounds for row in 1:height, col in 1:width
        u = 2 * (col - 0.5) / width - 1
        v = 1 - 2 * (row - 0.5) / height
        ray = camera_ray(cam, u, v, aspect)
        c   = trace(scene, ray; depth = depth)
        R[row, col] = clamp(c.x, 0.0, 1.0)
        G[row, col] = clamp(c.y, 0.0, 1.0)
        B[row, col] = clamp(c.z, 0.0, 1.0)
    end
    return (R, G, B)
end

"""
    basic_scene() -> (Scene, Camera)

A small reference scene I use for tests and the example script: a red
sphere, a blue sphere, a mirror sphere, a checkerboard floor, one warm
point light. Camera looking slightly down at the scene from a corner.
"""
function basic_scene()
    scene = Scene()
    floor = Plane(V3(0.0, -1.0, 0.0), V3(0.0, 1.0, 0.0),
                  Material(V3(0.8, 0.8, 0.8), 0.1, 0.7, 0.05, 32.0, 0.15),
                  V3(0.2, 0.2, 0.2); check_size = 1.0)
    push!(scene.planes, floor)
    push!(scene.spheres,
          Sphere(V3(-1.1, -0.3, 4.5), 0.7,
                 Material(V3(0.85, 0.25, 0.25), 0.1, 0.75, 0.5, 96.0, 0.0)))
    push!(scene.spheres,
          Sphere(V3(0.0, -0.5, 6.0), 0.5,
                 Material(V3(0.2, 0.4, 0.9), 0.1, 0.7, 0.5, 96.0, 0.0)))
    push!(scene.spheres,
          Sphere(V3(1.3, -0.2, 5.0), 0.8,
                 Material(V3(0.95, 0.95, 0.95), 0.05, 0.1, 0.9, 256.0, 0.8)))
    push!(scene.lights,
          PointLight(V3(-3.0, 5.0, 0.0), V3(1.0, 0.95, 0.85)))
    cam = Camera(V3(0.0, 0.7, 0.0), V3(0.0, -0.2, 5.0),
                 V3(0.0, 1.0, 0.0), 55.0)
    return (scene, cam)
end

end # module RayTracer
