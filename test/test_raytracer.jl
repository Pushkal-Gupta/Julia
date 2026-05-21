using Test
using ImageLab
using ImageLab.RayTracer

# Reach into the module for the few helpers I don't export but want to
# test directly.
const RT = ImageLab.RayTracer

vclose(a::V3, b::V3, tol = 1e-9) = RT.vnorm(a - b) < tol
std_local(A) = (μ = sum(A) / length(A); sqrt(sum((A .- μ) .^ 2) / length(A)))

@testset "RayTracer" begin

    @testset "V3 arithmetic" begin
        a = V3(1.0, 2.0, 3.0)
        b = V3(4.0, 5.0, 6.0)
        @test vclose(a + b, V3(5.0, 7.0, 9.0))
        @test vclose(a - b, V3(-3.0, -3.0, -3.0))
        @test vclose(-a, V3(-1.0, -2.0, -3.0))
        @test vclose(a * 2.0, V3(2.0, 4.0, 6.0))
        @test vclose(2.0 * a, V3(2.0, 4.0, 6.0))
        @test RT.vdot(a, b) ≈ 32.0
        @test RT.vnorm(V3(3.0, 4.0, 0.0)) ≈ 5.0
        @test RT.vnorm(RT.vunit(a)) ≈ 1.0
        # Right-handed cross product: x̂ × ŷ = ẑ
        @test vclose(RT.vcross(V3(1.0, 0.0, 0.0), V3(0.0, 1.0, 0.0)),
                     V3(0.0, 0.0, 1.0))
    end

    @testset "Ray normalizes its direction" begin
        r = Ray(V3(0.0, 0.0, 0.0), V3(0.0, 0.0, 5.0))
        @test RT.vnorm(r.direction) ≈ 1.0
        @test vclose(r.direction, V3(0.0, 0.0, 1.0))
        @test_throws ArgumentError Ray(V3(0.0, 0.0, 0.0), V3(0.0, 0.0, 0.0))
    end

    @testset "Ray-sphere intersection" begin
        mat = Material(V3(1.0, 0.0, 0.0))
        s   = Sphere(V3(0.0, 0.0, 5.0), 1.0, mat)
        # Ray straight at the sphere along +z: front face is at z = 4
        r = Ray(V3(0.0, 0.0, 0.0), V3(0.0, 0.0, 1.0))
        h = intersect_object(r, s)
        @test h !== nothing
        @test h.t ≈ 4.0
        @test vclose(h.point, V3(0.0, 0.0, 4.0))
        @test vclose(h.normal, V3(0.0, 0.0, -1.0))

        # Sideways ray that misses the sphere completely
        r_miss = Ray(V3(0.0, 0.0, 0.0), V3(1.0, 0.0, 0.0))
        @test intersect_object(r_miss, s) === nothing

        # Ray that grazes — origin inside the sphere should still produce
        # a forward exit hit, not nothing
        r_inside = Ray(V3(0.0, 0.0, 5.0), V3(0.0, 0.0, 1.0))
        hi = intersect_object(r_inside, s)
        @test hi !== nothing
        @test hi.t ≈ 1.0
    end

    @testset "Ray-plane intersection" begin
        mat = Material(V3(0.5, 0.5, 0.5))
        pl  = Plane(V3(0.0, 0.0, 0.0), V3(0.0, 1.0, 0.0), mat)
        # Ray pointing straight down, height 5 → t = 5
        r = Ray(V3(0.0, 5.0, 0.0), V3(0.0, -1.0, 0.0))
        h = intersect_object(r, pl)
        @test h !== nothing
        @test h.t ≈ 5.0
        @test vclose(h.normal, V3(0.0, 1.0, 0.0))    # flipped to face the ray
        # Parallel ray (along x) never hits a y-up plane
        r_par = Ray(V3(0.0, 5.0, 0.0), V3(1.0, 0.0, 0.0))
        @test intersect_object(r_par, pl) === nothing
        # Behind-camera intersection is rejected
        r_back = Ray(V3(0.0, 5.0, 0.0), V3(0.0, 1.0, 0.0))
        @test intersect_object(r_back, pl) === nothing
    end

    @testset "Plane checkerboard switches colour" begin
        mat = Material(V3(1.0, 1.0, 1.0))
        pl  = Plane(V3(0.0, 0.0, 0.0), V3(0.0, 1.0, 0.0), mat,
                    V3(0.0, 0.0, 0.0); check_size = 1.0)
        # Two adjacent unit cells on the floor should give opposite colours
        r1 = Ray(V3(0.3, 5.0, 0.3), V3(0.0, -1.0, 0.0))
        r2 = Ray(V3(1.3, 5.0, 0.3), V3(0.0, -1.0, 0.0))
        h1 = intersect_object(r1, pl)
        h2 = intersect_object(r2, pl)
        @test h1 !== nothing && h2 !== nothing
        @test h1.material.color != h2.material.color
    end

    @testset "Shadows occlude the light" begin
        scene = Scene()
        push!(scene.spheres,
              Sphere(V3(0.0, 0.0, 0.0), 0.5, Material(V3(1.0, 1.0, 1.0))))
        # Point below the sphere, light directly above: blocked
        ldir = RT.vunit(V3(0.0, 2.0, 0.0))
        @test RT.in_shadow(scene, V3(0.0, -1.0, 0.0), ldir, 2.0)
        # Same geometry but offset 5 units along z: nothing in the way
        @test !RT.in_shadow(scene, V3(0.0, -1.0, 5.0), ldir, 2.0)
    end

    @testset "Render returns clamped channels of the right size" begin
        scene = Scene()
        push!(scene.spheres,
              Sphere(V3(0.0, 0.0, 4.0), 1.0, Material(V3(0.8, 0.2, 0.2))))
        push!(scene.lights,
              PointLight(V3(2.0, 4.0, 0.0), V3(1.0, 1.0, 1.0)))
        cam = Camera(V3(0.0, 0.0, 0.0), V3(0.0, 0.0, 1.0),
                     V3(0.0, 1.0, 0.0), 60.0)
        R, G, B = render(scene, cam; width = 32, height = 32)
        @test size(R) == (32, 32)
        @test size(G) == (32, 32)
        @test size(B) == (32, 32)
        @test all(0.0 .≤ R .≤ 1.0)
        @test all(0.0 .≤ G .≤ 1.0)
        @test all(0.0 .≤ B .≤ 1.0)
        # Centre pixel should land on the red sphere
        @test R[16, 16] > 0.2
        @test R[16, 16] > G[16, 16]
        @test R[16, 16] > B[16, 16]
        # Top-left corner is sky, which is blue-dominant
        @test B[1, 1] > R[1, 1]
    end

    @testset "Mirror sphere reflects the sky" begin
        scene = Scene()
        # Pure mirror: no diffuse colour of its own to confuse the test
        push!(scene.spheres,
              Sphere(V3(0.0, 0.0, 4.0), 1.0,
                     Material(V3(0.0, 0.0, 0.0),
                              0.0, 0.0, 0.0, 1.0, 1.0)))
        push!(scene.lights,
              PointLight(V3(0.0, 10.0, 0.0), V3(1.0, 1.0, 1.0)))
        cam = Camera(V3(0.0, 0.0, 0.0), V3(0.0, 0.0, 1.0),
                     V3(0.0, 1.0, 0.0), 60.0)
        R, G, B = render(scene, cam; width = 32, height = 32, depth = 2)
        # Centre pixel hits the mirror. The only thing to reflect is the
        # sky (default Scene background is blue), so B should dominate.
        @test B[16, 16] > R[16, 16]
        @test B[16, 16] > 0.3
    end

    @testset "basic_scene smoke test" begin
        scene, cam = basic_scene()
        R, G, B = render(scene, cam; width = 32, height = 32, depth = 2)
        # The scene has a red sphere, a blue sphere, and a chrome sphere
        # over a checker floor. The render shouldn't be uniform or all-black.
        @test maximum(R) > 0.3
        @test maximum(G) > 0.3
        @test maximum(B) > 0.3
        @test std_local(R) > 0.05
    end
end
