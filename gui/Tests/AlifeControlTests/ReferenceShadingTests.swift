import AlifeCore
import Foundation
import Testing
import simd

@testable import AlifeControl

/// `ReferenceSphere` and its helpers, which the GPU test measures the rendered picture against, checked on the CPU: the
/// reference must say what the shader source says, and the GPU test must have agents to look at.
@Suite struct ReferenceShadingTests {
    /// A unit-radius agent at the origin seen face on from the -z side (the 2D viewer's view), facing `forward`.
    private func face(forward: SIMD3<Float>, color: SIMD3<Float> = SIMD3(1, 0.25, 0)) -> ReferenceSphere {
        ReferenceSphere(center: .zero, forward: simd_normalize(forward), color: color, radius: 1)
    }

    private func shade(_ sphere: ReferenceSphere, at offset: SIMD2<Float>) -> (color: SIMD3<Float>?, t: Float, steady: Bool) {
        // The eye 50 units below the plane, the ray through the point `offset` of the plane z = 0.
        let eye = SIMD3<Float>(0, 0, -50)
        return sphere.shade(eye: eye, dir: simd_normalize(SIMD3(offset.x, offset.y, 0) - eye))
    }

    @Test func theConstantsAreTheShaders() {
        func constant(_ name: String) -> Float? {
            Shaders.source.firstMatch(of: try! Regex("constant float \(name) = ([0-9.]+);"))
                .flatMap { $0.output[1].substring }.flatMap { Float($0) }
        }
        #expect(constant("noseOffset") == ReferenceSphere.noseOffset)
        #expect(constant("noseRadius") == ReferenceSphere.noseRadius)
        // The shader's headlight and the nose's paleness, as written (in `shadeSphere`, which `sphereFragment` and the World's
        // `agentFragmentA` both call).
        #expect(Shaders.source.contains("mix(baseColor, float3(1.0), 0.6)"))
        #expect(Shaders.source.contains("base * (0.3 + 0.7 * lambert)"))
    }

    @Test func theMiddleOfTheBodyIsItsColour() throws {
        let body = try #require(shade(face(forward: SIMD3(1, 0, 0)), at: .zero).color)
        #expect(simd_length(body - SIMD3(1, 0.25, 0)) < 0.01, "facing the eye the headlight is at full strength")
    }

    @Test func theNoseShowsBeyondTheBodyOnTheSideTheAgentFaces() throws {
        let sphere = face(forward: SIMD3(1, 0, 0))
        let ahead = shade(sphere, at: SIMD2(1.1, 0))
        let nose = try #require(ahead.color, "the nose reaches 1.25 radii")
        #expect(ahead.steady)
        #expect(nose.y > 0.4 && nose.z > 0.3, "paler than the body (1, 0.25, 0): toward white")
        #expect(shade(sphere, at: SIMD2(-1.1, 0)).color == nil, "nothing on the other side")
        #expect(shade(sphere, at: SIMD2(0, 1.1)).color == nil, "nothing to the side")
        #expect(shade(sphere, at: SIMD2(1.3, 0)).color == nil, "beyond the nose")
        // Turned the other way, the nose is on the other side.
        let behind = face(forward: SIMD3(-1, 0, 0))
        #expect(shade(behind, at: SIMD2(-1.1, 0)).color != nil && shade(behind, at: SIMD2(1.1, 0)).color == nil)
        // And a yaw of a quarter turn puts it on +y.
        let up = face(forward: SIMD3(0, 1, 0))
        #expect(shade(up, at: SIMD2(0, 1.1)).color != nil && shade(up, at: SIMD2(1.1, 0)).color == nil)
    }

    @Test func aNoseTurnedTowardTheEyeCoversTheMiddle() throws {
        let toward = try #require(shade(face(forward: SIMD3(0, 0, -1)), at: .zero).color)
        let away = try #require(shade(face(forward: SIMD3(0, 0, 1)), at: .zero).color)
        #expect(simd_length(away - SIMD3(1, 0.25, 0)) < 0.01, "a nose behind the body is hidden by it")
        #expect(toward.y > 0.25 + 0.3 && toward.z > 0.3, "a nose in front of the body is paler: toward white")
    }

    /// What the GPU test would compare for the fixtures: enough steady pixels, and every agent among them.
    @Test(arguments: ["2d", "3d"])
    func theGPUTestHasSomethingToCompareForEveryAgent(kind: String) throws {
        let fixture = try StreamFixtureData(kind: kind)
        let frame = fixture.frames[0]
        let camera = Camera.home(for: fixture.header, aspect: 1)
        let spheres = (0..<frame.rows).filter(frame.isAlive).map {
            ReferenceSphere(frame: frame, row: $0, radius: Float(fixture.header.radius))
        }
        #expect(spheres.count == frame.nAlive)
        let rays = PixelRays(camera: camera, width: 512, height: 512)
        let pixels = pixelsAround(frame, camera: camera, size: 512, half: kind == "2d" ? 16 : 11)
        var compared = 0, lit = 0, noseSeen = Set<Int>()
        var seen = Set<Int>()
        for p in pixels {
            let pixel = referencePixel(spheres, rays: rays, x: p.x, y: p.y)
            guard pixel.steady else { continue }
            compared += 1
            guard let color = pixel.color, let nearest = pixel.nearest else { continue }
            lit += 1
            seen.insert(nearest)
            // The nose is paler than the body: greener than a body whose green is below 0.5 (all the colour moves
            // toward white by 60 %, so a channel under 1 gains at least 0.6 times its distance from 1).
            let body = spheres[nearest].color
            if color.y > body.y + 0.1 && body.y < 0.6 { noseSeen.insert(nearest) }
        }
        #expect(compared > 1500 && lit > 300, "\(compared) comparable pixels, \(lit) lit")
        #expect(seen.count == spheres.count, "the reference picture shows agents \(seen.sorted()) of \(spheres.count)")
        #expect(noseSeen.count >= 3, "noses are seen on agents \(noseSeen.sorted())")
    }

    @Test func aPixelsRayLeavesTheEyeThroughThePixel() throws {
        let fixture = try StreamFixtureData(kind: "3d")
        let camera = Camera.home(for: fixture.header, aspect: 1)
        let rays = PixelRays(camera: camera, width: 512, height: 512)
        #expect(rays.eye == camera.position)
        // The middle of the picture looks at the target (half a pixel off).
        #expect(dot(rays.direction(x: 256, y: 256), camera.forward) > 0.9999)
        // A point on a pixel's ray projects back to that pixel.
        let world = rays.eye + 70 * rays.direction(x: 100, y: 400)
        let ndc = try #require(camera.ndc(of: world, aspect: 1))
        let column = Int((ndc.x * 0.5 + 0.5) * 512)
        let row = Int((0.5 - ndc.y * 0.5) * 512)
        #expect(column == 100 && row == 400)
    }
}
