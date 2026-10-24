import AlifeCore
import Foundation
import Testing
import simd

@testable import AlifeControl

/// The renderer's side of the globe that needs no GPU (DESIGN §36.10): the uniforms' layout and their shader mirror, the entry points, the
/// look's schedules, the tiles of the mesh, and what the shader source must say. The drawing is `GPUGlobeTests`.
struct GlobeRenderingTests {
    @Test func theGlobeUniformsHaveTheLayoutOfTheShader() {
        #expect(MemoryLayout<GlobeUniforms>.size == 160 && MemoryLayout<GlobeUniforms>.stride == 160)
        let offsets = [MemoryLayout<GlobeUniforms>.offset(of: \.a), MemoryLayout<GlobeUniforms>.offset(of: \.b),
                       MemoryLayout<GlobeUniforms>.offset(of: \.c), MemoryLayout<GlobeUniforms>.offset(of: \.right),
                       MemoryLayout<GlobeUniforms>.offset(of: \.up), MemoryLayout<GlobeUniforms>.offset(of: \.centre),
                       MemoryLayout<GlobeUniforms>.offset(of: \.copies), MemoryLayout<GlobeUniforms>.offset(of: \.look),
                       MemoryLayout<GlobeUniforms>.offset(of: \.shade), MemoryLayout<GlobeUniforms>.offset(of: \.light)]
        #expect(offsets == (0..<10).map { 16 * $0 })
        let block = Shaders.source.matches(of: #/struct GlobeU \{([^}]*)\};/#).first.map { String($0.1) } ?? ""
        let names = block.matches(of: #/float4\s+(\w+);/#).map { String($0.1) }
        #expect(names == ["a", "b", "c", "right", "up", "centre", "copies", "look", "shade", "light"])
    }

    @Test func theGlobeEntryPointsAreDefinedAndListed() {
        for name in Shaders.globeEntryPoints {
            #expect(Shaders.entryPoints.contains(name))
            #expect(Shaders.source.contains(" \(name)("), "\(name) is defined")
        }
        #expect(Shaders.source.contains("constant GlobeU& globe [[buffer(8)]]"), "the agents, bonds and points are placed by the morph")
        #expect(Shaders.source.contains("dot(place.nrm, scene.eye.xyz - place.foot) > 0.0"), "the facing test on the foot")
    }

    @Test func theShaderMorphIsTheSwiftMorphLineForLine() {
        // The vertex shader's arithmetic is `Morph`'s: the same constants and the same expressions.
        let source = Shaders.source
        for expression in ["float theta = min(r / rho, 6.2831853 - 0.2);", "float versine = 2.0 * sin(0.5 * theta) * sin(0.5 * theta);",
                           "return float3(rho * k * ux, rho * k * uy, R - rho * versine) + lift * nhat;",
                           "float cl = 1.0 / cosh(u);", "return mix(roll, (g.a.w + g.b.w * z) * globeDirection(e, n, g), w);"] {
            #expect(source.contains(expression), "\(expression)")
        }
        #expect(abs((2 * Float.pi - 0.2) - 6.0831853) < 1e-6)
    }

    @Test func theLookIsContinuousAndHasItsEnds() {
        let ss = stride(from: Float(0), through: 1, by: 0.001).map { $0 }
        for f in [GlobeLook.space, GlobeLook.stars, GlobeLook.fogStrength, GlobeLook.lit, GlobeLook.rim, GlobeLook.graticule, GlobeLook.caps,
                  { GlobeLook.fogClear(s: $0) * 4 }, { GlobeLook.fogWidth(s: $0) * 4 }, { GlobeLook.territoryBorder(s: $0) }] as [(Float) -> Float] {
            let v = ss.map(f)
            #expect(zip(v, v.dropFirst()).allSatisfy { abs($0 - $1) < 0.02 }, "no step")
        }
        #expect(GlobeLook.space(s: 0) == 0 && GlobeLook.space(s: 1) == 1)
        #expect(GlobeLook.fogStrength(s: 0) == 1 && GlobeLook.fogStrength(s: 1) == 0)
        #expect(GlobeLook.fogClear(s: 0) == WorldFog.clear && GlobeLook.fogWidth(s: 0) == WorldFog.width, "the flat map's haze at s = 0")
        #expect(GlobeLook.lit(s: 0) == 0 && GlobeLook.rim(s: 0) == 0 && GlobeLook.graticule(s: 0) == 0 && GlobeLook.caps(s: 0) == 0)
        #expect(GlobeLook.territoryBorder(s: 0) == 0.55 && abs(GlobeLook.territoryBorder(s: 1) - 0.25) < 1e-6)
        // The mesh's edge (half a world from the centre) is inside the fog's end wherever the tiles beside it have gone.
        for s: Float in stride(from: 0, through: 0.3, by: 0.05) {
            let x = (0.5 - GlobeLook.fogClear(s: s)) / GlobeLook.fogWidth(s: s)
            #expect(s >= 0.1 || x >= 0, "s \(s)")
        }
        let atEnd = (0.5 - GlobeLook.fogClear(s: 0.3)) / GlobeLook.fogWidth(s: 0.3)
        #expect(1 - exp(-atEnd * atEnd) > 0.99, "at s = 0.3 the edge of the central tile is hidden: \(atEnd)")
    }

    @Test func theTilesAreTheCentralOneAndTheCopiesWhileTheyExist() {
        func vantage(distance: Float) -> WorldVantage {
            WorldVantage(pose: WorldPose(x: 0, y: 0, distance: distance), size: 320, height: 80, aspect: 1.5, globe: true)
        }
        let flat = vantage(distance: 20)
        let plan = SeamCopies.plan(y: 0, extentY: 10, size: 320, s: flat.s)
        let g = GlobeUniforms(flat, copies: plan, fog: true)
        #expect(g.shade.w == 1 && g.look.y == 1 && g.shade.x == 0)
        #expect(g.tiles(wide: false) == [SIMD4(0, 0, 1, 0)], "no copy across the wrap: it is not reached")
        #expect(g.tiles(wide: true).count == 3, "the central tile and its neighbours in x")
        let near = GlobeUniforms(flat, copies: SeamCopies.plan(y: 150, extentY: 30, size: 320, s: flat.s), fog: true)
        #expect(near.tiles(wide: true).count == 6, "and the copy across the wrap, with its neighbours")
        let globe = vantage(distance: 0.5 * 320)
        let far = GlobeUniforms(globe, copies: SeamCopies.plan(y: 150, extentY: 30, size: 320, s: globe.s), fog: true)
        #expect(far.tiles(wide: true) == [SIMD4(0, 0, 1, 0)], "from s = 0.3 only the central tile")
        #expect(far.shade.z == 1 && far.look.x == 1, "caps and space at the globe")
        #expect(GlobeUniforms().shade.w == 0, "the flat path's constants are the flag off")
    }
}
