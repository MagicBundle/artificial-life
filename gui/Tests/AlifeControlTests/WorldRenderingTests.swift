import AlifeCore
import Foundation
import Testing
import simd

@testable import AlifeControl

/// The parts of the World's renderer that need no GPU: the uniform block and the shader source.
@Suite struct WorldRenderingTests {
    @Test func theWorldUniformsHaveTheLayoutOfTheShaders() {
        // `WorldU` in Shaders.swift: eight 16-byte vectors.
        #expect(MemoryLayout<WorldUniforms>.size == 128)
        #expect(MemoryLayout<WorldUniforms>.stride == 128)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.target) == 0)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.params) == 16)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.refs) == 32)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.flags) == 48)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.style) == 64)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.size) == 80)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.look) == 96)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.patch) == 112)
    }

    @Test func theShaderStructMirrorsTheSwiftOne() {
        let block = Shaders.source.matches(of: #/struct WorldU \{([^}]*)\};/#).first.map { String($0.1) } ?? ""
        let names = block.matches(of: #/(?:float4|uint4)\s+(\w+);/#).map { String($0.1) }
        #expect(names == ["target", "params", "refs", "flags", "style", "size", "look", "patch"])
        let types = block.matches(of: #/(float4|uint4)\s+\w+;/#).map { String($0.1) }
        #expect(types == ["float4", "float4", "float4", "uint4", "float4", "float4", "float4", "float4"], "flags is the one integer vector")
    }

    @Test func theWorldEntryPointsAreTheOnesTheRendererBuilds() {
        for name in [Shaders.agentVertexA, Shaders.agentFragmentA, Shaders.agentFragmentFlat, Shaders.bondVertexA,
                     Shaders.pointVertexA, Shaders.pointFragment, Shaders.mapVertex, Shaders.mapFragment] {
            #expect(Shaders.entryPoints.contains(name))
            #expect(Shaders.source.contains(" \(name)("), "\(name) is defined")
        }
        #expect(Set(Shaders.entryPoints).count == Shaders.entryPoints.count)
    }

    @Test func theWorldBitsAreTheOnesTheShaderTests() {
        #expect(WorldUniforms.mapBit == 1 && WorldUniforms.territoryBit == 2 && WorldUniforms.equatorBit == 4)
        for expression in ["world.flags.w & 1u", "world.flags.w & 2u", "world.flags.w & 4u"] {
            #expect(Shaders.source.contains(expression), "\(expression)")
        }
    }

    @Test func everyWorldDrawTakesTheMinimumImage() {
        // The agents, the bonds and the points are placed at the image nearest the camera target; the map is tiled.
        let source = Shaders.source
        #expect(source.contains("float3 minimumImage(float3 p, float2 target, float L)"))
        #expect(source.contains("d -= L * rint(d / L);"))
        let functions = [("AgentVaryingA", "agentVertexA"), ("LineVarying", "bondVertexA"), ("PointVarying", "pointVertexA")]
        for (result, name) in functions {
            guard let at = source.range(of: "vertex \(result) \(name)("),
                  let end = source.range(of: "\n    }\n", range: at.upperBound..<source.endIndex) else {
                Issue.record("\(name) not found")
                continue
            }
            #expect(source[at.lowerBound..<end.upperBound].contains("minimumImage("), "\(name) draws at the minimum image")
        }
        #expect(source.contains("address::repeat"), "the map's texture repeats, so the wrap in x is seamless")
    }

    @Test func theMapIsOnePatchOfOneWorldAndItsCopiesAcrossTheWrapInY() {
        let source = Shaders.source
        let start = source.range(of: "vertex MapVarying mapVertex(")!
        let end = source.range(of: "constexpr sampler planeSampler")!
        let body = String(source[start.lowerBound..<end.lowerBound])
        #expect(body.contains("[[instance_id]]") && body.contains("world.patch"), "three instances: the cell k - 1, k and k + 1")
        #expect(body.contains("corner * (0.5 * L)"), "one world wide, not five (the quad of v1 was 2.5 L on a side)")
        #expect(!body.contains("2.5 * L") && !body.contains("floor(world.target.xy"), "the 5 x 5 quad is gone")
        #expect(body.contains("float2(world.target.x, k * L)"), "centred on the camera in x, in the cell of the wrap in y")
        #expect(body.contains("alpha <= 0.0"), "a copy that is not wanted collapses")
        // The Swift side hands the shader the cell of the view's centre and the alphas of the three cells around it.
        let copies = [SeamCopies.Copy(k: 1, alpha: 1), SeamCopies.Copy(k: 2, alpha: 0.5)]
        #expect(Renderer.patchUniform(copies, y: 0.9 * 48, size: 48) == SIMD4(1, 0, 1, 0.5))
        #expect(Renderer.patchUniform([SeamCopies.Copy(k: 0, alpha: 1)], y: 10, size: 48) == SIMD4(0, 0, 1, 0))
        #expect(Renderer.patchUniform(nil, y: 10, size: 48) == SIMD4(0, 1, 1, 1), "no plan: all three")
    }

    @Test func theSphereShadingIsOneFunctionForTheLabAndTheWorld() {
        let source = Shaders.source
        #expect(source.components(separatedBy: "shadeSphere(").count - 1 == 3, "the definition and its two callers")
        #expect(source.contains("fragment SphereOut sphereFragment("))
        #expect(source.contains("fragment SphereOut agentFragmentA("))
    }

    // MARK: The constants of the shaders are the Swift mirror's

    /// The decimal literals of the first line of the shader source that contains `token`, in order.
    private func literals(inLineWith token: String) -> [Float] {
        guard let line = Shaders.source.split(separator: "\n").first(where: { $0.contains(token) }) else { return [] }
        return line.matches(of: #/-?\d+\.\d+/#).compactMap { Float(String($0.output)) }
    }

    private func floats(_ v: SIMD3<Float>) -> [Float] { [v.x, v.y, v.z] }

    @Test func theMapPaletteInTheShaderIsTheMirrorsInAlifeCore() {
        // (a token of the shader's line, the numbers MapPalette holds, in the order the shader writes them)
        let expected: [(String, [Float])] = [
            ("float3 sea = mix(", floats(MapPalette.coldSea) + floats(MapPalette.warmSea) + [MapPalette.seaEdge.x, MapPalette.seaEdge.y]),
            ("world.look.w);", floats(MapPalette.hemisphereSouth) + floats(MapPalette.hemisphereNorth) + [MapPalette.hemisphereMix]),
            ("smoothstep(0.0, 0.8, zeta)", floats(MapPalette.cyclonic) + [MapPalette.vorticityMix, 0, MapPalette.vorticityEdge]),
            ("smoothstep(0.0, 0.8, -zeta)", floats(MapPalette.anticyclonic) + [MapPalette.vorticityMix, 0, MapPalette.vorticityEdge]),
            ("1.4 * rainDepth", floats(MapPalette.rain) + [MapPalette.rainGain, 0, MapPalette.rainMax]),
            ("0.88 * smoothstep", floats(MapPalette.cloud) + [MapPalette.cloudMix, MapPalette.cloudEdge.x, MapPalette.cloudEdge.y]),
            ("float3 color = float3(0.045", floats(MapPalette.floor)),
            ("const float3 background", floats(MapPalette.background)),
            ("0.35 * line", floats(MapPalette.equator) + [MapPalette.equatorMix]),
        ]
        for (token, numbers) in expected {
            let found = literals(inLineWith: token)
            #expect(found.count == numbers.count && zip(found, numbers).allSatisfy { abs($0 - $1) < 1e-6 },
                    "\(token): the shader has \(found), MapPalette \(numbers)")
        }
    }

    @MainActor @Test func theClearColourIsThePalettesBackground() {
        let clear = Renderer.background
        #expect(abs(clear.red - Double(MapPalette.background.x)) < 1e-6 && abs(clear.green - Double(MapPalette.background.y)) < 1e-6)
        #expect(abs(clear.blue - Double(MapPalette.background.z)) < 1e-6)
    }

    @Test func theOutlineRuleInTheShaderIsOutlineContrasts() {
        let source = Shaders.source
        #expect(source.contains("return l > 0.17 ? float3(0.06) : float3(0.92);"))
        #expect(OutlineContrast.threshold == 0.17 && OutlineContrast.dark == SIMD3(repeating: 0.06) && OutlineContrast.light == SIMD3(repeating: 0.92))
        #expect(source.contains("float l = 0.2126 * srgbToLinear(shaded.r) + 0.7152 * srgbToLinear(shaded.g) + 0.0722 * srgbToLinear(shaded.b);"))
        #expect(source.contains("c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)"))
        // The pixel next to the ring: n.v at (ring + 0.5) px inside, the sphere's headlight 0.3 + 0.7 n.v, the disc's 1 - 0.3 r^2.
        #expect(source.contains("float k = clamp((rrPx - ring - 0.5) / max(rrPx, 1e-3), 0.0, 1.0);"))
        #expect(source.contains("outlineFor(base * (0.3 + 0.7 * sqrt(1.0 - k * k)))"))
        #expect(source.contains("outlineFor(base * (1.0 - 0.3 * k * k))"))
        #expect(OutlineContrast.width == 1)
    }

    @Test func theRendererGivesTheShaderTheSizePolicyAndTheLookAsUniforms() {
        // `WorldU.size` and `.look` carry the ramp, the depth cue and the territory border from the Swift constants.
        #expect(OrganismSizePolicy.rampLow == 1.5 && OrganismSizePolicy.rampHigh == 3.5)
        #expect(OutlineContrast.depthCueBase + OutlineContrast.depthCueSlope == 1, "the cue reaches 1 at the lid")
        #expect(MapPalette.territoryBorder == 0.55 && MapPalette.territoryTint == 0.10)
        let source = Shaders.source
        for expression in ["world.look.z + world.size.w * clamp(raw.z / scene.world.y, 0.0, 1.0)",
                           "smoothstep(world.size.y, world.size.z, drawnPx)",
                           "clamp(scene.world.z * perPixel, smallest, max(smallest, world.size.x))",
                           "world.params.z * here.a", "world.look.y * border", "color * world.look.x"] {
            #expect(source.contains(expression), "\(expression)")
        }
    }
}
