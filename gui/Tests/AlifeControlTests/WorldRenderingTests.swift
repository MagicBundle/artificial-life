import Foundation
import Testing

@testable import AlifeControl

/// The parts of the World's renderer that need no GPU: the uniform block and the shader source.
@Suite struct WorldRenderingTests {
    @Test func theWorldUniformsHaveTheLayoutOfTheShaders() {
        // `WorldU` in Shaders.swift: five 16-byte vectors.
        #expect(MemoryLayout<WorldUniforms>.size == 80)
        #expect(MemoryLayout<WorldUniforms>.stride == 80)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.target) == 0)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.params) == 16)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.refs) == 32)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.flags) == 48)
        #expect(MemoryLayout<WorldUniforms>.offset(of: \.style) == 64)
    }

    @Test func theShaderStructMirrorsTheSwiftOne() {
        let block = Shaders.source.matches(of: #/struct WorldU \{([^}]*)\};/#).first.map { String($0.1) } ?? ""
        let names = block.matches(of: #/(?:float4|uint4)\s+(\w+);/#).map { String($0.1) }
        #expect(names == ["target", "params", "refs", "flags", "style"])
        let types = block.matches(of: #/(float4|uint4)\s+\w+;/#).map { String($0.1) }
        #expect(types == ["float4", "float4", "float4", "uint4", "float4"], "flags is the one integer vector")
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
        #expect(source.contains("address::repeat"), "the map is one quad with a repeating sampler")
    }

    @Test func theSphereShadingIsOneFunctionForTheLabAndTheWorld() {
        let source = Shaders.source
        #expect(source.components(separatedBy: "shadeSphere(").count - 1 == 3, "the definition and its two callers")
        #expect(source.contains("fragment SphereOut sphereFragment("))
        #expect(source.contains("fragment SphereOut agentFragmentA("))
    }
}
