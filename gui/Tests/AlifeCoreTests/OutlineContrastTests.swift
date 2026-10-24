import Foundation
import Testing
import simd

@testable import AlifeCore

/// The outline rule of DESIGN §36.7: the colour is chosen from the shaded colour of the pixel next to the ring.
@Suite struct OutlineContrastTests {
    private func rgb(_ hex: UInt32) -> SIMD3<Float> {
        let (r, g, b) = WorldPalette.components(hex)
        return SIMD3(Float(r), Float(g), Float(b)) / 255
    }

    /// The colours of the tables of DESIGN §33.2.2: 16 kin slots, 3 reserved, 11 archetypes, 11 intents (the zeroes are `none`),
    /// and hurt red.
    private var tableColours: [UInt32] {
        WorldPalette.slots + [WorldPalette.unknown, WorldPalette.other, WorldPalette.none] + WorldPalette.archetypes
            + WorldPalette.intents + [0xFF0000]
    }

    private struct Generator: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    @Test func theLinearisationAndTheContrastAreWCAGs() {
        #expect(OutlineContrast.linear(0) == 0 && abs(OutlineContrast.linear(1) - 1) < 1e-6)
        #expect(abs(OutlineContrast.linear(0.5) - 0.2140) < 1e-3)
        #expect(abs(OutlineContrast.luminance(SIMD3(1, 1, 1)) - 1) < 1e-5)
        #expect(abs(OutlineContrast.contrast(SIMD3(0, 0, 0), SIMD3(1, 1, 1)) - 21) < 1e-3)
        #expect(OutlineContrast.contrast(SIMD3(0.3, 0.4, 0.5), SIMD3(0.3, 0.4, 0.5)) == 1)
        #expect(abs(OutlineContrast.luma(SIMD3(1, 1, 1)) - 1) < 1e-6)
    }

    @Test func theTwoOutlinesAreAtLeastThreeToOneAtTheThreshold() {
        let atThreshold = OutlineContrast.threshold
        // The luminance L = 0.17 is a linear value: a grey with that relative luminance.
        let grey = pow(atThreshold, 1 / 2.4) * 1.055 - 0.055
        let colour = SIMD3<Float>(repeating: grey)
        #expect(abs(OutlineContrast.luminance(colour) - atThreshold) < 2e-3)
        let darkContrast = OutlineContrast.contrast(colour, OutlineContrast.dark)
        let lightContrast = OutlineContrast.contrast(colour, OutlineContrast.light)
        #expect(darkContrast > 3.9 && darkContrast < 4.1, "the dark outline against L = 0.17: \(darkContrast)")
        #expect(lightContrast > 3.9 && lightContrast < 4.1, "the light one: \(lightContrast)")
        #expect(abs(darkContrast - lightContrast) < 0.1, "the threshold is where the two outlines are equal")
        // Just above the threshold the dark outline is chosen, at it and below the light one.
        #expect(OutlineContrast.outline(forShaded: SIMD3(repeating: grey + 0.01)) == OutlineContrast.dark)
        #expect(OutlineContrast.outline(forShaded: SIMD3(repeating: grey - 0.01)) == OutlineContrast.light)
        #expect(OutlineContrast.outline(forShaded: SIMD3(0, 0, 0)) == OutlineContrast.light)
        #expect(OutlineContrast.outline(forShaded: SIMD3(1, 1, 1)) == OutlineContrast.dark)
    }

    @Test func theRuleOnTheShadedColourGivesThreeToOneForEveryColourDepthAndShading() {
        var colours = tableColours.map(rgb)
        #expect(colours.count == 16 + 3 + 11 + 11 + 1)
        var rng = Generator(state: 1)
        for _ in 0..<1000 {  // the `tag` mode paints a colour per organism: the table is not enough
            colours.append(SIMD3(Float.random(in: 0...1, using: &rng), Float.random(in: 0...1, using: &rng), Float.random(in: 0...1, using: &rng)))
        }
        var worst = Float.infinity
        var cases = 0
        for base in colours {
            for cue in [Float(0.85), 0.92, 1.0] {
                for nv in [Float(0.2), 0.46, 1.0] {
                    let shaded = OutlineContrast.shadedSphere(base: base, depthCue: cue, normalDotView: nv)
                    let outline = OutlineContrast.outline(forShaded: shaded)
                    let k = OutlineContrast.contrast(shaded, outline)
                    worst = min(worst, k)
                    cases += 1
                    #expect(k >= 3, "base \(base) cue \(cue) n.v \(nv): \(k)")
                }
            }
        }
        #expect(cases == colours.count * 9)
        #expect(worst >= 3.9, "the worst case of the rule is 4.0 at the threshold: \(worst)")
    }

    /// The rule is evaluated at 1.5 px inside the edge; the pixel next to the ring is 1 to 2 px inside it, wherever the grid falls on
    /// the silhouette. Over that range of n.v, for the colours of the tables and a thousand random ones, at the radii the size policy
    /// draws (3.4, 7, 9 px) and the depth cues, the edge keeps 3:1 (the GPU test measures it on the pixels).
    @Test func theRuleHoldsWhereverThePixelGridFallsOnTheSilhouette() {
        var colours = tableColours.map(rgb)
        var rng = Generator(state: 7)
        for _ in 0..<1000 {
            colours.append(SIMD3(Float.random(in: 0...1, using: &rng), Float.random(in: 0...1, using: &rng), Float.random(in: 0...1, using: &rng)))
        }
        var worst = Float.infinity
        for base in colours {
            for radius: Float in [3.4, 7, 9, 17.5, 53] {
                for cue in [Float(0.85), 0.92, 1.0] {
                    let estimated = OutlineContrast.shadedSphere(base: base, depthCue: cue, normalDotView: OutlineContrast.ringNormalDotView(radiusPixels: radius))
                    let outline = OutlineContrast.outline(forShaded: estimated)
                    for inside in stride(from: Float(1.0), through: 2.0, by: 0.125) {
                        let k = min(max((radius - inside) / radius, 0), 1)
                        let shaded = OutlineContrast.shadedSphere(base: base, depthCue: cue, normalDotView: (1 - k * k).squareRoot())
                        let contrast = OutlineContrast.contrast(shaded, outline)
                        worst = min(worst, contrast)
                        #expect(contrast >= 3, "base \(base) r \(radius) cue \(cue) \(inside) px inside: \(contrast)")
                    }
                }
            }
        }
        #expect(worst >= 3, "the worst case: \(worst)")
    }

    @Test func aRuleOnTheBaseColourWouldNotHaveHeldUnderTheOldDepthCue() {
        // The first draft's rule, kept as the reason for the shaded one: the outline from the base colour of D55E00 (L just over
        // the threshold: dark), then the colour shaded at the old depth cue 0.62 and a rim: the dark outline loses.
        let base = rgb(0xD55E00)
        let wrong = OutlineContrast.outline(forShaded: base)
        let shaded = OutlineContrast.shadedSphere(base: base, depthCue: 0.62, normalDotView: 0.2)
        #expect(OutlineContrast.contrast(shaded, wrong) < 3)
        #expect(OutlineContrast.contrast(shaded, OutlineContrast.outline(forShaded: shaded)) >= 3)
    }

    @Test func theDepthCueIsSoftened() {
        #expect(OutlineContrast.depthCue(z: 0, height: 80) == 0.85)
        #expect(abs(OutlineContrast.depthCue(z: 80, height: 80) - 1) < 1e-6)
        #expect(abs(OutlineContrast.depthCue(z: 40, height: 80) - 0.925) < 1e-6)
        #expect(OutlineContrast.depthCue(z: 500, height: 80) == 1 && OutlineContrast.depthCue(z: -3, height: 80) == 0.85)
        #expect(OutlineContrast.depthCue(z: 5, height: 0) == 1, "a 2D world has no height to cue")
        // The recorded population (z quantiles 1.0 / 5.7 / 16.9 / 70.3 / 79 of 80) keeps at least 85% of its brightness.
        for z: Float in [1.0, 5.7, 16.9, 70.3, 79] { #expect(OutlineContrast.depthCue(z: z, height: 80) >= 0.85) }
    }

    @Test func theRingIsShadedAsThePixelNextToItIs() {
        // n.v at 1.5 px inside the edge: 0.55 for a 9 px sphere, 0.87 for 3 px; a sphere no larger than the ring and
        // the pixel next to it is face on (1), and none has none (0).
        #expect(abs(OutlineContrast.ringNormalDotView(radiusPixels: 9) - 0.553) < 0.005)
        #expect(abs(OutlineContrast.ringNormalDotView(radiusPixels: 3) - 0.866) < 0.005)
        #expect(OutlineContrast.ringNormalDotView(radiusPixels: 1.5) == 1 && OutlineContrast.ringNormalDotView(radiusPixels: 0) == 0)
        #expect(abs(OutlineContrast.ringDiscRadius(radiusPixels: 3) - 0.5) < 1e-6)
        #expect(OutlineContrast.sphereShade(normalDotView: 1) == 1 && OutlineContrast.sphereShade(normalDotView: -1) == 0.3)
        #expect(abs(OutlineContrast.discShade(r: 1) - 0.7) < 1e-6)
    }
}
