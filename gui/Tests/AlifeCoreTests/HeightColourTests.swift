import Foundation
import Testing
import simd

@testable import AlifeCore

/// The height colour mode's ramp (DESIGN §36.5).
@Suite struct HeightColourTests {
    private func linear(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }

    /// CIE L* of an sRGB colour.
    private func lightness(_ c: SIMD3<Float>) -> Float {
        let y = 0.2126 * linear(c.x) + 0.7152 * linear(c.y) + 0.0722 * linear(c.z)
        return y > 216 / 24389 ? 116 * cbrt(y) - 16 : 24389 / 27 * y
    }

    @Test func theRampIsViridisAt17StopsFromPurpleToYellow() {
        #expect(HeightColour.stops.count == 17 && HeightColour.hexStops.count == 17)
        #expect(HeightColour.hexStops.first == 0x440154 && HeightColour.hexStops.last == 0xFDE725)
        #expect(HeightColour.color(0) == HeightColour.stops[0] && HeightColour.color(1) == HeightColour.stops[16])
        let mid = HeightColour.color(0.5)
        #expect(mid == HeightColour.stops[8])
        let quarterStep = HeightColour.color(1.5 / 16)
        #expect(simd_length(quarterStep - (HeightColour.stops[1] + HeightColour.stops[2]) / 2) < 1e-6, "linear between stops")
    }

    @Test func itIsClampedAndSafeForWhatAShaderIsGiven() {
        #expect(HeightColour.color(-3) == HeightColour.stops[0] && HeightColour.color(7) == HeightColour.stops[16])
        #expect(HeightColour.color(.nan) == HeightColour.stops[0] && HeightColour.color(.infinity) == HeightColour.stops[16])
        #expect(HeightColour.fraction(z: 40, height: 80) == 0.5 && HeightColour.fraction(z: -5, height: 80) == 0 && HeightColour.fraction(z: 90, height: 80) == 1)
        #expect(HeightColour.fraction(z: 10, height: 0) == 0 && HeightColour.fraction(z: .nan, height: 80) == 0)
    }

    @Test func lightnessRisesSteadilyAlongTheRampSoItReadsInGreyAndThroughColourBlindness() {
        let l = HeightColour.stops.map(lightness)
        for i in 1..<l.count {
            let step = l[i] - l[i - 1]
            #expect(step > 3.5 && step < 6.5, "L* steps by \(step) at stop \(i): perceptually even")
        }
        #expect(l.first! < 20 && l.last! > 85)
    }

    /// CIELAB (D65) of an sRGB colour.
    private func lab(_ c: SIMD3<Float>) -> SIMD3<Float> {
        let (r, g, b) = (linear(c.x), linear(c.y), linear(c.z))
        let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
        func f(_ t: Float) -> Float { t > 216 / 24389 ? cbrt(t) : (24389 / 27 * t + 16) / 116 }
        return SIMD3(116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))
    }

    @Test func theEndsOfTheRampWhereMostOrganismsAreStandApartFromBothSeasByColourNotJustLuminance() {
        // (review finding: the dark end of the ramp is within 1.04 to 1.5 of the cold sea in WCAG luminance contrast, and half of the recorded
        // planet's organisms are in the lowest fifth of the volume. Luminance is the wrong measure for a purple on a blue: in CIELAB the
        // lowest stop is 36 from the cold sea and 67 from the warm one, and a render of the 740-point window shows purple interiors. A ramp
        // started lighter was rendered too and was worse: its blue is the sea's. The confusable band is the middle, z 20 to 40, where few
        // organisms are.)
        let cold = lab(MapPalette.coldSea), warm = lab(MapPalette.warmSea)
        for t in Array(stride(from: Float(0), through: 0.1, by: 0.025)) + Array(stride(from: Float(0.7), through: 1, by: 0.05)) {
            let c = lab(HeightColour.color(t))
            #expect(simd_distance(c, cold) > 25 && simd_distance(c, warm) > 25, "t = \(t): \(simd_distance(c, cold)) from cold, \(simd_distance(c, warm)) from warm")
        }
        let lowest = lab(HeightColour.color(0))
        #expect(simd_distance(lowest, cold) > 30 && simd_distance(lowest, warm) > 50)
        // The same stops, measured by luminance contrast, are the finding's numbers: that measure does not say they are alike.
        #expect(OutlineContrast.contrast(HeightColour.color(0), MapPalette.coldSea) < 1.5)
    }

    @Test func noStopIsNearTheEnginesHurtRedWhichEveryModeKeeps() {
        let red = SIMD3<Float>(1, 0, 0)
        for stop in HeightColour.stops { #expect(simd_distance(stop, red) > 0.7, "\(stop)") }
        #expect(HeightColour.isHurtRed(0xFF0000) && HeightColour.isHurtRed(0x00FF0000) && !HeightColour.isHurtRed(0xFE0000) && !HeightColour.isHurtRed(0x440154))
    }

    @Test func theModeIsOnlyForAWorldWithHeightAndIsNotTheDefault() {
        #expect(HeightColour.isAvailable(dims: 3, height: 80) && !HeightColour.isAvailable(dims: 2, height: 0) && !HeightColour.isAvailable(dims: 3, height: 0))
        #expect(HeightColour.modeName == "height" && HeightColour.title == "Height")
        #expect(!HeightColour.isDefaultIn3D, "the decision of DESIGN 36.5; this is the one constant that flips it")
    }

    @Test func theLegendHasTheHeightsAtItsTicks() {
        let ticks = HeightColour.ticks(height: 80)
        #expect(ticks.map(\.label) == ["0", "40", "80"] && ticks.map(\.fraction) == [0, 0.5, 1])
    }

    @Test func theShaderTableIsTheStopsToFivePlaces() {
        let table = HeightColour.shaderTable
        #expect(table.hasPrefix("float3(0.26667, 0.00392, 0.32941)"))
        #expect(table.components(separatedBy: "float3(").count - 1 == 17)
        let numbers = table.matches(of: #/([0-9]\.[0-9]{5})/#).map { Float($0.1)! }
        #expect(numbers.count == 51)
        for (i, v) in numbers.enumerated() { #expect(abs(v - HeightColour.stops[i / 3][i % 3]) < 6e-6) }
    }
}
