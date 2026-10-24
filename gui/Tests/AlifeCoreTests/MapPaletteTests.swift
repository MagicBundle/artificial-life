import Foundation
import Testing
import simd

@testable import AlifeCore

/// The map palette of DESIGN §36.6, against the clear colour: the arithmetic behind the brightness floors.
@Suite struct MapPaletteTests {
    private let background = MapPalette.background

    private func shown(_ colour: SIMD3<Float>, opacity: Float) -> SIMD3<Float> {
        MapPalette.eightBit(MapPalette.shown(MapPalette.eightBit(colour), opacity: opacity))
    }

    private func contrast(_ colour: SIMD3<Float>, opacity: Float) -> Float {
        OutlineContrast.contrast(shown(colour, opacity: opacity), MapPalette.eightBit(background))
    }

    @Test func theColdSeaIs20_59_107AndItsLumaIs0212() {
        let eight = MapPalette.eightBit(MapPalette.coldSea) * 255
        #expect(eight == SIMD3(20, 59, 107))
        #expect(abs(OutlineContrast.luma(MapPalette.eightBit(MapPalette.coldSea)) - 0.212) < 0.002)
        // v1's cold sea sat on the clear colour (5, 17, 35 against 8, 10, 14).
        let v1 = SIMD3<Float>(0.02, 0.07, 0.14)
        #expect(OutlineContrast.contrast(MapPalette.eightBit(v1), MapPalette.eightBit(background)) < 1.1)
    }

    @Test func theColdSeaIsBrightEnoughAtTheThreeOpacities() {
        // Opacities 1.0 / 0.9 / 0.7 (planet / region / organism): contrast against the clear colour 1.75 / 1.63 / 1.41, luma
        // 0.212 / 0.195 / 0.160 (v1: 0.156 / 0.139 / 0.104 at 1.0 / 0.85 / 0.55). The test's floors have a margin on them.
        for (opacity, minContrast, minLuma): (Float, Float, Float) in [(1.0, 1.70, 0.205), (0.9, 1.58, 0.188), (0.7, 1.40, 0.155)] {
            let c = contrast(MapPalette.coldSea, opacity: opacity)
            let y = OutlineContrast.luma(shown(MapPalette.coldSea, opacity: opacity))
            #expect(c >= minContrast, "opacity \(opacity): contrast \(c)")
            #expect(y >= minLuma, "opacity \(opacity): luma \(y)")
        }
    }

    @Test func theWarmSeaIsWellAboveTheClearColour() {
        // 3.75 / 3.23 / 2.38 in the arithmetic of the spec.
        for (opacity, minContrast): (Float, Float) in [(1.0, 3.6), (0.9, 3.1), (0.7, 2.3)] {
            let c = contrast(MapPalette.warmSea, opacity: opacity)
            #expect(c >= minContrast, "opacity \(opacity): \(c)")
        }
    }

    @Test func theMapDoesNotSitOnTheClearColourAnywhere() {
        // The whole range of the sea (no pool to a warm pool), with and without a hemisphere tint and the strongest signed
        // vorticity: at the organism level's opacity the map stays 1.35:1 or more from the clear colour.
        for t in stride(from: Float(0), through: 1, by: 0.1) {
            for zeta in [Float(-1), 0, 1] {
                for northness in [Float(0), 1] {
                    let c = MapPalette.color(sea: t, zeta: zeta, rain: 0, cloud: 0, northness: northness, hemisphereWeight: 1)
                    #expect(contrast(c, opacity: 0.7) >= 1.35, "T_s \(t) zeta \(zeta) north \(northness)")
                }
            }
        }
    }

    @Test func cloudAndRainAreBrighterThanTheSea() {
        let sea = MapPalette.color(sea: 0, zeta: 0, rain: 0, cloud: 0)
        let rain = MapPalette.color(sea: 0, zeta: 0, rain: 0.6, cloud: 0)
        let cloud = MapPalette.color(sea: 0, zeta: 0, rain: 0, cloud: 1)
        #expect(OutlineContrast.luma(rain) > OutlineContrast.luma(sea) + 0.1)
        #expect(OutlineContrast.luma(cloud) > OutlineContrast.luma(rain) + 0.15)
        #expect(OutlineContrast.luma(cloud) > 0.85, "cloud (216, 231, 241)-ish at full depth: \(cloud)")
        // Cloud is mixed in by 88% at most.
        let full = MapPalette.color(sea: 0, zeta: 0, rain: 0, cloud: 5)
        #expect(abs(full.x - (MapPalette.coldSea.x + (MapPalette.cloud.x - MapPalette.coldSea.x) * 0.88)) < 1e-5)
    }

    @Test func vorticityIsSignedAndHasNoEffectWhereItIsZero() {
        let none = MapPalette.color(sea: 0.5, zeta: 0, rain: 0, cloud: 0)
        let cyclonic = MapPalette.color(sea: 0.5, zeta: 0.8, rain: 0, cloud: 0)
        let anticyclonic = MapPalette.color(sea: 0.5, zeta: -0.8, rain: 0, cloud: 0)
        #expect(cyclonic.x > none.x + 0.05, "cyclonic is warm (more red)")
        #expect(anticyclonic.z > none.z + 0.01 && anticyclonic.x < cyclonic.x, "anticyclonic is the cool side")
        #expect(MapPalette.color(sea: 0.5, zeta: 0.0, rain: 0, cloud: 0) == none)
        // A faint vorticity (a few per cent of full scale) is a few per cent of the tint, not the invisible 0.35 zeta of v1.
        let faint = MapPalette.color(sea: 0.5, zeta: 0.3, rain: 0, cloud: 0)
        #expect(faint != none)
    }

    @Test func theHemisphereTintIsFaintAndGoneInTheFlatCloseUp() {
        #expect(MapPalette.hemisphereWeight(distance: 0.1) == 0 && MapPalette.hemisphereWeight(distance: 0.12) == 0)
        #expect(MapPalette.hemisphereWeight(distance: 0.2) == 1 && MapPalette.hemisphereWeight(distance: 1.1) == 1)
        var previous: Float = 0
        for i in 0...100 {
            let w = MapPalette.hemisphereWeight(distance: 0.12 + 0.08 * Float(i) / 100)
            #expect(w >= previous)
            previous = w
        }
        let north = MapPalette.color(sea: 0.3, zeta: 0, rain: 0, cloud: 0, northness: 1, hemisphereWeight: 1)
        let south = MapPalette.color(sea: 0.3, zeta: 0, rain: 0, cloud: 0, northness: 0, hemisphereWeight: 1)
        let off = MapPalette.color(sea: 0.3, zeta: 0, rain: 0, cloud: 0, northness: 1, hemisphereWeight: 0)
        #expect(north != south && north.z > south.z && south.y > north.y, "blue-violet in the north, teal in the south")
        #expect(simd_length(north - off) < 0.03 && simd_length(south - off) < 0.03, "3% of the way to the tint: faint")
    }

    @Test func theExposureScalesTheColourAndClamps() {
        let base = MapPalette.color(sea: 0.4, zeta: 0, rain: 0, cloud: 0)
        let bright = MapPalette.color(sea: 0.4, zeta: 0, rain: 0, cloud: 0, exposure: 1.5)
        let dim = MapPalette.color(sea: 0.4, zeta: 0, rain: 0, cloud: 0, exposure: 0.8)
        #expect(simd_length(bright - base * 1.5) < 1e-5 && simd_length(dim - base * 0.8) < 1e-5)
        let cloud = MapPalette.color(sea: 0.4, zeta: 0, rain: 0, cloud: 1, exposure: 1.5)
        #expect(cloud.max() == 1, "the exposure clamps to 1")
    }
}
