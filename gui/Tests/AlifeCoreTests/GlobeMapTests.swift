import Foundation
import Testing
import simd

@testable import AlifeCore

/// The mapping of the torus onto the globe (DESIGN §36.10; spec 2.1 and 7.1 `GlobeMapTests`).
@Suite struct GlobeMapTests {
    private let L: Float = 320
    private let H: Float = 80

    /// A deterministic generator (SplitMix64) so a failure reproduces.
    private struct Rng {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        mutating func next() -> Float {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return Float((z ^ (z >> 31)) >> 40) / Float(1 << 24)
        }
    }

    @Test func theRoundTripHoldsToAThousandthOfAUnitBelow85Degrees() {
        var rng = Rng()
        // |phi| < 85 degrees is |ybar| < 159.5; z over the whole height.
        var worst = SIMD3<Float>.zero
        var worstDouble = 0.0
        for _ in 0..<20_000 {
            let x = rng.next() * L, ybar = (rng.next() * 2 - 1) * 159.4, z = rng.next() * H
            let y = Torus.wrap(ybar, size: L)
            let t = GlobeMap.torus(from: GlobeMap.point(x: x, y: y, z: z, L: L, height: H), L: L, height: H)
            #expect(!t.seam)
            worst = simd_max(worst, SIMD3(abs(Torus.wrapDelta(t.x - x, size: L)), abs(Torus.wrapDelta(t.y - y, size: L)), abs(t.z - z)))
            let pd = GlobeMap.pointDouble(x: Double(x), y: Double(y), z: Double(z), L: Double(L), height: Double(H))
            let td = GlobeMap.torusDouble(from: pd, L: Double(L), height: Double(H))
            var dx = abs(td.x - Double(x)), dy = abs(td.y - Double(y))
            dx = min(dx, Double(L) - dx)
            dy = min(dy, Double(L) - dy)
            worstDouble = max(worstDouble, dx, dy, abs(td.z - Double(z)))
        }
        #expect(worst.x < 1e-3 && worst.y < 1e-3 && worst.z < 1e-3, "worst \(worst)")
        #expect(worstDouble < 1e-9, "Double path worst \(worstDouble)")
    }

    @Test func columnZeroAndColumnLAreTheSamePoint() {
        for y: Float in [10, 120, 200, 300] {
            let a = GlobeMap.point(x: 0, y: y, z: 20, L: L, height: H), b = GlobeMap.point(x: L, y: y, z: 20, L: L, height: H)
            #expect(simd_length(a - b) < 1e-3)
        }
    }

    @Test func theLatitudeIsOddIncreasingAndEndsAtPhiMax() {
        var last = -Float.infinity
        for i in stride(from: -159.5 as Float, through: 159.5, by: 0.5) {
            let phi = GlobeMap.latitude(ybar: i, L: L)
            #expect(phi > last, "strictly increasing at \(i)")
            last = phi
            #expect(abs(phi + GlobeMap.latitude(ybar: -i, L: L)) < 1e-6, "odd")
            #expect(GlobeMap.latitude(y: Torus.wrap(i, size: L), L: L) == phi, "the row y and the signed row agree")
        }
        #expect(abs(GlobeMap.latitude(ybar: L / 2, L: L) - GlobeMap.phiMax) < 1e-6)
        #expect(abs(GlobeMap.latitude(ybar: -L / 2, L: L) + GlobeMap.phiMax) < 1e-6)
        #expect(abs(GlobeMap.phiMax - Float(atan(sinh(Double.pi)))) < 1e-6, "phiMax is gd(pi)")
        #expect(GlobeMap.y(latitude: GlobeMap.phiMax, L: L) == L / 2 && GlobeMap.y(latitude: -1.5, L: L) == L / 2, "the seam row for both poles")
    }

    @Test func theMapIsConformal() {
        var rng = Rng()
        for _ in 0..<200 {
            let x = Double(rng.next() * L), ybar = Double((rng.next() * 2 - 1) * 150)
            let h = 1e-3
            func p(_ x: Double, _ yb: Double) -> SIMD3<Double> {
                GlobeMap.pointDouble(x: x, y: yb, z: 0, L: Double(L), height: Double(H))
            }
            let dx = simd_length(p(x + h, ybar) - p(x - h, ybar)) / (2 * h)
            let dy = simd_length(p(x, ybar + h) - p(x, ybar - h)) / (2 * h)
            #expect(abs(dx - dy) / dx < 1e-3, "equal length east and north")
            let m = Double(GlobeMap.scale(latitude: GlobeMap.latitude(ybar: Float(ybar), L: L), L: L))
            #expect(abs(dx - m) / m < 1e-3, "and equal to scale(latitude)")
        }
        #expect(abs(GlobeMap.scale(latitude: 0, L: L) - Morph.M) < 1e-5, "2.83 sphere units per torus unit at the equator")
    }

    @Test func theHemisphereIsTheSignOfTheCoriolisParameterAndOfTheLatitude() {
        var checked = 0
        for i in 1..<1000 {
            let y = Float(i) * L / 1000
            guard abs(y - L / 2) > 1e-3 else { continue }
            let hemi = GlobeMap.hemisphere(y: y, L: L)
            #expect((hemi == 0) == (sin(2 * .pi * y / L) > 0), "the sign of f0 sin(2 pi y / L) at \(y)")
            #expect((hemi == 0) == (GlobeMap.latitude(y: y, L: L) > 0), "northern is phi > 0")
            checked += 1
        }
        #expect(checked > 990)
        #expect(GlobeMap.hemisphere(y: 36, L: L) == 0 && GlobeMap.hemisphere(y: 284, L: L) == 1, "the storm pools")
    }

    @Test func theLidIsAShellAndTheCapIsFiveDegrees() {
        let R = GlobeMap.radius(L: L)
        #expect(R == 144)
        for z in [0, 25, 80] as [Float] {
            for (x, y) in [(10, 5), (200, 90), (310, 250)] as [(Float, Float)] {
                let r = simd_length(GlobeMap.point(x: x, y: y, z: z, L: L, height: H))
                #expect(abs(r - (R + GlobeMap.shellScale(L: L, height: H) * z)) < 1e-3)
            }
        }
        #expect(abs(GlobeMap.shellScale(L: L, height: H) * H - 0.06 * R) < 1e-4, "the lid is 6% of R")
        let cap = (Float.pi / 2 - GlobeMap.phiMax) * 180 / .pi
        #expect(abs(cap - 4.95) < 0.01, "the cap's angular radius, \(cap) degrees")
    }

    @Test func aPointInsideACapIsTheSeamRow() {
        let R = GlobeMap.radius(L: L)
        let p = SIMD3<Float>(0.001, 0.001, R)  // the pole
        let t = GlobeMap.torus(from: p, L: L, height: H)
        #expect(t.seam && t.y == L / 2)
        let south = GlobeMap.torus(from: SIMD3(0.001, 0.001, -R), L: L, height: H)
        #expect(south.seam && south.y == L / 2, "both poles are the one seam row")
        #expect(!GlobeMap.torus(from: R * GlobeMap.direction(x: 5, y: 150, L: L), L: L, height: H).seam)
    }
}
