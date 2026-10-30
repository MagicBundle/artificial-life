import Foundation
import Testing
import simd

@testable import AlifeCore

/// The HEALPix lookup and the pixel geometry, against the engine's tables (`geom.ang2pix`, `pix2vec`, `pix2xyf`, `xyf2vec`).
@Suite struct HealpixTests {
    private func table(_ nside: Int) throws -> [String: Any] {
        let all = try #require(SphereFixture.expected["healpix"] as? [String: Any])
        return try #require(all["\(nside)"] as? [String: Any])
    }

    static let nsides = [2, 4, 8, 21]

    @Test func theFixtureIsThere() {
        #expect(!SphereFixture.stream.isEmpty)
        #expect(!SphereFixture.expected.isEmpty)
    }

    @Test(arguments: nsides) func theDoubleLookupIsTheEnginesForRandomDirectionsPolesAndAxes(nside: Int) throws {
        let t = try table(nside)
        let points = try #require(t["points"] as? [Any]).map { SphereFixture.vector($0) }
        let pixels = try #require(t["pixels"] as? [Int])
        #expect(points.count == pixels.count && points.count >= 300)
        var wrong: [Int] = []
        for (i, p) in points.enumerated() where Healpix.pixel(of: p, nside: nside) != pixels[i] { wrong.append(i) }
        #expect(wrong.isEmpty, "directions \(wrong.prefix(5)) differ from geom.ang2pix at Nside \(nside)")
    }

    /// The shader's function is `pixelFloat` in MSL; here the Float arithmetic against the engine: at most a neighbour away, and
    /// the same pixel except within float rounding of a border.
    @Test(arguments: nsides) func theFloatLookupIsTheEnginesUpToABorder(nside: Int) throws {
        let t = try table(nside)
        let points = try #require(t["points"] as? [Any]).map { SphereFixture.vector($0) }
        let pixels = try #require(t["pixels"] as? [Int])
        var different = 0
        for (i, p) in points.enumerated() {
            let got = Healpix.pixelFloat(of: SIMD3<Float>(Float(p.x), Float(p.y), Float(p.z)), nside: nside)
            if got != pixels[i] {
                different += 1
                // a different pixel must be one whose centre is as near as a neighbour's
                let cGot = Healpix.centre(of: got, nside: nside), cWant = Healpix.centre(of: pixels[i], nside: nside)
                let angle = acos(min(1, simd_dot(cGot, cWant)))
                #expect(angle < 2.2 * (4 * Double.pi / Double(Healpix.npix(nside))).squareRoot(), "pixel \(got) is not adjacent to \(pixels[i])")
            }
        }
        #expect(different <= 3, "\(different) of \(points.count) directions fell on another pixel in Float")
    }

    @Test(arguments: nsides) func theCentresAreTheEnginesPixelCentres(nside: Int) throws {
        let t = try table(nside)
        let sample = try #require(t["sample"] as? [Int])
        let centres = try #require(t["centres"] as? [Any]).map { SphereFixture.vector($0) }
        for (p, want) in zip(sample, centres) {
            let got = Healpix.centre(of: p, nside: nside)
            #expect(simd_length(got - want) < 1e-12, "centre of pixel \(p) at Nside \(nside)")
            #expect(Healpix.pixel(of: want, nside: nside) == p, "the centre of \(p) is in \(p)")
        }
    }

    @Test(arguments: nsides) func theFacetCoordinatesAndCornersAreTheEngines(nside: Int) throws {
        let t = try table(nside)
        let sample = try #require(t["sample"] as? [Int])
        let xyf = try #require(t["xyf"] as? [[Int]])
        for (k, p) in sample.enumerated() {
            let got = Healpix.xyf(of: p, nside: nside)
            let want: [Int] = [xyf[0][k], xyf[1][k], xyf[2][k]]
            let have: [Int] = [got.ix, got.iy, got.face]
            #expect(have == want, "pix2xyf of \(p)")
        }
        let pixels = try #require(t["corner_pixels"] as? [Int])
        let corners = try #require(t["corners"] as? [[Any]])
        for (p, want) in zip(pixels, corners) {
            let got = Healpix.corners(of: p, nside: nside)
            for (a, b) in zip(got, want) { #expect(simd_length(a - SphereFixture.vector(b)) < 1e-12, "corner of \(p)") }
        }
    }

    @Test func everyPixelIsFoundAtItsCentreAndTheAreasAreEqual() {
        for nside in [2, 3, 8] {
            for p in 0..<Healpix.npix(nside) {
                #expect(Healpix.pixel(of: Healpix.centre(of: p, nside: nside), nside: nside) == p)
            }
        }
        #expect(abs(Healpix.pixelArea(nside: 8) * Double(Healpix.npix(8)) - 4 * .pi) < 1e-12)
    }

    @Test func aNonUnitVectorAndAZeroVectorAreHandled() {
        #expect(Healpix.pixel(of: SIMD3(0, 0, 5), nside: 4) == Healpix.pixel(of: SIMD3(0, 0, 1), nside: 4))
        #expect(Healpix.pixel(of: SIMD3(0, 0, 0), nside: 4) == 0)
    }
}
