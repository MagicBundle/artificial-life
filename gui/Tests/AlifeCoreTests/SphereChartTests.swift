import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The Mollweide chart of the whole planet (the overview and the minimap) and the colours it paints.
@Suite struct SphereChartTests {
    @Test func theProjectionOfKnownPoints() {
        #expect(Mollweide.project(lon: 0, lat: 0) == SIMD2(0, 0))
        let east = Mollweide.project(lon: .pi, lat: 0)
        #expect(abs(east.x - 1) < 1e-12 && abs(east.y) < 1e-12, "the equator's end is the chart's edge")
        let north = Mollweide.project(lon: 0, lat: .pi / 2)
        #expect(abs(north.x) < 1e-12 && abs(north.y - 0.5) < 1e-9, "the pole is the top of the ellipse")
        let south = Mollweide.project(lon: 1, lat: -.pi / 2)
        #expect(abs(south.x) < 1e-6 && abs(south.y + 0.5) < 1e-9)
        // recentred: the new centre longitude maps to x = 0
        #expect(abs(Mollweide.project(lon: 1.2, lat: 0.4, centre: 1.2).x) < 1e-12)
    }

    @Test(arguments: [(-80.0, -170.0), (-45.0, 33.0), (0.0, 179.0), (12.0, -179.5), (60.0, 0.0), (89.0, 90.0)])
    func theProjectionInvertsExactly(lat: Double, lon: Double) throws {
        for centre in [0.0, 1.0, -2.5] {
            let p = Mollweide.project(lon: lon * .pi / 180, lat: lat * .pi / 180, centre: centre)
            let back = try #require(Mollweide.invert(x: p.x, y: p.y, centre: centre))
            var dl = back.lon - lon * .pi / 180
            dl -= 2 * .pi * (dl / (2 * .pi)).rounded()
            #expect(abs(dl) < 1e-6 && abs(back.lat - lat * .pi / 180) < 1e-6, "lat \(lat) lon \(lon) centre \(centre)")
        }
    }

    @Test func outsideTheEllipseThereIsNothing() {
        #expect(Mollweide.invert(x: 0.99, y: 0.45) == nil)
        #expect(Mollweide.invert(x: 0, y: 0.51) == nil)
        #expect(Mollweide.direction(x: -1, y: 0.5) == nil)
        #expect(Mollweide.direction(x: 0, y: 0) != nil)
    }

    /// Equal area: a region of the chart has the area of the sphere's region it shows, up to a constant, so the pixels of a cell image
    /// divide the sphere in equal parts. The share of chart pixels that fall in each HEALPix cell is the same for every cell.
    @Test func theChartIsEqualArea() {
        let nside = 4
        let width = 720, height = 360
        let cells = Healpix.npix(nside)
        var count = [Int](repeating: 0, count: cells)
        var total = 0
        for py in 0..<height {
            let y = 0.5 - (Double(py) + 0.5) / Double(height)
            for px in 0..<width {
                let x = (Double(px) + 0.5) / Double(width) * 2 - 1
                guard let d = Mollweide.direction(x: x, y: y) else { continue }
                count[Healpix.pixelFloat(of: d, nside: nside)] += 1
                total += 1
            }
        }
        let mean = Double(total) / Double(cells)
        let worst = count.map { abs(Double($0) - mean) / mean }.max() ?? 1
        #expect(worst < 0.2, "every cell has the same share of the chart (worst \(worst))")
        // the ellipse of semi-axes 1 and 1/2 has the area pi/2 of a rectangle of area 2
        #expect(abs(Double(total) / Double(width * height) - .pi / 4) < 0.01)
    }

    @Test func theImageHoldsEachCellsColourAtItsCentre() {
        let nside = 3
        let colours = (0..<Healpix.npix(nside)).map { UInt32(0xFF000000) | UInt32($0) }
        let image = Mollweide.image(width: 200, height: 100, centre: 0, nside: nside, colours: colours)
        #expect(image[0] == 0 && image[199] == 0 && image[99 * 200] == 0, "the corners are clear")
        for p in [0, 7, 20, 50, 100, 107] {
            let c = Healpix.centre(of: p, nside: nside)
            let at = Mollweide.point(SIMD3(Float(c.x), Float(c.y), Float(c.z)), centre: 0, in: CGSize(width: 200, height: 100))
            let x = min(max(Int(at.x), 0), 199), y = min(max(Int(at.y), 0), 99)
            #expect(image[y * 200 + x] == colours[p], "the pixel at the centre of cell \(p)")
        }
        // north is up: a cell near the north pole is in the top rows
        let north = Mollweide.point(SIMD3(0.01, 0, 0.9999), centre: 0, in: CGSize(width: 200, height: 100))
        #expect(north.y < 5)
    }

    @Test func aRectangleOfTheChartMapsBackToTheSameDirection() {
        let size = CGSize(width: 220, height: 110)
        for d in [SIMD3<Float>(0.3, -0.4, 0.5), SIMD3(-0.9, 0.1, -0.3), SIMD3(0.1, 0.99, 0.05)].map({ simd_normalize($0) }) {
            let at = Mollweide.point(d, centre: 0.7, in: size)
            let back = Mollweide.direction(at: at, centre: 0.7, in: size)
            #expect(back != nil && SphereFrame.angle(back!, d) < 1e-4)
        }
    }
}
