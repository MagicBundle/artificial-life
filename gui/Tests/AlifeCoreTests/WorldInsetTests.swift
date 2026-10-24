import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The minimap's geometry (DESIGN §36.5, the spec's 4.1): the map of the whole torus, north up, and the footprint of the view.
@Suite struct WorldInsetTests {
    private let L: Float = 320
    private let inset = WorldInset(rect: CGRect(x: 500, y: 400, width: 200, height: 200), size: 320)

    private func pose(x: Float, y: Float, distance: Float = 120, heading: Float = 0, tilt: Float = 0, z: Float = 0) -> WorldPose {
        WorldPose(x: x, y: y, z: z, heading: heading, tilt: tilt, distance: distance)
    }

    private func contains(_ polygon: [CGPoint], _ p: CGPoint) -> Bool {
        var inside = false
        var j = polygon.count - 1
        for i in 0..<polygon.count {
            let (a, b) = (polygon[i], polygon[j])
            if (a.y > p.y) != (b.y > p.y), p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x { inside.toggle() }
            j = i
        }
        return inside
    }

    @Test func northIsUpEastIsRightAndTheEquatorIsThroughTheMiddle() {
        let box = inset.rect
        let origin = inset.point(forTorus: 0, y: 0)
        #expect(origin.x == box.minX && origin.y == box.midY)
        #expect(inset.point(forTorus: 160, y: 0).x == box.midX)
        #expect(inset.point(forTorus: 0, y: 36).y < box.midY, "the northern hemisphere (0 < y < L/2) is above the equator")
        #expect(inset.point(forTorus: 0, y: 284).y > box.midY, "and the southern below")
        #expect(abs(inset.point(forTorus: 0, y: 159.99).y - box.minY) < 0.1, "the wrap in y is the top edge")
        #expect(abs(inset.point(forTorus: 0, y: 160.01).y - box.maxY) < 0.1, "and the bottom edge")
        #expect(abs(inset.point(forTorus: 320, y: 0).x - box.minX) < 1e-3, "x = L is x = 0")
    }

    @Test func aPointOfTheBoxAndTheWorldPointUnderItAreEachOthersInverse() {
        var generator = SystemRandomNumberGenerator()
        for _ in 0..<1000 {
            let x = Float.random(in: 0..<320, using: &generator), y = Float.random(in: 0..<320, using: &generator)
            let p = inset.point(forTorus: x, y: y)
            let back = try! #require(inset.torus(atPoint: p))
            #expect(abs(Torus.wrapDelta(back.x - x, size: L)) < 1e-2 && abs(Torus.wrapDelta(back.y - y, size: L)) < 1e-2, "(\(x), \(y)) -> (\(back.x), \(back.y))")
        }
        #expect(inset.torus(atPoint: CGPoint(x: 499, y: 500)) == nil)
        #expect(inset.torus(atPoint: CGPoint(x: 600, y: 700)) == nil)
    }

    @Test func theFootprintIsTheProjectionOfTheViewsBorderAndContainsTheCentre() {
        for p in [pose(x: 160, y: 20), pose(x: 100, y: 300, distance: 80, heading: 0.9), pose(x: 40, y: 60, distance: 60, heading: -2, tilt: 0.7),
                  pose(x: 250, y: 5, distance: 100, tilt: 1.0)] {
            var p = p
            p.clamp(aspect: 1.5, size: L, height: 80)
            let polygon = try! #require(inset.footprint(of: p, aspect: 1.5))
            #expect(polygon.count == 64)
            #expect(contains(polygon, inset.centre(of: p, aspect: 1.5)), "the footprint contains the centre of the picture")
            // The corners of the view are the first samples of the four sides: where the ray meets the floor, or, where that is beyond the
            // clear box (or no floor), the clip of its line to the box.
            let scale = Float(inset.rect.width) / L
            for (index, ndc) in [(0, SIMD2<Float>(-1, 1)), (16, SIMD2(1, 1)), (32, SIMD2(1, -1)), (48, SIMD2(-1, -1))] {
                let at = p.clippedOffset(at: ndc, aspect: 1.5, planeZ: 0, limit: 0.5 * L)
                let expected = inset.point(x: p.x + at.x, ybar: p.ybar(size: L) + at.y)
                #expect(abs(polygon[index].x - expected.x) < 1e-3 && abs(polygon[index].y - expected.y) < 1e-3)
                if let hit = p.offset(at: ndc, aspect: 1.5, planeZ: 0), WorldFog.chebyshev(hit) <= 0.5 * L {
                    #expect(simd_length(hit - at) < 1e-3, "inside the box it is the floor point: \(hit) against \(at)")
                }
                #expect(WorldFog.chebyshev(at) <= 0.5 * L * 1.0001 && scale > 0)
            }
        }
    }

    @Test func straightDownTheFootprintIsARectangleOfTheViewsWidthInTheBoxsUnits() {
        let p = pose(x: 160, y: 0, distance: 100)
        let polygon = try! #require(inset.footprint(of: p, aspect: 2))
        let xs = polygon.map(\.x), ys = polygon.map(\.y)
        let halfHeight = 100 * tan(p.fovY / 2)  // units, at the floor
        let scale = Float(inset.rect.width) / L
        #expect(abs(Float(xs.max()! - xs.min()!) - 2 * 2 * halfHeight * scale) < 0.01)
        #expect(abs(Float(ys.max()! - ys.min()!) - 2 * halfHeight * scale) < 0.01)
    }

    @Test func aFootprintAcrossASeamIsTwoPiecesAndAtACornerFour() {
        func pieces(_ p: WorldPose) -> Int {
            inset.pieces(of: try! #require(inset.footprint(of: p, aspect: 1.5))).count
        }
        #expect(pieces(pose(x: 160, y: 0, distance: 60)) == 1)
        #expect(pieces(pose(x: 5, y: 0, distance: 60)) == 2, "across x = 0")
        #expect(pieces(pose(x: 315, y: 0, distance: 60)) == 2, "across x = L")
        #expect(pieces(pose(x: 160, y: 158, distance: 60)) == 2, "across the wrap in y")
        #expect(pieces(pose(x: 5, y: 158, distance: 60)) == 4, "at a corner")
        // Each piece is the same polygon shifted by whole boxes, and the first is the one in the box.
        let polygon = try! #require(inset.footprint(of: pose(x: 5, y: 158, distance: 60), aspect: 1.5))
        let all = inset.pieces(of: polygon)
        #expect(all[0] == polygon)
        for piece in all.dropFirst() {
            let dx = (piece[0].x - polygon[0].x) / inset.rect.width, dy = (piece[0].y - polygon[0].y) / inset.rect.height
            #expect(abs(dx.rounded() - dx) < 1e-6 && abs(dy.rounded() - dy) < 1e-6 && (dx != 0 || dy != 0))
            #expect(piece.indices.allSatisfy { abs(piece[$0].x - polygon[$0].x - dx * inset.rect.width) < 1e-6 })
        }
    }

    @Test func aHorizonInViewGivesAFootprintClippedToTheClearBoxAndItIsNeverMoreThanOneBoxAcross() throws {
        for (tilt, heading, distance) in [(Float(1.5), Float(0), Float(100)), (1.2, 0.7, 120), (.pi / 2, -2, 128), (1.0, 3, 150)] as [(Float, Float, Float)] {
            var p = pose(x: 160, y: 0, distance: distance, heading: heading, tilt: tilt)
            p.fovY = 1.0
            p.clamp(aspect: 1.5, size: L, height: 80)
            let polygon = try #require(inset.footprint(of: p, aspect: 1.5))
            #expect(polygon.count == 64 && polygon.allSatisfy { $0.x.isFinite && $0.y.isFinite })
            // Within the clear box about the centre, in the box's points; and it reaches the box's boundary (the far side is clipped).
            let half = CGFloat(0.5 * inset.rect.width)
            let centre = inset.point(x: p.x, ybar: p.ybar(size: L))
            #expect(polygon.allSatisfy { abs($0.x - centre.x) <= half * 1.0001 && abs($0.y - centre.y) <= half * 1.0001 }, "tilt \(tilt)")
            #expect(polygon.contains { abs(abs($0.x - centre.x) - half) < 1e-3 || abs(abs($0.y - centre.y) - half) < 1e-3 }, "it reaches the clip")
            // So its pieces are whole and few (at most the four of a corner).
            #expect(inset.pieces(of: polygon).count <= 4)
        }
    }

    @Test func aRayThatMissesTheFloorOrMeetsItFarAwayIsClippedAlongItsOwnLineToTheBoxOfTheClearView() throws {
        // Straight down every ray meets the floor in the box.
        let down = pose(x: 160, y: 0, distance: 60)
        let near = down.clippedOffset(at: SIMD2(0.5, 0.5), aspect: 1.5, planeZ: 0, limit: 0.5 * L)
        #expect(near == down.offset(at: SIMD2(0.5, 0.5), aspect: 1.5, planeZ: 0))
        // Edge-on at 40 units up: the ray above the horizon has no floor; below it, far down the line, there is a floor beyond the box.
        var side = pose(x: 160, y: 0, distance: 100, tilt: .pi / 2)
        side.z = 40
        let sky = side.clippedOffset(at: SIMD2(0, 0.6), aspect: 1.5, planeZ: 0, limit: 0.5 * L)
        #expect(abs(WorldFog.chebyshev(sky) - 0.5 * L) < 1e-3, "on the box: \(sky)")
        #expect(abs(sky.x) < 1e-3 && sky.y > 0, "straight ahead of a view that looks along +y (heading 0): the same line")
        let faraway = side.clippedOffset(at: SIMD2(0.9, -0.01), aspect: 1.5, planeZ: 0, limit: 0.5 * L)
        #expect(WorldFog.chebyshev(faraway) <= 0.5 * L * 1.0001)
        // A ray straight up has no line: the eye's own place.
        var up = pose(x: 0, y: 0, distance: 40, tilt: .pi / 2)
        up.fovY = 3.0
        let zenith = up.clippedOffset(at: SIMD2(0, 1), aspect: 1, planeZ: 0, limit: 0.5 * L)
        #expect(zenith.x.isFinite && zenith.y.isFinite && WorldFog.chebyshev(zenith) <= 0.5 * L * 1.0001)
    }

    @Test func theFloorFootprintOfAnElevatedViewIsWiderThanThePlaneThroughItsCentre() {
        let low = pose(x: 160, y: 0, distance: 60, z: 0)
        let high = pose(x: 160, y: 0, distance: 60, z: 40)
        let width = { (p: WorldPose) -> CGFloat in
            let polygon = try! #require(self.inset.footprint(of: p, aspect: 1.5))
            return polygon.map(\.x).max()! - polygon.map(\.x).min()!
        }
        #expect(width(high) > width(low) * 1.5, "the eye is at d + z over the floor")
    }
}
