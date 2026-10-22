import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The overlay's world-to-view projection: the camera on the free rectangle, at the minimum image of the target.
@Suite struct WorldProjectorTests {
    private let L: Float = 320
    private let view = CGSize(width: 900, height: 700)

    private func projector(target: SIMD3<Float>? = nil, distance: Float? = nil) -> WorldProjector {
        var camera = Camera.planet(size: L, aspect: Float(view.width / view.height))
        if let target { camera.target = target }
        if let distance { camera.distance = distance }
        return WorldProjector(camera: camera, size: L, view: view)
    }

    @Test func theTargetIsAtTheCentreOfTheView() throws {
        let p = projector()
        let at = try #require(p.screen(p.camera.target))
        #expect(abs(at.x - 450) < 1e-3 && abs(at.y - 350) < 1e-3)
    }

    @Test func thePlanetReadsLikeAMapXToTheRightYUp() throws {
        let p = projector()
        let origin = try #require(p.screen(SIMD3(0, 0, 0)))
        let east = try #require(p.screen(SIMD3(L / 2, 0, 0)))
        let north = try #require(p.screen(SIMD3(0, L / 2, 0)))
        #expect(east.x > origin.x && abs(east.y - origin.y) < 1e-2, "x grows to the right")
        #expect(north.y < origin.y && abs(north.x - origin.x) < 1e-2, "y grows upwards (smaller screen y)")
    }

    @Test func theWholeWorldIsInViewAtHome() throws {
        let p = projector()
        for corner in [SIMD3<Float>(0, 0, 0), SIMD3(L, 0, 0), SIMD3(0, L, 0), SIMD3(L, L, 0)] {
            let at = try #require(p.screen(corner))
            #expect(at.x >= 0 && at.x <= view.width && at.y >= 0 && at.y <= view.height, "\(corner) -> \(at)")
        }
    }

    @Test func aPointAcrossTheSeamIsDrawnNextToTheTarget() throws {
        // Target just inside the near edge; a storm just inside the far edge is two units to the left of it, not 318 to the right.
        let p = projector(target: SIMD3(1, 160, 0), distance: 60)
        let near = try #require(p.screen(SIMD3(1, 160, 0)))
        let across = try #require(p.screen(SIMD3(319, 160, 0)))
        #expect(across.x < near.x && near.x - across.x < 40, "\(across) against \(near)")
        let both = try #require(p.screen(SIMD3(-1, 160, 0)))
        #expect(abs(across.x - both.x) < 1e-2 && abs(across.y - both.y) < 1e-2, "the same point is the same image")
    }

    @Test func aGlyphsRadiusIsThePointsOfItsUnits() throws {
        let p = projector(distance: 80)
        let units: Float = 10
        let r = try #require(p.points(ofUnits: units, at: SIMD3(160, 160, 0)))
        // 2·d·tan(fov/2) units fill the view's height in the smaller dimension.
        let perUnit = CGFloat(view.height) / CGFloat(2 * 80 * tan(Camera.defaultFovY / 2))
        #expect(abs(r - perUnit * CGFloat(units)) < 0.5, "\(r) against \(perUnit * CGFloat(units))")
    }

    @Test func theLengthOfAUnitDoesNotDependOnWhereTheWrapHappens() throws {
        // A point half a world from the target is at the edge of the minimum image: its far end, one more unit on, would wrap
        // to the other side if it were wrapped again. The length is the same wherever in the view the point is.
        let p = projector(target: SIMD3(0.5, 160, 0), distance: 80)
        let near = try #require(p.points(ofUnits: 12, at: SIMD3(0.5, 160, 0)))
        for x: Float in [100, 150, 159, 160.5, 161] {  // up to and past the half world from the target at 0.5
            let r = try #require(p.points(ofUnits: 12, at: SIMD3(x, 160, 0)))
            #expect(abs(r - near) < 0.5, "x = \(x): \(r) against \(near)")
        }
    }

    // MARK: Storms

    private func storm(id: Int, x: Double, y: Double, r: Double = 6, track: [[Double]] = [], named: Bool = true,
                       state: String = "active") -> PlayStorms.Storm {
        PlayStorms.Storm(id: id, name: named ? "Merrow" : nil, named: named, hemi: 0, x: x, y: y, r: r, ext: [r, r],
                         zeta: 0.75, zetaNow: 0.6, wind: 0.06, rain: 0.002, age: 100, parent: 0, state: state,
                         track: track, pred: [])
    }

    @Test func aStormIsAGlyphWithItsRingAndName() throws {
        let p = projector(distance: 100)
        let record = PlayStorms(detectStep: 100, storms: [storm(id: 7, x: 170, y: 150, r: 12)], ended: [])
        let glyphs = p.glyphs(of: record)
        #expect(glyphs.count == 1)
        let g = try #require(glyphs.first)
        #expect(g.id == 7 && g.named && g.name == "Merrow" && !g.coasting)
        let at = try #require(p.screen(SIMD3(170, 150, 0)))
        #expect(g.center == at)
        let perUnit = CGFloat(view.height) / CGFloat(2 * 100 * tan(Camera.defaultFovY / 2))
        #expect(abs(g.radius - 12 * perUnit) < 0.5)
    }

    @Test func aTinyStormStillHasARingYouCanSee() throws {
        let p = projector(distance: 300)
        let g = try #require(p.glyphs(of: PlayStorms(detectStep: 0, storms: [storm(id: 1, x: 160, y: 160, r: 0.5, named: false)], ended: [])).first)
        #expect(g.radius == 5 && g.name == nil)
    }

    @Test func aCoastingStormSaysSo() throws {
        let p = projector()
        let g = try #require(p.glyphs(of: PlayStorms(detectStep: 0, storms: [storm(id: 1, x: 100, y: 100, state: "coasting")], ended: [])).first)
        #expect(g.coasting)
    }

    @Test func aTrackIsOneContinuousLineAcrossTheSeam() throws {
        // The storm has drifted east across the seam: its track is unwrapped (x runs past L) and its position is wrapped.
        let p = projector(target: SIMD3(2, 160, 0), distance: 60)
        let track: [[Double]] = [[0, 316, 160], [25, 318, 160], [50, 320.5, 160], [75, 322, 161]]
        let s = storm(id: 3, x: 2, y: 161, track: track)
        let points = p.track(s)
        #expect(points.count == 4)
        // Continuous: each step is a few units on screen, not a jump across the world.
        for (a, b) in zip(points, points.dropFirst()) { #expect(hypot(b.x - a.x, b.y - a.y) < 80, "\(a) -> \(b)") }
        // It ends at the storm.
        let end = try #require(points.last)
        let center = try #require(p.screen(SIMD3(2, 161, 0)))
        #expect(hypot(end.x - center.x, end.y - center.y) < 1e-2)
        #expect(points[0].x < end.x, "it came from the west")
    }

    @Test func aStormBehindTheEyeIsLeftOut() {
        var camera = Camera(target: SIMD3(160, 160, 0), distance: 10, yaw: .pi / 2, pitch: 0)
        camera.scene = Camera.tiledScene(size: L, height: 0)
        let p = WorldProjector(camera: camera, size: L, view: view)
        // Looking along +y from the south: a storm directly south of the eye is behind it (nearest image of 100 is 100).
        let behind = PlayStorms(detectStep: 0, storms: [storm(id: 1, x: 160, y: 140)], ended: [])
        #expect(p.glyphs(of: behind).isEmpty)
    }

    @Test func theLevelIsTheCamerasOnThisRectangle() {
        #expect(projector().level == .planet)
        #expect(projector(distance: 10).level == .organism)
        #expect(projector(distance: 90).level == .region)
    }

    // MARK: Territory cells

    @Test func aCellsCornersAreTheProjectedCorners() throws {
        // 32 cells of 10 units over L = 320: cell (3, 5) spans x 30 to 40 and y 50 to 60.
        let p = projector(target: SIMD3(35, 55, 0), distance: 100)
        let quad = try #require(p.cellCorners(cx: 3, cy: 5, nx: 32, ny: 32))
        #expect(quad.count == 4)
        let expected = [SIMD3<Float>(30, 50, 0), SIMD3(40, 50, 0), SIMD3(40, 60, 0), SIMD3(30, 60, 0)]
        for (corner, world) in zip(quad, expected) {
            let at = try #require(p.screen(world))
            #expect(abs(corner.x - at.x) < 1e-2 && abs(corner.y - at.y) < 1e-2)
        }
        // South-west is lower left in the planet pose (y up): the quad is drawn counter-clockwise on the screen.
        #expect(quad[0].x < quad[1].x && quad[0].y > quad[3].y)
    }

    @Test func aCellAcrossTheSeamIsDrawnNextToTheTargetAndItsCopiesAWorldAway() throws {
        let p = projector(target: SIMD3(2, 160, 0), distance: 100)
        let wrapped = try #require(p.cellCorners(cx: 31, cy: 16, nx: 32, ny: 32))  // x 310 to 320, a few units west of 2
        let direct = try #require(p.screenOfImage(SIMD3(-5, 165, 0)))
        let centre = CGPoint(x: wrapped.map(\.x).reduce(0, +) / 4, y: wrapped.map(\.y).reduce(0, +) / 4)
        #expect(abs(centre.x - direct.x) < 1e-2 && abs(centre.y - direct.y) < 1e-2)
        let east = try #require(p.cellCorners(cx: 31, cy: 16, nx: 32, ny: 32, shift: SIMD2(1, 0)))
        let eastCentre = CGPoint(x: east.map(\.x).reduce(0, +) / 4, y: east.map(\.y).reduce(0, +) / 4)
        let oneWorld = try #require(p.screenOfImage(SIMD3(-5 + 320, 165, 0)))
        #expect(abs(eastCentre.x - oneWorld.x) < 1e-2)
        #expect(p.cellCorners(cx: 0, cy: 0, nx: 0, ny: 32) == nil)
    }
}
