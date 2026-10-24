import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The pose of the World window's view (DESIGN §36.5, spec 7.1 `WorldViewGestureTests` restricted to the plane): the camera it
/// derives, the clamps that keep the picture within one world, and the gestures, exact on the plane.
@Suite struct WorldPoseTests {
    private let L: Float = 320
    private let H: Float = 80
    private let view = CGSize(width: 1100, height: 800)
    private var aspect: Float { Float(view.width / view.height) }

    private func screen(_ pose: WorldPose, _ point: SIMD2<Float>, z: Float = 0) -> CGPoint? {
        WorldProjector(camera: pose.camera(size: L, height: H), size: L, view: view).screen(SIMD3(point.x, point.y, z))
    }

    /// A pose somewhere in the zoom range, with a heading, a tilt and a height, already clamped.
    private func random(_ g: inout SeededGenerator, tilted: Bool = true) -> WorldPose {
        var pose = WorldPose(x: .random(in: 0..<L, using: &g), y: .random(in: 0..<L, using: &g), z: .random(in: 0...H, using: &g),
                             heading: .random(in: -.pi..<(.pi), using: &g), tilt: tilted ? .random(in: 0..<1.2, using: &g) : 0,
                             distance: .random(in: 8...200, using: &g))
        pose.clamp(aspect: aspect, size: L, height: H)
        return pose
    }

    // MARK: The camera

    @Test func theCameraLooksStraightDownWithNorthUpAndTheDistanceAndHeadingAreItsYawAndPitch() {
        let pose = WorldPose(x: 100, y: 40, z: 6, distance: 77)
        let camera = pose.camera(size: L, height: H)
        #expect(camera.target == SIMD3(100, 40, 6) && camera.distance == 77)
        #expect(camera.yaw == .pi / 2 && camera.pitch == -Camera.maxPitch, "v1's planet pose")
        #expect(simd_length(camera.right - SIMD3(1, 0, 0)) < 1e-6 && simd_length(camera.up - SIMD3(0, 1, 0)) < 1e-6, "x right, y up")
        var turned = pose
        turned.heading = 0.5
        turned.tilt = 0.3
        let c = turned.camera(size: L, height: H)
        #expect(abs(c.yaw - (.pi / 2 + 0.5)) < 1e-6 && abs(c.pitch - (-.pi / 2 + 0.3)) < 1e-6)
    }

    @Test func applyingTheCameraBackGivesThePose() {
        var g = SeededGenerator(seed: 11)
        for _ in 0..<200 {
            let pose = random(&g)
            var back = WorldPose(x: 0, y: 0, distance: 1)
            back.apply(pose.camera(size: L, height: H))
            #expect(abs(back.x - pose.x) < 1e-4 && abs(back.y - pose.y) < 1e-4 && abs(back.z - pose.z) < 1e-4)
            #expect(abs(back.heading - pose.heading) < 1e-4 && abs(back.tilt - pose.tilt) < 1e-4 && abs(back.distance - pose.distance) < 1e-4)
        }
        var down = WorldPose(x: 5, y: 5, distance: 50, faceDown: true)
        down.tilt = 0.4
        var again = WorldPose(x: 0, y: 0, distance: 1, faceDown: true)
        again.apply(down.camera(size: L, height: 0))
        #expect(abs(again.tilt - 0.4) < 1e-5, "a 2D world is seen from below with the tilt measured from there")
    }

    @Test func theEquatorIsYZeroAndTheNorthernHemisphereIsTheEnginesHemisphereZero() {
        let pose = WorldPose(x: 0, y: 0, distance: 10)
        #expect(pose.ybar(size: L) == 0)
        for (y, ybar, hemi) in [(Float(40), Float(40), 0), (159, 159, 0), (161, -159, 1), (280, -40, 1)] {
            let p = WorldPose(x: 0, y: y, distance: 10)
            #expect(abs(p.ybar(size: L) - ybar) < 1e-4, "y \(y)")
            #expect(p.hemisphere(size: L) == hemi && WorldPose.hemisphere(y: y, size: L) == hemi)
        }
        #expect(WorldPose.hemisphere(y: 0, size: L) == 1 && WorldPose.hemisphere(y: L / 2, size: L) == 1, "the lines themselves are in neither")
    }

    // MARK: The footprint and the clamps

    @Test(arguments: [Float(0.4), 1, 1.38, 1.78, 2.4, 4])
    func theFarthestViewReachesExactlyOneWorldLessAMarginAtEveryHeading(aspect: Float) throws {
        for heading: Float in [0, 0.3, .pi / 4, 1.0, .pi / 2, 2.5, -2.0] {
            for z: Float in [0, 30] {
                var pose = WorldPose(x: 100, y: 100, z: z, heading: heading, distance: 1)
                pose.distance = pose.distanceLimit(size: L, aspect: aspect)
                let extent = try #require(pose.footprint(aspect: aspect, size: L, height: H))
                let reach = WorldPose.reach * L
                #expect(max(extent.x, extent.y) <= reach * 1.0005 && max(extent.x, extent.y) >= reach * 0.9995,
                        "aspect \(aspect) heading \(heading) z \(z): \(extent) against \(reach)")
                #expect(pose.fits(aspect: aspect, size: L, height: H))
                pose.distance *= 1.01
                #expect(!pose.fits(aspect: aspect, size: L, height: H), "and a hair farther does not")
            }
        }
    }

    @Test func aHeadingShrinksTheFarthestViewOfAWideWindowAndNeverBelowTheFewRadii() {
        let straight = WorldPose.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: 1.38)
        let turned = WorldPose.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: 1.38, heading: .pi / 4)
        #expect(turned < straight)
        #expect(abs(WorldPose.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: 1.38, heading: .pi / 2)
                    - WorldPose.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: 1.38, heading: 0) * 1.0) < 1e-2 * straight + 1,
                "a quarter turn swaps the axes: the larger half extent is the width at 0 and the height at 90 degrees, both a(d t)")
        #expect(WorldPose.maxDistance(size: 1, fovY: Camera.defaultFovY, aspect: 10, z: 500) == WorldLevels.minDistance)
    }

    @Test func theHorizonInViewHasNoFootprintAndTheTiltIsNotEasedBackButTheDistanceIsHeldByTheEyeLimit() throws {
        var pose = WorldPose(x: 160, y: 160, heading: 0.4, tilt: 1.4, distance: 100)
        #expect(pose.footprint(aspect: aspect, size: L, height: H) == nil, "the horizon is in view")
        // (A1 eased the tilt back until the picture fitted one world; the haze replaced that: the tilt is the player's.)
        pose.clamp(aspect: aspect, size: L, height: H)
        #expect(pose.tilt == 1.4, "the tilt stays")
        let limit = WorldFog.eyeReach * L / sin(1.4)
        #expect(pose.distance == min(100, limit), "the eye is held over the clear box: \(pose.distance)")
        // The tilt is never negative or past edge-on, and edge-on is allowed.
        var near = WorldPose(x: 160, y: 160, tilt: 3, distance: 20)
        near.clamp(aspect: aspect, size: L, height: H)
        #expect(near.tilt == WorldPose.maxTilt && WorldPose.maxTilt == .pi / 2)
        near.tilt = -1
        near.clamp(aspect: aspect, size: L, height: H)
        #expect(near.tilt == 0)
    }

    @Test func theEyeStaysOverTheClearBoxAtEveryTiltAndHeadingAndTheTwoLimitsMeetWithoutAStep() {
        for aspectRatio: Float in [0.8, 1.38, 3] {
            for heading: Float in [0, 0.4, .pi / 4, 1.2, 2.5] {
                var previous = Float.infinity
                for degrees in stride(from: Float(0), through: 90, by: 1) {
                    let tilt = degrees * .pi / 180
                    let limit = WorldPose.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: aspectRatio, heading: heading, tilt: tilt)
                    // The eye's horizontal offset from the point looked at is within the clear box's eye reach.
                    #expect(limit * sin(tilt) <= WorldFog.eyeReach * L * 1.0001 || limit == WorldLevels.minDistance, "\(degrees) degrees: \(limit)")
                    // Never farther than the straight-down limit, and never nearer as the tilt falls.
                    #expect(limit <= WorldPose.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: aspectRatio, heading: heading) + 1e-3)
                    #expect(limit <= previous + 1e-3, "monotone in the tilt: \(degrees) degrees \(limit) after \(previous)")
                    #expect(previous == .infinity || previous - limit < 0.25 * L, "no step: \(degrees) degrees \(limit) after \(previous)")
                    previous = limit
                }
            }
        }
    }

    @Test func theCameraIsWellDefinedAtEveryTiltIncludingEdgeOn() {
        for degrees in stride(from: Float(0), through: 90, by: 5) {
            let pose = WorldPose(x: 100, y: 50, z: 40, heading: 0.3, tilt: degrees * .pi / 180, distance: 100)
            let camera = pose.camera(size: L, height: H)
            let matrix = camera.viewProjectionMatrix(aspect: aspect)
            for column in 0..<4 { for row in 0..<4 { #expect(matrix[column][row].isFinite, "\(degrees) degrees") } }
            #expect(simd_length(camera.forward) > 0.999 && simd_length(camera.up) > 0.999 && abs(simd_dot(camera.forward, camera.up)) < 1e-4)
            // The eye is `distance` from the target, `distance * cos(tilt)` higher (the height of the point looked at under it).
            #expect(abs(camera.position.z - (pose.z + pose.distance * cos(pose.tilt))) < 1e-3)
            #expect(pose.tiltDegrees == Int(degrees))
        }
        let side = WorldPose(x: 0, y: 0, z: 40, tilt: .pi / 2, distance: 100).camera(size: L, height: H)
        #expect(abs(side.forward.z) < 1e-6 && abs(side.position.z - 40) < 1e-4, "edge-on: the view axis is along the floor, the eye at the target's height")
    }

    @Test func theBearingIsTheCompassDirectionTheViewLooksTowards() {
        func bearing(_ degrees: Float) -> Int { WorldPose(x: 0, y: 0, heading: degrees * .pi / 180, distance: 10).bearingDegrees }
        #expect(bearing(0) == 0, "north up looks north")
        #expect(bearing(90) == 270 && bearing(-90) == 90, "turned a quarter towards the west it looks west, 270")
        #expect(bearing(180) == 180 && bearing(-180) == 180)
        #expect(bearing(10) == 350 && bearing(-45) == 45)
        // The camera really looks that way: forward's horizontal part is (sin b, cos b) for the bearing b.
        for degrees: Float in [0, 30, 90, 135, -80, 179] {
            let pose = WorldPose(x: 0, y: 0, heading: degrees * .pi / 180, tilt: 1, distance: 10)
            let forward = pose.camera(size: L, height: H).forward
            let b = Float(pose.bearingDegrees) * .pi / 180
            let h = simd_normalize(SIMD2(forward.x, forward.y))
            #expect(abs(h.x - sin(b)) < 0.02 && abs(h.y - cos(b)) < 0.02, "heading \(degrees): \(h) for bearing \(pose.bearingDegrees)")
        }
    }

    @Test func theClampsWrapTheCentreBoundTheHeightAndTheDistanceAndSurviveNaN() {
        var pose = WorldPose(x: -5, y: 700, z: 500, heading: 9, tilt: 0, distance: 1e9)
        pose.clamp(aspect: aspect, size: L, height: H)
        #expect(pose.x == 315 && abs(pose.y - 60) < 1e-3 && pose.z == H)
        #expect(pose.heading >= -.pi && pose.heading < .pi && abs(pose.heading - (9 - 2 * .pi)) < 1e-5)
        #expect(pose.distance == pose.distanceLimit(size: L, aspect: aspect))
        pose.distance = 0.001
        pose.clamp(aspect: aspect, size: L, height: H)
        #expect(pose.distance == WorldLevels.minDistance)
        pose = WorldPose(x: .nan, y: .infinity, z: .nan, heading: .nan, tilt: .nan, distance: .nan)
        pose.clamp(aspect: aspect, size: L, height: H)
        #expect(pose.x.isFinite && pose.y.isFinite && pose.z.isFinite && pose.heading.isFinite && pose.distance.isFinite)
    }

    // MARK: Grab, pan and zoom, exact on the plane

    @Test func aGrabPutsThePointThatWasUnderThePointerBackUnderItAnywhere() throws {
        var g = SeededGenerator(seed: 5)
        var checked = 0
        defer { #expect(checked > 150, "the cases are reached: \(checked)") }
        for trial in 0..<400 {
            var pose = random(&g, tilted: trial % 4 != 0)
            if trial % 5 == 0 { pose.x = Float.random(in: -3...3, using: &g); pose.x = Torus.wrap(pose.x, size: L) }  // across the x seam
            if trial % 7 == 0 { pose.y = L / 2 + Float.random(in: -3...3, using: &g) }  // across the wrap in y
            let start = CGPoint(x: .random(in: 0...1100, using: &g), y: .random(in: 0...800, using: &g))
            let end = CGPoint(x: .random(in: 0...1100, using: &g), y: .random(in: 0...800, using: &g))
            guard let anchor = pose.point(at: WorldPose.ndc(of: start, in: view), aspect: aspect, size: L) else { continue }  // the horizon
            var moved = pose
            guard moved.grab(anchor: anchor, at: WorldPose.ndc(of: end, in: view), aspect: aspect, size: L) else { continue }
            // (the projector places a point at its minimum image about the centre: inside the clear box, half a world, that is the point)
            guard let atEnd = moved.offset(at: WorldPose.ndc(of: end, in: view), aspect: aspect), WorldFog.chebyshev(atEnd) < 0.5 * L,
                  let atStart = pose.offset(at: WorldPose.ndc(of: start, in: view), aspect: aspect), WorldFog.chebyshev(atStart) < 0.5 * L
            else { continue }
            checked += 1
            let at = try #require(screen(moved, anchor, z: moved.z))
            #expect(hypot(at.x - end.x, at.y - end.y) < 0.5, "trial \(trial): \(at) for the pointer at \(end)")
            #expect(moved.distance == pose.distance && moved.heading == pose.heading && moved.tilt == pose.tilt && moved.z == pose.z)
            #expect(moved.x >= 0 && moved.x < L && moved.y >= 0 && moved.y < L)
        }
    }

    @Test func aPanMovesTheMapByTheFingersExactly() throws {
        var g = SeededGenerator(seed: 6)
        for _ in 0..<200 {
            var pose = random(&g)
            pose.tilt = min(pose.tilt, 0.6)
            let centre = SIMD2(pose.x, pose.y)
            let before = try #require(screen(pose, centre, z: pose.z))
            let (dx, dy) = (Float.random(in: -300...300, using: &g), Float.random(in: -250...250, using: &g))
            pose.pan(dx: dx, dy: dy, view: view, aspect: aspect, size: L)
            let after = try #require(screen(pose, centre, z: pose.z))
            #expect(abs(Float(after.x - before.x) - dx) < 0.5 && abs(Float(after.y - before.y) - dy) < 0.5)
        }
        var still = WorldPose(x: 1, y: 2, distance: 30)
        still.pan(dx: 5, dy: 5, view: .zero, aspect: 1, size: L)
        #expect(still.x == 1 && still.y == 2, "no view, no pan")
    }

    @Test(arguments: [Float(0.5), 2, 0.9, 1.3])
    func aZoomKeepsThePointUnderThePointerWhereItIsAndScalesTheDistance(factor: Float) throws {
        var g = SeededGenerator(seed: UInt64(factor * 100))
        var checked = 0
        for _ in 0..<300 {
            var pose = random(&g)
            pose.tilt = min(pose.tilt, 0.7)
            pose.distance = .random(in: 10...60, using: &g)  // room to zoom either way
            pose.clamp(aspect: aspect, size: L, height: H)
            let pointer = CGPoint(x: .random(in: 0...1100, using: &g), y: .random(in: 0...800, using: &g))
            let ndc = WorldPose.ndc(of: pointer, in: view)
            guard let anchor = pose.point(at: ndc, aspect: aspect, size: L) else { continue }
            let limit = pose.distanceLimit(size: L, aspect: aspect)
            let wanted = pose.distance * factor
            guard wanted > WorldLevels.minDistance, wanted < limit else { continue }  // not clamped
            let tiltBefore = pose.tilt
            pose.zoom(by: factor, at: ndc, aspect: aspect, size: L, height: H)
            #expect(abs(pose.distance / wanted - 1) < 1e-5)
            // (also when a zoom out eased the tilt: the grab is made last, with the orientation the view ends with)
            let at = try #require(screen(pose, anchor, z: pose.z))
            #expect(hypot(at.x - pointer.x, at.y - pointer.y) < 0.5, "\(at) for \(pointer), tilt \(tiltBefore) to \(pose.tilt)")
            checked += 1
        }
        #expect(checked > 100)
    }

    @Test func aZoomOutThatTheEyeLimitHoldsStillKeepsThePointUnderThePointer() throws {
        // (review finding of A1: the grab was made before the tilt was limited; the tilt is not limited now, the distance is)
        var held = 0
        for tilt: Float in [0.5, 0.7, 0.9, 1.1, 1.4] {
            var pose = WorldPose(x: 160, y: 40, heading: 0, tilt: tilt, distance: 100)
            pose.clamp(aspect: aspect, size: L, height: H)
            let ndc = SIMD2<Float>(-0.4, -0.6)  // below the middle: on the floor at every tilt
            let anchor = try #require(pose.point(at: ndc, aspect: aspect, size: L))
            pose.zoom(by: 3, at: ndc, aspect: aspect, size: L, height: H)
            if pose.distance < 300 * 0.999 { held += 1 }
            #expect(pose.tilt == tilt, "the tilt stays")
            let again = try #require(pose.point(at: ndc, aspect: aspect, size: L))
            let off = Torus.wrapDelta(again - anchor, size: L)
            #expect(off.x.magnitude < 1e-3 && off.y.magnitude < 1e-3, "tilt \(tilt): \(anchor) against \(again)")
        }
        #expect(held >= 3, "the case is reached: a tilted view is not allowed the whole zoom")
    }

    @Test func aZoomWithNoPointerIsAboutTheCentreAndTheClampsHold() {
        var pose = WorldPose(x: 12, y: 34, distance: 60)
        pose.zoom(by: 0.5, at: nil, aspect: aspect, size: L, height: H)
        #expect(pose.x == 12 && pose.y == 34 && pose.distance == 30)
        pose.zoom(by: 1e-9, at: nil, aspect: aspect, size: L, height: H)
        #expect(pose.distance == WorldLevels.minDistance)
        pose.zoom(by: 1e9, at: SIMD2(0.3, -0.2), aspect: aspect, size: L, height: H)
        #expect(pose.distance == pose.distanceLimit(size: L, aspect: aspect))
        let held = pose
        pose.zoom(by: .nan, at: nil, aspect: aspect, size: L, height: H)
        pose.zoom(by: -1, at: nil, aspect: aspect, size: L, height: H)
        pose.zoom(by: 0, at: nil, aspect: aspect, size: L, height: H)
        #expect(pose == held, "a factor that is not a positive number does nothing")
    }

    @Test func aPointerOffThePlaneGrabsAndZoomsNothingAboutIt() {
        var pose = WorldPose(x: 160, y: 160, heading: 0, tilt: 1.45, distance: 100)  // the horizon is in the top of the view
        let sky = SIMD2<Float>(0, 0.99)
        #expect(pose.offset(at: sky, aspect: aspect) == nil && pose.point(at: sky, aspect: aspect, size: L) == nil)
        let held = pose
        let grabbed = pose.grab(anchor: SIMD2(1, 1), at: sky, aspect: aspect, size: L)
        #expect(!grabbed && pose == held)
        pose.tilt = 0
        pose.zoom(by: 0.5, at: sky, aspect: aspect, size: L, height: H)
        #expect(pose.x == held.x && pose.distance == held.distance * 0.5, "straight down every ray meets the plane, so this one is a normal zoom")
    }

    @Test func theOffsetUnderThePointerIsTheProjectorsPlanePoint() throws {
        var g = SeededGenerator(seed: 8)
        for _ in 0..<100 {
            let pose = random(&g)
            let point = CGPoint(x: .random(in: 0...1100, using: &g), y: .random(in: 0...800, using: &g))
            let projector = WorldProjector(camera: pose.camera(size: L, height: H), size: L, view: view)
            guard let mine = pose.point(at: WorldPose.ndc(of: point, in: view), aspect: aspect, size: L) else { continue }
            let theirs = try #require(projector.world(at: point, height: pose.z))
            #expect(simd_length(Torus.wrapDelta(mine - theirs, size: L)) < 0.02, "\(mine) against \(theirs)")
        }
    }

    // MARK: Turning

    @Test func aHeadingDragTurnsTowardsTheDragAndATiltDragTiltsUpOrDownWithinItsRange() {
        var pose = WorldPose(x: 0, y: 0, distance: 50)
        pose.turn(dx: 100, dy: 0)
        #expect(abs(pose.heading - -0.8) < 1e-5, "0.008 rad a point, dragging right turns the view towards the right")
        pose.turn(dx: 0, dy: -50)
        #expect(abs(pose.tilt - 0.4) < 1e-5, "dragging up tilts")
        pose.turn(dx: 0, dy: 1e6)
        #expect(pose.tilt == 0)
        pose.turn(dx: 0, dy: -1e6)
        #expect(pose.tilt == WorldPose.maxTilt)
        pose.turn(dx: -2000, dy: 0)
        #expect(pose.heading >= -.pi && pose.heading < .pi || abs(pose.heading) <= .pi + 1e-5, "the heading stays in one turn: \(pose.heading)")
        var below = WorldPose(x: 0, y: 0, distance: 50, faceDown: true)
        below.turn(dx: 0, dy: 50)
        #expect(abs(below.tilt - 0.4) < 1e-5, "a world seen from below tilts the other way, as v1's orbit did")
    }

    // MARK: Home

    private func header(_ kind: String) throws -> StreamHeader {
        let found = try PlayFixture(kind: kind).events().compactMap { event -> StreamHeader? in
            if case .header(let h) = event { h } else { nil }
        }.first
        return try #require(found)
    }

    @Test func homeIsTheFlatOverviewFacingThePopulatedSide() throws {
        let h = try header("3d")
        let size = Float(h.size)
        let home = WorldPose.home(for: h, aspect: 1.38, populatedX: 40)
        #expect(home.x == 40 && home.y == 0 && home.z == 0 && home.heading == 0 && home.tilt == 0 && !home.faceDown)
        #expect(home.distance == home.distanceLimit(size: size, aspect: 1.38), "the farthest the view may be: the whole width of one world")
        #expect(WorldPose.home(for: h, aspect: 1.38).x == size / 2, "no organisms known: the middle")
        #expect(WorldPose.home(for: h, aspect: 1.38, populatedX: size + 5).x == 5, "wrapped")
        let flatHeader = try header("2d")
        let flat = WorldPose.home(for: flatHeader, aspect: 1)
        #expect(flat.faceDown && flat.y == Float(flatHeader.size) / 2)
    }

    @Test func thePopulatedSideIsTheCircularMeanSoARowAtEitherSideOfTheSeamMeansTheSeam() {
        let across = WorldFixtures.frame([SIMD3(318, 5, 3), SIMD3(2, 90, 3), SIMD3(319, 200, 3), SIMD3(1, 300, 3)])
        let mean = WorldPose.populatedX(frame: across, size: 320)
        #expect(mean != nil && min(abs(mean! - 0), abs(mean! - 320)) < 0.5, "a plain mean would say 160: \(mean ?? -1)")
        let apart = WorldFixtures.frame([SIMD3(80, 0, 0), SIMD3(240, 0, 0)])
        #expect(WorldPose.populatedX(frame: apart, size: 320) == nil || true, "two opposite sides have no mean (anything is fine, but no crash)")
        #expect(WorldPose.populatedX(frame: WorldFixtures.frame([]), size: 320) == nil)
        #expect(abs(WorldPose.populatedX(frame: WorldFixtures.frame([SIMD3(100, 1, 1), SIMD3(120, 2, 2)]), size: 320)! - 110) < 0.5)
    }

    @Test func theDistanceTheTiltAsksForIsGivenBackWhenTheTiltIsRelaxed() {
        // (the model keeps the wish: `WorldModel.distanceWish`; here the pure limit it uses)
        let far = WorldPose.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: aspect, tilt: 0)
        let low = WorldPose.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: aspect, tilt: 70 * .pi / 180)
        let side = WorldPose.maxDistance(size: L, fovY: Camera.defaultFovY, aspect: aspect, tilt: .pi / 2)
        #expect(far > low && low > side, "\(far) \(low) \(side)")
        #expect(abs(side - WorldFog.eyeReach * L) < 1e-3, "edge-on, the eye is at most eyeReach L from the point looked at: \(side)")
    }
}
