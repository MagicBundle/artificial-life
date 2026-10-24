import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The review of the camera controls (DESIGN §36.5, camera controls): the pure parts of what the review found. The edge-on lift of the
/// point looked at, the zoom given back by a view's end, the compass's needle and dead zone, the elliptical storm pick and the HUD's
/// readout.
@Suite struct WorldCameraReviewTests {
    private let L: Float = 320
    private let H: Float = 80
    private let aspect: Float = 1.6

    // MARK: The edge-on lift

    @Test func theEyeIsNotOnTheFloorAtEdgeOnWhateverWayTheTiltWasReached() {
        // Nothing before the Low view's 70 degrees, the whole way to the middle of the height at 90, rising one way only.
        #expect(WorldPose.edgeOnLift(tilt: 0) == 0 && WorldPose.edgeOnLift(tilt: WorldPose.liftFrom) == 0)
        #expect(abs(WorldPose.edgeOnLift(tilt: WorldPose.maxTilt) - 1) < 1e-6)
        var last: Float = 0
        for degrees in stride(from: Float(0), through: 90, by: 1) {
            let lift = WorldPose.edgeOnLift(tilt: degrees * .pi / 180)
            #expect(lift >= last - 1e-6 && lift <= 1)
            last = lift
        }
        for degrees: Float in [0, 30, 45, 70] {
            var pose = WorldPose(x: 100, y: 50, tilt: degrees * .pi / 180, distance: 100)
            pose.clamp(aspect: aspect, size: L, height: H)
            #expect(pose.lift == 0 && pose.lookZ == 0, "\(degrees) degrees: the floor is looked at")
        }
        // A pose that was edited to edge-on (the handle, an Option drag, a tilt of the tilt handle) from the floor: the point looked at
        // is the middle of the height, so is the eye (the picture has the life from 0 to H above and below the horizon).
        var edge = WorldPose(x: 100, y: 50, tilt: WorldPose.maxTilt, distance: 100)
        edge.clamp(aspect: aspect, size: L, height: H)
        #expect(edge.z == 0 && abs(edge.lookZ - H / 2) < 1e-4)
        let camera = edge.camera(size: L, height: H)
        #expect(abs(camera.target.z - H / 2) < 1e-4 && abs(camera.position.z - H / 2) < 1e-3, "the eye at the middle of the height")
        #expect(abs(edge.eyeHeight(size: L, height: H) - H / 2) < 1e-3)
        // It goes as the tilt is relaxed: the stored height is the floor still.
        edge.tilt = 60 * .pi / 180
        edge.clamp(aspect: aspect, size: L, height: H)
        #expect(edge.lift == 0 && edge.lookZ == 0)
        // Between, the eye's height is the sum of the lift and the tilt's own: the eye rises as the tilt goes to edge-on.
        var eyes: [Float] = []
        for degrees: Float in [70, 75, 80, 85, 90] {
            var pose = WorldPose(x: 100, y: 50, tilt: degrees * .pi / 180, distance: 100)
            pose.clamp(aspect: aspect, size: L, height: H)
            eyes.append(pose.eyeHeight(size: L, height: H))
        }
        #expect(eyes.last! > 0.9 * H / 2 - 2, "at edge-on the eye is at the middle: \(eyes)")
    }

    @Test func aFollowedHeightIsNotLiftedAndANeighbouringHeightIsNotLoweredAndA2DWorldHasNone() {
        var followed = WorldPose(x: 100, y: 50, z: 10, tilt: WorldPose.maxTilt, distance: 100)
        followed.followsHeight = true
        followed.clamp(aspect: aspect, size: L, height: H)
        #expect(followed.lift == 0 && followed.lookZ == 10, "the camera is on the organism")
        var high = WorldPose(x: 100, y: 50, z: 60, tilt: WorldPose.maxTilt, distance: 100)
        high.clamp(aspect: aspect, size: L, height: H)
        #expect(high.lift == 0 && high.lookZ == 60, "above the middle already: it is not lowered")
        var low = WorldPose(x: 100, y: 50, z: 10, tilt: WorldPose.maxTilt, distance: 100)
        low.clamp(aspect: aspect, size: L, height: H)
        #expect(abs(low.lookZ - H / 2) < 1e-4, "an organism's height left over from a follow rises as well")
        var flat = WorldPose(x: 100, y: 50, tilt: WorldPose.maxTilt, distance: 100, faceDown: true)
        flat.clamp(aspect: aspect, size: L, height: 0)
        #expect(flat.lift == 0)
        var none = WorldPose(x: 100, y: 50, tilt: WorldPose.maxTilt, distance: 100)
        none.clamp(aspect: aspect, size: L, height: 0)
        #expect(none.lift == 0, "no height, no lift")
        // `apply` is the inverse of `camera` with the lift in it.
        var edge = WorldPose(x: 100, y: 50, tilt: WorldPose.maxTilt, distance: 100)
        edge.clamp(aspect: aspect, size: L, height: H)
        var round = edge
        round.apply(edge.camera(size: L, height: H))
        #expect(abs(round.z - edge.z) < 1e-4 && abs(round.tilt - edge.tilt) < 1e-5 && abs(round.distance - edge.distance) < 1e-3)
    }

    // MARK: The zoom given back

    @Test func theEndOfAnEaseGivesBackTheZoomThatTheAnglesLimitHeldNearer() {
        var home = WorldPose(x: 160, y: 0, distance: 1)
        home.distance = home.distanceLimit(size: L, aspect: aspect)
        let low = ViewEase(preset: .low, from: home, wish: nil, aspect: aspect, size: L, height: H, keepHeight: false)
        #expect(low.toDistance < home.distance && low.endWish == home.distance, "the wish survives: \(low.toDistance) of \(home.distance), \(String(describing: low.endWish))")
        // Top from Low, with that wish, goes back out.
        var atLow = home
        atLow.tilt = ViewPreset.low.tilt
        atLow.distance = low.toDistance
        atLow.clamp(aspect: aspect, size: L, height: H)
        let top = ViewEase(preset: .top, from: atLow, wish: low.endWish, aspect: aspect, size: L, height: H, keepHeight: false)
        #expect(top.toDistance == home.distance && top.endWish == nil)
        // The ease of the tilt alone does the same (the 3D button, twice).
        let angled = ViewEase(tiltTo: ViewPreset.low.tilt, from: home, wish: nil, aspect: aspect, size: L)
        #expect(angled.toDistance < home.distance && angled.endWish == home.distance)
        // Nothing is held: no wish to give back. The Side view has a distance of its own.
        let oblique = ViewEase(preset: .oblique, from: home, wish: 40, aspect: aspect, size: L, height: H, keepHeight: false)
        #expect(oblique.endWish == nil && oblique.toDistance == 40)
        #expect(ViewEase(preset: .side, from: home, wish: nil, aspect: aspect, size: L, height: H, keepHeight: false).endWish == nil)
    }

    @Test func theTiltAloneEaseGoesToTheFloorAtTheTopViewUnlessAnOrganismIsFollowed() {
        var pose = WorldPose(x: 100, y: 50, z: 30, tilt: 0.8, distance: 100)
        pose.clamp(aspect: aspect, size: L, height: H)
        #expect(ViewEase(tiltTo: 0, from: pose, wish: nil, aspect: aspect, size: L, keepHeight: false).toZ == 0, "Top is the floor, as the Top preset's")
        #expect(ViewEase(tiltTo: 0, from: pose, wish: nil, aspect: aspect, size: L, keepHeight: true).toZ == 30, "a followed height stays")
        #expect(ViewEase(tiltTo: 0.8, from: pose, wish: nil, aspect: aspect, size: L, keepHeight: false).toZ == 30, "tilting is not about the height")
    }

    // MARK: The compass

    @Test func theNeedleSaysWhereNorthIsAtEveryTiltAndAtEdgeOnItIsTheHeadingNotNoise() {
        for heading: Float in [0, 0.5, -1.2, 2.9, .pi, -.pi / 2, .pi / 2] {
            // The Top view: the heading. Edge-on: the heading, as exactly (the projection is the noise of a zero over a zero).
            #expect(abs(WorldPose.wrap(Compass.needleAngle(heading: heading, tilt: 0) - heading)) < 1e-5, "Top, heading \(heading)")
            #expect(abs(WorldPose.wrap(Compass.needleAngle(heading: heading, tilt: WorldPose.maxTilt) - heading)) < 1e-5, "edge-on, heading \(heading)")
            // The needle moves smoothly with the tilt: no step of more than a few degrees between neighbouring tilts.
            var last = Compass.needleAngle(heading: heading, tilt: 0)
            for degrees in stride(from: Float(1), through: 90, by: 1) {
                let angle = Compass.needleAngle(heading: heading, tilt: degrees * .pi / 180)
                #expect(abs(WorldPose.wrap(angle - last)) < 0.12, "heading \(heading), \(degrees) degrees: \(angle) after \(last)")
                last = angle
            }
        }
        // Facing north and tilted: north is ahead, so up, at every tilt.
        for degrees in stride(from: Float(0), through: 90, by: 5) {
            #expect(abs(Compass.needleAngle(heading: 0, tilt: degrees * .pi / 180)) < 1e-6)
        }
        // Where the projection is well conditioned, the needle is the projection (the Oblique view).
        let pose = WorldPose(x: 0, y: 0, heading: 0.4, tilt: ViewPreset.oblique.tilt, distance: 50)
        #expect(abs(Compass.needleAngle(heading: 0.4, tilt: pose.tilt) - Compass.northAngle(camera: pose.camera(size: L, height: H))) < 0.08)
    }

    @Test func thePointerAtTheDialsCentreHasNoAngle() {
        let c = CGPoint(x: 100, y: 100)
        #expect(Compass.dialAngle(of: c, around: c) == nil)
        #expect(Compass.dialAngle(of: CGPoint(x: 103, y: 102), around: c) == nil, "within the dead zone")
        #expect(Compass.dialAngle(of: CGPoint(x: 100, y: 100 - Compass.deadZone - 1), around: c) == 0, "straight up outside it")
        #expect(abs((Compass.dialAngle(of: CGPoint(x: 100 + 10, y: 100), around: c) ?? 9) - .pi / 2) < 1e-6)
    }

    // MARK: The storm's ring

    @Test func aStormIsPickedWithinTheEllipseItIsDrawnAsAndNotAboveOrBelowIt() throws {
        let store = try WorldFixtures.store()
        let storms = try #require(store.storms?.storms)
        let storm = try #require(storms.storms.first { $0.id == 7 })
        let size: Float = 48
        var pose = WorldPose(x: Float(storm.x), y: Float(storm.y), heading: 0, tilt: 70 * .pi / 180, distance: 20)
        pose.clamp(aspect: 1.3, size: size, height: 24)
        let projector = WorldProjector(camera: pose.camera(size: size, height: 24), size: size, view: CGSize(width: 900, height: 700))
        let glyph = try #require(projector.glyphs(of: storms).first { $0.id == 7 })
        #expect(glyph.flatten < 0.8 && glyph.radius > 40, "a flattened ring: \(glyph.flatten), \(glyph.radius)")
        func picked(_ dx: CGFloat, _ dy: CGFloat) -> Bool {
            WorldPicker.storm(at: CGPoint(x: glyph.center.x + dx, y: glyph.center.y + dy), projector: projector, storms: storms) == 7
        }
        let ry = glyph.radius * glyph.flatten
        #expect(picked(0, 0) && picked(0.8 * glyph.radius, 0) && picked(0, 0.8 * ry) && picked(0, -0.8 * ry), "inside the ellipse")
        #expect(!picked(0, 0.8 * glyph.radius) && !picked(0, -0.8 * glyph.radius), "above and below the drawn ring is not the storm")
        #expect(!picked(1.2 * glyph.radius, 0), "and outside to the side")
        // Seen from straight above the ring is a circle and the test is the circle the picker always had.
        var top = pose
        top.tilt = 0
        top.clamp(aspect: 1.3, size: size, height: 24)
        let from = WorldProjector(camera: top.camera(size: size, height: 24), size: size, view: CGSize(width: 900, height: 700))
        let round = try #require(from.glyphs(of: storms).first { $0.id == 7 })
        #expect(abs(round.flatten - 1) < 1e-3)
        #expect(WorldPicker.storm(at: CGPoint(x: round.center.x, y: round.center.y + 0.8 * round.radius), projector: from, storms: storms) == 7)
    }

    // MARK: The readout of the angle

    @Test func thePlateOfTheAngleTakesThePressAndIsNotTheMap() {
        let hud = WorldHUD(free: CGSize(width: 1500, height: 900), minimap: true)
        let readout = hud.readout!
        #expect(hud.target(at: CGPoint(x: readout.midX, y: readout.midY)) == .readout)
        #expect(hud.target(at: CGPoint(x: readout.minX + 1, y: readout.minY + 1)) == .readout)
        #expect(hud.target(at: CGPoint(x: readout.minX - 3, y: readout.midY)) == .map)
        #expect(!hud.rects.contains { $0.target == .readout }, "it is not a control: it has no rectangle of the controls' list")
    }
}
