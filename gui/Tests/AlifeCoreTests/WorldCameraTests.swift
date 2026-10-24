import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The camera controls' pure parts (DESIGN §36.5, camera controls): the standard views and their ease, the horizon fade, the projector's
/// reading of the floor at oblique and grazing angles, and the ring of a storm on the floor.
@Suite struct WorldCameraTests {
    private let L: Float = 320
    private let H: Float = 80
    private let aspect: Float = 1.6

    // MARK: The standard views

    @Test func theFourViewsAreTopObliqueLowAndSideAtTheAnglesTheNamesSay() {
        #expect(ViewPreset.allCases == [.top, .oblique, .low, .side])
        #expect(ViewPreset.top.tilt == 0 && ViewPreset.side.tilt == .pi / 2)
        #expect(abs(ViewPreset.oblique.tilt - .pi / 4) < 1e-6 && abs(ViewPreset.low.tilt - 70 * .pi / 180) < 1e-6)
        #expect(ViewPreset.allCases.map(\.title) == ["Top", "Oblique", "Low", "Side"])
    }

    @Test func aPresetIsReachedExactlyAndTheEaseIsMonotoneSmoothAndTheShortWayRound() {
        var pose = WorldPose(x: 100, y: 50, z: 5, heading: 3.0, tilt: 0.3, distance: 100)
        pose.clamp(aspect: aspect, size: L, height: H)
        var ease = ViewEase(preset: .top, from: pose, wish: nil, aspect: aspect, size: L, height: H, keepHeight: false)
        #expect(ease.toTilt == 0 && ease.toHeading == 0 && ease.toZ == 0)
        var headings: [Float] = [], tilts: [Float] = []
        var last: (tilt: Float, heading: Float, z: Float, distance: Float, done: Bool)
        repeat {
            last = ease.step(dt: 1.0 / 60)
            headings.append(last.heading)
            tilts.append(last.tilt)
        } while !last.done
        #expect(headings.count == 30, "half a second at 60 Hz")
        #expect(last.tilt == 0 && last.heading == 0 && last.z == 0 && last.distance == ease.toDistance, "ends exactly")
        // 3.0 to 0 is the short way (down through 0), never through pi.
        #expect(zip(headings, headings.dropFirst()).allSatisfy { $1 <= $0 + 1e-6 && abs($0) <= 3.0 + 1e-6 })
        // And across the wrap: from 3.0 to -3.0 goes through pi, 0.28 radians, not 6.
        var across = ViewEase(fromTilt: 0, toTilt: 0, fromHeading: 3.0, toHeading: -3.0, fromZ: 0, toZ: 0, fromDistance: 10, toDistance: 10)
        let mid = across.step(dt: ViewEase.duration / 2)
        #expect(abs(abs(mid.heading) - .pi) < 0.01, "\(mid.heading)")
        // Smooth: zero speed at both ends.
        #expect(abs(tilts[0] - 0.3) < 0.003 && abs(tilts[tilts.count - 2]) < 0.003)
        // The reduced-motion ease is a quarter of a second.
        var short = ViewEase(preset: .oblique, from: pose, wish: nil, aspect: aspect, size: L, height: H, keepHeight: false, duration: ViewEase.reducedDuration)
        var frames = 0
        while !short.step(dt: 1.0 / 60).done { frames += 1 }
        #expect(frames + 1 == 15 || frames + 1 == 16)
    }

    @Test func eachViewKeepsOrSetsTheHeadingTheHeightAndTheDistanceByItsRule() {
        var pose = WorldPose(x: 100, y: 50, z: 0, heading: 1.0, tilt: 0.2, distance: 150)
        pose.clamp(aspect: aspect, size: L, height: H)
        func ease(_ preset: ViewPreset, keep: Bool = false, wish: Float? = nil, from: WorldPose? = nil) -> ViewEase {
            ViewEase(preset: preset, from: from ?? pose, wish: wish, aspect: aspect, size: L, height: H, keepHeight: keep)
        }
        #expect(ease(.top).toHeading == 0 && ease(.oblique).toHeading == 1.0 && ease(.low).toHeading == 1.0 && ease(.side).toHeading == 1.0)
        #expect(ease(.top).toZ == 0 && ease(.oblique).toZ == 0 && ease(.low).toZ == 0 && ease(.side).toZ == 0, "the Side view looks at the middle by the pose's lift, not by a stored height")
        var raised = pose
        raised.z = 30
        #expect(ease(.side, keep: true, from: raised).toZ == 30 && ease(.oblique, keep: true, from: raised).toZ == 30, "a followed organism's height stays")
        // The distance is the wish held by the limit of the angle: Top keeps it, Low comes in, Side is the one the layer needs.
        let top = ease(.top, wish: 150).toDistance, low = ease(.low, wish: 150).toDistance, side = ease(.side).toDistance
        #expect(top == min(150, WorldPose.maxDistance(size: L, fovY: pose.fovY, aspect: aspect, heading: 0, tilt: 0)))
        #expect(low == min(150, WorldPose.maxDistance(size: L, fovY: pose.fovY, aspect: aspect, heading: 1, tilt: ViewPreset.low.tilt)))
        #expect(abs(side - H / (2 * tan(pose.fovY / 2)) / ViewEase.sideFill) < 1e-3 && side < WorldFog.eyeReach * L, "\(side)")
        #expect(low <= top, "the eye is held nearer at the Low view, or as near")
        // (in a tall window the straight-down limit is far, so the eye's is what holds the Low view)
        let tall = ViewEase(preset: .low, from: pose, wish: 1e6, aspect: 0.8, size: L, height: H, keepHeight: false).toDistance
        #expect(abs(tall - WorldFog.eyeReach * L / sin(ViewPreset.low.tilt)) < 1e-2, "\(tall)")
        // The ease of the tilt alone keeps everything else.
        let flat = ViewEase(tiltTo: 0, from: pose, wish: nil, aspect: aspect, size: L)
        #expect(flat.toTilt == 0 && flat.toHeading == pose.heading && flat.toZ == pose.z && flat.toDistance == pose.distance)
        // Whatever the start, the end is a valid pose: finite, within the limit of its tilt.
        for preset in ViewPreset.allCases {
            var end = pose
            let e = ease(preset)
            end.tilt = e.toTilt; end.heading = e.toHeading; end.z = e.toZ; end.distance = e.toDistance
            let before = end
            end.clamp(aspect: aspect, size: L, height: H)
            #expect((end.tilt, end.heading, end.z, end.distance) == (before.tilt, before.heading, before.z, before.distance),
                    "\(preset): the pose it ends at is already clamped")
        }
    }

    // MARK: The haze

    @Test func theFogIsClearInsideTheBoxAndOver99PercentAtTheMeshsEdges() {
        #expect(WorldFog.amount(c: 0, size: L) == 0 && WorldFog.amount(c: 0.5 * L, size: L) == 0)
        #expect(WorldFog.amount(c: 0.49 * L * 1.0005, size: L) == 0, "the Top view's farthest picture is clear")
        for (fraction, expected) in [(Float(0.6), Float(0.221)), (0.65, 0.430), (0.7, 0.632), (0.9, 0.982), (1.0, 0.998)] {
            #expect(abs(WorldFog.amount(c: fraction * L, size: L) - expected) < 0.002, "\(fraction) L: \(WorldFog.amount(c: fraction * L, size: L))")
        }
        var last: Float = 0
        for i in 0...200 {
            let f = WorldFog.amount(c: Float(i) / 100 * L, size: L)
            #expect(f >= last && f <= 1)
            last = f
        }
        // The mesh's edges are where the fog is complete: x at 1.5 L, y at 1 L from the point looked at at the nearest (the centre is in its cell).
        #expect(WorldFog.amount(c: 1.0 * L, size: L) > 0.997 && WorldFog.meshHalfX * L >= L)
        #expect(WorldFog.amount(at: SIMD2(100, 100), centre: SIMD2(100 + 0.8 * L, 90), size: L) == WorldFog.amount(c: 0.8 * L, size: L), "the larger offset is the distance")
        #expect(WorldFog.amount(c: 5, size: 0) == 0)
    }

    @Test func theHazeIsTheSeasColourScaledByTheBrightnessAndTheMapsOpacityAndTheSkyFallsAwayFromIt() {
        let full = WorldFog.colour(exposure: 1, opacity: 1)
        #expect(full == WorldFog.hazeColour)
        #expect(WorldFog.colour(exposure: 1.5, opacity: 1).y > full.y && WorldFog.colour(exposure: 0.8, opacity: 1).y < full.y)
        let dim = WorldFog.colour(exposure: 1, opacity: 0.7)
        #expect(dim.y < full.y && dim.y > MapPalette.background.y, "the opacity mixes it back towards the clear colour as the map's")
        #expect(WorldFog.colour(exposure: 100, opacity: 1) == SIMD3(repeating: 1) || WorldFog.colour(exposure: 100, opacity: 1).max() <= 1)
        // The haze sits between the clear colour and the sea in luminance.
        func luma(_ c: SIMD3<Float>) -> Float { 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
        #expect(luma(MapPalette.background) < luma(WorldFog.hazeColour) && luma(WorldFog.hazeColour) < luma(MapPalette.coldSea))
        #expect(WorldFog.skyHaze(elevation: 0) == 1 && WorldFog.skyHaze(elevation: -1) == 1)
        #expect(WorldFog.skyHaze(elevation: 0.1) < 1 && WorldFog.skyHaze(elevation: 1.2) < 0.01)
        var last: Float = 1
        for i in 1...100 {
            let h = WorldFog.skyHaze(elevation: Float(i) / 100)
            #expect(h <= last)
            last = h
        }
    }

    // MARK: Reading the floor at oblique and grazing angles

    @Test func theProjectorReadsTheFloorWhereItIsAndNotInTheHazeOrAboveTheHorizon() {
        let pose = WorldPose(x: 160, y: 20, heading: 0, tilt: 80 * .pi / 180, distance: 100)
        let view = CGSize(width: 1000, height: 640)
        let projector = WorldProjector(camera: pose.camera(size: L, height: H), size: L, view: view)
        var near = 0, haze = 0, sky = 0
        for row in stride(from: 2.0, to: 640.0, by: 8) {
            let point = CGPoint(x: 500, y: row)
            let hit = pose.offset(at: WorldPose.ndc(of: point, in: view), aspect: Float(view.width / view.height), planeZ: 0)
            let found = projector.world(at: point, height: 0)
            switch hit.map(WorldFog.chebyshev) {
            case nil: sky += 1; #expect(found == nil)
            case let c? where c <= WorldFog.readReach * L: near += 1; #expect(found != nil)
            default: haze += 1; #expect(found == nil)
            }
        }
        #expect(near > 5 && haze > 3 && sky > 3, "\(near) \(haze) \(sky)")
    }

    @Test func aRingOnTheFloorIsFlatteningTowardsTheHorizonAndRoundFromAbove() {
        let view = CGSize(width: 1000, height: 640)
        var previous: CGFloat = 2
        for degrees: Float in [0, 20, 45, 70, 85] {
            let pose = WorldPose(x: 160, y: 20, heading: 0.4, tilt: degrees * .pi / 180, distance: 100)
            let projector = WorldProjector(camera: pose.camera(size: L, height: H), size: L, view: view)
            let ratio = projector.flatten(at: SIMD3(160, 20, 0))
            #expect(ratio <= previous + 1e-4 && ratio >= 0.12 && ratio <= 1, "\(degrees) degrees: \(ratio)")
            if degrees == 0 { #expect(abs(ratio - 1) < 1e-3, "from above a circle is a circle") }
            if degrees == 45 { #expect(abs(Float(ratio) - cos(Float.pi / 4)) < 0.06, "at 45 degrees about cos 45: \(ratio)") }
            previous = ratio
        }
    }

    // MARK: The readouts

    @Test func theCompassBearingAndTheDialAngleAgreeWithTheNeedle() {
        #expect(Compass.angle(of: CGPoint(x: 10, y: 0), around: CGPoint(x: 10, y: 10)) == 0, "straight up")
        #expect(abs(Compass.angle(of: CGPoint(x: 20, y: 10), around: CGPoint(x: 10, y: 10)) - .pi / 2) < 1e-6, "to the right is clockwise")
        #expect(abs(abs(Compass.angle(of: CGPoint(x: 10, y: 20), around: CGPoint(x: 10, y: 10))) - .pi) < 1e-6)
        // At the Top view the needle points where the dial's heading says: northAngle is the heading.
        for heading: Float in [0, 0.5, -1.2, 2.9] {
            let pose = WorldPose(x: 0, y: 0, heading: heading, distance: 50)
            #expect(abs(Compass.northAngle(camera: pose.camera(size: L, height: H)) - heading) < 1e-5, "heading \(heading)")
        }
        #expect(WorldReadout.viewText(pose: WorldPose(x: 0, y: 0, heading: .pi / 2, tilt: .pi / 4, distance: 10)) == "Tilt 45° · heading 270°")
    }
}
