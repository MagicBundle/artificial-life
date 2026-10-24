import AlifeCore
import Foundation
import Testing
import simd

@testable import AlifeControl

/// The World's camera and what it does to the level, on a played fixture (no renderer, no window): the model keeps the
/// camera, wraps its target, clamps the zoom, and fits the world to the free rectangle while the camera is home.
@MainActor
@Suite struct WorldModelTests {
    private let free = CGRect(x: 0, y: 0, width: 900, height: 700)

    /// A controller that has played the 3D fixture (L = 48, height 24) and stays live.
    private func live(_ scratch: Scratch) async throws -> PlayController {
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do :; done")
        let play = PlayController()
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                   arguments: [Repo.playFixtureURL("3d").path])
        #expect(await eventually { play.store.header != nil && play.store.clock != nil })
        return play
    }

    private func model(_ scratch: Scratch) async throws -> (WorldModel, PlayController) {
        let play = try await live(scratch)
        let model = WorldModel(play: play, defaults: try #require(UserDefaults(suiteName: "alife-worldmodel-\(UUID().uuidString)")))
        model.setFreeRect(free)
        model.reduceMotion = { false }
        model.goHome()
        return (model, play)
    }

    /// Runs the camera's own motion for up to `seconds`, a sixtieth of a second at a time; true if it arrived.
    private func arrive(_ model: WorldModel, seconds: Float = 4) -> Bool {
        for _ in 0..<Int(seconds * 60) where !model.advanceCamera(dt: 1.0 / 60) { return true }
        return false
    }

    /// The point of the floor under a point of the free rectangle.
    private func under(_ model: WorldModel, _ point: CGPoint) -> SIMD2<Float>? {
        model.projector(size: free.size).world(at: point)
    }

    private func close(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ tolerance: Float, size: Float = 48) -> Bool {
        simd_length(Torus.wrapDelta(a - b, size: size)) < tolerance
    }

    @Test func theViewStartsAtTheHomeViewFittedToTheFreeRectangle() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        let header = try #require(play.store.header)
        let home = model.homePose(header: header)
        #expect(model.pose == home && model.camera == home.camera(size: 48, height: 24))
        #expect(model.pose.heading == 0 && model.pose.tilt == 0 && model.pose.y == 0, "north up, straight down, on the equator")
        #expect(abs(model.pose.distance - model.distanceLimit) < 1e-3, "the whole width of one world: the farthest the view may be")
        #expect(model.cameraIsHome && model.level == .planet)
        #expect(abs(model.aspect - Float(900.0 / 700)) < 1e-6)
        #expect(model.worldSize == 48)
    }

    @Test func theFirstFrameOfAnUntouchedWorldTurnsTheHomeViewTowardsThePopulatedSide() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do :; done")
        let play = PlayController()
        defer { play.shutdown() }
        let model = WorldModel(play: play, defaults: try #require(UserDefaults(suiteName: "alife-firstframe-\(UUID().uuidString)")))
        model.setFreeRect(free)
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url, arguments: [Repo.playFixtureURL("3d").path])
        #expect(await eventually { play.store.header != nil && play.store.frame != nil })
        let frame = try #require(play.store.frame)
        let populated = try #require(WorldPose.populatedX(frame: frame, size: 48))
        #expect(abs(Torus.wrapDelta(model.pose.x - populated, size: 48)) < 1e-3 && abs(model.pose.x - 24) > 0.01, "faces the organisms: \(model.pose.x) for \(populated)")
        #expect(model.cameraIsHome && !model.homeAwaitsFrame)
    }

    @Test func aGrabKeepsThePointUnderThePointerWhereverItGoes() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.zoom(by: 0.4)  // a closer view, so that a drag moves the map by a good part of it
        model.editPose { $0.heading = 0.5; $0.tilt = 0.4 }
        let start = CGPoint(x: 300, y: 250)
        let anchor = try #require(under(model, start))
        model.beginGrab(at: start)
        // A long wandering drag: across the x seam, across the wrap in y, over most of the free rectangle.
        var at = start
        for step in 0..<120 {
            at = CGPoint(x: 100 + 700 * abs(sin(Double(step) / 19)), y: 100 + 500 * abs(cos(Double(step) / 23)))
            model.grab(to: at)
            let here = try #require(under(model, at))
            #expect(close(here, anchor, 0.01), "step \(step): \(here) against \(anchor)")
        }
        #expect(!model.cameraIsHome)
        model.endGrab()
        model.grab(to: CGPoint(x: 10, y: 10))  // no grab, no move
        #expect(close(try #require(under(model, at)), anchor, 0.01))
    }

    @Test func aDragWithoutAnAnchorPansByTheDeltaAndWrapsTheTarget() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        let before = model.camera
        model.drag(dx: 40, dy: 0, orbit: false)
        // The map follows the pointer: the target moves the other way, along the screen's x (east is right).
        #expect(model.camera.target.x < before.target.x || model.camera.target.x > 40, "west (or across the seam)")
        #expect(abs(Torus.wrapDelta(model.camera.target.y - before.target.y, size: 48)) < 1e-3)
        #expect(!model.cameraIsHome)
        for _ in 0..<40 { model.drag(dx: 60, dy: -30, orbit: false) }  // a long drag: the target crosses the seam
        let t = model.camera.target
        #expect(t.x >= 0 && t.x < 48 && t.y >= 0 && t.y < 48, "kept in the world: \(t)")
        #expect(model.camera.distance == before.distance && model.camera.yaw == before.yaw && model.camera.pitch == before.pitch)
    }

    @Test func anOrbitDragTurnsTheHeadingAndTiltsTheViewAboutTheTarget() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.zoom(by: 0.5)
        let before = model.pose
        model.drag(dx: 30, dy: -20, orbit: true)
        #expect(model.pose.x == before.x && model.pose.y == before.y && model.pose.distance == before.distance)
        #expect(model.pose.heading < before.heading, "dragging right turns the view towards the right")
        #expect(model.pose.tilt > 0, "dragging up tilts")
        model.drag(dx: 0, dy: 400, orbit: true)
        #expect(model.pose.tilt == 0, "and back to straight down, no further")
    }

    @Test func theZoomIsClampedBetweenAFewRadiiAndOneWorld() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        for _ in 0..<60 { model.zoom(by: 0.5) }
        #expect(model.camera.distance == WorldLevels.minDistance)
        for _ in 0..<60 { model.zoom(by: 2) }
        let limit = WorldLevels.maxDistance(size: 48, fovY: model.camera.fovY, aspect: model.aspect)
        #expect(abs(model.camera.distance - limit) < 1e-3)
        model.zoom(by: 0.5, at: CGPoint(x: 100, y: 100))
        #expect(model.camera.distance < limit)
        model.magnify(0.3)
        model.zoom(by: .infinity)
        model.zoom(by: .nan)
        model.zoom(by: -2)
        model.scroll(ScrollEvent(dy: -1_000_000, precise: false), at: nil)  // an absurd wheel: bounded, never infinite or NaN
        #expect(model.camera.distance.isFinite && model.camera.distance <= limit + 1e-3)
    }

    @Test func aZoomIsAboutThePointerAndAWheelNotchIsATenthOfALine() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.zoom(by: 0.5)  // room to zoom both ways
        let pointer = CGPoint(x: 700, y: 180)
        let under0 = try #require(under(model, pointer))
        let d0 = model.pose.distance
        model.scroll(ScrollEvent(dy: 3, precise: false, time: 100), at: pointer)  // three notches up: closer
        #expect(abs(model.pose.distance / d0 - exp(-0.3)) < 1e-4, "0.1 per line: \(model.pose.distance / d0)")
        #expect(close(try #require(under(model, pointer)), under0, 0.01), "the point under the pointer stays under it")
        let d1 = model.pose.distance
        model.magnify(-0.2, at: pointer)  // a pinch in: farther
        let after = try #require(under(model, pointer))
        #expect(model.pose.distance > d1 && close(after, under0, 0.01))
        #expect(model.scrollLaw.zoomed > 0)
    }

    @Test func theTrackpadsTwoFingerScrollPansTheMapByTheSummedDeltaMomentumIncluded() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.zoom(by: 0.5)
        let d0 = model.pose.distance
        let centre = CGPoint(x: free.midX, y: free.midY)
        let anchor = try #require(under(model, centre))
        let screen0 = try #require(model.projector(size: free.size).screen(SIMD3(anchor.x, anchor.y, 0)))
        var total = SIMD2<Float>.zero
        var time = 0.0
        func send(_ dx: Float, _ dy: Float, phase: ScrollGesturePhase = .changed, momentum: ScrollGesturePhase = .none) {
            time += 0.01
            model.scroll(ScrollEvent(dx: dx, dy: dy, precise: true, phase: phase, momentum: momentum, time: time), at: centre)
            total += SIMD2(dx, dy)
        }
        send(0, 0, phase: .began)
        for i in 0..<30 { send(Float(30 - i) * 0.5, Float(30 - i) * 0.3) }
        send(0, 0, phase: .ended)
        for i in 0..<60 { send(Float(60 - i) * 0.1, Float(60 - i) * 0.05, phase: .none, momentum: .changed) }
        #expect(model.pose.distance == d0, "a scroll that pans does not zoom, momentum or not")
        let screen1 = try #require(model.projector(size: free.size).screen(SIMD3(anchor.x, anchor.y, 0)))
        // The map followed the fingers: what was at the centre moved by the summed delta (+y is down on the screen).
        #expect(abs(Float(screen1.x - screen0.x) - total.x) < 0.5 && abs(Float(screen1.y - screen0.y) - total.y) < 0.5,
                "moved (\(screen1.x - screen0.x), \(screen1.y - screen0.y)) for \(total)")
    }

    @Test func aHardFlickWithScrollZoomsOnCrossesAtMostAFactorOfFourAndItsMomentumNothing() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.display.scrollZooms = true
        let d0 = model.pose.distance
        var time = 0.0
        func send(_ dy: Float, phase: ScrollGesturePhase, momentum: ScrollGesturePhase = .none) {
            time += 0.008
            model.scroll(ScrollEvent(dy: dy, precise: true, phase: phase, momentum: momentum, time: time), at: CGPoint(x: 400, y: 300))
        }
        send(0, phase: .began)
        for i in 0..<300 { send(60 * exp(-Float(i) / 90), phase: .changed) }  // a hard flick of the fingers, decaying
        send(0, phase: .ended)
        let afterFingers = model.pose.distance
        #expect(afterFingers >= d0 / 4 * (1 - 1e-3) && afterFingers < d0, "at most a factor of 4 for the gesture: \(d0) to \(afterFingers)")
        for i in 0..<400 { send(40 * exp(-Float(i) / 100), phase: .none, momentum: .changed) }
        #expect(model.pose.distance == afterFingers, "the momentum does not zoom")
        // The next gesture has its own allowance.
        send(0, phase: .began)
        send(20, phase: .changed)
        #expect(model.pose.distance < afterFingers)
    }

    @Test func theLevelFollowsTheZoom() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        var seen: [WorldLevel] = [model.level]
        for _ in 0..<40 {
            model.zoom(by: 0.9)
            if seen.last != model.level { seen.append(model.level) }
        }
        #expect(seen == [.planet, .region, .organism])
        for _ in 0..<80 {
            model.zoom(by: 1.1)
            if seen.last != model.level { seen.append(model.level) }
        }
        #expect(seen.suffix(3) == [.organism, .region, .planet])
    }

    @Test func whileTheCameraIsHomeAResizeFitsTheWorldAgain() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.setFreeRect(CGRect(x: 0, y: 0, width: 1200, height: 600))
        model.drawableResized(CGSize(width: 1200, height: 600))
        #expect(model.pose.distance == model.pose.distanceLimit(size: 48, aspect: 2), "the free rectangle changed shape: refit")
        #expect(model.cameraIsHome)
        model.drag(dx: 10, dy: 10, orbit: false)
        let moved = model.pose
        model.drawableResized(CGSize(width: 400, height: 800))
        #expect(model.pose == moved, "the user's view is kept")
        model.home()
        #expect(model.isFlying && arrive(model))
        let header = try #require(play.store.header)
        #expect(model.cameraIsHome && model.pose == model.homePose(header: header))
    }

    @Test func homeFliesToTheHomeViewAndAnInputStopsItWhereItIs() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        let header = try #require(play.store.header)
        let home = model.homePose(header: header)
        model.zoom(by: 0.2)
        model.editPose { $0.x = 5; $0.y = 33; $0.heading = 1.2; $0.tilt = 0.5 }
        model.home()
        #expect(model.isFlying && !model.cameraIsHome, "a flight, not a snap")
        var peak: Float = 0
        var frames = 0
        while model.advanceCamera(dt: 1.0 / 60) {
            peak = max(peak, model.pose.distance)
            frames += 1
            #expect(frames < 200, "a flight lasts under 1.8 s")
        }
        #expect(frames >= 27, "and over 0.45 s: \(frames) frames")
        #expect(model.pose == home && model.cameraIsHome && !model.isFlying)
        // An input stops a flight where it is.
        model.zoom(by: 0.3)
        model.editPose { $0.x = 40 }
        model.home()
        for _ in 0..<10 { _ = model.advanceCamera(dt: 1.0 / 60) }
        let held = model.pose
        #expect(model.isFlying && held != home)
        model.drag(dx: 1, dy: 0, orbit: false)
        #expect(!model.isFlying && model.flight == nil)
        #expect(!model.advanceCamera(dt: 1.0 / 60))
        model.home()
        for _ in 0..<5 { _ = model.advanceCamera(dt: 1.0 / 60) }
        model.zoom(by: 0.9)
        #expect(!model.isFlying, "a zoom stops the flight too")
    }

    @Test func reduceMotionMakesAFlightAStraightQuarterSecondEase() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.zoom(by: 0.2)
        model.editPose { $0.x = 5; $0.y = 33; $0.heading = 1.0 }
        let from = model.pose.distance
        model.reduceMotion = { true }
        model.home()
        #expect(model.flight?.path.duration == FlightPath.easeDuration)
        var frames = 0
        while model.advanceCamera(dt: 1.0 / 60) {
            frames += 1
            #expect(model.pose.distance <= model.homePose(header: model.play.store.header!).distance + 1e-3 && model.pose.distance >= from - 1e-3,
                    "no zoom out and in: the distance goes one way")
        }
        #expect(frames <= 16 && model.cameraIsHome)
    }

    @Test func aTurnAskedForDuringAFlightToAPlaceStandsAndNorthUpDuringAFlightHomeIsLeftToIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.zoom(by: 0.3)
        model.editPose { $0.heading = 1.0 }
        model.flyTo(SIMD3(40, 30, 0), distance: nil)
        model.northUp()
        #expect(model.arriveForTests() && model.pose.heading == 0, "the flight did not turn it back to 1.0: \(model.pose.heading)")
        model.editPose { $0.heading = 1.0 }
        model.home()
        model.northUp()
        #expect(model.headingEase == nil, "Home turns it north up by itself")
        #expect(model.arriveForTests() && model.pose.heading == 0 && model.cameraIsHome)
    }

    @Test func northUpEasesTheHeadingToZeroInUnderAThirdOfASecondAndADragStopsIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.zoom(by: 0.5)
        model.editPose { $0.heading = 2.0 }
        model.northUp()
        #expect(model.cameraMotionPending)
        var frames = 0
        var last = model.pose.heading
        while model.advanceCamera(dt: 1.0 / 60) {
            frames += 1
            #expect(model.pose.heading <= last + 1e-6, "monotone")
            last = model.pose.heading
            #expect(frames < 40)
        }
        #expect(model.pose.heading == 0 && frames <= 19, "0.3 s is 18 frames: \(frames)")
        model.editPose { $0.heading = -1 }
        model.northUp()
        _ = model.advanceCamera(dt: 1.0 / 60)
        model.drag(dx: 5, dy: 0, orbit: true)
        let held = model.pose.heading
        #expect(!model.advanceCamera(dt: 1.0 / 60) && model.pose.heading == held, "the drag keeps the heading it gave")
    }

    @Test func theTiltIsEasedOutWhereTheMapWouldShowItsEdge() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.zoom(by: 0.3)
        model.drag(dx: 0, dy: -400, orbit: true)  // tilt from straight down towards the horizon
        #expect(model.pose.tilt > 0.3, "the view is tilted: \(model.pose.tilt)")
        let tilted = model.pose.tilt
        for _ in 0..<30 {
            model.zoom(by: 1.15)
            #expect(model.pose.fits(aspect: model.aspect, size: 48, height: 24), "the picture stays within one world")
        }
        #expect(model.pose.tilt < tilted, "zooming out tilted the view back")
    }

    @Test func aPanOfATiltedViewStaysOnThePlaneThroughTheTarget() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.zoom(by: 0.3)
        model.drag(dx: 0, dy: -300, orbit: true)
        model.editPose { $0.z = 12 }
        for sign: Float in [1, -1] {
            for _ in 0..<40 {
                model.drag(dx: 0, dy: sign * 300, orbit: false)
                #expect(model.camera.target.z == 12, "\(model.camera.target)")
            }
        }
    }

    @Test func selectingBumpsTheOverlayOnceAndOnlyOnAChange() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        let r0 = model.overlay.revision
        model.setSelection(.organism(uid: 5_000_000_000))
        #expect(model.selectedUID == 5_000_000_000 && model.overlay.revision == r0 + 1)
        model.setSelection(.organism(uid: 5_000_000_000))
        #expect(model.overlay.revision == r0 + 1)
        model.setSelection(.none)
        #expect(model.selectedUID == nil && model.overlay.revision == r0 + 2)
    }

    @Test func theProjectorIsTheCameraOnTheGivenSize() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        let p = model.projector(size: CGSize(width: 900, height: 700))
        #expect(p.camera == model.camera && p.size == 48 && p.level == model.level)
    }
}

extension WorldModel {
    /// Runs the camera's own motion for up to four seconds of sixtieths; true if it arrived.
    func arriveForTests() -> Bool {
        for _ in 0..<240 where !advanceCamera(dt: 1.0 / 60) { return true }
        return false
    }
}
