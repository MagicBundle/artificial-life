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
        let model = WorldModel(play: play)
        model.setFreeRect(free)
        model.goHome()
        return (model, play)
    }

    @Test func theCameraStartsAboveTheWorldFittedToTheFreeRectangle() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        let header = try #require(play.store.header)
        #expect(model.camera == Camera.world(for: header, aspect: Float(900.0 / 700)))
        #expect(model.cameraIsHome && model.level == .planet)
        #expect(abs(model.aspect - Float(900.0 / 700)) < 1e-6)
        #expect(model.worldSize == 48)
    }

    @Test func aDragPansTheMapAndWrapsTheTarget() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        let before = model.camera
        model.drag(dx: 40, dy: 0, orbit: false)
        // The map follows the pointer: the target moves the other way, along the screen's x (east is right).
        #expect(model.camera.target.x < before.target.x && abs(model.camera.target.y - before.target.y) < 1e-3)
        #expect(!model.cameraIsHome)
        for _ in 0..<40 { model.drag(dx: 60, dy: -30, orbit: false) }  // a long drag: the target crosses the seam
        let t = model.camera.target
        #expect(t.x >= 0 && t.x < 48 && t.y >= 0 && t.y < 48, "kept in the world: \(t)")
        #expect(model.camera.distance == before.distance && model.camera.yaw == before.yaw && model.camera.pitch == before.pitch)
    }

    @Test func anOrbitDragTurnsTheViewAboutTheTarget() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        let before = model.camera
        model.drag(dx: 30, dy: 20, orbit: true)
        #expect(model.camera.target == before.target && model.camera.distance == before.distance)
        #expect(model.camera.yaw != before.yaw || model.camera.pitch != before.pitch)
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
        model.scroll(delta: 5, precise: true)
        #expect(model.camera.distance < limit, "scrolling up comes closer")
        model.magnify(0.3)
        model.scroll(delta: -1_000_000, precise: false)  // an absurd wheel: bounded, never infinite or NaN
        #expect(model.camera.distance.isFinite && model.camera.distance <= limit + 1e-3)
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
        let header = try #require(play.store.header)
        model.drawableResized(CGSize(width: 1200, height: 600))
        #expect(model.camera == Camera.world(for: header, aspect: 2), "the free rectangle changed shape: refit")
        #expect(model.cameraIsHome)
        model.drag(dx: 10, dy: 10, orbit: false)
        let moved = model.camera
        model.drawableResized(CGSize(width: 400, height: 800))
        #expect(model.camera == moved, "the user's view is kept")
        model.resetCamera()
        #expect(model.cameraIsHome && model.camera == Camera.world(for: header, aspect: model.aspect))
    }

    @Test func aTiltedViewCannotPanTheTargetOutOfTheWorldsHeight() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        model.drag(dx: 0, dy: -400, orbit: true)  // tilt from straight down towards the horizon
        #expect(model.camera.pitch > -Camera.maxPitch + 0.3, "the view is tilted: \(model.camera.pitch)")
        for sign: Float in [1, -1] {
            for _ in 0..<40 {
                model.drag(dx: 0, dy: sign * 300, orbit: false)
                #expect(model.camera.target.z >= 0 && model.camera.target.z <= 24, "\(model.camera.target)")
            }
        }
        // The pan along the screen's up really did lift the target (so the clamp was what held it).
        model.drag(dx: 0, dy: -300, orbit: false)
        #expect(model.camera.target.z == 24 || model.camera.target.z == 0)
    }

    @Test func selectingBumpsTheOverlayOnceAndOnlyOnAChange() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play) = try await model(scratch)
        defer { play.shutdown() }
        let r0 = model.overlay.revision
        model.select(uid: 5_000_000_000)
        #expect(model.selectedUID == 5_000_000_000 && model.overlay.revision == r0 + 1)
        model.select(uid: 5_000_000_000)
        #expect(model.overlay.revision == r0 + 1)
        model.select(uid: nil)
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
