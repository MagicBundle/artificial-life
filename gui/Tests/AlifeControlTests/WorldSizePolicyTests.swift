import AlifeCore
import Foundation
import Metal
import Testing
import simd

@testable import AlifeControl

/// The model's side of the organism size rule and the look settings (DESIGN §36.7): the rows in view are counted at most ten
/// times a second, the radius they give is smoothed, the brightness is kept between runs. No renderer, no window.
@MainActor
@Suite struct WorldSizePolicyTests {
    private let free = CGRect(x: 0, y: 0, width: 900, height: 700)

    private final class Clock { var t: Double = 1000 }

    /// A controller that has played the 3D fixture (L = 48, height 24, 7 alive rows).
    private func rig(_ scratch: Scratch, defaults: UserDefaults? = nil) async throws -> (WorldModel, PlayController, Clock) {
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do :; done")
        let play = PlayController()
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                   arguments: [Repo.playFixtureURL("3d").path])
        #expect(await eventually { play.store.header != nil && play.store.clock != nil && play.store.frame != nil })
        let suite = try defaults ?? #require(UserDefaults(suiteName: "alife-size-\(UUID().uuidString)"))
        let model = WorldModel(play: play, defaults: suite)
        let clock = Clock()
        model.now = { clock.t }
        model.setFreeRect(free)
        model.goHome()
        model.drawableResized(CGSize(width: 1800, height: 1400))  // a Retina drawable of the free rectangle
        return (model, play, clock)
    }

    private func expectedCover(_ model: WorldModel, _ play: PlayController) throws -> Float {
        let frame = try #require(play.store.frame)
        let rows = OrganismSizePolicy.rowsInView(frame: frame, camera: model.camera, aspect: Float(1800.0 / 1400),
                                                 size: model.worldSize)
        return OrganismSizePolicy.coverRadius(area: 1800 * 1400, rows: rows)
    }

    @Test func theDrawableSetsTheScaleAndTheFirstCountIsAtOnce() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play, _) = try await rig(scratch)
        defer { play.shutdown() }
        #expect(model.pixelsPerPoint == 2, "1400 pixels of a 700 point rectangle")
        #expect(model.drawableSize == CGSize(width: 1800, height: 1400))
        let cover = try expectedCover(model, play)
        #expect(cover.isFinite && model.coverTarget == cover, "the count is of the rows in view")
        #expect(model.coverRadius == cover, "the first value is taken as it is, not filtered from nothing")
        #expect(!model.sizeCountPending)
    }

    @Test func aCountInsideTheTenthOfASecondWaitsForTheDisplayLink() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play, clock) = try await rig(scratch)
        defer { play.shutdown() }
        let first = model.coverTarget
        // Zoom in on two of the rows so that fewer rows are in view; the clock has not moved: no new count yet.
        model.edit { $0.target = SIMD3(11, 10.5, 0); $0.distance = 10 }
        #expect(model.coverTarget == first, "inside the period, the count is not made")
        #expect(model.sizeCountPending && model.isAnimating, "but it is asked for")
        // Half a period later, still pending.
        clock.t += 0.05
        _ = model.advance()
        #expect(model.sizeCountPending && model.coverTarget == first)
        // A period after the first count: the display link makes it.
        clock.t += 0.06
        _ = model.advance()
        #expect(!model.sizeCountPending)
        #expect(model.coverTarget == (try expectedCover(model, play)))
        #expect(model.coverTarget != first, "fewer rows are in view: a larger cover radius")
    }

    @Test func theRadiusFollowsTheCountWithAOnePoleFilterAndSettles() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play, clock) = try await rig(scratch)
        defer { play.shutdown() }
        let start = model.coverRadius
        clock.t += 1
        model.edit { $0.target = SIMD3(11, 10.5, 0); $0.distance = 10 }  // a count at once: a second has passed
        let target = model.coverTarget
        #expect(target > start, "two rows in view against seven")
        #expect(model.coverRadius == start && model.isAnimating, "the radius does not jump")
        var ticks = 0
        while model.isAnimating, ticks < 600 {
            clock.t += 1.0 / 60
            _ = model.advance()
            ticks += 1
            #expect(model.coverRadius >= start - 1e-3 && model.coverRadius <= target + 1e-3, "monotone towards the target")
        }
        #expect(model.coverRadius == target && !model.isAnimating, "settled after \(ticks) ticks")
        #expect(ticks > 20 && ticks < 120, "about a second at a time constant of a quarter: \(ticks)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1", "set ALIFE_GPU_TESTS=1 (a Metal device)"))
    func aRendererAttachedLaterIsGivenTheCurrentSizePolicy() async throws {
        // Review of A0, finding 3: a new view on the existing model (the model outlives its view) drew at the absolute cap until
        // the first zoom, because only a change of the radius was pushed.
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play, _) = try await rig(scratch)
        defer { play.shutdown() }
        let device = try #require(MTLCreateSystemDefaultDevice(), "no Metal device")
        let first = try Renderer(device: device)
        model.attach(renderer: first, view: WorldMTKView(frame: .zero, device: nil))
        #expect(first.worldDraw.coverRadius == model.coverRadius && first.worldDraw.pixelsPerPoint == model.pixelsPerPoint)
        let second = try Renderer(device: device)
        #expect(second.worldDraw.coverRadius == .infinity, "a fresh renderer knows nothing")
        model.attach(renderer: second, view: WorldMTKView(frame: .zero, device: nil))
        #expect(model.coverRadius.isFinite)
        #expect(second.worldDraw.coverRadius == model.coverRadius, "the radius: \(second.worldDraw.coverRadius) vs \(model.coverRadius)")
        #expect(second.worldDraw.pixelsPerPoint == model.pixelsPerPoint)
    }

    @Test func thePhysicalSizeCountsNothingAndHasNoCap() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play, _) = try await rig(scratch)
        defer { play.shutdown() }
        #expect(model.coverRadius.isFinite && model.drawnRadiusCap.isFinite)
        model.display.sizeMode = .physical
        #expect(model.coverRadius == .infinity && model.drawnRadiusCap == .infinity)
        model.display.sizeMode = .capped
        #expect(model.coverRadius.isFinite && model.coverRadius == model.coverTarget)
        model.display.agents = false
        #expect(model.coverRadius == .infinity, "no organism layer, nothing to cap")
    }

    @Test func thePickersCapIsInPointsAndFollowsTheAbsoluteRule() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play, _) = try await rig(scratch)
        defer { play.shutdown() }
        // At the planet pose (far: 0.2 L or more) the absolute cap is 4.5 pt at the drawable's scale, and the 7-row fixture's
        // cover radius is much larger.
        #expect(model.camera.distance >= 0.2 * model.worldSize)
        #expect(abs(model.drawnRadiusCap - OrganismSizePolicy.farRadiusPoints) < 1e-4)
        // Closer than 0.08 L there is no absolute cap: the cover radius (in points) rules.
        model.edit { $0.distance = 0.05 * model.worldSize }
        #expect(abs(model.drawnRadiusCap - max(model.coverRadius, 2.5) / model.pixelsPerPoint) < 1e-3)
    }

    @Test func theBrightnessItemKnowsWhetherAMapIsInView() async throws {
        // Review of A0, finding 8: Brightness scales the weather map's colours, so with no map in view the menu disables it.
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, play, _) = try await rig(scratch)
        defer { play.shutdown() }
        #expect(play.store.map != nil && model.mapShown, "the 3D fixture holds a weather map")
        #expect(ViewMenuInputs(play: play, model: model).mapShown)
        model.display.map = false
        #expect(!ViewMenuInputs(play: play, model: model).mapShown, "the layer off: no map in view")
        model.display.map = true
        #expect(ViewMenuInputs(play: play, model: model).mapShown)
    }

    @Test func theBrightnessIsKeptBetweenRuns() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let suite = try #require(UserDefaults(suiteName: "alife-exposure-\(UUID().uuidString)"))
        let (first, play, _) = try await rig(scratch, defaults: suite)
        defer { play.shutdown() }
        #expect(first.display.exposure == 1, "the default")
        first.display.exposure = 1.25
        #expect(suite.double(forKey: WorldModel.exposureKey) == 1.25)
        let second = WorldModel(play: play, defaults: suite)
        #expect(second.display.exposure == 1.25 && second.display.territory == .tint && second.display.sizeMode == .capped)
        // An absurd stored value is bounded, a non-number ignored.
        suite.set(40.0, forKey: WorldModel.exposureKey)
        #expect(WorldModel(play: play, defaults: suite).display.exposure == 2)
        suite.set("bright", forKey: WorldModel.exposureKey)
        #expect(WorldModel(play: play, defaults: suite).display.exposure == 1)
        suite.set(-3.0, forKey: WorldModel.exposureKey)
        #expect(WorldModel(play: play, defaults: suite).display.exposure == 1)
    }
}
