import AlifeCore
import AppKit
import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeControl

/// What outlives the input on the globe, Home as the globe, the region distance (DESIGN §36.10): the inertia of a flick, the idle spin,
/// Keep north up, reduced motion. Through the hosted `WorldView`, with the model's clock in the test's hands.
extension WorldInputTests {
    private func motionWorld(globe: Bool? = true) async throws -> (HostedWorld, Clock) {
        let world = try await HostedWorld(stream: Repo.playFixtureURL("3d"), dockCollapsed: true, globe: globe)
        let clock = Clock()
        world.model.now = { clock.t }
        world.model.reduceMotion = { false }
        world.model.dismissHint()
        world.model.display.globe = true
        world.model.goHome()
        world.settle()
        return (world, clock)
    }

    final class Clock { var t = 100.0 }

    /// A drag of the left button from the middle: `steps` moves of `dx` points, `dt` seconds apart, ending `hold` seconds after the last.
    private func flick(_ world: HostedWorld, _ clock: Clock, steps: Int, dx: CGFloat, dt: Double, hold: Double = 0) {
        let model = world.model
        let start = CGPoint(x: model.freeRect.midX - 100, y: model.freeRect.midY)
        model.beginGrab(at: start)
        for k in 1...steps {
            clock.t += dt
            model.grab(to: CGPoint(x: start.x + dx * CGFloat(k), y: start.y))
        }
        clock.t += hold
        model.endGrab()
    }

    @Test func theAppOpensOnTheGlobeUnlessThePlayerChose() throws {
        let name = "alife-globe-default-\(UUID().uuidString)"
        let suite = try #require(UserDefaults(suiteName: name))
        defer { suite.removePersistentDomain(forName: name) }
        let play = PlayController()
        let fresh = WorldModel(play: play, defaults: suite)
        #expect(!fresh.display.globe, "a model made by a test is flat until it asks")
        fresh.adoptGlobeDefault()
        #expect(fresh.display.globe && fresh.globeEnabled)
        #expect(suite.object(forKey: WorldModel.globeKey) == nil, "the default is not stored as the player's choice")
        fresh.display.globe = false
        #expect(suite.object(forKey: WorldModel.globeKey) as? Bool == false, "the player's choice is stored")
        suite.set(false, forKey: WorldModel.globeKey)
        let chosen = WorldModel(play: play, defaults: suite)
        chosen.adoptGlobeDefault()
        #expect(!chosen.display.globe && !chosen.globeEnabled, "a stored No stays")
    }

    @Test func aWorldWithNoMapInViewKeepsTheFlatHomeAndTheFlatPoseRules() async throws {
        let (world, _) = try await motionWorld()
        defer { world.finish() }
        let model = world.model
        let globeDistance = model.pose.distance
        #expect(model.globeActive && model.vantage().isGlobe)
        // the map layer off: the view is flat, and so are Home, the limit and the level
        model.display.map = false
        world.settle()
        #expect(!model.globeActive && !model.vantage().isGlobe)
        let flatFit = model.pose.distanceLimit(size: model.worldSize, aspect: model.aspect)
        #expect(model.pose.distance > globeDistance * 1.5, "Home refits to the flat one-world fit (\(model.pose.distance))")
        #expect(abs(model.pose.distance - flatFit) < 1e-3 && abs(model.distanceLimit - flatFit) < 1e-3)
        model.goHome()
        #expect(abs(model.pose.distance - flatFit) < 1e-3 && !model.projector(size: model.freeRect.size).isMorphing)
        #expect(model.level != .planet)
        // and the map back on: Home is the globe again
        model.display.map = true
        world.settle()
        #expect(model.globeEnabled && abs(model.pose.distance - globeDistance) < 1e-3)
    }

    @Test func aFlickOfTheGlobeKeepsAnInertiaThatDecaysAndAnyInputStops() async throws {
        let (world, clock) = try await motionWorld()
        defer { world.finish() }
        let model = world.model
        #expect(model.vantage().s == 1)
        flick(world, clock, steps: 6, dx: 14, dt: 0.03)
        #expect(model.motion.isCoasting, "a drag of 0.18 s at 470 points a second flicks")
        let x0 = model.pose.x
        var travelled: [Float] = []
        for _ in 0..<40 {
            clock.t += 1.0 / 60
            _ = model.advanceCamera(dt: 1.0 / 60)
            travelled.append(Torus.wrapDelta(model.pose.x - x0, size: model.worldSize))
        }
        #expect(travelled.last! != 0, "the globe turned on after the release")
        // the speed decays: later steps are shorter than earlier ones
        let early = abs(travelled[5] - travelled[0]), late = abs(travelled[39] - travelled[34])
        #expect(late < early * 0.4, "early \(early) late \(late)")
        for _ in 0..<600 where model.motion.isCoasting { _ = model.advanceCamera(dt: 1.0 / 60) }
        #expect(!model.motion.isCoasting, "and it ends")
        // any input stops it where it is
        flick(world, clock, steps: 6, dx: 14, dt: 0.03)
        #expect(model.motion.isCoasting)
        model.zoom(by: 0.9)
        #expect(!model.motion.isCoasting)
        flick(world, clock, steps: 6, dx: 14, dt: 0.03)
        model.beginGrab(at: CGPoint(x: model.freeRect.midX, y: model.freeRect.midY))
        #expect(!model.motion.isCoasting)
        model.endGrab()
    }

    @Test func noInertiaAfterAShortDragASlowOneReducedMotionOrOnTheFlatMap() async throws {
        let (world, clock) = try await motionWorld()
        defer { world.finish() }
        let model = world.model
        flick(world, clock, steps: 2, dx: 30, dt: 0.03)
        #expect(!model.motion.isCoasting, "a drag of 0.06 s is not a flick")
        flick(world, clock, steps: 20, dx: 0.5, dt: 0.02)
        #expect(!model.motion.isCoasting, "a slow drag")
        flick(world, clock, steps: 6, dx: 14, dt: 0.03, hold: 0.3)
        #expect(!model.motion.isCoasting, "the pointer rested before the release")
        model.reduceMotion = { true }
        flick(world, clock, steps: 6, dx: 14, dt: 0.03)
        #expect(!model.motion.isCoasting, "reduced motion")
        model.reduceMotion = { false }
        model.display.globe = false
        model.goHome()
        flick(world, clock, steps: 6, dx: 14, dt: 0.03)
        #expect(!model.motion.isCoasting, "the flat map has none")
    }

    @Test func keepNorthUpEasesTheHeadingBackAfterADragOfTheGlobe() async throws {
        let (world, clock) = try await motionWorld()
        defer { world.finish() }
        let model = world.model
        model.pose.heading = 0.6
        model.pushCamera()
        model.beginGrab(at: CGPoint(x: model.freeRect.midX, y: model.freeRect.midY))
        clock.t += 0.05
        model.grab(to: CGPoint(x: model.freeRect.midX + 40, y: model.freeRect.midY + 5))
        let held = model.pose.heading
        #expect(abs(held) > 0.1)
        model.endGrab()
        #expect(model.headingEase != nil)
        var frames = 0
        for n in 1...120 where !model.advanceCamera(dt: 1.0 / 60) { frames = n; break }
        #expect(frames >= 22 && frames <= 28 && model.pose.heading == 0, "0.4 s: \(frames) frames, heading \(model.pose.heading)")
        // off: the heading stays
        model.display.keepNorthUp = false
        model.pose.heading = 0.6
        model.beginGrab(at: CGPoint(x: model.freeRect.midX, y: model.freeRect.midY))
        model.grab(to: CGPoint(x: model.freeRect.midX + 40, y: model.freeRect.midY + 5))
        model.endGrab()
        #expect(model.headingEase == nil && model.pose.heading != 0)
        // reduced motion: at once
        model.display.keepNorthUp = true
        model.reduceMotion = { true }
        model.beginGrab(at: CGPoint(x: model.freeRect.midX, y: model.freeRect.midY))
        model.grab(to: CGPoint(x: model.freeRect.midX + 40, y: model.freeRect.midY + 5))
        model.endGrab()
        #expect(model.pose.heading == 0 && model.headingEase == nil)
    }

    @Test func theIdleSpinTurnsTheGlobeWestToEastOnlyWhenRestedAndNotBusy() async throws {
        let (world, clock) = try await motionWorld()
        defer { world.finish() }
        let model = world.model
        #expect(!model.spinWanted, "off by default")
        model.display.spin = true
        #expect(model.spinWanted && model.isAnimating)
        let x0 = model.pose.x
        clock.t += 1
        _ = model.advanceCamera(dt: 1.0 / 60)
        #expect(model.pose.x == x0, "an input just now: it waits")
        clock.t += GlobeMotion.spinRest + 0.1
        for _ in 0..<60 { _ = model.advanceCamera(dt: 1.0 / 60) }
        let turned = Torus.wrapDelta(model.pose.x - x0, size: model.worldSize)
        #expect(abs(turned + model.worldSize * 0.5 / 360) < 1e-3, "0.5 degree of longitude in a second, going west: \(turned)")
        // a drag stops it for the rest
        model.beginGrab(at: CGPoint(x: model.freeRect.midX, y: model.freeRect.midY))
        model.endGrab()
        let x1 = model.pose.x
        _ = model.advanceCamera(dt: 1.0 / 60)
        #expect(model.pose.x == x1)
        // not while following, not reduced, not on the flat map
        clock.t += 10
        model.reduceMotion = { true }
        #expect(!model.spinWanted)
        model.reduceMotion = { false }
        model.display.globe = false
        model.goHome()
        #expect(!model.spinWanted)
    }

    @Test func theRegionFlyToIsAFifteenthOfThePlanetWithTheGlobeAndTheFlatMapsOtherwise() async throws {
        let (world, _) = try await motionWorld()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        #expect(abs(model.flyDistance(.region) - max(0.15 * L, WorldLevels.minDistance)) < 1e-3)
        #expect(abs(model.flyDistance(.organism) - max(0.08 * L, WorldLevels.minDistance)) < 1e-3)
        model.display.globe = false
        #expect(model.flyDistance(.region) > max(0.15 * L, WorldLevels.minDistance), "the flat map's own (0.2 half extents)")
    }
}
