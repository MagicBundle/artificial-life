import AlifeCore
import AppKit
import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeControl

/// The camera controls on the globe, through the hosted `WorldView` (DESIGN §36.10; the coordinator's promise to the owner: "on the globe
/// the same controls tilt and turn around the planet"). `globeEnabled` is the model's switch for the globe (off until the renderer draws
/// it): these tests switch it on and drive the keys, the 3D button, the drag, the compass and the grab, and read the pose and the
/// projector the picker uses.
extension WorldInputTests {
    private func globeWorld() async throws -> HostedWorld {
        let world = try await HostedWorld(stream: Repo.playFixtureURL("3d"), dockCollapsed: true)
        world.model.reduceMotion = { false }
        world.model.display.minimap = true
        world.model.dismissHint()
        world.model.globeEnabled = true
        world.model.goHome()
        world.settle()
        return world
    }

    @discardableResult
    private func arrive(_ model: WorldModel, frames: Int = 900) -> Int {
        for n in 1...frames where !model.advanceCamera(dt: 1.0 / 60) { return n }
        return frames
    }

    @Test func homeIsTheGlobeOverviewAndTheProjectorIsTheMorphs() async throws {
        let world = try await globeWorld()
        defer { world.finish() }
        let model = world.model
        let dg = Morph.globeDistance(L: model.worldSize, fovY: model.pose.fovY, aspect: model.aspect)
        #expect(abs(model.pose.distance - dg) < 1e-3 && model.pose.tilt == 0 && model.pose.heading == 0)
        let projector = model.projector(size: model.freeRect.size)
        #expect(projector.isMorphing && projector.s == 1 && projector.level == .planet && model.level == .planet)
        // a click at the middle names the place under the middle
        let middle = CGPoint(x: model.freeRect.width / 2, y: model.freeRect.height / 2)
        let place = try #require(projector.world(at: middle))
        #expect(abs(Torus.wrapDelta(place.x - model.pose.x, size: model.worldSize)) < 0.05)
        // the flat map: the same model with the switch off keeps the flat Home
        model.globeEnabled = false
        model.goHome()
        #expect(model.pose.distance > dg && !model.projector(size: model.freeRect.size).isMorphing)
    }

    @Test func theStandardViewsAndTheThreeDButtonTiltAboutThePlanetAndKeepItsDistance() async throws {
        let world = try await globeWorld()
        defer { world.finish() }
        let model = world.model
        let dg = model.pose.distance
        for preset in ViewPreset.allCases + [ViewPreset.top] {
            #expect(world.press(preset.key))
            arrive(model)
            #expect(abs(model.pose.tilt - preset.tilt) < 1e-5, "\(preset): tilt \(model.pose.tilt)")
            #expect(abs(model.pose.distance - dg) < 1e-2, "\(preset): the globe keeps its distance (\(model.pose.distance))")
            let projector = model.projector(size: model.freeRect.size)
            #expect(projector.s == 1)
            // the look-at point stays at the middle of the view, at any tilt
            let middle = try #require(projector.vantage?.screen(e: 0, n: 0, z: model.pose.lookZ, in: model.freeRect.size))
            #expect(abs(middle.point.x - model.freeRect.width / 2) < 0.5 && abs(middle.point.y - model.freeRect.height / 2) < 0.5)
        }
        // Side: the planet below, the sky above the limb
        #expect(world.press("s"))
        arrive(model)
        let side = model.projector(size: model.freeRect.size)
        #expect(side.world(at: CGPoint(x: model.freeRect.width / 2, y: model.freeRect.height * 0.9)) != nil)
        #expect(side.world(at: CGPoint(x: model.freeRect.width / 2, y: model.freeRect.height * 0.1)) == nil)
        // the 3D button: from Side to Top and back, the distance untouched, not held by the flat map's eye reach
        model.toggle3D()
        arrive(model)
        #expect(model.pose.tilt == 0 && abs(model.pose.distance - dg) < 1e-2)
        model.toggle3D()
        arrive(model)
        #expect(model.pose.tilt > 0.3 && abs(model.pose.distance - dg) < 1e-2)
    }

    @Test func anOptionDragAndTheCompassTurnAndTiltTheGlobe() async throws {
        let world = try await globeWorld()
        defer { world.finish() }
        let model = world.model
        model.drag(dx: 60, dy: -50, orbit: true)
        #expect(abs(model.pose.heading + 60 * WorldPose.radiansPerPoint) < 1e-5 && abs(model.pose.tilt - 50 * WorldPose.radiansPerPoint) < 1e-5)
        let centre = model.projector(size: model.freeRect.size).vantage!.centreDirection
        model.northUp()
        arrive(model)
        #expect(abs(model.pose.heading) < 1e-5, "the compass: north up")
        #expect(simd_length(model.projector(size: model.freeRect.size).vantage!.centreDirection - centre) < 1e-5, "turning keeps the place")
    }

    @Test func aDragTurnsTheGlobeAndTheGrabHoldsThePointExactly() async throws {
        let world = try await globeWorld()
        defer { world.finish() }
        let model = world.model
        let size = model.freeRect.size
        let start = CGPoint(x: size.width * 0.42, y: size.height * 0.5)
        model.beginGrab(at: start)
        let anchor = try #require(model.grabAnchor)
        for step in 1...6 {
            model.grab(to: CGPoint(x: start.x + Double(step) * 18, y: start.y + Double(step) * 7))
        }
        model.endGrab()
        let end = CGPoint(x: start.x + 108, y: start.y + 42)
        let projector = model.projector(size: size)
        let at = try #require(projector.screen(SIMD3(anchor.x, anchor.y, anchor.z)))
        #expect(hypot(at.x - end.x, at.y - end.y) < 0.5, "the grabbed point is under the pointer: \(at) \(end)")
        #expect(projector.s == 1 && model.pose.distance > 0)
    }

    @Test func theWheelZoomsAboutThePointerOntoTheFlatCloseUpAndBackAlongTheMorph() async throws {
        let world = try await globeWorld()
        defer { world.finish() }
        let model = world.model
        let size = model.freeRect.size
        let pointer = CGPoint(x: size.width * 0.55, y: size.height * 0.45)
        let before = try #require(model.projector(size: size).world(at: pointer))
        var last = model.projector(size: size).s
        for _ in 0..<40 {
            model.zoom(by: 0.93, at: pointer)
            let projector = model.projector(size: size)
            #expect(projector.s <= last + 1e-6, "zooming in only lowers s")
            last = projector.s
            let under = try #require(projector.world(at: pointer))
            #expect(abs(Torus.wrapDelta(under.x - before.x, size: model.worldSize)) < 0.05
                    && abs(Torus.wrapDelta(under.y - before.y, size: model.worldSize)) < 0.05, "the point under the pointer stays (s \(projector.s))")
        }
        #expect(last == 0, "the close-up is flat: \(last)")
        for _ in 0..<80 { model.zoom(by: 1.07, at: pointer) }
        #expect(model.pose.distance <= Morph.maxDistanceFraction * model.worldSize + 1e-3)
        #expect(model.projector(size: size).s == 1)
    }

    @Test func aHomeFlightOnTheGlobeIsTheGreatCircle() async throws {
        let world = try await globeWorld()
        defer { world.finish() }
        let model = world.model
        model.editPose { $0.x = 10; $0.y = 100 }
        model.home()
        #expect(model.flight?.path.greatCircle == true)
        arrive(model)
        let dg = Morph.globeDistance(L: model.worldSize, fovY: model.pose.fovY, aspect: model.aspect)
        #expect(abs(model.pose.distance - dg) < 1e-2 && model.cameraIsHome)
    }
}
