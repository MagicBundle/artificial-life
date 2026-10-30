import AlifeCore
import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeControl

fileprivate final class SphereClock {
    var t: Double = 1000
}

@MainActor
fileprivate struct SphereRig {
    let model: WorldModel
    let play: PlayController
    let commands: URL
    let clock: SphereClock
    var sphere: SphereController { model.sphere }
    var sent: [String] { ((try? String(contentsOf: commands, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init) }

    /// Runs the camera's own motion for up to `seconds` of the hand-cranked clock, 1/60 s a tick; true when it stopped by itself.
    @MainActor
    func settle(seconds: Double = 6) -> Bool {
        var elapsed = 0.0
        while elapsed < seconds {
            clock.t += 1.0 / 60
            elapsed += 1.0 / 60
            if !sphere.advance() { return true }
        }
        return false
    }
}

/// A live world playing the committed sphere fixture (`tests/data/play_sphere.bin`) that records the control lines it receives.
@MainActor
fileprivate func makeSphereRig(_ scratch: Scratch) async throws -> SphereRig {
    let commands = scratch.url.appendingPathComponent("commands.txt")
    let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
    let play = PlayController()
    play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
               arguments: [Repo.streamData.appendingPathComponent("play_sphere.bin").path, commands.path])
    #expect(await eventually { play.store.header != nil && play.store.clock != nil && play.store.frame != nil && play.store.storms != nil })
    // (its own defaults: a test that turns Scroll zooms on must not leave it on for the next run)
    let model = WorldModel(play: play, defaults: UserDefaults(suiteName: "sphere-tests-\(UUID().uuidString)") ?? .standard)
    let clock = SphereClock()
    model.now = { clock.t }
    model.reduceMotion = { false }
    model.setFreeRect(CGRect(x: 0, y: 0, width: 900, height: 600))
    model.deliver([.header, .frame, .map, .storms, .environment, .territory, .material, .soil])
    return SphereRig(model: model, play: play, commands: commands, clock: clock)
}

/// What a click, a key and a flight do to the World's model on a planet: no renderer, no window; the camera of the planet is the
/// `SphereController`'s, and the dock's entry points (selectStorm and the rest) are routed to it.
@MainActor
@Suite struct SphereInteractionTests {
    private let free = CGRect(x: 0, y: 0, width: 900, height: 600)

    private func rig() async throws -> (SphereRig, Scratch) {
        let scratch = try Scratch()
        return (try await makeSphereRig(scratch), scratch)
    }

    @Test func aPlanetWorldIsRecognisedAndOpensOnItsHomeView() async throws {
        let (rig, scratch) = try await rig()
        defer { scratch.cleanup() }
        #expect(rig.model.sphereActive && rig.sphere.isActive)
        #expect(rig.sphere.cameraIsHome && rig.sphere.camera.pose.tilt == 0 && rig.sphere.camera.pose.isNorthUp)
        #expect(rig.sphere.camera.planetRadius == 20 && rig.sphere.camera.lidRadius == 28)
        #expect(rig.sphere.level == .planet)
        // Home faces the side the organisms are on
        let frame = try #require(rig.play.store.frame)
        var sum = SIMD3<Float>.zero
        for i in 0..<frame.n { sum += simd_normalize(frame.position(i)) }
        #expect(SphereFrame.angle(rig.sphere.camera.pose.up, simd_normalize(sum)) < 1e-4)
    }

    @Test func aBoxWorldIsNotAPlanet() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let commands = scratch.url.appendingPathComponent("c.txt")
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
        let play = PlayController()
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url, arguments: [Repo.playFixtureURL("3d").path, commands.path])
        #expect(await eventually { play.store.header != nil && play.store.frame != nil })
        let model = WorldModel(play: play)
        #expect(!model.sphereActive && play.store.header?.isSphere == false)
    }

    @Test func aClickOnAnOrganismSelectsAndFollowsIt() async throws {
        let (rig, scratch) = try await rig()
        defer { scratch.cleanup() }
        let frame = try #require(rig.play.store.frame)
        // fly in over organism 0 at once, then click where it is
        rig.sphere.flyTo(frame.position(0), distance: 14)
        #expect(rig.settle())
        let at = try #require(SpherePicker.screen(frame.position(0), camera: rig.sphere.camera, view: free.size))
        rig.sphere.click(at: at, shift: false, clicks: 1)
        #expect(rig.model.selection == .organism(uid: frame.uid[0]))
        #expect(await eventually { rig.sent.contains("follow \(frame.uid[0])") })
        #expect(rig.model.followTarget == .organism(uid: frame.uid[0]))
        // Esc: the selection and the follow go
        rig.model.perform(.escape)
        #expect(rig.model.selection.isNone && rig.model.followTarget.isNone)
        // a shift click selects only: focus, no follow, the camera stays
        rig.sphere.click(at: at, shift: true, clicks: 1)
        #expect(await eventually { rig.sent.contains("focus \(frame.uid[0])") })
        #expect(rig.model.followTarget.isNone)
    }

    @Test func aClickOnTheFloorMarksTheCellAndAClickOnNothingClearsIt() async throws {
        let (rig, scratch) = try await rig()
        defer { scratch.cleanup() }
        var display = rig.model.display
        display.territory = .off
        rig.model.display = display
        rig.sphere.flyTo(SIMD3(4, 4, 18), distance: 40)
        #expect(rig.settle())
        rig.sphere.click(at: CGPoint(x: 450, y: 300), shift: false, clicks: 1)
        let cell = try #require(rig.sphere.selectedCell)
        let words = try #require(rig.sphere.cellText)
        #expect(words.contains("(cell \(cell))"))
        rig.sphere.click(at: CGPoint(x: 2, y: 2), shift: false, clicks: 1)
        #expect(rig.sphere.selectedCell == nil, "a click on nothing clears the cell")
        // a double click on the floor flies there
        rig.sphere.click(at: CGPoint(x: 600, y: 350), shift: false, clicks: 2)
        #expect(rig.settle())
        #expect(rig.sphere.selectedCell != nil)
    }

    @Test func aStormInTheDockFliesTheCameraToIt() async throws {
        let (rig, scratch) = try await rig()
        defer { scratch.cleanup() }
        let storm = try #require(rig.play.store.storms?.storms.storms.first)
        let direction = try #require(SphereOverlay.direction(of: storm))
        rig.model.selectStorm(storm.id, fly: true, follow: false)
        #expect(rig.model.selection == .storm(id: storm.id))
        #expect(rig.sphere.camera.pose.distance > 0)
        #expect(rig.settle())
        #expect(SphereFrame.angle(rig.sphere.camera.pose.up, direction) < 1e-4, "the camera looks at the storm")
        #expect(rig.sphere.camera.pose.distance <= rig.sphere.flyDistanceRegion() + 1e-3)
        // a double click follows it: the camera stays on it
        rig.model.selectStorm(storm.id, fly: true, follow: true)
        #expect(rig.model.followTarget == .storm(id: storm.id))
        // a chronicle row of the event flies to where it happened (the planet-frame place of the event)
        rig.model.perform(.escape)
        let event = try #require(rig.play.store.chronicle.events.first)
        rig.model.activate(event)
        #expect(rig.settle())
        let want = simd_normalize(SIMD3(Float(event.x!), Float(event.y!), Float(event.z!)))
        #expect(SphereFrame.angle(rig.sphere.camera.pose.up, want) < 1e-3 || rig.model.selection == .storm(id: event.a))
    }

    @Test func homeFliesBackAndLeavesTheSelection() async throws {
        let (rig, scratch) = try await rig()
        defer { scratch.cleanup() }
        let home = rig.sphere.camera.pose
        rig.sphere.flyTo(SIMD3(-1, -1, -1), distance: 20)
        #expect(rig.settle() && !rig.sphere.cameraIsHome)
        rig.model.home()
        #expect(rig.settle())
        let now = rig.sphere.camera.pose
        #expect(SphereFrame.angle(now.up, home.up) < 1e-4 && abs(now.distance - home.distance) < 0.01 && now.isNorthUp)
        #expect(rig.sphere.cameraIsHome)
    }

    @Test func theViewKeysEaseTheTiltAndNorthUpTurnsTheHeading() async throws {
        let (rig, scratch) = try await rig()
        defer { scratch.cleanup() }
        rig.model.perform(.viewPreset(.low))
        #expect(rig.settle())
        #expect(abs(rig.sphere.camera.pose.tilt - ViewPreset.low.tilt) < 1e-3)
        rig.model.perform(.viewPreset(.side))
        #expect(rig.settle())
        #expect(abs(rig.sphere.camera.pose.tilt - SpherePose.maxTilt) < 1e-3, "the side view is as low as the planet's camera goes")
        rig.model.perform(.viewPreset(.top))
        #expect(rig.settle() && rig.sphere.camera.pose.tilt < 1e-3)
        rig.sphere.drag(dx: 120, dy: 0, orbit: true)
        #expect(!rig.sphere.camera.pose.isNorthUp)
        rig.model.perform(.northUp)
        #expect(rig.settle() && rig.sphere.camera.pose.isNorthUp)
        // the 3D button toggles
        rig.sphere.toggle3D()
        #expect(rig.settle() && rig.sphere.camera.pose.tilt > 0.1)
        rig.sphere.toggle3D()
        #expect(rig.settle() && rig.sphere.camera.pose.tilt < 1e-3)
    }

    @Test func theGrabTheScrollAndThePinchMoveThePlanetTheOwnersWay() async throws {
        let (rig, scratch) = try await rig()
        defer { scratch.cleanup() }
        let start = rig.sphere.camera.pose
        // a grab drags the point under the pointer
        let a = CGPoint(x: 450, y: 300), b = CGPoint(x: 520, y: 330)
        let anchor = try #require(rig.sphere.camera.floorDirection(ndc: SpherePicker.ndc(of: a, in: free.size), aspect: rig.sphere.aspect))
        rig.sphere.beginGrab(at: a)
        rig.sphere.grab(to: b)
        let under = try #require(rig.sphere.camera.floorDirection(ndc: SpherePicker.ndc(of: b, in: free.size), aspect: rig.sphere.aspect))
        #expect(SphereFrame.angle(under, anchor) < 3e-4)
        rig.sphere.endGrab()
        // a trackpad's two-finger scroll pans (the content follows the fingers) and does not zoom
        let beforeScroll = rig.sphere.camera.pose
        rig.sphere.scroll(ScrollEvent(dx: 0, dy: 40, precise: true, phase: .began), at: a)
        #expect(abs(rig.sphere.camera.pose.distance - beforeScroll.distance) < 1e-4 && SphereFrame.angle(rig.sphere.camera.pose.up, beforeScroll.up) > 1e-3)
        // a wheel zooms about the pointer
        rig.sphere.scroll(ScrollEvent(dy: 3, precise: false), at: a)
        #expect(rig.sphere.camera.pose.distance < beforeScroll.distance)
        // View > Scroll zooms: the trackpad zooms
        var display = rig.model.display
        display.scrollZooms = true
        rig.model.display = display
        let beforeZoom = rig.sphere.camera.pose.distance
        rig.sphere.scroll(ScrollEvent(dy: 60, precise: true, phase: .began), at: a)
        #expect(rig.sphere.camera.pose.distance < beforeZoom, "scrolling up comes closer")
        // a pinch zooms
        let beforePinch = rig.sphere.camera.pose.distance
        rig.sphere.magnify(0.3, at: a)
        #expect(rig.sphere.camera.pose.distance < beforePinch)
        // out and in again stay within the limits
        rig.sphere.zoom(by: 1e9)
        #expect(rig.sphere.camera.pose.distance <= rig.sphere.camera.maxDistance(aspect: rig.sphere.aspect) + 1e-3)
        rig.sphere.zoom(by: 1e-9)
        #expect(rig.sphere.camera.pose.distance == SpherePose.minDistance)
        _ = start
    }

    @Test func theControlsOverThePlanetAreTheBoxWindowsAndTheMinimapIsTheChart() async throws {
        let (rig, scratch) = try await rig()
        defer { scratch.cleanup() }
        let hud = rig.sphere.hud
        let chart = try #require(hud.minimap, "the minimap is on")
        #expect(abs(chart.height - chart.width / 2) < 1e-6, "the Mollweide chart is twice as wide as high")
        #expect(rig.sphere.pressTarget(at: CGPoint(x: chart.midX, y: chart.midY)) == .minimap)
        #expect(rig.sphere.pressTarget(at: CGPoint(x: 5, y: 300)) == .map)
        // a click on the minimap flies to the place under it
        rig.sphere.hudRelease(.minimap, at: CGPoint(x: chart.minX + chart.width * 0.25, y: chart.midY), clicks: 1, dragged: false)
        #expect(rig.settle())
        let (lat, lon) = SphereFrame.latLon(of: rig.sphere.camera.pose.up)
        #expect(abs(lat) < 2 && abs(lon - -90) < 3, "a quarter of the way across the chart is 90 degrees west")
        // the zoom buttons ease the distance
        let before = rig.sphere.camera.pose.distance
        rig.sphere.hudRelease(.zoomIn, at: CGPoint(x: hud.zoomIn!.midX, y: hud.zoomIn!.midY), clicks: 1, dragged: false)
        #expect(rig.settle())
        #expect(abs(rig.sphere.camera.pose.distance - before * exp(-WorldHUD.zoomStep)) < 0.05 * before)
        // the readout names the place
        #expect(rig.sphere.readout.contains("Equator") || rig.sphere.readout.contains("hemisphere"))
        #expect(rig.sphere.viewText.hasPrefix("Tilt"))
    }

    @Test func theHoverNamesWhatIsUnderThePointerAndTheFloorsPlace() async throws {
        let (rig, scratch) = try await rig()
        defer { scratch.cleanup() }
        let frame = try #require(rig.play.store.frame)
        rig.sphere.flyTo(frame.position(1), distance: 12)
        #expect(rig.settle())
        let at = try #require(SpherePicker.screen(frame.position(1), camera: rig.sphere.camera, view: free.size))
        rig.sphere.hover(at: at)
        #expect(rig.sphere.hover?.text.contains("organism") == true)
        #expect(rig.model.overlay.pointerText?.contains("organism") == true)
        rig.clockAdvance()
        rig.sphere.hover(at: CGPoint(x: 430, y: 280))
        rig.sphere.hover(at: nil)
        #expect(rig.sphere.hover == nil && rig.model.overlay.pointerText == nil)
    }
}

fileprivate extension SphereRig {
    /// Moves the hand-cranked clock past the hover period, so that the next hover is looked up.
    func clockAdvance() { clock.t += 1 }
}
