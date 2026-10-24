@testable import AlifeCore
import AppKit
import Combine
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// A menu or a pop-up that is evaluated again while it is open is rebuilt by AppKit under the pointer: it flashes in and out of
/// existence (the owner's report on the View menu). `PlayController` publishes with every record that is not a frame (the clock four
/// times a second), `StreamController` with every frame. These tests host the real views offscreen, deliver what a world plays and
/// count the evaluations of their bodies (`ViewProbe`): a menu is evaluated again when what it shows changes, never when only the
/// clock does (DESIGN §33.4.4). Serialized: the probe is one hook for the process.
@MainActor
@Suite(.serialized) struct MenuStabilityTests {
    // MARK: Rig

    /// Evaluations of the probed bodies since `reset()`.
    private final class Counts {
        var byName: [String: Int] = [:]
        subscript(_ name: String) -> Int { byName[name, default: 0] }
        func reset() { byName = [:] }
    }

    /// Counts the evaluations of the views made for `owner` (the suites of the other tests host views too, at the same time).
    private func counting(_ owner: AnyObject) -> Counts {
        let counts = Counts()
        let id = ObjectIdentifier(owner)
        ViewProbe.onEvaluate = { name, who in
            if who == id { counts.byName[name, default: 0] += 1 }
        }
        return counts
    }

    /// A view in a window that is never shown, kept alive with it.
    private struct Hosted {
        let host: NSView
        let window: NSWindow
    }

    private func hosted<V: View>(_ view: V, width: CGFloat, height: CGFloat) -> Hosted {
        let host = NSHostingView(rootView: view.frame(width: width, height: height, alignment: .topLeading))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: height), styleMask: [.borderless], backing: .buffered, defer: true)
        window.contentView = host
        host.frame = NSRect(x: 0, y: 0, width: width, height: height)
        host.layoutSubtreeIfNeeded()
        return Hosted(host: host, window: window)
    }

    /// A world that has sent its header and is live (the fixture of the G3b stream), and the window model over it.
    private func rig(_ scratch: Scratch) async throws -> (play: PlayController, model: WorldModel) {
        let commands = scratch.url.appendingPathComponent("commands.txt")
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
        let play = PlayController()
        let suite = try #require(UserDefaults(suiteName: "alife-menu-\(UUID().uuidString)"))
        let model = WorldModel(play: play, defaults: suite)
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                   arguments: [Repo.playFixtureURL("3d_g3b").path, commands.path])
        #expect(await eventually { play.store.header != nil && play.store.plans.records.count == 7 && play.store.clock?.reason == .alert })
        return (play, model)
    }

    /// The world plays on: `n` clocks, each a little later and otherwise the same (the host sends one every 250 ms).
    private func tick(_ play: PlayController, from clock: PlayClock, _ n: Int, flags: UInt16? = nil) async throws {
        for i in 0..<n {
            var next = clock
            next.head = RecordHead(step: clock.head.step + UInt64(i + 1), seq: 0, version: 1, flags: 0)
            if let flags { next.flags = flags }
            var delivery = StreamDelivery()
            delivery.records["CLCK"] = .clck(next)
            play.reduce(delivery)
            try await Task.sleep(for: .milliseconds(15))
        }
    }

    private func frames(_ play: PlayController, _ n: Int, colour: Int = 0, from step: Int = 100) async throws {
        for i in 0..<n {
            var frame = TestFrames.frame([SIMD3<Float>(1, 1, 1)], step: UInt64(step + i))
            frame.colorIndex = colour
            var delivery = StreamDelivery()
            delivery.records["FRMA"] = .frma(frame)
            play.reduce(delivery)
            try await Task.sleep(for: .milliseconds(15))
        }
    }

    private func dock(_ play: PlayController, _ model: WorldModel) -> Hosted {
        hosted(ControlCenterView(play: play, model: model, collapsed: .constant(false), side: .right, height: 700), width: 360, height: 700)
    }

    // MARK: The View menu

    @Test func theViewMenuIsNotRebuiltWhileTheWorldPlays() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown(); ViewProbe.onEvaluate = nil }
        let counts = counting(play)
        let view = dock(play, model)
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] >= 1, "the menu is built once")
        let clock = try #require(play.store.clock)
        counts.reset()
        let changes = model.viewMenu.changes
        var published = 0
        let sub = play.objectWillChange.sink { published += 1 }
        defer { sub.cancel() }

        try await frames(play, 20)
        try await tick(play, from: clock, 20)
        model.tell("a note")  // the window model publishes too
        model.tell(nil)
        try await Task.sleep(for: .milliseconds(200))

        #expect(published >= 20, "the controller did publish with each clock: this is what rebuilt the menu")
        #expect(counts["ViewMenu"] == 0, "the menu was evaluated again \(counts["ViewMenu"]) times while the world played")
        #expect(model.viewMenu.changes == changes)
        withExtendedLifetime(view) {}
    }

    @Test func aChangeOfWhatTheMenuShowsRebuildsItOnce() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown(); ViewProbe.onEvaluate = nil }
        let counts = counting(play)
        let view = dock(play, model)
        try await Task.sleep(for: .milliseconds(200))
        let modes = try #require(play.store.header?.colorModes)
        try #require(modes.count >= 3)

        // The colour mode: the frame says which it is, and a frame does not publish.
        try await frames(play, 3, colour: 0)
        try await Task.sleep(for: .milliseconds(200))
        #expect(model.viewMenu.value.current == modes[0])
        counts.reset()
        try await frames(play, 3, colour: 2, from: 200)
        #expect(await eventually { model.viewMenu.value.current == modes[2] })
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] == 1)
        try await frames(play, 10, colour: 2, from: 300)
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] == 1, "frames of the same mode change nothing")

        // A layer.
        counts.reset()
        model.display.bonds.toggle()
        #expect(await eventually { model.viewMenu.value.display.bonds == model.display.bonds })
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] == 1)

        // The chapter pause, from the clock (the flag is bit 0 of the clock's flags).
        counts.reset()
        let clock = try #require(play.store.clock)
        let armed = clock.chapterPauseArmed
        try await tick(play, from: clock, 3, flags: clock.flags ^ 1)
        #expect(await eventually { model.viewMenu.value.chapterPause == !armed })
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] == 1)

        // The labels switch.
        counts.reset()
        play.send(.labels(!play.labelsOn))
        #expect(await eventually { model.viewMenu.value.labelsOn == play.labelsOn })
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] == 1)
        withExtendedLifetime(view) {}
    }

    @Test func theLookSettingsAreFieldsOfTheMenuAndZoomingNeverRebuildsIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown(); ViewProbe.onEvaluate = nil }
        let counts = counting(play)
        let view = dock(play, model)
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] >= 1)
        try await frames(play, 3)  // a frame in the store, so that the camera has rows to count
        model.setFreeRect(CGRect(x: 0, y: 0, width: 900, height: 700))
        model.goHome()
        model.drawableResized(CGSize(width: 1800, height: 1400))
        try await Task.sleep(for: .milliseconds(200))

        // The camera is not an input: a long zoom inside one level (a continuous distance, which v2 keys the map's opacity,
        // the instances and the size rule on) rebuilds nothing, and neither does the count of the rows in view.
        counts.reset()
        let changes = model.viewMenu.changes
        let level = model.level
        for i in 0..<30 {
            model.zoom(by: i % 2 == 0 ? 1.03 : 0.97)
            try await Task.sleep(for: .milliseconds(10))
        }
        model.drag(dx: 40, dy: 10, orbit: false)
        try await Task.sleep(for: .milliseconds(200))
        #expect(model.level == level, "the zoom stayed inside one level")
        #expect(counts["ViewMenu"] == 0, "the menu was evaluated again \(counts["ViewMenu"]) times while the camera moved")
        #expect(model.viewMenu.changes == changes)

        // Nor does anything else the camera does, the level of the view included (v2 makes it a function of the continuous distance, and
        // the menu's "Agent points" no longer depends on it): a zoom across the levels, a grab, a scroll that pans, a turn, north up, a
        // flight to Home and the clock of the animation.
        counts.reset()
        model.goHome()
        var levels: [WorldLevel] = [model.level]
        for _ in 0..<60 {
            model.zoom(by: 0.93, at: CGPoint(x: 300, y: 200))
            if levels.last != model.level { levels.append(model.level) }
        }
        #expect(levels.suffix(2) == [.region, .organism], "the zoom crossed the levels (this fixture has no weather map: no planet level): \(levels)")
        model.beginGrab(at: CGPoint(x: 400, y: 300))
        for i in 0..<20 { model.grab(to: CGPoint(x: 400 + 5 * Double(i), y: 300 - 3 * Double(i))) }
        model.endGrab()
        model.scroll(ScrollEvent(dx: 5, dy: 9, precise: true, phase: .changed, time: 1), at: CGPoint(x: 100, y: 100))
        model.drag(dx: 30, dy: -10, orbit: true)
        model.northUp()
        model.home()
        for _ in 0..<200 where !model.advanceCamera(dt: 1.0 / 60) { break }
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] == 0, "the menu was evaluated again \(counts["ViewMenu"]) times while the camera moved, flew and crossed levels")
        #expect(model.viewMenu.changes == changes)

        // Brightness, Territory and Organism size are fields of the inputs (through the display): one evaluation each.
        let edits: [(String, () -> Void, () -> Bool)] = [
            ("Brightness", { model.display.exposure = 1.25 }, { model.viewMenu.value.display.exposure == 1.25 }),
            ("Territory", { model.display.territory = .borders }, { model.viewMenu.value.display.territory == .borders }),
            ("Organism size", { model.display.sizeMode = .physical }, { model.viewMenu.value.display.sizeMode == .physical }),
            ("Scroll zooms", { model.display.scrollZooms = true }, { model.viewMenu.value.display.scrollZooms }),
            // The layer is a field of the inputs and so is whether a map is in view (it disables Brightness, A0 review finding 8).
            ("Weather map", { model.display.map = false },
             { !model.viewMenu.value.display.map && model.viewMenu.value.mapShown == model.mapShown }),
        ]
        for (name, change, shown) in edits {
            counts.reset()
            change()
            #expect(await eventually { shown() }, "\(name) reaches the menu's inputs")
            try await Task.sleep(for: .milliseconds(200))
            #expect(counts["ViewMenu"] == 1, "\(name): \(counts["ViewMenu"]) evaluations")
        }
        withExtendedLifetime(view) {}
    }

    @Test func theControlsOverTheMapNeverRebuildTheMenuAndTheMinimapAndTheHeightColourAreFieldsOfIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown(); ViewProbe.onEvaluate = nil }
        let counts = counting(play)
        let view = dock(play, model)
        try await Task.sleep(for: .milliseconds(200))
        try await frames(play, 3)
        model.setFreeRect(CGRect(x: 0, y: 0, width: 900, height: 700))
        model.goHome()
        model.drawableResized(CGSize(width: 1800, height: 1400))
        try await Task.sleep(for: .milliseconds(200))

        // The pointer over the map and over each control, a press of each, the zoom buttons easing, the hint shown and taken away, a wheel
        // and a pinch over the minimap's place: the pointer, the controls and the hint are not inputs of the menu.
        counts.reset()
        let changes = model.viewMenu.changes
        let hud = model.hud
        for p in [CGPoint(x: 300, y: 300), CGPoint(x: 310, y: 305)] + hud.rects.map({ CGPoint(x: $0.rect.midX, y: $0.rect.midY) }) { model.hover(at: p) }
        model.hover(at: nil)
        model.showHint()
        model.dismissHint()
        for (target, rect) in hud.rects {
            model.hudPress(target)
            model.hudRelease(target, at: CGPoint(x: rect.midX, y: rect.midY), clicks: 1, dragged: false)
        }
        model.zoomStep(closer: true)
        model.scroll(ScrollEvent(dy: 1, precise: false, time: 1), at: CGPoint(x: 800, y: 600))
        model.magnify(0.1, at: CGPoint(x: 800, y: 600))
        for _ in 0..<60 where !model.advanceCamera(dt: 1.0 / 60) { break }
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] == 0, "the menu was evaluated again \(counts["ViewMenu"]) times by the pointer, the controls and the hint")
        #expect(model.viewMenu.changes == changes)

        // The minimap and the height colour are fields of the inputs: one evaluation each.
        let edits: [(String, () -> Void, () -> Bool)] = [
            ("Minimap", { model.display.minimap = false }, { !model.viewMenu.value.display.minimap }),
            ("Height colour", { model.setColourMode(HeightColour.modeName) },
             { model.viewMenu.value.current == HeightColour.modeName && model.viewMenu.value.display.heightColour }),
        ]
        for (name, change, shown) in edits {
            counts.reset()
            change()
            #expect(await eventually { shown() }, "\(name) reaches the menu's inputs")
            try await Task.sleep(for: .milliseconds(200))
            #expect(counts["ViewMenu"] == 1, "\(name): \(counts["ViewMenu"]) evaluations")
        }
        #expect(model.viewMenu.value.colours.last?.mode == HeightColour.modeName)
        withExtendedLifetime(view) {}
    }

    @Test func theCameraControlsNeverRebuildTheViewMenuAndItsViewAngleItemsAreTheStandardViews() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown(); ViewProbe.onEvaluate = nil }
        let counts = counting(play)
        let view = dock(play, model)
        try await Task.sleep(for: .milliseconds(200))
        try await frames(play, 3)
        model.setFreeRect(CGRect(x: 0, y: 0, width: 900, height: 700))
        model.reduceMotion = { false }
        model.goHome()
        model.drawableResized(CGSize(width: 1800, height: 1400))
        try await Task.sleep(for: .milliseconds(200))
        // The items are the four standard views, each with its key, and the 3D button; the submenu is on in a 3D world.
        #expect(ViewMenu.viewAngleTitles == ["Top (T)", "Oblique (O)", "Low (L)", "Side (S)", "Tilt on or off (3D button)"])
        #expect(model.viewMenu.value.tiltable)

        // Every way of changing the angle: the keys' actions, the strip, the 3D button, the handle and the compass dial, the eases, and the
        // readouts they change. The camera is not an input of the menu, and neither is its angle.
        counts.reset()
        let changes = model.viewMenu.changes
        let hud = model.hud
        for preset in ViewPreset.allCases + ViewPreset.allCases.reversed() {
            model.perform(.viewPreset(preset))
            for _ in 0..<40 where !model.advanceCamera(dt: 1.0 / 60) { break }
        }
        for entry in hud.presets {
            model.hudPress(.preset(entry.preset), at: CGPoint(x: entry.rect.midX, y: entry.rect.midY))
            model.hudRelease(.preset(entry.preset), at: CGPoint(x: entry.rect.midX, y: entry.rect.midY), clicks: 1, dragged: false)
        }
        for target in [HUDTarget.toggle3D, .toggle3D] {
            let rect = try #require(hud.rects.first { $0.target == target }?.rect)
            model.hudPress(target, at: CGPoint(x: rect.midX, y: rect.midY))
            model.hudRelease(target, at: CGPoint(x: rect.midX, y: rect.midY), clicks: 1, dragged: false)
            for _ in 0..<40 where !model.advanceCamera(dt: 1.0 / 60) { break }
        }
        let track = try #require(hud.tiltTrack), compass = try #require(hud.compass)
        model.hudPress(.tilt, at: CGPoint(x: track.midX, y: track.maxY - 3))
        for y in stride(from: track.maxY - 3, through: track.minY, by: -8) { model.hudDrag(.tilt, to: CGPoint(x: track.midX, y: y)) }
        model.hudRelease(.tilt, at: CGPoint(x: track.midX, y: track.minY), clicks: 1, dragged: true)
        model.hudPress(.compass, at: CGPoint(x: compass.midX, y: compass.minY + 2))
        for x in stride(from: compass.midX, through: compass.maxX, by: 5) { model.hudDrag(.compass, to: CGPoint(x: x, y: compass.midY)) }
        model.hudRelease(.compass, at: CGPoint(x: compass.maxX, y: compass.midY), clicks: 1, dragged: true)
        for point in [CGPoint(x: track.midX, y: track.midY), CGPoint(x: compass.midX, y: compass.midY)] + hud.presets.map({ CGPoint(x: $0.rect.midX, y: $0.rect.midY) }) {
            model.hover(at: point)
        }
        model.editPose { $0.tilt = .pi / 2 }
        model.drag(dx: 0, dy: 30, orbit: true)
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewMenu"] == 0, "the menu was evaluated again \(counts["ViewMenu"]) times by the camera controls")
        #expect(model.viewMenu.changes == changes)
        withExtendedLifetime(view) {}
    }

    @Test func aDefaultTheMenuReadsIsReReadWhenItIsSet() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model) = try await rig(scratch)
        defer { play.shutdown() }
        #expect(model.viewMenu.value.skipConfirm == false)
        model.defaults.set(true, forKey: WorldModel.skipConfirmKey)
        model.viewMenu.refresh()  // (what the menu's toggle does after it writes the default)
        #expect(model.viewMenu.value.skipConfirm)
        #expect(model.viewMenu.value.live)
    }

    // MARK: The pieces

    private final class Source: ObservableObject {
        var number = 0
        func bump() {
            objectWillChange.send()
            number += 1
        }
    }

    @Test func aBurstOfPublishesIsOneReadAndAnEqualReadPublishesNothing() async throws {
        let source = Source()
        var reads = 0
        let state = DistinctState(initial: 0, sources: [source.objectWillChange]) { () -> Int? in
            reads += 1
            return source.number / 10  // changes only every ten bumps
        }
        var published = 0
        let sub = state.objectWillChange.sink { published += 1 }
        defer { sub.cancel() }

        for _ in 0..<9 { source.bump() }
        try await Task.sleep(for: .milliseconds(50))
        #expect(reads == 1, "nine publishes in one turn are one read")
        #expect(published == 0 && state.value == 0)

        source.bump()  // the tenth: the value changes
        try await Task.sleep(for: .milliseconds(50))
        #expect(published == 1 && state.value == 1 && state.changes == 1)

        state.refresh()  // by hand, with nothing changed
        #expect(published == 1)
    }

    @Test func aStateWhoseOwnerIsGoneKeepsItsLastValue() async throws {
        let source = Source()
        let state = DistinctState(initial: 5, sources: [source.objectWillChange]) { () -> Int? in nil }
        source.bump()
        try await Task.sleep(for: .milliseconds(50))
        #expect(state.value == 5 && state.changes == 0)
    }

    // MARK: The viewer's colour pop-up

    @Test func theViewersColourPopUpIsNotRebuiltByFrames() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let commands = scratch.url.appendingPathComponent("commands.txt")
        let script = try scratch.script(
            "alife-stand-in", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
        let stream = StreamController()
        stream.start(StreamSource(run: "r", follow: false), executable: script, directory: scratch.url,
                     arguments: [Repo.streamFixtureURL("3d").path, commands.path])
        defer { stream.stop(); ViewProbe.onEvaluate = nil }
        #expect(await eventually { stream.phase == .live && stream.frameSerial >= 1 })
        let model = ViewerModel(stream: stream)
        let counts = counting(stream)
        let view = hosted(ViewerColourPicker(state: model.colour, stream: stream), width: 200, height: 40)
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewerColourPicker"] >= 1)
        counts.reset()
        var published = 0
        let sub = stream.objectWillChange.sink { published += 1 }
        defer { sub.cancel() }

        for _ in 0..<20 {
            stream.objectWillChange.send()  // a frame: `frame` and `frameSerial` are published
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(published >= 20)
        #expect(counts["ViewerColourPicker"] == 0, "the pop-up was evaluated again \(counts["ViewerColourPicker"]) times")

        let modes = try #require(stream.header?.colorModes)
        let other = try #require(modes.first { $0 != stream.control.color })
        stream.setColor(other)
        #expect(await eventually { model.colour.value.current == other })
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["ViewerColourPicker"] == 1)
        withExtendedLifetime(view) {}
    }

    // MARK: The launcher

    @Test func theLaunchersPlanetFormIsNotRebuiltByTheBuildsOutput() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // A world being built: lines on stderr (the planet's progress), no header.
        let script = try scratch.script("alife-building", "for i in 1 2 3 4 5 6 7 8; do echo \"line $i\" >&2; sleep 0.05; done\nread -r _")
        let play = PlayController()
        let launcher = WorldLauncherModel(runsDirectory: { nil }, launch: { _ in })
        let counts = counting(launcher)
        defer { play.shutdown(); ViewProbe.onEvaluate = nil }
        let view = hosted(WorldLauncherView(model: launcher, play: play, close: {}), width: 640, height: 600)
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["LauncherPlanetBox"] >= 1)
        counts.reset()
        var published = 0
        let sub = play.objectWillChange.sink { published += 1 }
        defer { sub.cancel() }

        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url, arguments: [])
        #expect(await eventually { play.progress.count >= 8 })
        try await Task.sleep(for: .milliseconds(200))
        #expect(published >= 8, "the controller published with the build's output")
        #expect(counts["LauncherPlanetBox"] == 0, "the form was evaluated again \(counts["LauncherPlanetBox"]) times")

        launcher.setStorm(!launcher.planet.storm)  // its own model changing is another matter
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts["LauncherPlanetBox"] == 1)
        withExtendedLifetime(view) {}
    }
}
