@testable import AlifeCore
import AppKit
import CoreGraphics
import Foundation
import SwiftUI
import Testing
import simd

@testable import AlifeControl

/// Where the pointer goes in the World window (DESIGN §33.4.4, §33.4.8): the whole `WorldView` is hosted in an
/// `NSHostingView`, and AppKit's own routing is exercised, a hit test at a point and the event sent to the view it finds.
/// The map's Metal view must get every click, drag, wheel and pinch over it; the divider only its 6-point strip; the dock its
/// controls; and the overlays over the map only what they draw. Mouse events go through `NSWindow.sendEvent`; a wheel or a
/// drag with deltas cannot be built for a window (an `NSEvent` made from a `CGEvent` has no window), so those are sent to the
/// view the hit test finds, which is the whole of what AppKit does with them.
///
/// The window is ordered in (SwiftUI's gestures see no events in one that is not) but invisible, far off every screen, and
/// ignores the real mouse: nothing a person does can reach it.
private let windowSize = CGSize(width: 1100, height: 700)

/// Lets the main run loop turn (CoreAnimation commits, SwiftUI updates) from a test.
@MainActor private func spin(_ seconds: Double) { RunLoop.current.run(until: Date().addingTimeInterval(seconds)) }

@MainActor
private final class Rig {
    let play = PlayController()
    let host: NSHostingView<WorldView>
    let window: NSWindow
    let scratch: Scratch
    private var saved: [String: Any?] = [:]
    static let keys = ["world.dockSide", "world.dockExtent", "world.dockExtentBottom", "world.dockCollapsed"]

    /// The extent the dock has when nothing was dragged.
    static func defaultExtent(_ side: DockSide) -> CGFloat { DockLayout.defaultExtent(for: side) }

    var frames: DockLayout.Frames

    init(side: DockSide, collapsed: Bool, kind: String) throws {
        scratch = try Scratch()
        for key in Self.keys { saved[key] = UserDefaults.standard.object(forKey: key) }
        UserDefaults.standard.set(side.rawValue, forKey: "world.dockSide")
        UserDefaults.standard.set(collapsed, forKey: "world.dockCollapsed")
        UserDefaults.standard.removeObject(forKey: "world.dockExtent")
        UserDefaults.standard.removeObject(forKey: "world.dockExtentBottom")
        frames = DockLayout.frames(in: windowSize, side: side, extent: Self.defaultExtent(side), collapsed: collapsed)
        let commands = scratch.url.appendingPathComponent("commands.txt")
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
        // (a world that is starting when the view is made: no launcher sheet, which would list the owner's runs)
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                   arguments: [Repo.playFixtureURL(kind).path, commands.path])
        host = NSHostingView(rootView: WorldView(play: play))
        window = NSWindow(contentRect: NSRect(origin: .zero, size: windowSize), styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = host
        host.frame = NSRect(origin: .zero, size: windowSize)
        window.alphaValue = 0
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.orderFrontRegardless()
    }

    static func make(side: DockSide = .right, collapsed: Bool = false, kind: String = "3d") async throws -> Rig {
        let rig = try Rig(side: side, collapsed: collapsed, kind: kind)
        let play = rig.play
        #expect(await eventually { play.store.header != nil && play.store.clock != nil })
        #expect(await eventually { rig.settle(); return rig.metal != nil })
        rig.settle()
        rig.model.reduceMotion = { false }  // (a flight is an arc whatever the machine's accessibility settings)
        return rig
    }

    func finish() {
        window.orderOut(nil)
        play.shutdown()
        for key in Self.keys {
            if let value = saved[key] ?? nil { UserDefaults.standard.set(value, forKey: key) } else { UserDefaults.standard.removeObject(forKey: key) }
        }
        scratch.cleanup()
    }

    /// Lets SwiftUI lay out and commit what changed.
    func settle(_ turns: Int = 8) {
        for _ in 0..<turns {
            host.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            spin(0.02)
        }
    }

    var metal: WorldMTKView? { Self.find(in: host) }

    var model: WorldModel { metal!.model! }

    private static func find(in view: NSView) -> WorldMTKView? {
        if let hit = view as? WorldMTKView { return hit }
        for sub in view.subviews { if let hit = find(in: sub) { return hit } }
        return nil
    }

    /// The window's point for a point of the content, origin at the top left (SwiftUI's).
    func windowPoint(_ p: CGPoint) -> NSPoint { NSPoint(x: p.x, y: host.bounds.height - p.y) }

    /// The view an event at `p` goes to.
    func hit(_ p: CGPoint) -> NSView? { host.hitTest(host.superview!.convert(windowPoint(p), from: nil)) }

    /// Whether the Metal view is what a pointer at `p` finds.
    func isMap(_ p: CGPoint) -> Bool { hit(p) is WorldMTKView }

    func mouse(_ type: NSEvent.EventType, _ p: CGPoint, clicks: Int = 1) -> NSEvent {
        NSEvent.mouseEvent(with: type, location: windowPoint(p), modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                           windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: clicks, pressure: 1)!
    }

    /// A press and release of the left button at `p` through the window, as AppKit sends them.
    func click(_ p: CGPoint, clicks: Int = 1) {
        window.sendEvent(mouse(.leftMouseDown, p, clicks: clicks))
        spin(0.05)
        window.sendEvent(mouse(.leftMouseUp, p, clicks: clicks))
        spin(0.05)
    }

    /// A drag of the left button from `from` to `to` through the window, in `steps` steps.
    func drag(from: CGPoint, to: CGPoint, steps: Int = 4) {
        window.sendEvent(mouse(.leftMouseDown, from))
        spin(0.05)
        for i in 1...steps {
            let t = CGFloat(i) / CGFloat(steps)
            window.sendEvent(mouse(.leftMouseDragged, CGPoint(x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t)))
            spin(0.05)  // (the divider takes at most 30 steps a second)
        }
        window.sendEvent(mouse(.leftMouseUp, to))
        spin(0.1)
        settle()
    }

    /// A wheel or trackpad scroll of `delta` points, sent to the view under `p`.
    func scroll(at p: CGPoint, delta: Int32) {
        let cg = CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1, wheel1: delta, wheel2: 0, wheel3: 0)!
        hit(p)?.scrollWheel(with: NSEvent(cgEvent: cg)!)
    }

    /// Where the pointer at the content's point `p` is on the screen, in the coordinates of a `CGEvent` (origin at the top left of the
    /// main display): an event made from one has no window, so the view finds its place from that.
    func screenLocation(_ p: CGPoint) -> CGPoint {
        let cocoa = window.convertPoint(toScreen: windowPoint(p))
        return CGPoint(x: cocoa.x, y: (NSScreen.screens.first?.frame.height ?? 0) - cocoa.y)
    }

    /// A trackpad scroll event (continuous: points, with the phases of a two-finger gesture and of its momentum) sent to the view under `p`.
    func trackpad(at p: CGPoint, dx: Int32 = 0, dy: Int32, phase: CGScrollPhase? = nil, momentum: CGMomentumScrollPhase = .none) {
        let cg = CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 2, wheel1: dy, wheel2: dx, wheel3: 0)!
        cg.location = screenLocation(p)
        cg.setIntegerValueField(.scrollWheelEventIsContinuous, value: 1)
        cg.setIntegerValueField(.scrollWheelEventScrollPhase, value: Int64(phase?.rawValue ?? 0))
        cg.setIntegerValueField(.scrollWheelEventMomentumPhase, value: Int64(momentum.rawValue))
        hit(p)?.scrollWheel(with: NSEvent(cgEvent: cg)!)
    }

    /// A mouse wheel notch event (lines) sent to the view under `p`.
    func notch(at p: CGPoint, lines: Int32) {
        let cg = CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1, wheel1: lines, wheel2: 0, wheel3: 0)!
        cg.location = screenLocation(p)
        hit(p)?.scrollWheel(with: NSEvent(cgEvent: cg)!)
    }

    func key(_ characters: String) -> NSEvent {
        NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: window.windowNumber, context: nil,
                         characters: characters, charactersIgnoringModifiers: characters, isARepeat: false, keyCode: 0)!
    }

    /// The point of the plane through the view's centre (at its height: a grab keeps the point on that plane under the pointer) under a
    /// point of the free rectangle, by the projector the picker uses.
    func under(_ p: CGPoint) -> SIMD2<Float>? { model.projector(size: model.freeRect.size).world(at: p, height: model.pose.z) }

    /// Runs the camera's own motion until it arrives (or `frames` pass).
    func arrive(frames: Int = 400) {
        for _ in 0..<frames where !model.advanceCamera(dt: 1.0 / 60) { return }
    }

    /// A move of a button held down by (`dx`, `dy`), sent to the view under `p`.
    func dragged(at p: CGPoint, _ type: CGEventType, button: CGMouseButton, dx: Int64, dy: Int64) -> Bool {
        guard let view = hit(p) else { return false }
        let cg = CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: CGPoint(x: 5000, y: 5000), mouseButton: button)!
        cg.setIntegerValueField(.mouseEventDeltaX, value: dx)
        cg.setIntegerValueField(.mouseEventDeltaY, value: dy)
        let event = NSEvent(cgEvent: cg)!
        switch type {
        case .rightMouseDragged: view.rightMouseDragged(with: event)
        case .otherMouseDragged: view.otherMouseDragged(with: event)
        default: view.mouseDragged(with: event)
        }
        return view is WorldMTKView
    }

    /// Points of the free rectangle on a grid, not within a point of the divider's strip.
    func mapSamples(step: CGFloat = 25) -> [CGPoint] {
        var out: [CGPoint] = []
        var y = frames.free.minY + step / 2
        while y < frames.free.maxY {
            var x = frames.free.minX + step / 2
            while x < frames.free.maxX {
                let p = CGPoint(x: x, y: y)
                if !frames.divider.insetBy(dx: -1, dy: -1).contains(p) { out.append(p) }
                x += step
            }
            y += step
        }
        return out
    }
}

@MainActor
@Suite(.serialized) struct WorldInputTests {
    private let uid0: Int64 = 5_000_000_000

    // MARK: Hit tests

    /// Every point of the map, but the divider's strip, finds the Metal view: in each dock position, with the dock collapsed,
    /// and after the window has re-rendered for a change of the model (the dock's materials gave their hit area a margin of
    /// 13 points that reached into the map then).
    @Test(arguments: [(DockSide.right, false), (.left, false), (.bottom, false), (.right, true)])
    func everyPointOfTheMapFindsTheMetalView(side: DockSide, collapsed: Bool) async throws {
        let rig = try await Rig.make(side: side, collapsed: collapsed)
        defer { rig.finish() }
        let samples = rig.mapSamples()
        #expect(samples.count > 300)
        #expect(samples.allSatisfy { rig.isMap($0) }, "before: \(samples.filter { !rig.isMap($0) }.prefix(4))")
        rig.model.objectWillChange.send()  // (any change of the model redraws the window)
        rig.settle()
        let missed = samples.filter { !rig.isMap($0) }
        #expect(missed.isEmpty, "after a change: \(missed.count) points, from \(missed.prefix(4))")
        // and the strip's outer edge: a point 3 points from it on the map's side, where the 13 points were
        if !collapsed {
            let outer = side == .bottom ? CGPoint(x: 400, y: rig.frames.divider.minY - 3) : CGPoint(x: side == .right ? rig.frames.divider.minX - 3 : rig.frames.divider.maxX + 3, y: 300)
            #expect(rig.isMap(outer), "\(outer)")
        }
    }

    @Test(arguments: DockSide.allCases)
    func theDividerFindsTheHostingViewOnlyOnItsStripAndTheDockItsOwnViews(side: DockSide) async throws {
        let rig = try await Rig.make(side: side)
        defer { rig.finish() }
        let strip = rig.frames.divider
        for p in [CGPoint(x: strip.midX, y: 300), CGPoint(x: 400, y: strip.midY)] where strip.contains(p) {
            #expect(rig.hit(p) === rig.host, "the strip is SwiftUI's: \(p)")
        }
        let dock = rig.frames.dock
        let inDock = CGPoint(x: dock.midX, y: dock.midY)
        #expect(!rig.isMap(inDock) && rig.hit(inDock) != nil)
        // two points either side of the strip, far from its ends
        let across: [CGPoint] = side == .bottom
            ? [CGPoint(x: 600, y: strip.minY - 8), CGPoint(x: 600, y: strip.maxY + 8)]
            : [CGPoint(x: strip.minX - 8, y: 300), CGPoint(x: strip.maxX + 8, y: 300)]
        let mapSide = across[side == .left ? 1 : 0]
        let dockSide = across[side == .left ? 0 : 1]
        #expect(rig.isMap(mapSide) && !rig.isMap(dockSide))
    }

    /// What is drawn over the map takes the pointer only where it is drawn: the ticker, the chapter card (full and folded),
    /// the alerts over the map of a bottom dock, the clock of a collapsed dock.
    @Test(arguments: [DockSide.right, .bottom])
    func theOverlaysTakeTheMouseOnlyWhereTheyAreDrawn(side: DockSide) async throws {
        let rig = try await Rig.make(side: side, kind: "3d_g3b")
        defer { rig.finish() }
        let play = rig.play, model = rig.model
        #expect(await eventually { play.store.plans.records.count == 7 && play.store.clock?.reason == .alert })
        rig.settle()
        let free = rig.frames.free
        // the free rectangle's corners, away from every overlay but the bottom dock's alerts at the top left
        let corners = [CGPoint(x: free.maxX - 20, y: free.minY + 20), CGPoint(x: free.maxX - 20, y: free.maxY - 20),
                       CGPoint(x: free.minX + 20, y: free.maxY - 20), CGPoint(x: free.maxX - 20, y: free.midY)]
        func expectMap(_ label: String) {
            rig.settle(12)
            #expect(corners.allSatisfy { rig.isMap($0) }, "\(label): \(corners.filter { !rig.isMap($0) })")
        }
        expectMap("alerts")
        model.ticker.show(id: 0, text: "Storm Merrow reached named strength.", at: 0)
        model.tickerLine = model.ticker.line
        expectMap("ticker")
        let line = CGPoint(x: free.midX, y: free.maxY - 12 - 14)
        #expect(!rig.isMap(line), "the ticker is where it is drawn")
        model.tickerLine = nil
        let clock = PlayClock(head: RecordHead(step: 12_000, seq: 0, version: 1, flags: 0), rate: 0, achieved: 0, stepMs: 15,
                              frameMs: 3, lateSteps: 0, saveStallSteps: 0, reason: .chapter, saving: false, flags: 1, fe: 1, chapter: 2,
                              inChapter: 3000, chapterLen: 6000)
        play.store.apply(.clck(clock))
        expectMap("chapter card")
        #expect(!rig.isMap(CGPoint(x: free.midX, y: 80)), "the card is where it is drawn")
        model.cardCollapsed = true
        expectMap("folded card")
        model.cardCollapsed = false
    }

    @Test func theCollapsedDocksClockOverTheMapTakesNothing() async throws {
        let rig = try await Rig.make(collapsed: true)
        defer { rig.finish() }
        let free = rig.frames.free
        // the badge is at the top right, 10 points in
        #expect(rig.isMap(CGPoint(x: free.maxX - 40, y: 20)) && rig.isMap(CGPoint(x: free.maxX - 90, y: 20)))
    }

    // MARK: Clicks and drags

    private func loner(_ rig: Rig) throws -> CGPoint {
        let frame = try #require(rig.play.store.frame)
        let index = try #require(rig.play.store.listedIndex(ofUID: uid0 + 147))
        return try #require(rig.model.projector(size: rig.model.freeRect.size).screen(frame.position(index)))
    }

    private func frameCamera(_ rig: Rig) {
        rig.model.goHome()
        rig.model.edit {
            $0.distance = 20
            $0.target = SIMD3(40, 5, 9)  // on the isolated organism (row 21), which is then under the middle of the map
        }
    }

    private func freeWidth(_ rig: Rig) -> CGFloat { rig.model.freeRect.width }

    @Test(arguments: DockSide.allCases)
    func aClickOnTheMapSelectsAndNeverMovesTheDock(side: DockSide) async throws {
        let rig = try await Rig.make(side: side)
        defer { rig.finish() }
        frameCamera(rig)
        let free = rig.model.freeRect
        let before = free
        let at = try loner(rig)
        let offset = CGPoint(x: rig.frames.free.minX, y: rig.frames.free.minY)  // (the left dock moves the map's origin)
        let onWindow = CGPoint(x: at.x + offset.x, y: at.y + offset.y)
        // empty places first: they select nothing and must not move the dock
        for p in rig.mapSamples(step: 200) { rig.click(p) }
        #expect(rig.model.freeRect == before, "the dock stayed: \(rig.model.freeRect) against \(before)")
        for key in ["world.dockExtent", "world.dockExtentBottom"] {
            #expect(UserDefaults.standard.object(forKey: key) == nil, "\(key) is untouched")
        }
        rig.click(onWindow)
        #expect(rig.model.selection == .organism(uid: uid0 + 147), "the click reached the map: \(rig.model.selection)")
        #expect(rig.model.freeRect == before)
        // the follow badge is drawn over the map, and takes nothing
        rig.settle()
        #expect(rig.isMap(CGPoint(x: rig.frames.free.minX + 60, y: rig.frames.free.maxY - 25)))
    }

    @Test func aDoubleClickOnTheMapFliesIn() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        frameCamera(rig)
        let at = try loner(rig)
        rig.click(at, clicks: 1)
        #expect(!rig.model.isFlying)
        rig.click(at, clicks: 2)
        #expect(rig.model.isFlying, "the second click of a double click reached the map")
    }

    @Test func dragsOnTheMapGrabOrbitAndPanAndReachNothingElse() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        frameCamera(rig)
        let p = CGPoint(x: 300, y: 300)
        let before = rig.model.camera
        // the left button grabs: the press and the moves go through the window to the Metal view, as AppKit sends them
        let anchor = try #require(rig.under(p))
        let to = CGPoint(x: 340, y: 310)
        rig.drag(from: p, to: to)
        #expect(rig.model.camera.target != before.target, "a left drag pans")
        #expect(simd_length(Torus.wrapDelta(try #require(rig.under(to)) - anchor, size: 48)) < 0.01, "the point grabbed is under the pointer")
        let panned = rig.model.camera
        #expect(rig.dragged(at: p, .rightMouseDragged, button: .right, dx: 20, dy: 8), "a right drag goes to the map")
        #expect(rig.model.camera.yaw != panned.yaw || rig.model.camera.pitch != panned.pitch, "a right drag turns and tilts")
        let turned = rig.model.camera
        #expect(rig.dragged(at: p, .otherMouseDragged, button: .center, dx: 15, dy: 5), "a middle drag goes to the map")
        #expect(rig.model.camera.target != turned.target, "a middle drag pans")
        #expect(UserDefaults.standard.object(forKey: "world.dockExtent") == nil)
    }

    @Test func aPressThatMovesLessThanTheSlopIsAClickAndMovesNothingAndADragThatStartsOnAnOrganismStillGrabsTheMap() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        frameCamera(rig)
        let before = rig.model.pose
        let at = try loner(rig)
        rig.window.sendEvent(rig.mouse(.leftMouseDown, at))
        rig.window.sendEvent(rig.mouse(.leftMouseDragged, CGPoint(x: at.x + 2, y: at.y + 1)))  // inside the slop
        rig.window.sendEvent(rig.mouse(.leftMouseUp, CGPoint(x: at.x + 2, y: at.y + 1)))
        spin(0.05)
        #expect(rig.model.pose == before || rig.model.pose.distance == before.distance, "a click, not a drag: the view stays")
        #expect(rig.model.selection == .organism(uid: uid0 + 147), "and it picked")
        rig.model.clearSelection()
        let anchor = try #require(rig.under(at))
        rig.drag(from: at, to: CGPoint(x: at.x + 60, y: at.y - 40))
        #expect(rig.model.selection.isNone, "a drag selects nothing")
        #expect(simd_length(Torus.wrapDelta(try #require(rig.under(CGPoint(x: at.x + 60, y: at.y - 40))) - anchor, size: 48)) < 0.01)
    }

    @Test func aDragCancelsAFollowAndAFlightAndKeepsThePoseReached() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        frameCamera(rig)
        rig.model.perform(.home)
        for _ in 0..<12 { _ = rig.model.advanceCamera(dt: 1.0 / 60) }
        #expect(rig.model.isFlying)
        let p = CGPoint(x: 300, y: 300)
        rig.drag(from: p, to: CGPoint(x: 330, y: 300))
        #expect(!rig.model.isFlying && rig.model.flight == nil && !rig.model.cameraIsHome)
        let held = rig.model.pose
        rig.arrive()
        #expect(rig.model.pose == held, "nothing moves it any more")
    }

    // MARK: The wheel

    @Test(arguments: DockSide.allCases)
    func theWheelAndAScrollWithScrollZoomsOnZoomTheMapAnywhereOverItAboutThePointer(side: DockSide) async throws {
        let rig = try await Rig.make(side: side)
        defer { rig.finish() }
        frameCamera(rig)
        rig.model.zoom(by: 1.4)  // room both ways
        let free = rig.frames.free
        let offset = CGPoint(x: free.minX, y: free.minY)  // (a left dock moves the map's origin)
        for local in [CGPoint(x: free.width / 2, y: free.height / 2), CGPoint(x: 30, y: 30), CGPoint(x: free.width - 30, y: free.height - 30)] {
            let p = CGPoint(x: local.x + offset.x, y: local.y + offset.y)
            let anchor = try #require(rig.under(local))
            let d0 = rig.model.camera.distance
            rig.notch(at: p, lines: 2)
            let near = rig.model.camera.distance
            #expect(near < d0, "scrolling one way zooms in at \(p): \(d0) to \(near)")
            #expect(simd_length(Torus.wrapDelta(try #require(rig.under(local)) - anchor, size: 48)) < 0.01, "about the pointer")
            rig.notch(at: p, lines: -2)
            #expect(rig.model.camera.distance > near, "and the other way zooms out at \(p)")
        }
        // and the trackpad, once the switch is on (it pans otherwise)
        rig.model.display.scrollZooms = true
        let d0 = rig.model.camera.distance
        let p = CGPoint(x: free.midX, y: free.midY)
        rig.trackpad(at: p, dy: 0, phase: .began)
        rig.trackpad(at: p, dy: 40, phase: .changed)
        #expect(rig.model.camera.distance < d0, "Scroll zooms: the trackpad zooms")
        rig.model.display.scrollZooms = false
    }

    @Test func aTrackpadsTwoFingerScrollPansTheMapAndItsMomentumToo() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        frameCamera(rig)
        rig.model.zoom(by: 1.4)
        let p = CGPoint(x: 300, y: 300)
        let d0 = rig.model.camera.distance
        let centre = try #require(rig.under(CGPoint(x: rig.model.freeRect.midX, y: rig.model.freeRect.midY)))
        let screen0 = try #require(rig.model.projector(size: rig.model.freeRect.size).screen(SIMD3(centre.x, centre.y, rig.model.pose.z)))
        rig.trackpad(at: p, dy: 0, phase: .began)
        for _ in 0..<10 { rig.trackpad(at: p, dx: 6, dy: 4, phase: .changed) }
        rig.trackpad(at: p, dy: 0, phase: .ended)
        for _ in 0..<10 { rig.trackpad(at: p, dx: 2, dy: 1, momentum: .continuous) }
        #expect(rig.model.camera.distance == d0, "the default law pans: it does not zoom")
        let screen1 = try #require(rig.model.projector(size: rig.model.freeRect.size).screen(SIMD3(centre.x, centre.y, rig.model.pose.z)))
        #expect(screen1.x - screen0.x > 70 && screen1.y - screen0.y > 40, "the map followed the fingers and the glide: (\(screen1.x - screen0.x), \(screen1.y - screen0.y))")
    }

    @Test func theWheelOverTheDividerOrTheDockDoesNotZoomTheMap() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        frameCamera(rig)
        let d0 = rig.model.camera.distance
        rig.scroll(at: CGPoint(x: rig.frames.divider.midX, y: 300), delta: 30)
        rig.scroll(at: CGPoint(x: rig.frames.dock.midX, y: 14), delta: 30)
        #expect(rig.model.camera.distance == d0)
    }

    // MARK: Keys

    @Test func hNAndIReachTheModelWhateverHasTheFocus() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        frameCamera(rig)
        let metal = try #require(rig.metal)
        #expect(rig.window.makeFirstResponder(rig.host), "the focus is not on the map")
        rig.model.editPose { $0.heading = 1.0 }
        #expect(metal.handleKey(rig.key("n")) && rig.model.cameraMotionPending)
        rig.arrive()
        #expect(rig.model.pose.heading == 0)
        #expect(metal.handleKey(rig.key("h")) && rig.model.isFlying && rig.model.flight?.home == true)
        rig.arrive()
        #expect(rig.model.cameraIsHome)
        frameCamera(rig)
        #expect(metal.handleKey(rig.key("r")) && rig.model.isFlying, "r is Home too")
        rig.model.cancelCameraMotion()
        let on = rig.model.display.minimap
        #expect(metal.handleKey(rig.key("i")) && rig.model.display.minimap == !on)
        #expect(!metal.handleKey(rig.key("x")))
    }

    // MARK: The divider

    @Test func grabbingTheDividerAtTheEdgeOfItsStripDoesNotMoveIt() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        let width = freeWidth(rig)
        #expect(width == rig.frames.free.width)
        for x in [rig.frames.divider.minX + 1, rig.frames.divider.maxX - 1] {
            rig.click(CGPoint(x: x, y: 300))
            rig.settle()
            #expect(freeWidth(rig) == width, "a click on the strip at \(x) moved the dock: \(freeWidth(rig)) against \(width)")
        }
    }

    @Test func draggingTheDividerFromItsEdgeMovesItByTheDragAndNoMore() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        let strip = rig.frames.divider
        let start = CGPoint(x: strip.minX + 1, y: 300)  // 2 points left of the centre
        rig.drag(from: start, to: CGPoint(x: start.x - 100, y: 300))
        #expect(freeWidth(rig) == rig.frames.free.width - 100, "100 points to the left: \(freeWidth(rig))")
        // the last step is kept when the throttle skipped it: a quick flick still ends where it ended
        let strip2 = CGRect(x: freeWidth(rig) - 3, y: 0, width: 6, height: 700)
        let from2 = CGPoint(x: strip2.midX, y: 300)
        rig.window.sendEvent(rig.mouse(.leftMouseDown, from2))
        spin(0.05)
        rig.window.sendEvent(rig.mouse(.leftMouseDragged, CGPoint(x: from2.x + 40, y: 300)))
        rig.window.sendEvent(rig.mouse(.leftMouseDragged, CGPoint(x: from2.x + 80, y: 300)))  // (inside the same 33 ms)
        rig.window.sendEvent(rig.mouse(.leftMouseUp, CGPoint(x: from2.x + 80, y: 300)))
        spin(0.1)
        rig.settle()
        #expect(freeWidth(rig) == strip2.midX + 80, "the free rectangle ended under the pointer: \(freeWidth(rig))")
    }

    @Test func theDockIsNeverNarrowerThanItsMinimumWhateverTheDrag() async throws {
        let rig = try await Rig.make()
        defer { rig.finish() }
        let strip = rig.frames.divider
        // dragged far to the right the dock is no narrower than its minimum
        rig.drag(from: CGPoint(x: strip.midX, y: 300), to: CGPoint(x: 1090, y: 300))
        #expect(freeWidth(rig) == windowSize.width - DockLayout.minExtent)
    }
}
