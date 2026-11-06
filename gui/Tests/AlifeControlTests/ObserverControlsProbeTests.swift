@testable import AlifeCore
import AlifeTestSupport
import AppKit
import Combine
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The research Watch's controls and page are not redone with every frame of the stream (the review of PR-F measured 20 evaluations in 20
/// frames, against the repository's rule of 0: a `Menu` that is evaluated again while it is open is rebuilt under the pointer). The views are
/// hosted off screen, the stream announces a change 20 times, and the bodies are counted (`ViewProbe`). **These tests are part of
/// `MenuStabilityTests`' suite** (an extension): the probe is one hook for the process, and two suites that set it at the same time spoil each
/// other's counts (the first full run of these tests made `aChangeOfWhatTheMenuShowsRebuildsItOnce` fail that way).
@MainActor
extension MenuStabilityTests {
    private final class Counts {
        var byName: [String: Int] = [:]
        func reset() { byName = [:] }
    }

    private func observerCounting(_ owner: AnyObject) -> Counts {
        let counts = Counts()
        let id = ObjectIdentifier(owner)
        ViewProbe.onEvaluate = { name, who in if who == id { counts.byName[name, default: 0] += 1 } }
        return counts
    }

    private func observerHost<V: View>(_ view: V, width: CGFloat, height: CGFloat) -> (NSView, NSWindow) {
        let host = NSHostingView(rootView: view.frame(width: width, height: height, alignment: .topLeading))
        let window = QuietWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: height), styleMask: [.borderless], backing: .buffered, defer: true)
        window.alphaValue = 0
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.contentView = host
        host.frame = NSRect(x: 0, y: 0, width: width, height: height)
        host.layoutSubtreeIfNeeded()
        return (host, window)
    }

    /// Waits until the bodies have stopped being evaluated (the library's scan, the reader's first read and the like settle a moment after the
    /// page is built): what is counted afterwards is what the frames did. A quiet that is only a stalled main loop is not a quiet (the first CI
    /// run, DESIGN §22.0 I91, held the loop for seconds, the late evaluations came during the frames, and 17 of them were counted): a wait that
    /// took far longer than it asked for starts the count of quiet samples again, and `TestTime.scale` asks for that many quiet samples in a row
    /// (one at the default scale, as before).
    private func observerSettle(_ counts: Counts) async throws {
        var last = -1
        var quiet = 0
        let needed = Int(TestTime.scale.rounded(.up))
        let deadline = ContinuousClock.now + TestTime.limit(.seconds(20))  // (50 samples of 400 ms)
        while ContinuousClock.now < deadline {
            let now = counts.byName.values.reduce(0, +)
            if now == last {
                quiet += 1
                if quiet >= needed { return }
            } else {
                quiet = 0
            }
            last = now
            let began = ContinuousClock.now
            try await Task.sleep(for: .milliseconds(400))
            if ContinuousClock.now - began > .milliseconds(1200) { quiet = 0 }  // (the loop was held up: what it did not do says nothing)
        }
    }

    private func observerFrames(_ stream: StreamController, _ n: Int) async throws {
        for _ in 0..<n {
            stream.objectWillChange.send()  // (a frame: what the stream does with every one)
            try await Task.sleep(for: .milliseconds(15))
        }
    }

    @Test func theObserversControlsAreNotEvaluatedAgainWhileFramesArrive() async throws {
        let stream = StreamController()
        let model = ViewerModel(stream: stream)
        defer { ViewProbe.onEvaluate = nil }
        let counts = observerCounting(stream)
        let (view, window) = observerHost(ViewerCompactControls(stream: stream, model: model) { _ in }, width: 525, height: 90)
        withExtendedLifetime((view, window)) {}
        try await Task.sleep(for: .milliseconds(200))
        #expect((counts.byName["ViewerCompactControls", default: 0]) >= 1, "built once")
        try await observerSettle(counts)
        counts.reset()
        let changes = model.controls.changes
        var published = 0
        let sub = stream.objectWillChange.sink { published += 1 }
        defer { sub.cancel() }
        try await observerFrames(stream, 20)
        try await Task.sleep(for: .milliseconds(100))
        #expect(published == 20, "the stream announced every frame: \(published)")
        #expect(counts.byName["ViewerCompactControls", default: 0] == 0, "evaluated \(counts.byName["ViewerCompactControls", default: 0]) times in 20 frames (the rule: 0)")
        #expect(model.controls.changes == changes && model.page.changes == 0)
    }

    @Test func theObserversControlsAreEvaluatedAgainWhenWhatTheyShowChanges() async throws {
        let stream = StreamController()
        let model = ViewerModel(stream: stream)
        defer { ViewProbe.onEvaluate = nil }
        let counts = observerCounting(stream)
        let (view, window) = observerHost(ViewerCompactControls(stream: stream, model: model) { _ in }, width: 525, height: 90)
        withExtendedLifetime((view, window)) {}
        try await Task.sleep(for: .milliseconds(200))
        counts.reset()
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let script = try scratch.script("live", "cat \"$1\"\nwhile IFS= read -r line; do :; done")
        stream.start(StreamSource(run: "w", follow: false), executable: script, directory: scratch.url, arguments: [Repo.streamFixtureURL("3d").path])
        defer { stream.stop() }
        #expect(await eventually { stream.phase == .live })
        try await Task.sleep(for: .milliseconds(200))
        #expect(counts.byName["ViewerCompactControls", default: 0] >= 1, "going live is a change the controls show")
        #expect(model.controls.value.live && model.controls.value.hasFrame)
    }

    @Test func thePageBodyIsNotRedoneWithEveryFrame() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let root = scratch.url.appendingPathComponent("folder")
        try FileManager.default.createDirectory(at: root.appendingPathComponent(".venv/bin"), withIntermediateDirectories: true)
        let alife = try scratch.script("alife", "exit 0")
        try FileManager.default.moveItem(at: alife, to: root.appendingPathComponent(".venv/bin/alife"))
        try LibraryTestFolders.research(root.appendingPathComponent("runs"), "w")
        let controller = RunController(storedPath: root.path, savePath: { _ in }, candidates: [], isApp: false)
        let suite = "alife-probe-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let shell = ShellModel(session: WorldSession(play: PlayController(), defaults: defaults), defaults: defaults)
        let stream = StreamController()
        let observer = ObserverSession(controller: controller, stream: stream)
        observer.startStream = { _, _ in }
        let runs: () -> URL? = { root.appendingPathComponent("runs") }
        let library = WorldLibrary(runsDirectory: runs, observesApp: false, lockHeld: { _ in false })
        library.refresh()
        #expect(await eventually { library.hasScanned })
        let setup = SetupModel.forApp(controller: controller, play: shell.session.play, library: library, defaults: defaults)
        defer { ViewProbe.onEvaluate = nil }
        let counts = observerCounting(stream)
        let page = ResearchWatchPage(world: "w", shell: shell, observer: observer, stream: stream, controller: controller, library: library,
                                     reader: WorldReader(runsDirectory: runs, interval: .seconds(60)), setup: setup)
        let (view, window) = observerHost(page, width: 885, height: 700)
        withExtendedLifetime((view, window)) {}
        try await Task.sleep(for: .milliseconds(300))
        #expect(counts.byName["ResearchWatchPage", default: 0] >= 1, "built")
        try await observerSettle(counts)
        counts.reset()
        try await observerFrames(stream, 20)
        try await Task.sleep(for: .milliseconds(100))
        #expect(counts.byName["ResearchWatchPage", default: 0] == 0, "the page's body ran \(counts.byName["ResearchWatchPage", default: 0]) times in 20 frames (the rule: 0); all bodies: \(counts.byName)")
    }
}

/// Which keys are the observer's, and which are not (the monitor that makes them work whichever view has the focus).
@MainActor
@Suite("The observer's keys follow the window")
struct ViewerKeyRoutingTests {
    private let window = QuietWindow(contentRect: NSRect(x: -20000, y: -20000, width: 100, height: 100), styleMask: [.borderless], backing: .buffered, defer: true)

    private func key(_ characters: String, flags: NSEvent.ModifierFlags = []) -> NSEvent {
        NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: flags, timestamp: 0, windowNumber: 0, context: nil,
                         characters: characters, charactersIgnoringModifiers: characters, isARepeat: false, keyCode: 0)!
    }

    private func action(_ characters: String, flags: NSEvent.ModifierFlags = [], responder: NSResponder? = nil) -> ViewerAction? {
        ViewerKeyMonitor.action(for: key(characters, flags: flags), eventWindow: window, in: window, firstResponder: responder)
    }

    @Test func aKeyOfTheTableIsTheObserversWhateverHasTheFocus() {
        #expect(action(" ") == .togglePause)
        #expect(action("c", responder: NSView()) == .cycleColor)
        #expect(action("S", responder: NSButton()) == .snapshot, "a focused button does not take Space or a letter from the picture")
    }

    @Test func aTextBeingEditedKeepsItsKeys() {
        let text = NSTextView()
        #expect(action("c", responder: text) == nil)
        #expect(action(" ", responder: text) == nil)
    }

    @Test func aChordAndAKeyOutsideTheTableAreNotTheObservers() {
        #expect(action("c", flags: .command) == nil)
        #expect(action("r", flags: .control) == nil)
        #expect(action("x") == nil)
        // the arrow keys are the focused view's own: the sidebar's list moves with them
        #expect(action(String(UnicodeScalar(NSUpArrowFunctionKey)!)) == nil)
    }

    @Test func anotherWindowsKeysAreNotTheObservers() {
        let other = QuietWindow(contentRect: NSRect(x: -20000, y: -20000, width: 100, height: 100), styleMask: [.borderless], backing: .buffered, defer: true)
        let e = key("c")
        #expect(ViewerKeyMonitor.action(for: e, eventWindow: other, in: window, firstResponder: nil) == nil, "the Settings or the Help window has its own keys")
        #expect(ViewerKeyMonitor.action(for: e, eventWindow: window, in: nil, firstResponder: nil) == nil)
        #expect(ViewerKeyMonitor.action(for: e, eventWindow: nil, in: window, firstResponder: nil) == nil)
        #expect(ViewerKeyMonitor.action(for: e, eventWindow: window, in: window, firstResponder: nil) == .cycleColor)
    }

    @Test func everyKeyOfTheTableReachesItsAction() {
        for character in ViewerKeys.characters {
            #expect(action(character) == ViewerKeys.action(for: character), "'\(character)'")
        }
    }

    @Test func stepsPerFrameWhileFollowingSaySoInsteadOfDoingNothing() throws {
        let stream = StreamController()
        let model = ViewerModel(stream: stream)
        model.perform(.faster)  // (no stream: it is the stream's own no-op)
        #expect(model.note == nil)
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let script = try scratch.script("slow", "sleep \(Int(TestTime.limit(seconds: 5)))")
        stream.start(StreamSource(run: "w", follow: true), executable: script, directory: scratch.url, arguments: [])
        defer { stream.stop() }
        model.perform(.slower)
        #expect(model.note == WatchGuide.stepsPerFrameNote)
        model.say("x")
        model.perform(.faster)
        #expect(model.note == WatchGuide.stepsPerFrameNote)
    }
}
