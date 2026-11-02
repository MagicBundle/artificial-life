@testable import AlifeCore
import AppKit
import Foundation
import Testing

@testable import AlifeControl

/// What the `alife` window does with the world (docs/APP.md §2.4, PR-B): the model exists before any page, the world's keys and attention
/// follow the Watch page, and the page follows the world. No renderer and no real window; the world is a stand-in process that plays the
/// G3b fixture and records the control lines it receives.
@MainActor
private struct ShellRig {
    let scratch: Scratch
    let play = PlayController()
    let session: WorldSession
    let shell: ShellModel
    let defaults: UserDefaults
    let suite: String
    let commands: URL
    let script: URL
    var sent: [String] { ((try? String(contentsOf: commands, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init) }

    init(page: ShellPage? = nil) throws {
        scratch = try Scratch()
        commands = scratch.url.appendingPathComponent("commands.txt")
        script = try scratch.script("alife", "cat \"\(Repo.playFixtureURL("3d_g3b").path)\"\nwhile IFS= read -r line; do echo \"$line\" >> \"\(commands.path)\"; done")
        suite = "alife-shell-\(UUID().uuidString)"
        defaults = try #require(UserDefaults(suiteName: suite))
        if let page { defaults.set(page.rawValue, forKey: AppShell.pageKey) }
        session = WorldSession(play: play, defaults: defaults)
        shell = ShellModel(session: session, defaults: defaults)
    }

    func start() {
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url, arguments: [])
    }

    func finish() {
        play.shutdown()
        defaults.removePersistentDomain(forName: suite)
        scratch.cleanup()
    }

    /// A World key the way the Metal view's monitor hands it over.
    func key(_ characters: String) -> NSEvent {
        NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0, context: nil,
                         characters: characters, charactersIgnoringModifiers: characters, isARepeat: false, keyCode: 0)!
    }
}

@MainActor
@Suite("The alife window and the world") struct ShellTests {
    // MARK: The model exists before any page

    @Test func aWorldStartedByTheLaunchArgumentIsDeliveredWithNoPageEverShown() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let engine = scratch.url.appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: engine, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: scratch.url.appendingPathComponent("runs/r"), withIntermediateDirectories: true)
        let alife = try scratch.script("alife", "cat \"\(Repo.playFixtureURL("3d_g3b").path)\"\nwhile IFS= read -r line; do :; done")
        try FileManager.default.moveItem(at: alife, to: engine.appendingPathComponent("alife"))
        let controller = RunController(storedPath: scratch.url.path, savePath: { _ in }, candidates: [], isApp: false)
        #expect(controller.repoReady && controller.runs == ["r"])

        let play = PlayController()
        let suite = "alife-shell-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let session = WorldSession(play: play, defaults: defaults)  // (made first, as the app does: no view has been made)
        var asked = 0
        session.model.windowIsKey = { false }
        session.model.requestAttention = { asked += 1 }
        let shell = ShellModel(session: session, defaults: defaults)
        defer { play.shutdown() }

        shell.handleLaunch(arguments: ["alife", "--world", "r"], controller: controller)
        #expect(await eventually { play.store.header != nil && play.store.clock?.reason == .alert })
        // The fixture ends on an alert stop; only a model that existed when it arrived heard it and asked the dock.
        #expect(asked == 1, "the stop at the end of the backlog reached the model")
        #expect(shell.launchNote == nil)
        #expect(await eventually { shell.page == .watch }, "a world that goes live is shown")
    }

    @Test func theLaunchArgumentIsHandledOnceWhateverBecomesOfTheWindow() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try ShellRig()
        defer { rig.finish() }
        let controller = RunController(storedPath: scratch.url.path, savePath: { _ in }, candidates: [], isApp: false)
        rig.shell.handleLaunch(arguments: ["alife", "--world", "nosuch"], controller: controller)
        let first = rig.shell.launchNote
        #expect(first != nil)
        rig.shell.launchNote = nil
        rig.shell.handleLaunch(arguments: ["alife", "--world", "nosuch"], controller: controller)  // a rebuilt window asking again
        #expect(rig.shell.launchNote == nil)
    }

    @Test func aLaunchArgumentThatCannotBeStartedSaysWhy() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try ShellRig()
        defer { rig.finish() }
        let controller = RunController(storedPath: scratch.url.path, savePath: { _ in }, candidates: [], isApp: false)
        rig.shell.handleLaunch(arguments: ["alife", "--world", "nosuch"], controller: controller)
        #expect(rig.shell.launchNote?.contains("alife folder is not set up") == true)
        let other = try ShellRig()
        defer { other.finish() }
        other.shell.handleLaunch(arguments: ["alife", "--world", "../x"], controller: controller)
        #expect(other.shell.launchNote?.contains("needs the name of a world") == true)
        let none = try ShellRig()
        defer { none.finish() }
        none.shell.handleLaunch(arguments: ["alife"], controller: controller)
        #expect(none.shell.launchNote == nil)
    }

    // MARK: Pages and the world's keys

    @Test func theWindowOpensOnWorldsAndWatchIsNotActive() throws {
        let rig = try ShellRig()
        defer { rig.finish() }
        #expect(rig.shell.page == .worlds && !rig.session.model.watchActive)
    }

    @Test func aRememberedPageComesBackExceptWatch() throws {
        let lab = try ShellRig(page: .lab)
        defer { lab.finish() }
        #expect(lab.shell.page == .lab)
        let watch = try ShellRig(page: .watch)
        defer { watch.finish() }
        #expect(watch.shell.page == .worlds)
    }

    @Test func selectingAPageRemembersItAndSetsWatchActive() throws {
        let rig = try ShellRig()
        defer { rig.finish() }
        rig.shell.select(.activity)
        #expect(rig.defaults.string(forKey: AppShell.pageKey) == "activity")
        rig.shell.select(.watch)
        #expect(rig.session.model.watchActive)
        #expect(rig.defaults.string(forKey: AppShell.pageKey) == "activity", "Watch is never the page to come back to")
        rig.shell.select(.lab)
        #expect(!rig.session.model.watchActive)
    }

    @Test func aWorldKeyOnAnotherPageSendsNothingAndOnWatchItDoes() async throws {
        let rig = try ShellRig()
        defer { rig.finish() }
        rig.start()
        #expect(await eventually { rig.play.store.header != nil })
        let view = WorldMTKView(frame: CGRect(x: 0, y: 0, width: 900, height: 700), device: nil)
        view.model = rig.session.model
        rig.shell.select(.worlds)
        for chars in [" ", ".", "g", "k", "1"] {
            #expect(!view.handleKey(rig.key(chars)), "'\(chars)' on the Worlds page falls through to the page")
        }
        rig.shell.select(.activity)
        #expect(!view.handleKey(rig.key(" ")))
        try await Task.sleep(for: .milliseconds(300))
        #expect(rig.sent.isEmpty, "no control line went to the world: \(rig.sent)")
        // On Watch the same keys act. (The stream's own header and clock are not control lines.)
        rig.shell.select(.watch)
        #expect(view.handleKey(rig.key(" ")))
        #expect(await eventually { !rig.sent.isEmpty }, "Space on Watch reaches the world")
    }

    @Test func theCanvasTakesNoFocusAndDrawsNothingWhileAnotherPageIsShown() throws {
        let rig = try ShellRig()
        defer { rig.finish() }
        let view = WorldMTKView(frame: CGRect(x: 0, y: 0, width: 900, height: 700), device: nil)
        view.model = rig.session.model
        rig.session.model.view = view
        #expect(!view.canvasVisible && !view.acceptsFirstResponder)
        view.needsDisplay = true
        #expect(view.drawDeferred, "a draw asked for while hidden is deferred")
        rig.shell.select(.watch)
        #expect(view.canvasVisible && view.acceptsFirstResponder && !view.drawDeferred, "and made when Watch is shown")
        #expect(!view.isHidden)
        rig.shell.select(.worlds)
        #expect(view.isHidden)
    }

    // MARK: Attention: another page is not "looking"

    @Test func anAlertWhileAnotherPageIsShownIsUnreadEvenInTheKeyWindow() async throws {
        let rig = try ShellRig()
        defer { rig.finish() }
        var badge: String?
        rig.session.model.windowIsKey = { true }  // the window is in front and the app is active
        rig.session.model.setBadge = { badge = $0 }
        rig.start()
        #expect(await eventually { rig.play.store.header != nil && rig.play.phase == .live })
        rig.shell.select(.worlds)  // the player goes to look at something else
        #expect(!rig.session.model.viewerIsLooking)
        let alerts = try [12_000_600, 12_000_601, 12_000_602, 11_500_603, 11_750_605].map {
            AlertChange(kind: .raised, alert: try #require(rig.play.store.alerts.alert(id: UInt64($0))))
        }
        rig.session.model.handle(alerts: alerts, backlog: false)
        #expect(badge == "2" && rig.session.model.attention.unread == 2, "the key window on another page is not looking: \(badge ?? "-")")
        // Showing Watch is looking: the unread alerts and the badge go, as when a window becomes key.
        rig.shell.select(.watch)
        #expect(rig.session.model.viewerIsLooking && badge == nil && rig.session.model.attention.unread == 0)
        rig.session.model.handle(alerts: alerts, backlog: false)
        #expect(badge == nil, "on Watch, in the key window, nothing is unread")
    }

    @Test func withWatchShownAndTheWindowInFrontTheDockIsLeftAlone() async throws {
        let rig = try ShellRig()
        defer { rig.finish() }
        var asked = 0
        rig.session.model.windowIsKey = { true }
        rig.session.model.requestAttention = { asked += 1 }
        rig.shell.select(.watch)
        rig.start()
        #expect(await eventually { rig.play.store.header != nil && rig.play.store.clock?.reason == .alert })
        #expect(asked == 0)
    }

    // MARK: The page follows the world

    @Test func aWorldThePlayerStartedOnTheWorldsPageIsShownWhenLiveAndCloseWorldGoesBack() async throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        #expect(!rig.shell.worldIsOpen)
        rig.shell.playerStartsWorld()  // Play or Make planet on the Worlds page
        rig.start()
        #expect(await eventually { rig.shell.page == .watch })
        #expect(rig.shell.worldIsOpen)
        rig.shell.closeWorld()
        #expect(rig.shell.page == .worlds)
        #expect(await eventually { !rig.shell.worldIsOpen })
    }

    @Test func aWorldIsNotPulledInFrontOfSomeoneWhoWentElsewhere() async throws {
        let lab = try ShellRig(page: .lab)
        defer { lab.finish() }
        lab.start()  // nobody pressed Play
        #expect(await eventually { lab.play.phase == .live })
        #expect(lab.shell.page == .lab)

        let away = try ShellRig(page: .worlds)
        defer { away.finish() }
        away.shell.playerStartsWorld()
        away.shell.select(.activity)  // pressed Play, then went to look at Activity during the long start
        away.start()
        #expect(await eventually { away.play.phase == .live })
        #expect(away.shell.page == .activity)
    }

    // MARK: A world that does not open is said where the player is (review H1)

    @Test func aWorldThatFailsAtOnceIsToldOnTheWorldsPage() throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        rig.shell.playerStartsWorld()
        rig.play.start(.resume(PlayResume(run: "w1")), executable: URL(fileURLWithPath: "/nonexistent/alife"), directory: rig.scratch.url)
        guard case .failed = rig.play.phase else { Issue.record("expected a failed start"); return }
        let failure = try #require(rig.shell.startFailure)
        #expect(failure.world == "w1" && failure.message.contains("Failed to launch"))
        #expect(failure.explanation.contains("w1 did not open") && failure.explanation.contains("Settings"))
        #expect(rig.shell.page == .worlds, "the player stays where the message is")
        rig.shell.startFailure = nil
        #expect(rig.shell.startFailure == nil)
    }

    @Test func aWorldWhoseEngineStopsWhileItStartsIsToldToo() async throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        let dies = try rig.scratch.script("dies", "echo 'engine: something went wrong' >&2\nexit 2")
        rig.shell.playerStartsWorld()
        rig.play.start(.resume(PlayResume(run: "w2")), executable: dies, directory: rig.scratch.url, arguments: [])
        #expect(await eventually { rig.shell.startFailure != nil })
        let failure = try #require(rig.shell.startFailure)
        #expect(failure.world == "w2" && failure.message.contains("refused the request"))
        #expect(failure.explanation.contains("w2 did not open"))
        #expect(rig.shell.page == .worlds)
        // the next world to start clears it
        rig.shell.playerStartsWorld()
        #expect(rig.shell.startFailure == nil)
    }

    @Test func aWorldThatWasLiveAndThenDiedIsNotAFailureToStart() async throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        // plays the fixture, then exits with an error: live first, then failed
        let script = try rig.scratch.script("late", "cat \"\(Repo.playFixtureURL("3d_g3b").path)\"\nsleep 2\nexit 1")
        rig.shell.playerStartsWorld()
        rig.play.start(.resume(PlayResume(run: "w3")), executable: script, directory: rig.scratch.url, arguments: [])
        #expect(await eventually { rig.play.phase == .live })
        #expect(await eventually { if case .failed = rig.play.phase { return true } else { return false } })
        #expect(rig.shell.startFailure == nil, "its Watch page tells it; the Worlds page does not claim it never opened")
    }

    // MARK: What the Unfinished card knows of this session's builds

    @Test func aBuildThatFailsWhileItStartsIsRecordedAsFailed() async throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        var notes: [String: String] = [:]
        rig.shell.buildEnded = { notes[$0] = $1 }
        let dies = try rig.scratch.script("dies", "echo 'engine: something went wrong' >&2\nexit 2")
        rig.play.start(.resume(PlayResume(run: "built")), executable: dies, directory: rig.scratch.url, arguments: [])
        #expect(await eventually { notes["built"] != nil })
        #expect(notes["built"] == "It failed")
    }

    @Test func aBuildTheUserStopsWhileItStartsIsRecordedAsCancelled() async throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        var notes: [String: String] = [:]
        rig.shell.buildEnded = { notes[$0] = $1 }
        let silent = try rig.scratch.script("quiet", "trap 'exit 130' INT\nwhile true; do sleep 0.1; done")
        rig.play.start(.resume(PlayResume(run: "built")), executable: silent, directory: rig.scratch.url, arguments: [])
        #expect(rig.play.phase == .starting)
        rig.play.stop()
        #expect(await eventually { notes["built"] != nil })
        #expect(notes["built"] == "You cancelled it")
    }

    @Test func aWorldThatWentLiveIsNotABuildThatEnded() async throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        var notes: [String: String] = [:]
        rig.shell.buildEnded = { notes[$0] = $1 }
        rig.start()
        #expect(await eventually { rig.play.phase == .live })
        rig.play.stop()
        #expect(await eventually { rig.play.phase != .live })
        try await Task.sleep(for: .milliseconds(300))
        #expect(notes.isEmpty)
    }

    @Test func aWorldThatIsStillStartingDoesNotMoveThePage() throws {
        let rig = try ShellRig()
        defer { rig.finish() }
        let silent = try rig.scratch.script("quiet", "while IFS= read -r line; do :; done")
        rig.play.start(.resume(PlayResume(run: "r")), executable: silent, directory: rig.scratch.url, arguments: [])
        #expect(rig.play.phase == .starting && rig.shell.worldIsOpen)
        #expect(rig.shell.page == .worlds, "progress and Cancel are on the Worlds page until the world is live")
    }

    @Test func theFormRequestOfTheFileMenuShowsWorldsAndCountsOnce() throws {
        let rig = try ShellRig(page: .lab)
        defer { rig.finish() }
        let before = rig.shell.planetFormRequests
        rig.shell.requestPlanetForm()
        #expect(rig.shell.page == .worlds && rig.shell.planetFormRequests == before + 1)
    }
}
