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
        let lab = try ShellRig(page: .setup)
        defer { lab.finish() }
        #expect(lab.shell.page == .setup)
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
        rig.shell.select(.setup)
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
        let lab = try ShellRig(page: .setup)
        defer { lab.finish() }
        lab.start()  // nobody pressed Play
        #expect(await eventually { lab.play.phase == .live })
        #expect(lab.shell.page == .setup)

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
        let rig = try ShellRig(page: .setup)
        defer { rig.finish() }
        let before = rig.shell.planetFormRequests
        rig.shell.requestPlanetForm()
        #expect(rig.shell.page == .worlds && rig.shell.planetFormRequests == before + 1)
    }
}

// MARK: - The two forms of Watch (PR-F)

@MainActor
@Suite("The two forms of Watch") struct WatchFormTests {
    @Test func withNothingOpenWatchShowsNothingAndTheWorldModelHasTheKeys() throws {
        let rig = try ShellRig()
        defer { rig.finish() }
        #expect(rig.shell.watchForm == nil && rig.shell.observed == nil)
        rig.shell.select(.watch)
        #expect(rig.session.model.watchActive, "no research world in view: the world model keeps its keys on the Watch page, as before")
    }

    @Test func chosingAResearchWorldToWatchShowsItAsAnObserverAndGivesTheKeysToTheObserver() throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        rig.shell.shapeOf = { $0 == "r" ? .planet : nil }
        rig.shell.observe("r")
        #expect(rig.shell.observed == "r" && rig.shell.watchForm == .research && rig.shell.page == .watch && rig.shell.observedShape == .planet)
        #expect(!rig.session.model.watchActive, "the world model's keys must not act on the page of an observer")
        rig.shell.select(.worlds)
        #expect(!rig.session.model.watchActive)
        #expect(rig.shell.watchForm == .research, "the choice is kept while the player is elsewhere; the stream is not (the page ends it)")
    }

    @Test func aPlayedWorldThatOpensReplacesTheObservedOneAndIsNotBroughtBack() async throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        rig.shell.observe("research")
        #expect(rig.shell.watchForm == .research)
        rig.start()
        #expect(await eventually { rig.play.phase == .live })
        #expect(rig.shell.watchForm == .played && rig.shell.observed == nil)
        #expect(rig.session.model.watchActive, "Watch is the page and the form is the world's")
        rig.play.stop()
        #expect(await eventually { rig.play.phase != .live })
        #expect(rig.shell.observed == nil, "the research world seen before is not brought back behind the world")
    }

    @Test func aWorldThatIsOpenIsWhatWatchShowsAndNoObserverCanReplaceIt() async throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        rig.start()
        #expect(await eventually { rig.play.phase == .live })
        rig.shell.select(.worlds)
        rig.shell.observe("research")
        #expect(rig.shell.observed == nil && rig.shell.watchForm == .played && rig.shell.page == .worlds, "a card's Watch cannot replace a world that is open (R1)")
        rig.shell.researchStarted("research")
        #expect(rig.shell.watchForm == .played && rig.shell.page == .worlds, "a run started meanwhile does not either: it is kept for when the world is closed")
    }

    @Test func aRunStartedFromTheSetUpPageTakesThePersonToWatchButNoOneElse() throws {
        let rig = try ShellRig(page: .setup)
        defer { rig.finish() }
        rig.shell.researchStarted("r1")
        #expect(rig.shell.page == .watch && rig.shell.observed == "r1" && rig.shell.watchForm == .research)

        let other = try ShellRig(page: .activity)
        defer { other.finish() }
        other.shell.researchStarted("r1")
        #expect(other.shell.page == .activity && other.shell.observed == "r1", "a person who went on to another page is not moved")
        other.shell.select(.watch)
        #expect(other.shell.watchForm == .research)
    }

    @Test func aRunStartedWhileAWorldIsOpenIsWhatWatchShowsWhenTheWorldIsClosed() async throws {
        let rig = try ShellRig(page: .setup)
        defer { rig.finish() }
        rig.start()
        #expect(await eventually { rig.play.phase == .live })
        rig.shell.researchStarted("r1")
        #expect(rig.shell.watchForm == .played && rig.shell.observed == "r1", "the world stays what Watch shows; the run is kept, not lost")
        #expect(rig.shell.page == .setup, "nobody is moved in front of a world being played")
        rig.play.stop()
        #expect(await eventually { rig.play.phase != .live })
        #expect(rig.shell.observed == "r1", "phase \(rig.play.phase)")
        #expect(rig.shell.watchForm == .research, "the run is what Watch shows once the world is closed (phase \(rig.play.phase))")
    }

    @Test func theObservedWorldIsGrowingOnlyWhenTheRunIsItsOwn() throws {
        let rig = try ShellRig(page: .worlds)
        defer { rig.finish() }
        rig.shell.observe("planet-1-grow")
        rig.shell.runningRun = nil
        #expect(!rig.shell.observedGrowing)
        rig.shell.runningRun = "other"
        #expect(!rig.shell.observedGrowing, "another world's run is not this one's")
        rig.shell.runningRun = "PLANET-1-GROW"
        #expect(rig.shell.observedGrowing, "capitals do not tell worlds apart")
    }

    @Test func theFormTableCoversEveryPhase() {
        #expect(ShellModel.form(of: .idle, observed: nil) == nil)
        #expect(ShellModel.form(of: .idle, observed: "r") == .research)
        #expect(ShellModel.form(of: .starting, observed: "r") == .played)
        #expect(ShellModel.form(of: .live, observed: nil) == .played)
        #expect(ShellModel.form(of: .ended("x"), observed: nil) == .played)
        #expect(ShellModel.form(of: .ended("x"), observed: "r") == .research)
        #expect(ShellModel.form(of: .failed("x"), observed: nil) == .played)
        #expect(ShellModel.form(of: .failed("x"), observed: "r") == .research)
    }

    @Test func theWorldMenuIsEnabledOnlyOnWatchWithSomethingInIt() {
        #expect(WorldCommands.itemsAreEnabled(page: .watch, form: .played))
        #expect(WorldCommands.itemsAreEnabled(page: .watch, form: .research))
        #expect(!WorldCommands.itemsAreEnabled(page: .watch, form: nil))
        for page in [ShellPage.worlds, .activity, .setup, .details] {
            #expect(!WorldCommands.itemsAreEnabled(page: page, form: .played), "\(page)")
            #expect(!WorldCommands.itemsAreEnabled(page: page, form: .research), "\(page)")
        }
    }

    @Test func theObserverMenusExtraItemsKnowWhenTheyCanWork() {
        // Stop is for the run Watch shows, and only from Watch: never from another page, never for another world's run
        #expect(WorldCommands.canStop(page: .watch, form: .research, observedGrowing: true))
        #expect(!WorldCommands.canStop(page: .watch, form: .research, observedGrowing: false))
        #expect(!WorldCommands.canStop(page: .watch, form: .played, observedGrowing: true))
        #expect(!WorldCommands.canStop(page: .watch, form: nil, observedGrowing: true))
        for page in [ShellPage.worlds, .activity, .setup, .details] { #expect(!WorldCommands.canStop(page: page, form: .research, observedGrowing: true), "\(page)") }
        #expect(WorldCommands.canCheck(runGoing: false, repoReady: true))
        #expect(!WorldCommands.canCheck(runGoing: true, repoReady: true), "a run holds the engine's one job")
        #expect(!WorldCommands.canCheck(runGoing: false, repoReady: false))
        #expect(WorldCommands.canOpenClassic(shape: .flat, repoReady: true))
        #expect(!WorldCommands.canOpenClassic(shape: .planet, repoReady: true) && !WorldCommands.canOpenClassic(shape: .box, repoReady: true))
        #expect(!WorldCommands.canOpenClassic(shape: nil, repoReady: true) && !WorldCommands.canOpenClassic(shape: .flat, repoReady: false))
    }

    @Test func aMenuItemGoesToTheWorldOrToTheViewerAccordingToItsForm() {
        var world: [WorldAction] = [], viewer: [ViewerAction] = []
        for form in [WatchForm.played, .research] {
            for entry in ControlTables.menu(form) { WorldCommands.dispatch(entry, world: { world.append($0) }, observer: { viewer.append($0) }) }
        }
        #expect(world == ControlTables.menu(.played).compactMap { if case .world(let a)? = $0.action { a } else { nil } })
        #expect(viewer == ControlTables.menu(.research).compactMap { if case .observer(let a)? = $0.action { a } else { nil } })
        #expect(world.count == 12 && viewer.count == 9)
        // a row with no single action does nothing
        WorldCommands.dispatch(ControlTables.played.first { $0.action == nil }!, world: { _ in Issue.record("acted") }, observer: { _ in Issue.record("acted") })
    }

    @Test func theHintSaysOverviewInTheNewWindowAndHomeInTheOldApps() throws {
        let rig = try ShellRig()
        defer { rig.finish() }
        #expect(rig.session.model.usesOverviewWord && rig.session.model.plainNames)
        #expect(rig.session.model.hint(planet: false).contains("H overview") && rig.session.model.hint(planet: true).contains("H overview"))
        let old = WorldModel(play: PlayController())
        #expect(!old.usesOverviewWord && !old.plainNames)
        #expect(old.hint(planet: false) == WorldModel.hintText && old.hint(planet: true).contains("H home"))
    }
}
