import AlifeCore
import AppKit
import SwiftUI

/// Makes the app a regular foreground app (needed when run unbundled with `swift run`; harmless in the .app
/// bundle) and stops a running simulation cleanly when the app quits. Shared by both modes (`LaunchPlan`).
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// The delegate of the running app, for the Quit command (SwiftUI wraps the adaptor's delegate, so `NSApp.delegate` is not it).
    static weak var current: AppDelegate?

    /// Holds the quit while a played world, a run or another job saves or ends (`TerminationGate`).
    let terminationGate = TerminationGate()
    /// Shown while the quit waits. Owned here, not by a scene, so that it exists when the last window has been closed.
    let quitPanel: QuitPanel
    var plan = LaunchMode.current().plan
    private let run: RunController
    private let play: PlayController

    /// The delegate of the app: the shared controllers. The tests give it their own.
    override convenience init() { self.init(run: .shared, play: .shared) }

    /// `quitPanel` is the panel shown while the quit waits (the tests give one that is placed off every screen).
    init(run: RunController, play: PlayController, quitPanel: QuitPanel = QuitPanel()) {
        self.run = run
        self.play = play
        self.quitPanel = quitPanel
        super.init()
        Self.current = self
    }
    /// The question before a run is stopped by a quit (tests replace it, so that no alert is shown).
    var confirmStopRun: (String) -> Bool = { QuitPanel.askToStopRun($0, shell: LaunchMode.current().plan.isShell) }
    /// Why the quit was asked for: the reason of the quit Apple event being handled (none for a menu Quit). Tests replace it.
    var quitReason: () -> QuitReason = { AppDelegate.reason(of: NSAppleEventManager.shared().currentAppleEvent) }
    /// Ends the app through AppKit; tests replace it.
    var terminate: () -> Void = { NSApp.terminate(nil) }
    /// Answers AppKit's `.terminateLater`; tests replace it (a reply outside a quit must not end the test process).
    var replyToTerminate: () -> Void = { NSApp.reply(toApplicationShouldTerminate: true) }
    /// The panel appears only when the save is not instant.
    static let panelAfter: Duration = .milliseconds(400)

    /// The reason a quit was asked for by a Quit Apple event carries, `.user` without one.
    static func reason(of event: NSAppleEventDescriptor?) -> QuitReason {
        let code = event?.paramDescriptor(forKeyword: AEKeyword(kAEQuitReason))?.enumCodeValue
        return QuitReason(code: code == 0 ? nil : code)
    }

    private var pendingReason: QuitReason?
    private var questionOpen = false
    private var questionAnswered = false
    private var lossCache: [String: Int] = [:]
    private var waitLimitSeconds = QuitWording.otherWaitSeconds

    func applicationWillFinishLaunching(_ notification: Notification) {
        // The Dock's Quit and the system's log out, restart and shut down send a quit Apple event; AppKit would call `terminate`
        // for it, which is ignored while a sheet is attached. This handler ends the sheets first (`requestQuit`).
        NSAppleEventManager.shared().setEventHandler(
            self, andSelector: #selector(handleQuitEvent(_:withReplyEvent:)),
            forEventClass: AEEventClass(kCoreEventClass), andEventID: AEEventID(kAEQuitApplication))
    }

    @objc func handleQuitEvent(_ event: NSAppleEventDescriptor, withReplyEvent reply: NSAppleEventDescriptor) {
        pendingReason = Self.reason(of: event)
        requestQuit()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // The 3D viewer writes control lines to the stdin of `alife stream`; a write to a pipe whose reader has gone must
        // fail with EPIPE (an error `FileHandle.write(contentsOf:)` throws), not kill the app with SIGPIPE.
        signal(SIGPIPE, SIG_IGN)
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // The `alife` window closed with "Keep it running": the run is the reason to stay (docs/APP.md D3). The flag is set when the window
        // is asked to close and read once, here.
        let stays = plan.asksWhenClosingWithRun && keepsAliveForRun
        return plan.quitsAfterLastWindow && !stays
    }

    // MARK: Closing the window (the `alife` window, D3)

    /// What the person chose last time they were asked ("Remember my choice"), read from the app's preferences; the tests replace it.
    var closePreference: () -> ClosePreference = { ClosePreference(stored: UserDefaults.standard.string(forKey: AppShell.closeWithRunKey)) }
    var storeClosePreference: (ClosePreference) -> Void = { UserDefaults.standard.set($0.rawValue, forKey: AppShell.closeWithRunKey) }
    /// The question put to the person who closes the window with a run going; tests replace it, so that no alert is shown.
    var askClose: (CloseQuestion) -> (answer: CloseAnswer, remember: Bool) = { QuitPanel.askToCloseWindow($0) }
    /// The window is closing and the run goes on: the app lives on with no window. Read by `applicationShouldTerminateAfterLastWindowClosed`.
    private(set) var keepsAliveForRun = false
    /// Told when the window closed and the run stays: the world that had been open, if any, and whether it was still starting. Set by the shell.
    var onKeptRunning: ((String?, Bool) -> Void)?

    /// What the app knows when the window is asked to close.
    var closeFacts: CloseFacts {
        let starting = play.phase == .starting
        var making = false
        if starting, case .planet? = play.request { making = true }
        return CloseFacts(runGoing: run.hasRunJob, runName: run.runningRunName, runStopping: run.hasRunJob && run.stopRequested,
                          world: play.isActive ? play.world : nil, worldStarting: starting, planetBeingMade: making)
    }

    /// `windowShouldClose` of the main window: with a run going, asks "Keep it running / Stop and quit / Cancel" (or keeps it running when
    /// that was remembered); with a planet being made, asks "Stop making 'X' and quit?". True lets the window close. "Stop and Quit" does not
    /// close it: the quit goes on with the window up, through the same gate and panel as a Command-Q, and the question is not asked again.
    /// Called once per attempt to close, so a Cancel leaves everything as it was, window included. **The facts are read again after the
    /// question**: a run that ended while the person read it leaves nothing to keep, and a planet that began meanwhile is a different question.
    func mainWindowShouldClose() -> Bool {
        guard plan.asksWhenClosingWithRun else { return true }
        guard !questionOpen else { return false }  // (a second attempt while the question is up is the first one's business)
        return decideClose(attempt: 0)
    }

    private func decideClose(attempt: Int) -> Bool {
        let facts = closeFacts
        switch SessionRules.decideClose(facts, preference: closePreference()) {
        case .closeAndQuit:
            keepsAliveForRun = false
            return true
        case .keepRunning:
            keepsAliveForRun = true
            return true
        case .stopAndQuit:
            beginStopAndQuit()
            return false
        case .ask(let question):
            questionOpen = true
            let (answer, remember) = askClose(question)
            questionOpen = false
            if let preference = SessionRules.remembered(answer, remember: remember) { storeClosePreference(preference) }
            if answer != .cancel, closeFacts.situation != facts.situation, attempt < 2 {
                return decideClose(attempt: attempt + 1)  // the question no longer describes what is going on
            }
            switch answer {
            case .keepRunning:
                keepsAliveForRun = true
                return true
            case .stopAndQuit:
                beginStopAndQuit()
                return false
            case .cancel:
                return false
            }
        }
    }

    /// The person chose to stop the run and quit from the close question: the quit is asked for without asking again.
    private func beginStopAndQuit() {
        keepsAliveForRun = false
        questionAnswered = true
        requestQuit()
    }

    /// The window has closed and the run stays: what only the window needed ends (R3). The open world is saved and closed through its
    /// ladder (the app lives on while it saves), the viewer's stream stops, and the shell is told so that the Worlds page can say what became
    /// of the world when the window comes back. The run is not touched.
    private func closeWhatTheWindowNeeded() {
        let world = play.isActive ? play.world : nil
        let starting = play.phase == .starting
        play.stop()
        stopStream()
        onKeptRunning?(world, starting)
    }

    /// Stops the 3D viewer's stream (the shared one; tests replace it).
    var stopStream: () -> Void = { StreamController.shared.stop() }

    // MARK: Reopening (the alife window)

    /// Brings the main window of the `alife` window back (set by the shell when its content appears: it is the `openWindow` action of the
    /// scene, which still works after the window has closed). Docs/APP.md spike S3: SwiftUI does not reopen a closed lone `Window` on a
    /// Dock click by itself.
    var reopenMainWindow: (() -> Void)?

    /// A click on the Dock icon (or `open` on the running app) with the main window not showing: it comes back, whatever kept the app
    /// alive (a Cancel at the quit question, a window that was left open). AppKit's own `hasVisibleWindows` is not used: on this macOS the
    /// quit question's alert leaves a glass-effect window that counts as visible long after it was answered (seen in the running app: the
    /// reopen event found "visible windows" and did nothing). The old apps keep AppKit's own handling.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        guard plan.isShell, reopenMainWindow != nil else { return true }
        // (shown: AppKit does the rest; not shown: the window is asked for, and AppKit is told it has nothing to do)
        return !showMainWindowIfNeeded()
    }

    /// The main window of the `alife` window is on screen.
    var mainWindowIsShown: Bool {
        windowsProvider().contains { $0.isVisible && !$0.isMiniaturized && $0.identifier?.rawValue == AppShell.mainWindowID }
    }

    /// The windows the reopen and the close look at (tests replace it).
    var windowsProvider: () -> [NSWindow] = { NSApp.windows }

    /// The main window of the `alife` window closed: it is the app's window, so closing it is closing the last one. The other windows
    /// it leaves (Settings, the 3D Viewer) go with it, which lets AppKit end the app as it does after any last window
    /// (`applicationShouldTerminateAfterLastWindowClosed`), through the question and the wait of the quit. `windows` is injectable.
    func mainWindowClosed(windows: (() -> [NSWindow])? = nil) {
        guard plan.isShell else { return }
        for window in (windows ?? windowsProvider)() where window.isVisible && !(window is NSPanel) && window.identifier?.rawValue != AppShell.mainWindowID {
            window.close()
        }
        if plan.asksWhenClosingWithRun, keepsAliveForRun { closeWhatTheWindowNeeded() }
    }

    /// Brings the main window back if it is not showing (a Dock click does it through `applicationShouldHandleReopen`; the menus do it
    /// when a command needs a window and there is none). True when the window was asked for.
    @discardableResult
    func showMainWindowIfNeeded() -> Bool {
        guard plan.isShell, let reopenMainWindow else { return false }
        guard !mainWindowIsShown else { return false }
        // A window in the Dock is brought out of it; `openWindow` does not do that for a window that exists.
        if let minimised = windowsProvider().first(where: { $0.isMiniaturized && $0.identifier?.rawValue == AppShell.mainWindowID }) {
            minimised.deminiaturize(nil)
            return true
        }
        reopenMainWindow()
        return true
    }

    // MARK: Quit

    /// Whether the user is to be asked "Stop 'X' and quit?": a run is going and not already stopping, and no wait is in progress.
    private var questionNeeded: Bool {
        plan.confirmsRunOnQuit && !terminationGate.isWaiting && run.hasRunJob && !run.stopRequested
    }

    /// The Quit command and the quit Apple event. While a sheet is attached to a window AppKit ignores `terminate` and never
    /// asks the delegate (the launcher sheet of Storm Worlds, the preset comparison of the Lab): the sheets are ended and
    /// `terminate` follows on the next turn, so that the question and the panel can appear. The question comes first when it is
    /// to be asked, so that Cancel leaves the sheets where they were.
    func requestQuit() {
        let attached = NSApp.windows.flatMap { window in window.sheets.map { (window, $0) } }
        if !attached.isEmpty {
            let reason = pendingReason ?? quitReason()
            if reason.asksQuestion, questionNeeded, !questionAnswered {
                guard ask() else { pendingReason = nil; return }
                questionAnswered = true
            }
            // A SwiftUI sheet is dismissed through its own cancel action (Escape), which keeps its state in step; a sheet that does
            // not answer that is ended by AppKit after half a second.
            for (_, sheet) in attached { Self.escape(sheet) }
        }
        // `terminate` is asked for once the sheets have really gone: ending a sheet animates, and AppKit would ignore it until then.
        Task { @MainActor [weak self] in
            var polls = 0
            while NSApp.windows.contains(where: { !$0.sheets.isEmpty }), polls < 60 {
                try? await Task.sleep(for: .milliseconds(50))
                polls += 1
                if polls == 10 {
                    for window in NSApp.windows { for sheet in window.sheets { window.endSheet(sheet) } }
                }
            }
            // From a run-loop block, neither from inside this task nor from a dispatch block: `terminate` blocks in AppKit's own
            // loop while the quit waits (`.terminateLater`), and the gate's polling is a main-actor task on the main dispatch
            // queue, which cannot run while another job or block of it (this one) is still going. Found by running the app.
            RunLoop.main.perform(inModes: [.common]) { MainActor.assumeIsolated { self?.terminate() } }
        }
    }

    /// Sends Escape to a sheet: the cancel action of a sheet that has one ("Done", "Close").
    private static func escape(_ sheet: NSWindow) {
        guard let event = NSEvent.keyEvent(
            with: .keyDown, location: .zero, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
            windowNumber: sheet.windowNumber, context: nil, characters: "\u{1b}", charactersIgnoringModifiers: "\u{1b}",
            isARepeat: false, keyCode: 53) else { return }
        if !sheet.performKeyEquivalent(with: event) { sheet.cancelOperation(nil) }
    }

    private func ask() -> Bool {
        questionOpen = true
        defer { questionOpen = false }
        return confirmStopRun(run.runningRunName ?? "this run")
    }

    /// 1. A run is going and the user quit: ask "Stop 'X' and quit?" first (Cancel keeps everything as it was); not at the end of
    /// the session, where nobody is there to answer. 2. The gate: a played world, a run, a setup or a check is asked to stop
    /// and the quit waits for it, with the panel; the user can give up waiting, and the wait has a limit (`QuitWording`).
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // A second quit while the question is up (the Dock's, in the modal loop) is the first one's business.
        if questionOpen { return .terminateCancel }
        let reason = pendingReason ?? quitReason()
        pendingReason = nil
        let answered = questionAnswered
        questionAnswered = false
        if reason.asksQuestion, !answered, questionNeeded {
            guard ask() else { return .terminateCancel }
        }
        let (play, run) = (play, run)
        let waitsForRun = plan.quitWaitsForRun
        let limit = QuitWording.waitSeconds(runGoing: waitsForRun && run.hasRunJob)
        if !terminationGate.isWaiting {
            waitLimitSeconds = limit
            lossCache = [:]
        }
        terminationGate.limit = { .seconds(limit) }
        terminationGate.onWaiting = { [weak self] elapsed in self?.showQuitPanel(after: elapsed) }
        quitPanel.onQuitNow = { [weak self] in self?.terminationGate.abandonWait() }
        return terminationGate.shouldTerminate(
            hasProcess: { play.hasProcess || (waitsForRun && run.hasJob) },
            stop: {
                play.stopForQuit()
                if waitsForRun { run.stopForQuit() }
            },
            reply: { [weak self] in
                self?.quitPanel.close()
                self?.replyToTerminate()
            })
    }

    func applicationWillTerminate(_ notification: Notification) {
        // The last resort for whatever the gate's wait left running (it timed out, or the user quit without saving).
        run.shutdown(escalating: plan.quitWaitsForRun)
    }

    /// What the panel shows `elapsed` into the wait; nil while the save is still quick.
    func quitPanelModel(after elapsed: Duration) -> QuitPanelModel? {
        guard elapsed >= Self.panelAfter else { return nil }
        let worlds = play.retiredWorlds
        return QuitPanelModel(
            worlds: worlds, hasRun: run.hasRunJob, runName: run.runningRunName, other: run.otherJobName,
            elapsedSeconds: Int(elapsed.components.seconds), limitSeconds: waitLimitSeconds,
            lossMinutes: QuitWording.lossMinutes(worlds: worlds, repo: play.directory, cache: &lossCache))
    }

    private func showQuitPanel(after elapsed: Duration) {
        if let model = quitPanelModel(after: elapsed) { quitPanel.show(model) }
    }
}

/// The Quit command of the app menu, in place of SwiftUI's: it goes through `AppDelegate.requestQuit`, which ends attached sheets
/// first (AppKit ignores `terminate` while one is attached).
struct QuitCommands: Commands {
    let appName: String

    var body: some Commands {
        CommandGroup(replacing: .appTermination) {
            Button("Quit \(appName)") {
                if let delegate = AppDelegate.current { delegate.requestQuit() } else { NSApp.terminate(nil) }
            }
            .keyboardShortcut("q")
        }
    }
}

/// The entry point: one executable, three apps (`LaunchMode`): `alife` (one window with a sidebar, `AlifeApp`), and the two it will replace. The Lab is today's app (the main window, the viewer and the
/// World window as a second window); Storm Worlds opens straight into the World window and has the Lab one menu item away
/// (`StormWorldsApp`). The mode is read from the bundle, so `swift run` and a bundle from before the key start the Lab.
@main
enum AlifeEntry {
    static func main() {
        let mode = LaunchMode.current()
        switch mode {
        case .alife:
            // The preferences of the two old apps are brought in once, before anything reads the folder setting. Only a bundle made by
            // make-app.sh has its own preferences domain: an unbundled run (`swift run`) keeps whatever it had, as it always did.
            if Bundle.main.object(forInfoDictionaryKey: LaunchMode.infoKey) != nil { PreferenceMigration.forApp()?.run() }
            AlifeApp.main()
        case .world: StormWorldsApp.main()
        case .lab: AlifeControlApp.main()
        }
    }
}

struct AlifeControlApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var controller = RunController.shared

    var body: some Scene {
        WindowGroup("Alife Control") {
            ContentView(controller: controller)
                .frame(minWidth: 960, minHeight: 700)
        }
        .windowResizability(.contentMinSize)
        .commands { QuitCommands(appName: LaunchMode.lab.plan.appName) }

        Viewer3DScene()

        // The game: the map and the organisms with the control center docked beside them. Opened from the Lab (Open
        // World) and, with `--world RUN`, at launch (`ContentView` opens it: a window that is not the first scene is not
        // opened by SwiftUI at launch).
        Window("World", id: AppWindow.world.rawValue) {
            WorldView(play: PlayController.shared)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1280, height: 800)
    }
}

/// The 3D viewer window, a second window in both apps.
struct Viewer3DScene: Scene {
    var body: some Scene {
        Window("3D Viewer", id: AppWindow.viewer3d.rawValue) {
            Viewer3DView(stream: StreamController.shared)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1000, height: 760)
    }
}
