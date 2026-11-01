import AlifeCore
import AppKit
import SwiftUI

/// Makes the app a regular foreground app (needed when run unbundled with `swift run`; harmless in the .app
/// bundle) and stops a running simulation cleanly when the app quits. Shared by both modes (`LaunchPlan`).
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Holds the quit while a played world or a run saves its checkpoint (`TerminationGate`).
    let terminationGate = TerminationGate()
    /// Shown while the quit waits. Owned here, not by a scene, so that it exists when the last window has been closed.
    let quitPanel = QuitPanel()
    var plan = LaunchMode.current().plan
    private let run: RunController
    private let play: PlayController

    /// The delegate of the app: the shared controllers. The tests give it their own.
    override convenience init() { self.init(run: .shared, play: .shared) }

    init(run: RunController, play: PlayController) {
        self.run = run
        self.play = play
        super.init()
    }
    /// The question before a run is stopped by a quit (tests replace it, so that no alert is shown).
    var confirmStopRun: (String) -> Bool = { QuitPanel.askToStopRun($0) }
    /// The panel appears only when the save is not instant.
    static let panelAfter: Duration = .milliseconds(400)

    func applicationDidFinishLaunching(_ notification: Notification) {
        // The 3D viewer writes control lines to the stdin of `alife stream`; a write to a pipe whose reader has gone must
        // fail with EPIPE (an error `FileHandle.write(contentsOf:)` throws), not kill the app with SIGPIPE.
        signal(SIGPIPE, SIG_IGN)
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        plan.quitsAfterLastWindow
    }

    /// 1. A run is going: ask "Stop 'X' and quit?" first (Cancel keeps everything as it was). 2. The gate: a played world or a
    /// run is asked to stop and the quit waits for it, with the panel; the user can give up waiting.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let (play, run) = (play, run)
        let waitsForRun = plan.quitWaitsForRun
        if plan.confirmsRunOnQuit, !terminationGate.isWaiting, run.hasRunJob, !run.stopRequested {
            guard confirmStopRun(run.runningRunName ?? "this run") else { return .terminateCancel }
        }
        terminationGate.onWaiting = { [weak self] elapsed in self?.showQuitPanel(after: elapsed) }
        quitPanel.onQuitNow = { [weak self] in self?.terminationGate.abandonWait() }
        return terminationGate.shouldTerminate(
            hasProcess: { play.hasProcess || (waitsForRun && run.hasRunJob) },
            stop: {
                play.stopForQuit()
                if waitsForRun { run.stop() }
            },
            reply: { [weak self] in
                self?.quitPanel.close()
                NSApp.reply(toApplicationShouldTerminate: true)
            })
    }

    func applicationWillTerminate(_ notification: Notification) {
        // The last resort for whatever the gate's wait left running (it timed out, or the user quit without saving).
        run.shutdown(escalating: plan.quitWaitsForRun)
    }

    private func showQuitPanel(after elapsed: Duration) {
        guard elapsed >= Self.panelAfter else { return }
        let worlds = play.retiredWorlds
        let repo = play.directory
        let minutes = worlds.compactMap { world -> Int? in
            guard let repo else { return nil }
            let file = repo.appendingPathComponent("runs/\(world)/play.json")
            return QuitWording.lossMinutes(baseRate: QuitWording.baseRate(playFile: file))
        }.max()
        quitPanel.show(QuitPanelModel(
            worlds: worlds, run: run.hasRunJob ? run.runningRunName : nil,
            elapsedSeconds: Int(elapsed.components.seconds), lossMinutes: minutes))
    }
}

/// The entry point: one executable, two apps (`LaunchMode`). The Lab is today's app (the main window, the viewer and the
/// World window as a second window); Storm Worlds opens straight into the World window and has the Lab one menu item away
/// (`StormWorldsApp`). The mode is read from the bundle, so `swift run` and a bundle from before the key start the Lab.
@main
enum AlifeEntry {
    static func main() {
        switch LaunchMode.current().plan.firstScene {
        case .world: StormWorldsApp.main()
        case .lab, .viewer3d: AlifeControlApp.main()
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
