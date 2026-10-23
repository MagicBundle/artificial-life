import AlifeCore
import AppKit
import SwiftUI

/// Makes the app a regular foreground app (needed when run unbundled with `swift run`; harmless in the .app
/// bundle) and stops a running simulation cleanly when the app quits.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Holds the quit while a played world saves its checkpoint (`TerminationGate`).
    let terminationGate = TerminationGate()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // The 3D viewer writes control lines to the stdin of `alife stream`; a write to a pipe whose reader has gone must
        // fail with EPIPE (an error `FileHandle.write(contentsOf:)` throws), not kill the app with SIGPIPE.
        signal(SIGPIPE, SIG_IGN)
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        LaunchMode.current().plan.quitsAfterLastWindow
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        terminationGate.shouldTerminate(
            hasProcess: { PlayController.shared.hasProcess },
            stop: { PlayController.shared.stopForQuit() },
            reply: { NSApp.reply(toApplicationShouldTerminate: true) })
    }

    func applicationWillTerminate(_ notification: Notification) {
        RunController.shared.shutdown()  // the last resort for whatever the gate's wait left running
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
