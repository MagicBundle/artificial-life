import AppKit
import SwiftUI

/// Makes the app a regular foreground app (needed when run unbundled with `swift run`; harmless in the .app
/// bundle) and stops a running simulation cleanly when the app quits.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // The 3D viewer writes control lines to the stdin of `alife stream`; a write to a pipe whose reader has gone must
        // fail with EPIPE (an error `FileHandle.write(contentsOf:)` throws), not kill the app with SIGPIPE.
        signal(SIGPIPE, SIG_IGN)
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        RunController.shared.shutdown()
    }
}

@main
struct AlifeControlApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var controller = RunController.shared

    var body: some Scene {
        WindowGroup("Alife Control") {
            ContentView(controller: controller)
                .frame(minWidth: 960, minHeight: 700)
        }
        .windowResizability(.contentMinSize)

        Window("3D Viewer", id: "viewer3d") {
            Viewer3DView(stream: StreamController.shared)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1000, height: 760)
    }
}
