import AppKit
import SwiftUI

/// Makes the app a regular foreground app (needed when run unbundled with `swift run`; harmless in the .app
/// bundle) and stops a running simulation cleanly when the app quits.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
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
    }
}
