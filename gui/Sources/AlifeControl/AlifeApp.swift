import AlifeCore
import AppKit
import SwiftUI

/// `alife`: one window, a sidebar of pages (docs/APP.md §2.4, PR-B). Built beside the two old apps and installed as `alife Preview.app`;
/// the old modes (`AlifeControlApp`, `StormWorldsApp`) are untouched.
struct AlifeApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var controller = RunController.shared
    @StateObject private var shell = ShellModel(session: .shared)
    private let appName = (Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String) ?? LaunchMode.alife.plan.appName

    var body: some Scene {
        Window(appName, id: AppShell.mainWindowID) {
            AppShellView(shell: shell, controller: controller)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1280, height: 800)
        .commands {
            QuitCommands(appName: appName)
            ShellCommands(shell: shell, controller: controller)
        }

        // The Lab page's "Open 3D viewer" opens this window until PR-F makes the observer a form of Watch.
        Viewer3DScene()

        Settings {
            GeneralSettingsView(controller: controller)
        }
    }
}

/// The menu bar of `alife`: File (make a planet, close the world, reveal in Finder) and the Window menu's pages with their Command keys.
/// It observes the shell only (not the play controller, which publishes with every frame: a menu rebuilt that often is unusable).
struct ShellCommands: Commands {
    @ObservedObject var shell: ShellModel
    let controller: RunController

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Make Planet…") { shell.requestPlanetForm() }
            Divider()
            Button("Close World") { shell.closeWorld() }
                .disabled(!shell.worldIsOpen)
            Button("Reveal in Finder") { controller.revealInFinder() }
                .keyboardShortcut("r", modifiers: [.command, .shift])
        }
        CommandGroup(before: .windowList) {
            ForEach(ShellPage.allCases.sorted { $0.key < $1.key }) { page in
                Button(page.title) { shell.select(page) }
                    .keyboardShortcut(KeyEquivalent(page.key), modifiers: .command)
            }
            Divider()
        }
    }
}
