import AlifeCore
import AppKit
import SwiftUI

/// `alife`: one window, a sidebar of pages (docs/APP.md §2.4, PR-B). Built beside the two old apps and installed as `alife Preview.app`;
/// the old modes (`AlifeControlApp`, `StormWorldsApp`) are untouched.
struct AlifeApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var controller = RunController.shared
    @StateObject private var shell = ShellModel(session: .shared)
    @StateObject private var library: WorldLibrary
    @StateObject private var setup: SetupModel
    private let appName = (Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String) ?? LaunchMode.alife.plan.appName

    init() {
        // (the set-up model reads the library, so both are made here and handed to the scenes)
        let library = WorldLibrary.forApp(run: .shared)
        _library = StateObject(wrappedValue: library)
        _setup = StateObject(wrappedValue: SetupModel.forApp(controller: .shared, play: PlayController.shared, library: library))
    }

    var body: some Scene {
        Window(appName, id: AppShell.mainWindowID) {
            AppShellView(shell: shell, controller: controller, library: library, setup: setup)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1280, height: 800)
        .commands {
            QuitCommands(appName: appName)
            ShellCommands(shell: shell, controller: controller, library: library, setup: setup)
        }

        // The set-up page in a window of its own, so that a form can be composed beside a Watch (docs/APP.md §3.7). One instance.
        Window("Set up", id: AppShell.setupWindowID) {
            SetupWindowView(setup: setup, controller: controller, library: library)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 820, height: 760)

        // The 3D viewer of a research world's card ("Watch in 3D viewer") opens this window until PR-F makes the observer a form of Watch.
        Viewer3DScene()

        Settings {
            GeneralSettingsView(controller: controller)
        }
    }
}

/// The set-up page in its window: the same model as the page of the main window.
struct SetupWindowView: View {
    @ObservedObject var setup: SetupModel
    @ObservedObject var controller: RunController
    @ObservedObject var library: WorldLibrary

    var body: some View {
        SetupPage(model: setup, controller: controller, coordinator: setup.coordinator, library: library, launcher: setup.launcher, inWindow: true)
            .frame(minWidth: 640, minHeight: 560)
    }
}

/// The menu bar of `alife`: File (new world, make a planet, close the world, reveal in Finder) and the Window menu's pages with their Command keys.
/// It observes the shell only (not the play controller, which publishes with every frame: a menu rebuilt that often is unusable).
struct ShellCommands: Commands {
    @ObservedObject var shell: ShellModel
    let controller: RunController
    let library: WorldLibrary
    let setup: SetupModel

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New World…") { setup.openCreate() }
                .keyboardShortcut("n")
            Button("Make Planet…") { shell.requestPlanetForm() }
            Divider()
            Button("Close World") { shell.closeWorld() }
                .disabled(!shell.worldIsOpen)
            Button("Reveal in Finder") {
                // The world that is chosen while the Worlds page is in view (a world that is gone is told, and no other folder is shown in
                // its place); on any other page the Lab's run, else the alife folder.
                Task { await library.revealFromMenu(page: shell.page) { controller.revealInFinder() } }
            }
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
