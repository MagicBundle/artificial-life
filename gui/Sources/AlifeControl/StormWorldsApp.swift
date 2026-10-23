import AlifeCore
import AppKit
import SwiftUI

/// Storm Worlds: the same executable as the Lab (`LaunchMode.world`), opening straight into the game. The World window is
/// the first scene, so it is the one SwiftUI opens at launch (on its launcher sheet); the Lab is a second window, one
/// Window-menu item away (Command-2) and not reopened at the next launch. The World window's first scene is also why
/// `WorldLaunchArgument` is handled there (`WorldWindowRoot`) and not in `ContentView`.
struct StormWorldsApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var controller = RunController.shared
    private let plan = LaunchMode.world.plan

    var body: some Scene {
        Window(plan.appName, id: AppWindow.world.rawValue) {
            WorldWindowRoot(controller: controller, play: PlayController.shared)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1280, height: 800)
        // A `Window` scene adds an item to the Window menu that opens it; those two items (without key equivalents) are
        // replaced by `WindowMenuCommands`', which have them.
        .commands { WindowMenuCommands(items: plan.windowMenu) }

        Window("Alife Control", id: AppWindow.lab.rawValue) {
            ContentView(controller: controller)
                .frame(minWidth: 960, minHeight: 700)
        }
        .windowResizability(.contentMinSize)

        Viewer3DScene()

        // Settings (Command-comma): the repository folder, without opening the Lab.
        Settings {
            RepoSettingsView(controller: controller)
        }
    }
}

/// The Window menu's entries that open the World and Lab windows, with Command-1 and Command-2. SwiftUI lists every `Window`
/// scene there already, by title and without a key equivalent; `before: .windowList` keeps those and "Bring All to Front"
/// (`replacing:` and `commandsRemoved()` both lose it).
struct WindowMenuCommands: Commands {
    let items: [WindowMenuItem]
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(before: .windowList) {
            ForEach(items) { item in
                Button(item.title) { openWindow(id: item.window.rawValue) }
                    .keyboardShortcut(KeyEquivalent(item.key), modifiers: .command)
            }
            Divider()
        }
    }
}

/// The content of Storm Worlds' World window. While there is no usable repository folder and no world is open, the launcher
/// has nothing to list, so the window asks for the folder instead (`RepoGateView`); the World view is created once there
/// is one. A world that is open keeps its window whatever happens to the folder.
struct WorldWindowRoot: View {
    @ObservedObject var controller: RunController
    @ObservedObject var play: PlayController
    /// `--world RUN` is honoured once, when the window first has a usable repository.
    @State private var argumentHandled = false

    var body: some View {
        Group {
            if controller.repoReady || play.phase != .idle {
                WorldView(play: play)
                    .onAppear { startWorldFromArgument() }
            } else {
                RepoGateView(controller: controller)
            }
        }
    }

    private func startWorldFromArgument() {
        guard !argumentHandled else { return }
        argumentHandled = true
        if let run = WorldLaunchArgument.parse(CommandLine.arguments) { _ = controller.openWorldAtLaunch(run) }
    }
}

/// The repository folder: the field, "Choose...", and what is wrong with it. The same words as the Lab's Repository box.
struct RepoFolderForm: View {
    @ObservedObject var controller: RunController

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                TextField(RepoFolder.defaultPath, text: $controller.repoPath)
                    .textFieldStyle(.roundedBorder)
                Button("Choose...") { controller.chooseRepo() }
            }
            if controller.repoReady {
                Label("Found .venv/bin/alife", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.caption)
            } else {
                Label(controller.repoProblem ?? ".venv/bin/alife not found in this folder.",
                      systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.caption)
            }
        }
        // The folder can be fixed outside the app (a `uv sync`, a restored drive): look again when the app is brought back.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            controller.refreshRepo()
        }
    }
}

/// What the World window shows instead of the game while the repository folder is missing or has no engine.
struct RepoGateView: View {
    @ObservedObject var controller: RunController
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(spacing: 16) {
            Text("Storm Worlds").font(.largeTitle)
            Text("The game plays the worlds of an artificial-life repository, with its Python environment. Choose the folder that holds it.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            RepoFolderForm(controller: controller)
                .frame(maxWidth: 520)
            HStack {
                Button("Re-check") { controller.refreshRepo() }
                if controller.repoFolderExists {
                    Button("Open the Lab to set up the environment") { openWindow(id: AppWindow.lab.rawValue) }
                        .help("The Lab's Repository box has \"Set up environment\", which runs uv sync in the folder.")
                }
            }
            Text("Also in Settings (Command-comma). The Lab uses the same folder.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .frame(minWidth: 1100, minHeight: 700)
        .navigationTitle("Storm Worlds")
    }
}

/// The Settings window of Storm Worlds.
struct RepoSettingsView: View {
    @ObservedObject var controller: RunController

    var body: some View {
        Form {
            Section("Repository folder") {
                RepoFolderForm(controller: controller)
                Text("The folder with .venv/bin/alife and runs/. The Lab app uses the same one.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding(8)
        .frame(width: 520)
    }
}
