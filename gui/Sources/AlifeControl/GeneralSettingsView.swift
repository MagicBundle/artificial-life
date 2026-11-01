import AlifeCore
import SwiftUI

/// Settings > General of the `alife` window (docs/APP.md §2.4): where the alife folder is and whether its engine is set up. The same
/// folder form as the Worlds page's first card; the old apps keep their own Settings until they are retired.
struct GeneralSettingsView: View {
    @ObservedObject var controller: RunController
    /// What closing the window does while a run is going (D3), and whether a world paused by leaving Watch continues when the player comes
    /// back (D4). The app's own preferences; the question's "Remember my choice" sets the first.
    @AppStorage(AppShell.closeWithRunKey) private var closeWithRun = ClosePreference.ask.rawValue
    @AppStorage(AppShell.pickUpKey) private var pickUp = true

    var body: some View {
        TabView {
            Form {
                Section("alife folder") {
                    RepoFolderForm(controller: controller)
                    Text("The folder with .venv/bin/alife and runs/. This preview keeps its own choice: the other alife apps are not changed by it.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("While you are away") {
                    Picker("When I close the window while a run is going", selection: $closeWithRun) {
                        ForEach(ClosePreference.allCases) { Text($0.title).tag($0.rawValue) }
                    }
                    Text(ClosePreference(stored: closeWithRun).effect).font(.caption).foregroundStyle(.secondary)
                    Toggle("Pick up where I left off", isOn: $pickUp)
                    Text(pickUp
                         ? "Leaving the Watch page pauses the world you are playing, and it continues when you come back. A world you paused yourself, or that stopped at a chapter or an alert, stays paused."
                         : "Leaving the Watch page pauses the world you are playing, and it stays paused until you press Run in Watch.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                Section("Engine") {
                    if controller.repoReady {
                        Label("The engine is set up", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    } else {
                        Label(controller.repoFolderExists ? "The engine is not set up in this folder." : "Choose the alife folder first.",
                              systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                        HStack {
                            Button("Set up engine") { controller.setupEnvironment() }
                                .disabled(!controller.repoFolderExists || controller.job != nil)
                            if !controller.repoFolderExists {
                                Text("It needs an existing folder.").font(.caption).foregroundStyle(.secondary)
                            } else if controller.job == .setup {
                                Text("Setting up. Progress is in Activity.").font(.caption).foregroundStyle(.secondary)
                            } else if controller.job != nil {
                                Text("Another job is running.").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        Text("Runs `uv sync` in the folder. It installs what is missing and removes nothing.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)
            .tabItem { Label("General", systemImage: "gearshape") }
        }
        .frame(width: 540, height: 620)
    }
}
