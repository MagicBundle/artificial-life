import AlifeCore
import SwiftUI

/// Settings > General of the `alife` window (docs/APP.md §2.4): where the alife folder is and whether its engine is set up. The same
/// folder form as the Worlds page's first card; the old apps keep their own Settings until they are retired.
struct GeneralSettingsView: View {
    @ObservedObject var controller: RunController

    var body: some View {
        TabView {
            Form {
                Section("alife folder") {
                    RepoFolderForm(controller: controller)
                    Text("The folder with .venv/bin/alife and runs/. The other alife apps use the same one.")
                        .font(.caption).foregroundStyle(.secondary)
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
        .frame(width: 540, height: 360)
    }
}
