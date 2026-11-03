import AlifeCore
import AppKit
import SwiftUI

/// The alife folder: the field, "Choose...", and what is wrong with it (Settings, and Welcome where a folder is to be chosen).
struct RepoFolderForm: View {
    @ObservedObject var controller: RunController

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                TextField(RepoFolder.fieldPrompt, text: $controller.repoPath)
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
                DetectedFolderButton(controller: controller)
            }
        }
        // The folder can be fixed outside the app (a `uv sync`, a restored drive): look again when the app is brought back.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            controller.refreshRepo()
        }
    }
}

/// "Use <folder>": offered while the chosen folder is missing and a usual place (`RepoFolder.candidates`) holds the engine. The
/// missing folder is never replaced without this click (it may be on a drive that is not connected); "Choose..." picks another.
struct DetectedFolderButton: View {
    @ObservedObject var controller: RunController

    var body: some View {
        if let detected = controller.detectedRepo {
            Button("Use \(detected) instead") { controller.useDetectedFolder() }
                .font(.caption)
                .help("The folder above stays chosen until you click this; it is not replaced when it comes back.")
        }
    }
}
