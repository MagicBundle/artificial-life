import AlifeCore
import SwiftUI

/// The Lab as a page: today's form, kept until the set-up page replaces it (PR-D). It keeps its own labels (Resume, Fork, Open viewer)
/// until then; the note at its top says what the page is for, in plain words.
struct LabPage: View {
    @ObservedObject var controller: RunController
    var openPage: (ShellPage) -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "info.circle").foregroundStyle(.secondary)
                Text("Research runs start here. A run with a save point can become a planet on the Worlds page.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.4))
            ContentView(controller: controller, embedded: true, openPage: openPage)
        }
    }
}
