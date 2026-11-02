import AlifeCore
import SwiftUI

/// Welcome (docs/APP.md §3.1, PR-G): the card the Worlds page shows instead of its next-step card while the alife folder, the engine or the
/// engine's answer is missing. A checklist of three things, what is wrong in a sentence and the one thing to do. The folder form is the
/// Worlds page's old one (`RepoFolderForm`), shown where a folder is to be chosen; the set-up is the controller's `uv sync` job, with its
/// output beside it. Nothing here starts by itself except the check that the engine answers.
struct WelcomeCard: View {
    @ObservedObject var model: WelcomeModel
    @ObservedObject var controller: RunController
    let step: WelcomeGuide.Step

    private var facts: WelcomeGuide.Facts { model.facts(controller) }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(WelcomeGuide.title(step), systemImage: symbol)
                .font(.title2.weight(.semibold))
                .foregroundStyle(tint)
            Text(WelcomeGuide.explanation(step))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            checklist
            Divider()
            actions
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
        // `uv` may have been installed, and the folder mended, while the app was in the background.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            model.refresh(controller)
        }
    }

    private var symbol: String {
        switch step {
        case .chooseFolder, .notAlife: "folder.badge.questionmark"
        case .setUpEngine(let uvFound): uvFound ? "wrench.and.screwdriver" : "shippingbox"
        case .settingUp, .checking: "hourglass"
        case .setupFailed, .checkFailed: "exclamationmark.triangle.fill"
        }
    }

    private var tint: Color {
        switch step {
        case .setupFailed, .checkFailed: .orange
        default: .primary
        }
    }

    // MARK: The checklist

    private var checklist: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(WelcomeGuide.checklist(facts).enumerated()), id: \.offset) { index, item in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Image(systemName: mark(item.mark))
                        .foregroundStyle(markColour(item.mark))
                        .frame(width: 18)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("\(index + 1). \(item.title)")
                            .fontWeight(item.mark == .current ? .semibold : .regular)
                            .foregroundStyle(item.mark == .waiting ? .secondary : .primary)
                        if let detail = item.detail {
                            Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                        }
                    }
                }
                .accessibilityElement(children: .combine)
                .accessibilityValue(item.mark == .done ? "done" : item.mark == .current ? "next" : "waiting")
            }
        }
    }

    private func mark(_ mark: WelcomeGuide.Mark) -> String {
        switch mark {
        case .done: "checkmark.circle.fill"
        case .current: "arrow.right.circle.fill"
        case .waiting: "circle"
        }
    }

    private func markColour(_ mark: WelcomeGuide.Mark) -> Color {
        switch mark {
        case .done: .green
        case .current: .accentColor
        case .waiting: .secondary
        }
    }

    // MARK: What to do

    @ViewBuilder private var actions: some View {
        switch step {
        case .chooseFolder, .notAlife:
            RepoFolderForm(controller: controller)
            HStack {
                Button("Check again") { model.checkAgain(controller) }
                Text("You can change the folder later in Settings (Command-comma).").font(.caption).foregroundStyle(.secondary)
            }
            fetchSteps
        case .setUpEngine(let uvFound):
            if uvFound {
                HStack {
                    Button("Set up engine") { controller.setupEnvironment() }
                        .buttonStyle(.borderedProminent)
                        .disabled(controller.job != nil)
                    Button("Check again") { model.checkAgain(controller) }
                    if controller.job != nil {
                        Text("Another job is running; wait for it to finish.").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Text("Runs `\(WelcomeGuide.setupCommand)` in the folder.").font(.caption).foregroundStyle(.secondary)
            } else {
                uvGuidance
            }
            folderLine
        case .settingUp:
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text("Working...").foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") { controller.cancelSetup() }
                    .help("Stops uv at a clean point. Nothing is removed; Set up again carries on.")
            }
            lines(WelcomeModel.progressLines(controller))
            folderLine
        case .setupFailed(let trouble):
            lines(trouble.lines)
            HStack {
                Button("Set up again") { controller.setupEnvironment() }
                    .buttonStyle(.borderedProminent)
                    .disabled(controller.job != nil || !facts.uvFound)
                Button("Copy log") { model.copyReport(trouble) }
                Button("Check again") { model.checkAgain(controller) }
                if !facts.uvFound { Text("uv was not found: install it first.").font(.caption).foregroundStyle(.secondary) }
            }
            if !facts.uvFound { uvGuidance }
            folderLine
        case .checking:
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text("Waiting for the engine's answer...").foregroundStyle(.secondary)
            }
            folderLine
        case .checkFailed(let trouble):
            lines(trouble.lines)
            HStack {
                Button("Set up again") { controller.setupEnvironment() }
                    .buttonStyle(.borderedProminent)
                    .disabled(controller.job != nil || !facts.uvFound)
                Button("Check again") { model.checkAgain(controller) }
                Button("Copy log") { model.copyReport(trouble) }
            }
            folderLine
        }
    }

    /// "I don't have it": the four lines that fetch alife and set it up, copyable (D2: the app does not fetch it).
    private var fetchSteps: some View {
        DisclosureGroup("I don't have alife yet") {
            VStack(alignment: .leading, spacing: 8) {
                Text(WelcomeGuide.fetchExplanation).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Text(WelcomeGuide.fetchSteps())
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.background.opacity(0.6), in: RoundedRectangle(cornerRadius: 6))
                Button("Copy these lines") { model.copyFolderSteps() }.controlSize(.small)
            }
            .padding(.top, 4)
        }
    }

    /// uv is not on this Mac: where to get it, and the Homebrew line.
    private var uvGuidance: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(WelcomeGuide.uvInstallCommand)
                .font(.callout.monospaced())
                .textSelection(.enabled)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.background.opacity(0.6), in: RoundedRectangle(cornerRadius: 6))
            HStack {
                Button("Copy this line") { model.copyUVCommand() }
                Button("Open the uv page") { model.openUVPage() }
                Button("Check again") { model.checkAgain(controller) }.buttonStyle(.borderedProminent)
            }
            Text("Run the line in Terminal (it needs brew); the uv page has other ways to install uv. Then come back and press Check again.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    /// A way to another folder, once one is chosen (the checklist names the folder; the form is shown only where a folder is to be chosen).
    private var folderLine: some View {
        HStack(spacing: 6) {
            Text("Not this folder?").font(.caption).foregroundStyle(.secondary)
            Button("Choose another...") { controller.chooseRepo() }.buttonStyle(.link).font(.caption)
        }
    }

    @ViewBuilder private func lines(_ text: [String]) -> some View {
        if !text.isEmpty {
            Text(text.joined(separator: "\n"))
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .lineLimit(8)
                .truncationMode(.middle)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.background.opacity(0.6), in: RoundedRectangle(cornerRadius: 6))
        }
    }
}
