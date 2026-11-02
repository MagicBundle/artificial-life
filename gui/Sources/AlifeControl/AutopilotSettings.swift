import AlifeCore
import SwiftUI

/// Settings > Autopilot (docs/APP.md §2.4, §3.8 row 14): the address and the name of the local language model the autopilot may ask, and the
/// button that tests them. They used to be set per run, in the Lab; now each new world starts from these, and a world can still change them under
/// Advanced > Autopilot. Kept in the app's own preferences; empty or unusable values mean the engine's defaults.
@MainActor
final class AutopilotSettings: ObservableObject {
    @Published var url: String { didSet { save(url, AppShell.autopilotURLKey); forgetResult() } }
    @Published var model: String { didSet { save(model, AppShell.autopilotModelKey); forgetResult() } }
    @Published private(set) var check: LLMCheckState = .idle

    private let defaults: UserDefaults
    private let controller: RunController
    /// Runs `alife llm-check`; the tests give their own so that nothing is contacted.
    var runCheck: (URL, [String], URL) async -> LLMCheckResult = { exe, args, dir in await LLMCheck.run(executable: exe, arguments: args, directory: dir) }

    init(controller: RunController, defaults: UserDefaults = .standard) {
        self.controller = controller
        self.defaults = defaults
        url = defaults.string(forKey: AppShell.autopilotURLKey) ?? DirectorDefaults.url
        model = defaults.string(forKey: AppShell.autopilotModelKey) ?? DirectorDefaults.model
    }

    private func save(_ value: String, _ key: String) { defaults.set(value, forKey: key) }

    private func forgetResult() { if check != .running { check = .idle } }

    var trimmedURL: String { url.trimmingCharacters(in: .whitespacesAndNewlines) }
    var trimmedModel: String { model.trimmingCharacters(in: .whitespacesAndNewlines) }
    var endpointValid: Bool { DirectorSettings.isHTTPURL(trimmedURL) && !trimmedModel.isEmpty }
    var isDefault: Bool { trimmedURL == DirectorDefaults.url && trimmedModel == DirectorDefaults.model }

    func useDefaults() {
        url = DirectorDefaults.url
        model = DirectorDefaults.model
    }

    /// What a new world's form starts from: these values when they are usable, else the engine's (a half-typed address never reaches a command).
    func apply(to director: inout DirectorSettings) {
        guard endpointValid else { return }
        director.llmURL = trimmedURL
        director.llmModel = trimmedModel
    }

    /// Why the test cannot run, in words; nil when it can.
    var testReason: String? {
        if !controller.repoReady { return "Set up the engine first." }
        if !endpointValid { return "Type an http or https address and a model name." }
        if check == .running { return "Testing…" }
        return nil
    }

    func test() {
        guard testReason == nil else { return }
        let url = trimmedURL, model = trimmedModel
        let executable = controller.alifeURL, directory = controller.repoURL
        check = .running
        Task {
            let result = await runCheck(executable, ["llm-check", "--llm-url=\(url)", "--llm-model=\(model)"], directory)
            // The fields may have been edited while it ran; a result for other values would mislead.
            guard trimmedURL == url, trimmedModel == model else {
                check = .idle
                return
            }
            check = .result(result)
        }
    }
}

struct AutopilotSettingsView: View {
    @ObservedObject var settings: AutopilotSettings

    var body: some View {
        Form {
            Section("Language model") {
                Text("The autopilot can ask a language model that runs on your Mac (for example in LM Studio) to choose its moves. Nothing is sent anywhere else. Each new world starts from this address and name; a world can change them under Advanced > Autopilot.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                TextField("Address", text: $settings.url, prompt: Text(DirectorDefaults.url))
                    .textFieldStyle(.roundedBorder)
                TextField("Model name", text: $settings.model, prompt: Text(DirectorDefaults.model))
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Button("Test connection") { settings.test() }
                        .disabled(settings.testReason != nil)
                        .help("Sends one tiny request to the address and shows whether the model answers (alife llm-check).")
                    Button("Use the defaults") { settings.useDefaults() }
                        .disabled(settings.isDefault)
                    result
                }
                if let reason = settings.testReason, settings.check != .running {
                    Text(reason).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    @ViewBuilder private var result: some View {
        switch settings.check {
        case .idle:
            EmptyView()
        case .running:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Testing…").font(.caption).foregroundStyle(.secondary)
            }
        case .result(let result):
            Label(result.text, systemImage: result.ok ? "checkmark.circle.fill" : "xmark.octagon.fill")
                .font(.caption)
                .foregroundStyle(result.ok ? Color.green : Color.red)
                .textSelection(.enabled)
        }
    }
}
