import SwiftUI

// MARK: - Settings (left column)

/// The "Director" group of the settings column: mode, persona, goal, cadence and the LLM tier.
struct DirectorSettingsBox: View {
    @ObservedObject var controller: RunController

    private var busy: Bool { controller.job != nil }
    private var context: DirectorContext { controller.settings.directorContext }

    var body: some View {
        GroupBox("Director") {
            VStack(alignment: .leading, spacing: 8) {
                if context.available {
                    modePicker
                    modeNote
                    if controller.settings.director.mode == .live { liveControls }
                } else {
                    Text("This run was created without the director; fork it to add one.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let warning = controller.directorWarning {
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .disabled(busy)
    }

    // MARK: Mode

    private var modes: [DirectorMode] {
        controller.settings.mode == .resume ? [.off, .live, .replay] : [.off, .live]
    }

    private var modePicker: some View {
        Picker("Mode", selection: $controller.settings.director.mode) {
            ForEach(modes) { Text($0.label).tag($0) }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
    }

    @ViewBuilder private var modeNote: some View {
        switch (controller.settings.mode, controller.settings.director.mode) {
        case (.fork, _):
            note("The settings are pre-filled from the source run, including Live if it ran live. A fork does not "
                 + "inherit the director: Live is passed as --director live, and the environment levers restart "
                 + "inert (the source's director log is not carried over).")
        case (_, .off):
            note(controller.settings.mode == .resume
                 ? "Passes --director off if the run was live; otherwise nothing."
                 : "No director: the run is the plain engine.")
        case (.resume, .replay):
            note("Replays the run's own director.jsonl from the checkpoint (bitwise identical; the model is never "
                 + "called, and the log is not changed).")
        default:
            EmptyView()
        }
    }

    // MARK: Live controls

    @ViewBuilder private var liveControls: some View {
        HStack {
            Text("Persona").frame(width: 110, alignment: .leading)
            Picker("", selection: $controller.settings.director.persona) {
                ForEach(Persona.allCases) { Text($0.label).tag($0) }
            }
            .labelsHidden()
            .pickerStyle(.menu)
        }
        if controller.settings.director.persona == .random { yokeControls }
        if controller.settings.director.persona == .architect {
            field("Goal", prompt: "e.g. bodies of at least 100 agents", text: $controller.settings.director.goal)
            note("The Architect acts through the LLM tier: the rule tier applies the safety rules only, and the "
                 + "goal reaches the model as data.")
        }
        field("Rule every", prompt: "steps", text: $controller.settings.director.ruleEvery)
        note(ruleEveryHelp)
        if controller.settings.director.persona != .random {  // the random persona has no LLM tier (forced off)
            Divider()
            Toggle("LLM tier (local LM Studio)", isOn: $controller.settings.director.llmOn)
            if controller.settings.director.llmOn { llmControls }
        }
        ForEach(controller.settings.director.notes(context: context), id: \.self) { message in
            Label(message, systemImage: "info.circle").font(.caption).foregroundStyle(.orange)
        }
    }

    /// The runs a random director can follow: those with a `director.jsonl`, but not the run being resumed, plus
    /// the one the settings already name (a source config may follow a run that is gone: validation says so).
    private var yokeOptions: [String] {
        var options = context.yokeRuns.filter { $0 != context.ownRun }
        let current = controller.settings.director.yokeRun
        if !current.isEmpty && !options.contains(current) { options.append(current) }
        return options
    }

    @ViewBuilder private var yokeControls: some View {
        HStack {
            Text("Yoked to run").frame(width: 110, alignment: .leading)
            Picker("", selection: $controller.settings.director.yokeRun) {
                Text("Choose a run").tag("")
                ForEach(yokeOptions, id: \.self) { Text($0).tag($0) }
            }
            .labelsHidden()
        }
        let log = controller.settings.director.yokeLog
        if !log.isEmpty && YokeLog.run(from: log) == nil {
            note("The source config follows \(log); choose a run to replace it.")
        }
        note("Arm R of the evaluation: it makes the same number of discretionary moves as that run's director, at "
             + "the same steps, but picks the actions and their parameters at random. Reverts and the safety rules "
             + "run as for the Gardener, with the Gardener's budget and cooldowns. No LLM tier. Run the yoked run "
             + "first, and give this one its fork point, seed and Rule every.")
    }

    private var ruleEveryHelp: String {
        var text = "Steps between rule-tier decisions (default \(DirectorDefaults.ruleEvery)); a positive multiple of "
        if let c = context.cadence {
            text += "run.metrics_every \(c.metricsEvery) and run.heavy_metrics_every \(c.heavyMetricsEvery)."
        } else {
            text += "run.metrics_every and run.heavy_metrics_every."
        }
        return text
    }

    @ViewBuilder private var llmControls: some View {
        field("LLM every", prompt: "steps", text: $controller.settings.director.llmEvery)
        note("Steps between model calls (a multiple of the rule cadence). Between calls the rule tier keeps its "
             + "safety rules; a failed call falls back to the rule tier.")
        field("URL", prompt: DirectorDefaults.url, text: $controller.settings.director.llmURL)
            .onChange(of: controller.settings.director.llmURL) { controller.clearLLMCheck() }
        field("Model", prompt: DirectorDefaults.model, text: $controller.settings.director.llmModel)
            .onChange(of: controller.settings.director.llmModel) { controller.clearLLMCheck() }
        HStack {
            Button("Test connection") { controller.checkLLM() }
                .disabled(!controller.repoReady || controller.llmCheck == .running
                          || !controller.settings.director.llmEndpointValid)
                .help("Runs `alife llm-check` with this URL and model: one tiny structured-output request.")
            checkResult
        }
    }

    @ViewBuilder private var checkResult: some View {
        switch controller.llmCheck {
        case .idle:
            EmptyView()
        case .running:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Testing...").font(.caption).foregroundStyle(.secondary)
            }
        case .result(let result):
            Label(result.text, systemImage: result.ok ? "checkmark.circle.fill" : "xmark.octagon.fill")
                .font(.caption)
                .foregroundStyle(result.ok ? Color.green : Color.red)
                .textSelection(.enabled)
        }
    }

    private func field(_ title: String, prompt: String, text: Binding<String>) -> some View {
        HStack {
            Text(title).frame(width: 110, alignment: .leading)
            TextField(prompt, text: text).textFieldStyle(.roundedBorder)
        }
    }

    private func note(_ text: String) -> some View {
        Text(text).font(.caption).foregroundStyle(.secondary)
    }
}

// MARK: - Live views (right column)

private enum DirectorTab: String, CaseIterable, Identifiable {
    case interventions = "Interventions"
    case journal = "Journal"
    var id: String { rawValue }
}

/// The last interventions and the journal of the tracked or selected run, tailed from its files.
struct DirectorLiveView: View {
    @ObservedObject var controller: RunController
    @State private var tab: DirectorTab = .interventions

    var body: some View {
        GroupBox("Director") {
            VStack(alignment: .leading, spacing: 6) {
                if let snapshot = controller.director, snapshot.hasLog {
                    Picker("", selection: $tab) {
                        ForEach(DirectorTab.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 280)
                    Group {
                        switch tab {
                        case .interventions: InterventionList(rows: snapshot.interventions)
                        case .journal: JournalList(rows: snapshot.journal)
                        }
                    }
                    .frame(height: 170)
                    .background(Color(nsColor: .textBackgroundColor))
                } else {
                    Text("Director off for this run (no director.jsonl).").foregroundStyle(.secondary)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private func sourceStyle(_ source: String) -> (symbol: String, color: Color) {
    switch source {
    case "rule": return ("gearshape.fill", .blue)
    case "llm": return ("sparkles", .purple)
    case "revert": return ("arrow.uturn.backward", .orange)
    default: return ("questionmark.circle", .secondary)
    }
}

private struct InterventionList: View {
    let rows: [InterventionRow]

    var body: some View {
        if rows.isEmpty {
            Text("No interventions yet.").foregroundStyle(.secondary).padding(6)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        } else {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(rows) { InterventionRowView(row: $0) }
                }
                .padding(6)
            }
        }
    }
}

private struct InterventionRowView: View {
    let row: InterventionRow

    var body: some View {
        let style = sourceStyle(row.source)
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 6) {
                Text(String(row.step)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Label(row.source, systemImage: style.symbol)
                    .font(.caption).foregroundStyle(style.color)
                Text(row.action).font(.callout.bold())
                Text(row.params).font(.system(.caption, design: .monospaced)).lineLimit(1)
                    .truncationMode(.tail)
            }
            if !row.reason.isEmpty {
                Text(row.reason).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    .truncationMode(.tail)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .help(row.reason.isEmpty ? row.params : "\(row.params)\n\(row.reason)")
    }
}

private struct JournalList: View {
    let rows: [JournalRow]

    var body: some View {
        if rows.isEmpty {
            Text("The journal is empty.").foregroundStyle(.secondary).padding(6)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        } else {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(rows) { row in
                        switch row {
                        case .note(let note): JournalNoteView(note: note)
                        case .call(let call): LLMCallView(call: call)
                        }
                    }
                }
                .padding(6)
            }
        }
    }
}

private struct JournalNoteView: View {
    let note: JournalNote

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(String(note.step)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            Text(note.text).font(.caption).textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct LLMCallView: View {
    let call: LLMCall

    private var outcomeColor: Color {
        switch call.outcome {
        case "ok": return .green
        case "none": return .secondary
        default: return .red
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Text(String(call.step)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Label("llm", systemImage: "sparkles").font(.caption).foregroundStyle(.purple)
                Text(call.outcome).font(.caption.bold()).foregroundStyle(outcomeColor)
                if let latency = call.latency {
                    Text(String(format: "%.1f s", latency)).font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Text("\(call.accepted) accepted, \(call.rejected) rejected").font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let fallback = call.fallback {
                Text("Fallback to the rule tier: " + (fallback.isEmpty ? "no action" : fallback.joined(separator: ", ")))
                    .font(.caption).foregroundStyle(.orange)
            }
            if let error = call.error, !error.isEmpty {
                Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
            }
            if let journal = call.journal, !journal.isEmpty {
                Text(journal).font(.caption).textSelection(.enabled)
            }
            DisclosureGroup("Raw prompt and response") {
                VStack(alignment: .leading, spacing: 6) {
                    raw("Prompt", call.prompt)
                    raw("Response", call.response)
                }
            }
            .font(.caption)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.purple.opacity(0.08)))
    }

    private func raw(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            ScrollView {
                Text(text)
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(4)
            }
            .frame(maxHeight: 140)
            .background(Color(nsColor: .controlBackgroundColor))
        }
    }
}
