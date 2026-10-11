import AppKit
import SwiftUI

struct ContentView: View {
    @ObservedObject var controller: RunController

    private var busy: Bool { controller.job != nil }

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    repoSection
                    runSection
                    leverSection
                    timerSection
                }
                .padding(.trailing, 4)
            }
            .frame(width: 400)

            VStack(alignment: .leading, spacing: 12) {
                commandSection
                controlSection
                statusSection
                toolSection
                LogView(controller: controller)
            }
            .frame(maxWidth: .infinity)
        }
        .padding(16)
    }

    // MARK: Repository

    private var repoSection: some View {
        GroupBox("Repository") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    TextField("~/artificial-life", text: $controller.repoPath)
                        .textFieldStyle(.roundedBorder)
                    Button("Choose...") { controller.chooseRepo() }
                }
                if controller.repoReady {
                    Label("Found .venv/bin/alife", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.caption)
                } else {
                    Label(".venv/bin/alife not found in this folder.", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                        .font(.caption)
                    HStack {
                        Button("Set up environment") { controller.setupEnvironment() }
                            .disabled(busy)
                        Button("Re-check") { controller.refreshRepo() }
                    }
                    Text("Runs `uv sync --extra viewer --extra video` in the repository and streams the output to the log.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .disabled(busy)
    }

    // MARK: Run settings

    private var runSection: some View {
        GroupBox("Run") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Mode", selection: $controller.settings.mode) {
                    ForEach(RunMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .onChange(of: controller.settings.mode) { _, newMode in
                    // A fork keeps the source state's RNG unless a seed is given.
                    if newMode == .fork { controller.settings.seed = "" }
                    if newMode == .new && controller.settings.seed.isEmpty { controller.settings.seed = "1" }
                }

                if controller.settings.mode != .new {
                    HStack {
                        Picker(controller.settings.mode == .resume ? "Resume run" : "Fork from",
                               selection: $controller.settings.sourceRun) {
                            ForEach(controller.runs, id: \.self) { Text($0).tag($0) }
                        }
                        Button("Refresh") { controller.refreshRuns() }
                    }
                    if controller.runs.isEmpty {
                        Text("No runs found in runs/.").font(.caption).foregroundStyle(.secondary)
                    }
                }
                if controller.settings.mode != .resume {
                    HStack {
                        TextField("New run name", text: $controller.settings.name)
                            .textFieldStyle(.roundedBorder)
                        Button("New name") { controller.suggestNewName() }
                    }
                }
                if controller.settings.mode == .new {
                    Picker("Preset", selection: $controller.settings.preset) {
                        ForEach(Preset.allCases) { Text($0.rawValue).tag($0) }
                    }
                }
                if controller.settings.mode == .new {
                    // A fork keeps the source's brain size, which the fast overlay would change.
                    Toggle("Fast overlay (--overlay fast)", isOn: $controller.settings.fast)
                }
                if controller.settings.mode != .resume {
                    field("Seed", text: $controller.settings.seed,
                          prompt: controller.settings.mode == .fork ? "blank = keep source RNG" : "blank = preset default")
                }
                field(controller.settings.mode == .resume ? "Additional steps" : "Steps",
                      text: $controller.settings.steps,
                      prompt: controller.settings.mode == .resume ? "blank = original target" : "blank = preset default")
                if controller.settings.mode == .fork {
                    Text("A fork starts a new run from the source run's latest checkpoint; the source config is the base, so the preset cannot be changed.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .disabled(busy)
    }

    private func field(_ title: String, text: Binding<String>, prompt: String) -> some View {
        HStack {
            Text(title).frame(width: 110, alignment: .leading)
            TextField(prompt, text: text).textFieldStyle(.roundedBorder)
        }
    }

    // MARK: Levers

    private var leverSection: some View {
        let isResume = controller.settings.mode == .resume
        return GroupBox("Keep inventing levers") {
            VStack(alignment: .leading, spacing: 10) {
                if isResume {
                    Text("Resume cannot change the dynamics. Start a new run or a fork to use these levers.")
                        .font(.caption).foregroundStyle(.orange)
                }
                lever("Mutation rate floor", key: "mutation.rate_min", text: $controller.settings.levers.rateMin,
                      help: "Lowest value the evolvable mutation rate can drift to, in (0, 1). Default 0.001.")
                lever("Mutation scale floor", key: "mutation.sigma_min", text: $controller.settings.levers.sigmaMin,
                      help: "Lowest value the evolvable mutation step size can drift to, > 0. Default 0.002.")
                lever("Patch drift", key: "fields.patch_drift", text: $controller.settings.levers.patchDrift,
                      help: "Resource-patch drift speed (units/step); higher means a less predictable environment. Default 0.005.")
                lever("Immigration floor", key: "population.immigration_floor",
                      text: $controller.settings.levers.immigrationFloor,
                      help: "Immigrants (random founders) are added only while the population is below this floor. 0 = off.")
                lever("Immigrants per step", key: "population.immigrants_per_step",
                      text: $controller.settings.levers.immigrantsPerStep,
                      help: "Maximum immigrants added per step while below the floor; must be >= 1 when the floor is above 0. Default 64.")
                lever("Immigration every", key: "population.immigration_every",
                      text: $controller.settings.levers.immigrationEvery,
                      help: "Add random founders every N steps regardless of population size. 0 = off.")
                lever("Immigrants per event", key: "population.immigration_count",
                      text: $controller.settings.levers.immigrationCount,
                      help: "Founders added at each periodic event (1 to max births per step); needed when the period is above 0.")
                lever("Season period", key: "fields.season_period",
                      text: $controller.settings.levers.seasonPeriod,
                      help: "Length of one resource season cycle in steps. 0 = no seasons.")
                lever("Season amplitude", key: "fields.season_amplitude",
                      text: $controller.settings.levers.seasonAmplitude,
                      help: "Strength of the seasonal swing in regrowth, 0 to 1 (1 = regrowth falls to zero at the trough).")
                HStack {
                    Text("Season mode").frame(width: 140, alignment: .leading)
                    Picker("", selection: $controller.settings.levers.seasonMode) {
                        Text("Global").tag("global")
                        Text("Per patch").tag("per_patch")
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                }
                Text("Per patch: each resource patch peaks at a different time, so good seasons move across the map.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("Reset levers to defaults") { controller.settings.levers.reset() }
                    .font(.caption)
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .disabled(isResume)
        }
        .disabled(busy)
    }

    private func lever(_ title: String, key: String, text: Binding<String>, help: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).frame(width: 140, alignment: .leading)
                TextField(key, text: text).textFieldStyle(.roundedBorder)
            }
            Text(help).font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: Timer

    private var timerSection: some View {
        GroupBox("Timer") {
            VStack(alignment: .leading, spacing: 6) {
                Toggle("Stop after (HH:MM)", isOn: $controller.settings.timerEnabled)
                if controller.settings.timerEnabled {
                    HStack {
                        Stepper(value: $controller.settings.timerHours, in: 0...99) {
                            Text(String(format: "%02d h", controller.settings.timerHours))
                        }
                        Stepper(value: $controller.settings.timerMinutes, in: 0...55, step: 5) {
                            Text(String(format: "%02d min", controller.settings.timerMinutes))
                        }
                    }
                    Text("Sends one SIGINT when the time is up; the engine finishes the step, writes a checkpoint and exits. The timer starts when the run starts.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .disabled(busy)
    }

    // MARK: Right column

    private var commandSection: some View {
        GroupBox("Command") {
            VStack(alignment: .leading, spacing: 6) {
                Text(controller.commandLine)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if !busy {
                    ForEach(controller.validationErrors, id: \.self) { message in
                        Label(message, systemImage: "exclamationmark.circle")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
            }
            .padding(6)
        }
    }

    private var controlSection: some View {
        HStack {
            if controller.job == .run {
                Button(controller.stopRequested ? "Stopping..." : "Stop") { controller.stop() }
                    .disabled(controller.stopRequested)
                    .keyboardShortcut(".", modifiers: .command)
                if controller.stopRequested {
                    Button("Force stop") { controller.forceStop() }
                }
            } else if busy {
                Text(controller.job == .setup ? "Setting up environment..." : "Inspecting...")
                    .foregroundStyle(.secondary)
                Button("Cancel") { controller.forceStop() }
            } else {
                Button("Start") { controller.start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!controller.canStart)
            }
            Spacer()
            if let exit = controller.lastExit {
                Label(exit.text, systemImage: exit.success ? "checkmark.circle.fill" : "xmark.octagon.fill")
                    .foregroundStyle(exit.success ? .green : .red)
                    .font(.callout)
            }
        }
    }

    private var statusSection: some View {
        GroupBox("Status") {
            VStack(alignment: .leading, spacing: 4) {
                if controller.job == .run {
                    HStack {
                        Text("Elapsed \(formatDuration(controller.elapsed))")
                        if let remaining = controller.remaining {
                            Text("| remaining \(formatDuration(remaining))")
                        }
                    }
                    .font(.callout.monospacedDigit())
                }
                if let m = controller.metrics {
                    Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 4) {
                        GridRow {
                            stat("Step", String(m.step))
                            stat("Population", number(m.nAlive, "%.0f"))
                            stat("Steps/s", number(m.stepsPerSecond, "%.1f"))
                        }
                        GridRow {
                            stat("Clades", number(m.clades, "%.0f"))
                            stat("Generation max", number(m.generationMax, "%.0f"))
                            VStack(alignment: .leading, spacing: 0) {
                                Text("Ledger error").font(.caption).foregroundStyle(.secondary)
                                Text(m.ledgerError.map { $0 == 0 ? "OK (0)" : String(format: "%.0f", $0) } ?? "-")
                                    .font(.title3.monospacedDigit())
                                    .foregroundStyle((m.ledgerError ?? 0) == 0 ? Color.primary : Color.red)
                            }
                        }
                    }
                } else {
                    Text("No metrics yet for the selected run.").foregroundStyle(.secondary)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.monospacedDigit())
        }
    }

    private func number(_ value: Double?, _ format: String) -> String {
        value.map { String(format: format, $0) } ?? "-"
    }

    private func formatDuration(_ t: TimeInterval) -> String {
        let s = Int(t.rounded())
        return String(format: "%02d:%02d:%02d", s / 3600, (s / 60) % 60, s % 60)
    }

    private var toolSection: some View {
        HStack {
            if controller.viewerRunning {
                Button("Close viewer") { controller.closeViewer() }
            } else {
                Button("Open viewer") { controller.openViewer() }
                    .disabled(!controller.targetRunExists || controller.job == .setup)
                    .help("Opens the pygame viewer on the run's latest checkpoint. It shares the GPU with a running simulation and will slow it down.")
            }
            Button("Inspect") { controller.inspect() }
                .disabled(busy || !controller.targetRunExists)
                .help("Runs `alife inspect --check` and prints the result in the log.")
            Button("Reveal in Finder") { controller.revealInFinder() }
            Spacer()
            if controller.job == .run {
                Text("Viewer shares the GPU with the run.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Log

struct LogView: View {
    @ObservedObject var controller: RunController

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 4) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(controller.logLines) { line in
                                Text(line.text)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(line.isError ? Color.red : Color.primary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .id(line.id)
                            }
                        }
                        .textSelection(.enabled)
                        .padding(6)
                    }
                    .background(Color(nsColor: .textBackgroundColor))
                    .onChange(of: controller.logLines.last?.id) { _, lastID in
                        if let lastID { proxy.scrollTo(lastID, anchor: .bottom) }
                    }
                }
                HStack {
                    Button("Clear") { controller.clearLog() }
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(controller.logText, forType: .string)
                    }
                    Spacer()
                    Text("\(controller.logLines.count) lines").font(.caption).foregroundStyle(.secondary)
                }
                .font(.caption)
            }
        } label: {
            Text("Log")
        }
        .frame(maxHeight: .infinity)
    }
}
