import Foundation

// MARK: - Director settings

/// `run.director`: what the ecosystem director does in a run (docs/DIRECTOR.md, DESIGN.md §26).
package enum DirectorMode: String, CaseIterable, Identifiable, Sendable {
    case off, live, replay
    package var id: String { rawValue }
    package var label: String { rawValue.capitalized }
}

/// `director.persona`. `random` is arm R of the evaluation (docs/DIRECTOR.md §9, §15): it makes as many random
/// moves as another run's director made, at the same steps (`director.yoke_log`), and has no LLM tier.
package enum Persona: String, CaseIterable, Identifiable, Sendable {
    case gardener, trickster, architect, random
    package var id: String { rawValue }
    package var label: String { rawValue.capitalized }
}

/// `director.yoke_log`: the `director.jsonl` of the run a random director follows. The panel offers runs of the
/// repository and passes the path relative to it, where the engine runs.
package enum YokeLog {
    package static func path(run: String) -> String { "runs/\(run)/director.jsonl" }

    /// The run a path of the form `runs/<run>/director.jsonl` names; nil for any other path.
    package static func run(from path: String) -> String? {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0] == "runs", !parts[1].isEmpty, parts[2] == "director.jsonl" else {
            return nil
        }
        return String(parts[1])
    }
}

/// Engine defaults of the director keys (src/alife/config.py: DirectorConfig and RunConfig).
package enum DirectorDefaults {
    package static let ruleEvery = 1000
    /// What a bare `--llm` sets (cli.LLM_DEFAULT_EVERY); also the value offered when the tier is switched on.
    package static let llmEvery = 25_000
    package static let url = "http://localhost:1234/v1/chat/completions"
    package static let model = "gemma-4-26b-a4b-it-mlx@8bit"
    package static let metricsEvery = 100
    package static let heavyMetricsEvery = 1000
}

/// `run.metrics_every` and `run.heavy_metrics_every`: `director.rule_every` must be a multiple of both.
package struct Cadence: Equatable, Sendable {
    package var metricsEvery: Int
    package var heavyMetricsEvery: Int

    package static let engineDefaults = Cadence(
        metricsEvery: DirectorDefaults.metricsEvery, heavyMetricsEvery: DirectorDefaults.heavyMetricsEvery)

    /// The cadence a preset file (`src/alife/presets/NAME.toml`) gives: the engine defaults, overridden by the
    /// two keys in its `[run]` table (a value below 1 is ignored).
    package init(presetTOML text: String) {
        self.init(preset: PresetFile(text: text))
    }

    package init(preset file: PresetFile) {
        self = .engineDefaults
        if let n = file.int("run", "metrics_every"), n >= 1 { metricsEvery = n }
        if let n = file.int("run", "heavy_metrics_every"), n >= 1 { heavyMetricsEvery = n }
    }

    package init(metricsEvery: Int, heavyMetricsEvery: Int) {
        self.metricsEvery = metricsEvery
        self.heavyMetricsEvery = heavyMetricsEvery
    }

    /// The cadence of the named preset in the repository, or nil if its file cannot be read.
    package static func preset(_ name: String, repo: URL) -> Cadence? {
        PresetFile.preset(name, repo: repo).map(Cadence.init(preset:))
    }
}

/// The director's settings as edited in the panel. Values are kept as text where they are bound to text
/// fields. Only values that differ from the context's baseline are passed to the engine.
package struct DirectorSettings: Equatable {
    package var mode: DirectorMode = .off
    package var persona: Persona = .gardener
    package var goal = ""
    package var ruleEvery = String(DirectorDefaults.ruleEvery)
    /// The LLM tier: off is `director.llm_every = 0`; on uses `llmEvery`.
    package var llmOn = false
    package var llmEvery = String(DirectorDefaults.llmEvery)
    package var llmURL = DirectorDefaults.url
    package var llmModel = DirectorDefaults.model
    /// The random persona's `director.yoke_log` (empty for every other persona).
    package var yokeLog = ""

    package init() {}

    /// The run the random persona follows, as the picker shows it ("" when none, or when the path is not of
    /// the form `runs/<run>/director.jsonl`).
    package var yokeRun: String {
        get { YokeLog.run(from: yokeLog) ?? "" }
        set { yokeLog = newValue.isEmpty ? "" : YokeLog.path(run: newValue) }
    }

    /// The LLM tier is in use: switched on, and not the random persona (which has none; it is forced off).
    package var usesLLM: Bool { llmOn && persona != .random }

    /// `director.llm_every` as the engine would see it (0 when the tier is off), or nil if the field is invalid.
    package var effectiveLLMEvery: Int? { usesLLM ? Parse.integer(llmEvery) : 0 }

    private var trimmedGoal: String { goal.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var trimmedURL: String { llmURL.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var trimmedModel: String { llmModel.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var trimmedYokeLog: String { yokeLog.trimmingCharacters(in: .whitespacesAndNewlines) }

    /// True for an `http://` or `https://` URL with a host (the only schemes the LLM client contacts;
    /// config.is_http_url).
    package static func isHTTPURL(_ text: String) -> Bool {
        guard let parts = URLComponents(string: text), let scheme = parts.scheme?.lowercased(),
              scheme == "http" || scheme == "https", let host = parts.host, !host.isEmpty else { return false }
        return true
    }

    /// The URL and model are usable for a connection test.
    package var llmEndpointValid: Bool { Self.isHTTPURL(trimmedURL) && !trimmedModel.isEmpty }

    /// Human-readable problems; mirrors the director part of `validate()` in src/alife/config.py. Nothing is
    /// checked unless the director will run live (off passes nothing; a replay passes only its mode).
    package func validate(context: DirectorContext) -> [String] {
        guard context.available else { return [] }
        if mode == .replay {
            return context.hasLog ? [] : ["There is no director.jsonl in this run to replay."]
        }
        guard mode == .live else { return [] }
        var errors: [String] = []
        let rule = Parse.integer(ruleEvery)
        if let rule, rule >= 1 {
            if let c = context.cadence,
               rule % c.metricsEvery != 0 || rule % c.heavyMetricsEvery != 0 {
                errors.append("director.rule_every \(rule) must be a multiple of run.metrics_every "
                              + "\(c.metricsEvery) and run.heavy_metrics_every \(c.heavyMetricsEvery).")
            }
        } else {
            errors.append("director.rule_every must be a positive integer.")
        }
        if persona == .random {  // mirrors validate(): a yoke log, and no LLM tier
            if trimmedYokeLog.isEmpty {
                errors.append("director.persona random needs a run to follow (director.yoke_log).")
            } else if let run = YokeLog.run(from: trimmedYokeLog) {
                if run == context.ownRun {
                    errors.append("A run cannot follow its own director.jsonl.")
                } else if !context.yokeRuns.contains(run) {
                    errors.append("runs/\(run) has no director.jsonl to follow.")
                }
            }
        } else if llmOn {
            if let every = Parse.integer(llmEvery), every >= 1 {
                if let rule, rule >= 1, every % rule != 0 {
                    errors.append("director.llm_every must be 0 (off) or a positive multiple of "
                                  + "director.rule_every \(rule).")
                }
            } else {
                errors.append("director.llm_every must be a positive integer (switch the LLM off for 0).")
            }
            if !Self.isHTTPURL(trimmedURL) { errors.append("run.llm_url must be an http(s) URL.") }
            if trimmedModel.isEmpty { errors.append("run.llm_model must not be empty.") }
        }
        return errors
    }

    /// Advice that does not block Start.
    package func notes(context: DirectorContext) -> [String] {
        guard context.available, mode == .live else { return [] }
        var notes: [String] = []
        if context.cadence == nil {
            notes.append("run.metrics_every and run.heavy_metrics_every could not be read, so rule_every is not "
                         + "checked against them here; the engine rejects a mismatch (exit code 2).")
        }
        return notes
    }

    /// Arguments for the director settings that differ from `context.baseline`. The mode is `--director` (which
    /// on `run` also enables the director); the LLM and persona settings are passed only in live mode, where
    /// the engine accepts them. Free-text values use the `--flag=value` form so that a leading "-" is not
    /// mistaken for an option.
    package func arguments(context: DirectorContext) -> [String] {
        guard context.available else { return [] }
        let base = context.baseline
        var args: [String] = []
        if mode != base.mode { args += ["--director", mode.rawValue] }
        guard mode == .live else { return args }
        if persona != base.persona { args += ["--persona", persona.rawValue] }
        if persona == .architect, trimmedGoal != base.trimmedGoal { args.append("--goal=\(trimmedGoal)") }
        // Only the random persona may hold a yoke log: leaving it clears the one the source had.
        let wantLog = persona == .random ? trimmedYokeLog : ""
        if wantLog != base.trimmedYokeLog { args.append("--yoke-log=\(wantLog)") }
        if let now = effectiveLLMEvery, now != base.effectiveLLMEvery { args += ["--llm-every", "\(now)"] }
        if usesLLM {
            if trimmedURL != base.trimmedURL { args.append("--llm-url=\(trimmedURL)") }
            if trimmedModel != base.trimmedModel { args.append("--llm-model=\(trimmedModel)") }
        }
        if let rule = Parse.integer(ruleEvery), rule != Parse.integer(base.ruleEvery) {
            args += ["--set", "director.rule_every=\(rule)"]
        }
        return args
    }
}

// MARK: - What the panel is diffed against

/// The reference for the director settings of the selected run mode, plus what the run allows.
package struct DirectorContext: Equatable {
    /// The values the engine would use without any director argument (and the panel's pre-fill): the engine
    /// defaults for a new run, the source's values for a fork or resume. A fork never inherits the mode.
    package var baseline = DirectorSettings()
    /// What the mode picker starts at. A fork pre-fills the source's mode (Live if it ran live) although the
    /// engine starts a fork with the director off: `--director live` is then passed, as the baseline is off.
    package var initialMode: DirectorMode = .off
    /// False when the director cannot be used: a resumed run whose config has `director.enabled = false` (its
    /// compiled step cannot take director actions), or whose config could not be read.
    package var available = true
    /// Resume: the run has a `director.jsonl` that a replay can apply.
    package var hasLog = true
    /// For the `rule_every` multiple check; nil if unknown (the engine then does the check).
    package var cadence: Cadence? = .engineDefaults
    /// Set when the source config could not be read.
    package var warning: String?
    /// The runs of the repository that have a `director.jsonl`: what the random persona can follow.
    package var yokeRuns: [String] = []
    /// Resume: the run being resumed, which cannot follow itself.
    package var ownRun: String?

    package init(baseline: DirectorSettings = DirectorSettings(), initialMode: DirectorMode = .off,
                 available: Bool = true, hasLog: Bool = true, cadence: Cadence? = .engineDefaults,
                 warning: String? = nil, yokeRuns: [String] = [], ownRun: String? = nil) {
        self.baseline = baseline
        self.initialMode = initialMode
        self.available = available
        self.hasLog = hasLog
        self.cadence = cadence
        self.warning = warning
        self.yokeRuns = yokeRuns
        self.ownRun = ownRun
    }

    /// A new run: the engine defaults, with the preset's cadence.
    package static func newRun(cadence: Cadence?) -> DirectorContext {
        DirectorContext(baseline: DirectorSettings(), available: true, hasLog: false, cadence: cadence,
                        warning: nil)
    }

    /// Fork or resume from the run directory `runDirectory`, reading its `config.json`.
    package static func forRun(at runDirectory: URL, mode: RunMode) -> DirectorContext {
        let file = runDirectory.appendingPathComponent("config.json")
        let hasLog = FileManager.default.fileExists(
            atPath: runDirectory.appendingPathComponent("director.jsonl").path)
        do {
            guard let data = try? Data(contentsOf: file) else { throw LeverConfigError(message: "cannot read it") }
            let source = try DirectorSource(configJSON: data)
            var baseline = source.settings
            var initial = baseline.mode
            if mode == .fork {  // a fork starts with the director off (DESIGN §26), whatever the source ran
                baseline.mode = .off
                if initial == .replay { initial = .off }  // a replay needs its log; the GUI forks only live
            }
            if mode == .resume && !source.enabled { baseline.mode = .off }
            return DirectorContext(baseline: baseline, initialMode: initial, available: mode == .fork || source.enabled,
                                   hasLog: hasLog, cadence: source.cadence, warning: nil,
                                   ownRun: mode == .resume ? runDirectory.lastPathComponent : nil)
        } catch {
            let why = (error as? LeverConfigError)?.message ?? "\(error)"
            return DirectorContext(
                baseline: DirectorSettings(), available: mode == .fork, hasLog: hasLog, cadence: nil,
                warning: "Could not read \(runDirectory.lastPathComponent)/config.json (\(why)); "
                    + "the director settings show the engine defaults, not the source's values.")
        }
    }
}

/// The director-related part of a run's `config.json`.
package struct DirectorSource: Equatable {
    package var settings: DirectorSettings
    /// `director.enabled`: whether the run's compiled step takes director actions.
    package var enabled: Bool
    package var cadence: Cadence

    /// Missing sections or keys mean the engine default (older runs lack the director keys); a present value of
    /// the wrong type throws, as the engine would reject it too.
    package init(configJSON data: Data) throws {
        let root: Any
        do {
            root = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw LeverConfigError(message: "not valid JSON")
        }
        guard let config = root as? [String: Any] else { throw LeverConfigError(message: "top level is not an object") }

        func value(_ section: String, _ key: String) throws -> Any? {
            guard let raw = config[section] else { return nil }
            guard let table = raw as? [String: Any] else { throw LeverConfigError(message: "\(section) is not an object") }
            return table[key]
        }
        func integer(_ section: String, _ key: String) throws -> Int? {
            guard let raw = try value(section, key) else { return nil }
            guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
                  let v = Int(exactly: n.doubleValue) else {
                throw LeverConfigError(message: "\(section).\(key) is not an integer")
            }
            return v
        }
        func string(_ section: String, _ key: String) throws -> String? {
            guard let raw = try value(section, key) else { return nil }
            guard let s = raw as? String else { throw LeverConfigError(message: "\(section).\(key) is not a string") }
            return s
        }
        func bool(_ section: String, _ key: String) throws -> Bool? {
            guard let raw = try value(section, key) else { return nil }
            guard let n = raw as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else {
                throw LeverConfigError(message: "\(section).\(key) is not a boolean")
            }
            return n.boolValue
        }

        var s = DirectorSettings()
        if let name = try string("run", "director") {
            guard let mode = DirectorMode(rawValue: name) else {
                throw LeverConfigError(message: "run.director is not off, live or replay")
            }
            s.mode = mode
        }
        if let name = try string("director", "persona") {
            guard let persona = Persona(rawValue: name) else {
                throw LeverConfigError(message: "director.persona is not a known persona")
            }
            s.persona = persona
        }
        if let g = try string("director", "goal") { s.goal = g }
        if let y = try string("director", "yoke_log") { s.yokeLog = y }
        if let n = try integer("director", "rule_every") { s.ruleEvery = String(n) }
        if let n = try integer("director", "llm_every"), n > 0 {
            s.llmOn = true
            s.llmEvery = String(n)
        }
        if let u = try string("run", "llm_url") { s.llmURL = u }
        if let m = try string("run", "llm_model") { s.llmModel = m }
        settings = s
        enabled = try bool("director", "enabled") ?? false
        cadence = Cadence(
            metricsEvery: try integer("run", "metrics_every") ?? DirectorDefaults.metricsEvery,
            heavyMetricsEvery: try integer("run", "heavy_metrics_every") ?? DirectorDefaults.heavyMetricsEvery)
    }
}

// MARK: - Connection test

/// The one-line outcome of `alife llm-check`.
package struct LLMCheckResult: Equatable, Sendable {
    package let ok: Bool
    package let text: String

    /// `output` is the command's combined stdout and stderr. The result line is the last one that starts with
    /// "OK" or "FAIL"; otherwise (a usage or config error, a crash) the last non-empty line is shown.
    package static func parse(status: Int32, reason: Process.TerminationReason, output: String) -> LLMCheckResult {
        if reason == .uncaughtSignal { return LLMCheckResult(ok: false, text: "llm-check was terminated by signal \(status).") }
        let lines = output.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        if let line = lines.last(where: { $0.hasPrefix("OK ") || $0 == "OK" || $0.hasPrefix("FAIL ") }) {
            return LLMCheckResult(ok: status == 0 && line.hasPrefix("OK"), text: line)
        }
        return LLMCheckResult(ok: false, text: lines.last ?? "llm-check exited with code \(status) and no output.")
    }
}

package enum LLMCheck {
    /// Runs `executable arguments` in `directory` and waits for it asynchronously (no thread is blocked).
    package static func run(executable: URL, arguments: [String], directory: URL) async -> LLMCheckResult {
        await withCheckedContinuation { continuation in
            let process = Process()
            process.executableURL = executable
            process.arguments = arguments
            process.currentDirectoryURL = directory
            process.standardInput = FileHandle.nullDevice
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            let reader = pipe.fileHandleForReading
            process.terminationHandler = { finished in
                // The command prints one short line, so the pipe cannot fill up before it exits.
                let data = reader.readDataToEndOfFile()
                continuation.resume(returning: LLMCheckResult.parse(
                    status: finished.terminationStatus, reason: finished.terminationReason,
                    output: String(decoding: data, as: UTF8.self)))
            }
            do {
                try process.run()
            } catch {
                continuation.resume(returning: LLMCheckResult(
                    ok: false, text: "Failed to launch \(executable.path): \(error.localizedDescription)"))
            }
        }
    }
}
