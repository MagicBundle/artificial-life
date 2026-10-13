import Foundation

@testable import AlifeCore

/// One representative 2D GUI configuration and the argument list `RunSettings.arguments(existingRuns:)` builds for
/// it today. `ArgvIdentityTests` compares the lists with the ones captured before the 3D work.
struct ArgvCase {
    let name: String
    let args: [String]
}

/// A run directory with only a `config.json`, for the fork and resume cases.
private func makeRun(_ root: URL, _ name: String, config: String, director: Bool = false) throws -> URL {
    let dir = root.appendingPathComponent(name, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try config.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    if director { try "".write(to: dir.appendingPathComponent("director.jsonl"), atomically: true, encoding: .utf8) }
    return dir
}

private let liveSourceConfig = """
{"mutation": {"rate_min": 0.01, "sigma_min": 0.002},
 "fields": {"patch_drift": 0.02, "season_period": 5000, "season_amplitude": 0.5, "season_mode": "per_patch"},
 "population": {"immigration_floor": 100, "immigrants_per_step": 32, "immigration_every": 0, "immigration_count": 0},
 "run": {"director": "live", "llm_url": "http://localhost:1234/v1/chat/completions", "llm_model": "m",
         "metrics_every": 10, "heavy_metrics_every": 100},
 "director": {"enabled": true, "persona": "trickster", "rule_every": 500, "llm_every": 5000}}
"""

private let plainSourceConfig = """
{"mutation": {"rate_min": 0.001}, "fields": {"patch_drift": 0.005}, "run": {"metrics_every": 100}}
"""

/// The argument lists of the 2D configurations the GUI must keep producing byte for byte.
func argvCases() throws -> [ArgvCase] {
    var out: [ArgvCase] = []
    func add(_ name: String, _ s: RunSettings, runs: Set<String> = []) {
        out.append(ArgvCase(name: name, args: s.arguments(existingRuns: runs)))
    }
    func newRun(_ edit: (inout RunSettings) -> Void = { _ in }) -> RunSettings {
        var s = RunSettings()
        s.mode = .new
        s.name = "t"
        s.seed = "1"
        s.directorContext = .newRun(cadence: .engineDefaults)
        edit(&s)
        return s
    }

    // New run: presets, seed, steps, overlay.
    for p in [Preset.tiny, .dev, .k64, .k256] {
        add("new/\(p.rawValue)", newRun { $0.preset = p })
    }
    add("new/fast", newRun { $0.fast = true; $0.preset = .tiny })
    add("new/blank-seed-steps", newRun { $0.seed = ""; $0.steps = "2000" })
    add("new/steps-padded", newRun { $0.steps = " 750 "; $0.name = " spaced " })

    // New run: levers, one at a time and together.
    add("new/rate_min", newRun { $0.levers.rateMin = "0.01" })
    add("new/sigma_min", newRun { $0.levers.sigmaMin = "0.01" })
    add("new/patch_drift", newRun { $0.levers.patchDrift = "0.02" })
    add("new/patch_drift-zero", newRun { $0.levers.patchDrift = "0" })
    add("new/immigration_floor", newRun { $0.levers.immigrationFloor = "500" })
    add("new/immigrants_per_step", newRun { $0.levers.immigrantsPerStep = "8" })
    add("new/immigration_every+count", newRun { $0.levers.immigrationEvery = "1000"; $0.levers.immigrationCount = "16" })
    add("new/season", newRun {
        $0.levers.seasonPeriod = "20000"
        $0.levers.seasonAmplitude = "0.6"
        $0.levers.seasonMode = "per_patch"
    })
    add("new/all-levers", newRun {
        $0.levers.rateMin = "0.002"
        $0.levers.sigmaMin = "0.004"
        $0.levers.patchDrift = "0.01"
        $0.levers.immigrationFloor = "300"
        $0.levers.immigrantsPerStep = "16"
        $0.levers.immigrationEvery = "2000"
        $0.levers.immigrationCount = "4"
        $0.levers.seasonPeriod = "10000"
        $0.levers.seasonAmplitude = "1"
        $0.levers.seasonMode = "per_patch"
    })
    // Numeric comparison against the baseline: spelled differently but equal means no argument.
    add("new/levers-equal-to-defaults", newRun {
        $0.levers.rateMin = "0.0010"
        $0.levers.sigmaMin = "2e-3"
        $0.levers.patchDrift = "5e-3"
        $0.levers.immigrationFloor = "00"
        $0.levers.seasonAmplitude = "0.0"
    })
    add("new/lever-spellings", newRun {
        $0.levers.rateMin = "5e-05"
        $0.levers.sigmaMin = "0.00500"
        $0.levers.patchDrift = "1e-2"
    })
    add("new/invalid-levers-skipped", newRun {
        $0.levers.rateMin = "abc"
        $0.levers.immigrationFloor = "-1"
        $0.levers.seasonMode = "global"
    })

    // New run: the director.
    add("new/director-live", newRun { $0.director.mode = .live })
    add("new/director-replay", newRun { $0.director.mode = .replay })
    add("new/director-trickster-rule", newRun {
        $0.director.mode = .live
        $0.director.persona = .trickster
        $0.director.ruleEvery = "2000"
    })
    add("new/director-architect-goal", newRun {
        $0.director.mode = .live
        $0.director.persona = .architect
        $0.director.goal = "  keep clades above 5  "
    })
    add("new/director-llm", newRun {
        $0.director.mode = .live
        $0.director.llmOn = true
        $0.director.llmEvery = "50000"
        $0.director.llmURL = "http://127.0.0.1:8080/v1/chat/completions"
        $0.director.llmModel = "other-model"
    })
    add("new/director-llm-defaults", newRun { $0.director.mode = .live; $0.director.llmOn = true })
    add("new/director-random", newRun {
        $0.director.mode = .live
        $0.director.persona = .random
        $0.director.yokeRun = "arm-a"
        $0.director.llmOn = true  // forced off for the random persona
    })
    add("new/director-with-levers-fast", newRun {
        $0.fast = true
        $0.preset = .k64
        $0.director.mode = .live
        $0.director.persona = .gardener
        $0.levers.seasonPeriod = "5000"
        $0.levers.seasonAmplitude = "0.5"
    })

    // Run names that are taken.
    let taken: Set<String> = ["t", "t2", "run007", "baseline", "baseline-2"]
    add("name/taken-counter", newRun { $0.name = "t" }, runs: taken)
    add("name/taken-padded", newRun { $0.name = "run007" }, runs: taken)
    add("name/taken-suffix", newRun { $0.name = "baseline" }, runs: taken)
    add("name/free", newRun { $0.name = "fresh" }, runs: taken)

    // Fork and resume read their source's config.json.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("alife-argv-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let live = try makeRun(root, "src-live", config: liveSourceConfig, director: true)
    let plain = try makeRun(root, "src-plain", config: plainSourceConfig)
    func fork(_ dir: URL, _ edit: (inout RunSettings) -> Void = { _ in }) -> RunSettings {
        var s = RunSettings()
        s.mode = .fork
        s.name = "child"
        s.seed = ""
        s.sourceRun = dir.lastPathComponent
        let base = Levers.forkBaseline(runDirectory: dir)
        s.levers = base.levers
        s.leverBaseline = base.levers
        let context = DirectorContext.forRun(at: dir, mode: .fork)
        s.director = context.baseline
        s.director.mode = context.initialMode
        s.directorContext = context
        edit(&s)
        return s
    }
    func resume(_ dir: URL, _ edit: (inout RunSettings) -> Void = { _ in }) -> RunSettings {
        var s = RunSettings()
        s.mode = .resume
        s.sourceRun = dir.lastPathComponent
        let context = DirectorContext.forRun(at: dir, mode: .resume)
        s.director = context.baseline
        s.director.mode = context.initialMode
        s.directorContext = context
        edit(&s)
        return s
    }
    add("fork/live-untouched", fork(live))
    add("fork/live-seed-steps", fork(live) { $0.seed = "7"; $0.steps = "300000" })
    add("fork/live-lever-changes", fork(live) {
        $0.levers.rateMin = "0.002"
        $0.levers.patchDrift = "0.005"  // back to the engine default, still a change from the source
        $0.levers.seasonMode = "global"
        $0.levers.immigrationFloor = "0"
    })
    add("fork/live-director-off", fork(live) { $0.director.mode = .off })
    add("fork/live-persona-llm", fork(live) {
        $0.director.persona = .architect
        $0.director.goal = "diversify"
        $0.director.llmOn = false
        $0.director.ruleEvery = "1000"
    })
    add("fork/plain-untouched", fork(plain))
    add("fork/plain-director-live", fork(plain) { $0.director.mode = .live })
    add("fork/plain-levers", fork(plain) { $0.levers.seasonPeriod = "100"; $0.levers.seasonAmplitude = "0.1" })
    add("resume/live-untouched", resume(live))
    add("resume/live-steps", resume(live) { $0.steps = "50000" })
    add("resume/live-off", resume(live) { $0.director.mode = .off })
    add("resume/live-replay", resume(live) { $0.director.mode = .replay })
    add("resume/live-llm-on", resume(live) { $0.director.llmOn = true; $0.director.llmEvery = "10000" })
    add("resume/plain", resume(plain))
    add("resume/plain-steps", resume(plain) { $0.steps = "1000" })
    return out
}
