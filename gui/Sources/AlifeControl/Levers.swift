import Foundation

// MARK: - Input parsing helpers

enum Parse {
    /// Plain non-negative decimal ("0.001", "5e-4"); rejects NaN, inf, signs, commas and empty input.
    static func decimal(_ text: String) -> Double? {
        let t = text.trimmingCharacters(in: .whitespaces)
        guard t.range(of: #"^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$"#, options: .regularExpression) != nil,
              let v = Double(t), v.isFinite else { return nil }
        return v
    }

    /// Non-negative integer with at most nine digits.
    static func integer(_ text: String) -> Int? {
        let t = text.trimmingCharacters(in: .whitespaces)
        guard t.range(of: #"^[0-9]{1,9}$"#, options: .regularExpression) != nil else { return nil }
        return Int(t)
    }
}

// MARK: - "Keep inventing" levers

/// The dynamics levers. Values are kept as text (they are bound to text fields); only values that differ
/// from the engine default are passed to the engine, each as one `--set section.key=value`.
struct Levers: Equatable {
    static let defaultRateMin = 0.001
    static let defaultSigmaMin = 0.002
    static let defaultPatchDrift = 0.005
    static let defaultImmigrationFloor = 0
    static let defaultImmigrantsPerStep = 64

    var rateMin = "0.001"
    var sigmaMin = "0.002"
    var patchDrift = "0.005"
    var immigrationFloor = "0"
    var immigrantsPerStep = "64"
    var immigrationEvery = "0"
    var immigrationCount = "0"
    var seasonPeriod = "0"
    var seasonAmplitude = "0"
    var seasonMode = "global"

    /// Human-readable problems; ranges mirror `validate()` in src/alife/config.py.
    func validate() -> [String] {
        var errors: [String] = []
        if let v = Parse.decimal(rateMin) {
            if !(v > 0 && v < 1) { errors.append("mutation.rate_min must be in (0, 1).") }
        } else {
            errors.append("mutation.rate_min must be a plain decimal number in (0, 1).")
        }
        if let v = Parse.decimal(sigmaMin) {
            if !(v > 0) { errors.append("mutation.sigma_min must be > 0.") }
        } else {
            errors.append("mutation.sigma_min must be a plain decimal number > 0.")
        }
        if Parse.decimal(patchDrift) == nil {
            errors.append("fields.patch_drift must be a plain decimal number >= 0.")
        }
        let floor = Parse.integer(immigrationFloor)
        let perStep = Parse.integer(immigrantsPerStep)
        if floor == nil { errors.append("population.immigration_floor must be an integer >= 0.") }
        if perStep == nil { errors.append("population.immigrants_per_step must be an integer >= 0.") }
        if let floor, let perStep, floor > 0, perStep < 1 {
            errors.append("population.immigrants_per_step must be >= 1 when immigration_floor > 0.")
        }
        let every = Parse.integer(immigrationEvery)
        let count = Parse.integer(immigrationCount)
        if every == nil { errors.append("population.immigration_every must be an integer >= 0.") }
        if count == nil { errors.append("population.immigration_count must be an integer >= 0.") }
        if let every, let count, every > 0, count < 1 {
            errors.append("population.immigration_count must be >= 1 when immigration_every > 0.")
        }
        let period = Parse.integer(seasonPeriod)
        if period == nil { errors.append("fields.season_period must be an integer >= 0.") }
        if let a = Parse.decimal(seasonAmplitude) {
            if a > 1 { errors.append("fields.season_amplitude must be in [0, 1].") }
            if a > 0, let period, period < 1 {
                errors.append("fields.season_period must be >= 1 when season_amplitude > 0.")
            }
        } else {
            errors.append("fields.season_amplitude must be a plain decimal number in [0, 1].")
        }
        return errors
    }

    /// `--set` arguments for the levers that differ from their defaults (invalid fields are skipped).
    func setArguments() -> [String] {
        var args: [String] = []
        func add(_ key: String, _ value: String) { args += ["--set", "\(key)=\(value)"] }
        if let v = Parse.decimal(rateMin), v != Levers.defaultRateMin { add("mutation.rate_min", "\(v)") }
        if let v = Parse.decimal(sigmaMin), v != Levers.defaultSigmaMin { add("mutation.sigma_min", "\(v)") }
        if let v = Parse.decimal(patchDrift), v != Levers.defaultPatchDrift { add("fields.patch_drift", "\(v)") }
        if let v = Parse.integer(immigrationFloor), v != Levers.defaultImmigrationFloor {
            add("population.immigration_floor", "\(v)")
        }
        if let v = Parse.integer(immigrantsPerStep), v != Levers.defaultImmigrantsPerStep {
            add("population.immigrants_per_step", "\(v)")
        }
        if let v = Parse.integer(immigrationEvery), v != 0 { add("population.immigration_every", "\(v)") }
        if let v = Parse.integer(immigrationCount), v != 0 { add("population.immigration_count", "\(v)") }
        if let v = Parse.integer(seasonPeriod), v != 0 { add("fields.season_period", "\(v)") }
        if let v = Parse.decimal(seasonAmplitude), v != 0 { add("fields.season_amplitude", "\(v)") }
        if seasonMode != "global" { add("fields.season_mode", seasonMode) }
        return args
    }

    mutating func reset() { self = Levers() }
}

// MARK: - Run settings

enum RunMode: String, CaseIterable, Identifiable {
    case new = "New run"
    case resume = "Resume"
    case fork = "Fork"
    var id: String { rawValue }
}

enum Preset: String, CaseIterable, Identifiable {
    case tiny, dev
    case k64 = "64k"
    case k256 = "256k"
    var id: String { rawValue }
}

struct RunSettings: Equatable {
    var mode: RunMode = .new
    /// Output run name (new run, fork). The run directory is `runs/<name>`.
    var name = ""
    /// Existing run to resume, or to fork from (its latest checkpoint).
    var sourceRun = ""
    var preset: Preset = .dev
    var fast = false
    /// New run: blank = preset default. Fork: blank = keep the source state's RNG.
    var seed = "1"
    /// New run: total steps (blank = preset default). Fork: steps to run from the source checkpoint (blank = the
    /// source config's `run.steps`). Resume: additional steps (blank = original target).
    var steps = ""
    var levers = Levers()
    var timerEnabled = false
    var timerHours = 1
    var timerMinutes = 0

    /// The run the status, viewer, inspect and reveal buttons refer to.
    var targetRun: String { mode == .resume ? sourceRun : name.trimmingCharacters(in: .whitespaces) }

    var timerDuration: TimeInterval { TimeInterval(timerHours * 3600 + timerMinutes * 60) }

    func validate(existingRuns: Set<String>) -> [String] {
        var errors: [String] = []
        let trimmedName = name.trimmingCharacters(in: .whitespaces)
        if mode == .resume || mode == .fork {
            if sourceRun.isEmpty {
                errors.append("Choose an existing run to \(mode == .resume ? "resume" : "fork").")
            } else if !existingRuns.contains(sourceRun) {
                errors.append("Run 'runs/\(sourceRun)' does not exist.")
            }
        }
        if mode != .resume {
            if trimmedName.isEmpty {
                errors.append("Enter a run name.")
            } else if trimmedName.range(of: #"^[A-Za-z0-9][A-Za-z0-9._-]*$"#, options: .regularExpression) == nil {
                errors.append("Run name may use letters, digits, '.', '_' and '-' only.")
            } else if existingRuns.contains(trimmedName) {
                errors.append("'runs/\(trimmedName)' already exists; pick another name.")
            }
            let trimmedSeed = seed.trimmingCharacters(in: .whitespaces)
            if !trimmedSeed.isEmpty && Parse.integer(trimmedSeed) == nil {
                errors.append("Seed must be an integer >= 0 (or blank).")
            }
        }
        let trimmedSteps = steps.trimmingCharacters(in: .whitespaces)
        if !trimmedSteps.isEmpty {
            if let n = Parse.integer(trimmedSteps) {
                if n < 1 { errors.append("Steps must be >= 1.") }
            } else {
                errors.append("Steps must be a positive integer (or blank).")
            }
        }
        if mode != .resume { errors += levers.validate() }
        if timerEnabled && timerDuration <= 0 { errors.append("Timer duration must be greater than zero.") }
        return errors
    }

    /// Arguments after the executable. Call only when `validate` returned no errors.
    func arguments() -> [String] {
        let trimmedName = name.trimmingCharacters(in: .whitespaces)
        let trimmedSeed = seed.trimmingCharacters(in: .whitespaces)
        let trimmedSteps = steps.trimmingCharacters(in: .whitespaces)
        var args: [String]
        switch mode {
        case .new:
            args = ["run", "--preset", preset.rawValue]
            if fast { args += ["--overlay", "fast"] }
            if let s = Parse.integer(trimmedSeed) { args += ["--seed", "\(s)"] }
            if let n = Parse.integer(trimmedSteps) { args += ["--steps", "\(n)"] }
            args += ["--out", "runs/\(trimmedName)"]
            args += levers.setArguments()
        case .fork:
            args = ["run", "--init-from", "runs/\(sourceRun)"]
            if let s = Parse.integer(trimmedSeed) { args += ["--seed", "\(s)"] }
            if let n = Parse.integer(trimmedSteps) { args += ["--steps", "\(n)"] }
            args += ["--out", "runs/\(trimmedName)"]
            args += levers.setArguments()
        case .resume:
            args = ["resume", "runs/\(sourceRun)"]
            if let n = Parse.integer(trimmedSteps) { args += ["--steps", "\(n)"] }
        }
        return args
    }
}

/// Quotes an argument for display in a shell-style command line.
func shellQuote(_ s: String) -> String {
    if s.range(of: #"^[A-Za-z0-9_@%+=:,./-]+$"#, options: .regularExpression) != nil { return s }
    return "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
}
