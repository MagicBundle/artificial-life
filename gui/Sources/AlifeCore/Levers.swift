import Foundation

// MARK: - Input parsing helpers

package enum Parse {
    /// Plain non-negative decimal ("0.001", "5e-4"); rejects NaN, inf, signs, commas and empty input.
    package static func decimal(_ text: String) -> Double? {
        let t = text.trimmingCharacters(in: .whitespaces)
        guard t.range(of: #"^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$"#, options: .regularExpression) != nil,
              let v = Double(t), v.isFinite else { return nil }
        return v
    }

    /// Non-negative integer with at most nine digits.
    package static func integer(_ text: String) -> Int? {
        let t = text.trimmingCharacters(in: .whitespaces)
        guard t.range(of: #"^[0-9]{1,9}$"#, options: .regularExpression) != nil else { return nil }
        return Int(t)
    }
}

// MARK: - "Keep inventing" levers

/// The dynamics levers. Values are kept as text (they are bound to text fields). Only values that differ from a
/// baseline are passed to the engine, each as one `--set section.key=value`: the engine defaults for a new run,
/// the source run's values for a fork.
package struct Levers: Equatable {
    package var rateMin = "0.001"
    package var sigmaMin = "0.002"
    package var patchDrift = "0.005"
    package var immigrationFloor = "0"
    package var immigrantsPerStep = "64"
    package var immigrationEvery = "0"
    package var immigrationCount = "0"
    package var seasonPeriod = "0"
    package var seasonAmplitude = "0"
    package var seasonMode = "global"
    /// The 3D world's levers; inert unless `world3D.dims` is 3. Their baseline is the preset's, not the engine
    /// defaults (see `World3DLevers`), which is why a `Levers` value carries them and `RunSettings` picks the baseline.
    package var world3D = World3DLevers()

    package init() {}

    package init(world3D: World3DLevers) { self.world3D = world3D }

    /// Human-readable problems; ranges mirror `validate()` in src/alife/config.py.
    package func validate() -> [String] {
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
        return errors + world3D.validate()
    }

    /// `--set` arguments for the levers whose value differs from `baseline` (compared as numbers, so 5e-05 equals
    /// 0.00005 and 0 equals 0.0). The default baseline is the engine defaults, except for the 3D levers, whose
    /// baseline is the preset's or the source's (the caller puts it in `baseline.world3D`). Invalid fields are skipped.
    package func setArguments(baseline: Levers = Levers()) -> [String] {
        var args: [String] = []
        func decimal(_ key: String, _ now: String, _ base: String) {
            guard let v = Parse.decimal(now), v != Parse.decimal(base) else { return }
            args += ["--set", "\(key)=\(v)"]
        }
        func integer(_ key: String, _ now: String, _ base: String) {
            guard let v = Parse.integer(now), v != Parse.integer(base) else { return }
            args += ["--set", "\(key)=\(v)"]
        }
        decimal("mutation.rate_min", rateMin, baseline.rateMin)
        decimal("mutation.sigma_min", sigmaMin, baseline.sigmaMin)
        decimal("fields.patch_drift", patchDrift, baseline.patchDrift)
        integer("population.immigration_floor", immigrationFloor, baseline.immigrationFloor)
        integer("population.immigrants_per_step", immigrantsPerStep, baseline.immigrantsPerStep)
        integer("population.immigration_every", immigrationEvery, baseline.immigrationEvery)
        integer("population.immigration_count", immigrationCount, baseline.immigrationCount)
        integer("fields.season_period", seasonPeriod, baseline.seasonPeriod)
        decimal("fields.season_amplitude", seasonAmplitude, baseline.seasonAmplitude)
        if seasonMode != baseline.seasonMode { args += ["--set", "fields.season_mode=\(seasonMode)"] }
        return args + world3D.setArguments(baseline: baseline.world3D)
    }

    /// Back to the engine defaults, except for the 3D levers, which go to `world3D` (a 3D world has no valid all-zero
    /// defaults: the preset's or the source's values are its defaults).
    package mutating func reset(world3D: World3DLevers = World3DLevers()) { self = Levers(world3D: world3D) }
}

// MARK: - Lever values of an existing run

/// Why a source run's levers could not be read.
package struct LeverConfigError: Error, Equatable {
    package let message: String
}

extension Levers {
    /// The levers of a run's `config.json` (the full nested config written by the engine). A missing section or
    /// key means the engine default, as in `config_from_dict`: runs from before `POST_V01_KEYS` simply lack
    /// the newer keys. A present value of the wrong type throws, because the engine would reject it too.
    package init(configJSON data: Data) throws {
        let root: Any
        do {
            root = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw LeverConfigError(message: "not valid JSON")
        }
        guard let config = root as? [String: Any] else { throw LeverConfigError(message: "top level is not an object") }
        self.init()

        func value(_ section: String, _ key: String) throws -> Any? {
            guard let raw = config[section] else { return nil }
            guard let table = raw as? [String: Any] else { throw LeverConfigError(message: "\(section) is not an object") }
            return table[key]
        }
        func number(_ section: String, _ key: String) throws -> NSNumber? {
            guard let raw = try value(section, key) else { return nil }
            guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else {
                throw LeverConfigError(message: "\(section).\(key) is not a number")
            }
            return n
        }
        func decimalText(_ section: String, _ key: String) throws -> String? {
            guard let n = try number(section, key) else { return nil }
            let v = n.doubleValue
            return v == v.rounded() && abs(v) < 1e15 ? String(Int64(v)) : "\(v)"
        }
        func integerText(_ section: String, _ key: String) throws -> String? {
            guard let n = try number(section, key) else { return nil }
            guard let v = Int(exactly: n.doubleValue) else {
                throw LeverConfigError(message: "\(section).\(key) is not an integer")
            }
            return String(v)
        }

        if let t = try decimalText("mutation", "rate_min") { rateMin = t }
        if let t = try decimalText("mutation", "sigma_min") { sigmaMin = t }
        if let t = try decimalText("fields", "patch_drift") { patchDrift = t }
        if let t = try integerText("population", "immigration_floor") { immigrationFloor = t }
        if let t = try integerText("population", "immigrants_per_step") { immigrantsPerStep = t }
        if let t = try integerText("population", "immigration_every") { immigrationEvery = t }
        if let t = try integerText("population", "immigration_count") { immigrationCount = t }
        if let t = try integerText("fields", "season_period") { seasonPeriod = t }
        if let t = try decimalText("fields", "season_amplitude") { seasonAmplitude = t }
        if let raw = try value("fields", "season_mode") {
            guard let mode = raw as? String else { throw LeverConfigError(message: "fields.season_mode is not a string") }
            seasonMode = mode
        }

        // The 3D world (DESIGN §27.1): a run from before it lacks these keys and is a 2D world with the 3D defaults.
        if let t = try integerText("world", "dims"), let dims = Int(t) { world3D.dims = dims }
        if let t = try decimalText("world", "height") { world3D.height = t }
        if let t = try decimalText("physics", "gravity") { world3D.gravity = t }
        if let t = try decimalText("physics", "buoyancy_max") { world3D.buoyancyMax = t }
        if let t = try integerText("energy", "buoyancy_cost") { world3D.buoyancyCost = t }
        if let t = try integerText("energy", "buoyancy_cap_cost") { world3D.buoyancyCapCost = t }
        if let t = try decimalText("fields", "benthic_lambda") { world3D.benthicLambda = t }
        if let t = try decimalText("fields", "photic_amp") { world3D.photicAmp = t }
        if let t = try decimalText("fields", "photic_lambda") { world3D.photicLambda = t }
        if let t = try decimalText("mutation", "buoy_init") { world3D.buoyInit = t }
        if let t = try decimalText("mutation", "buoy_sigma") { world3D.buoySigma = t }
    }

    /// The levers of the run directory `runDirectory`, or the engine defaults plus a short warning if its
    /// `config.json` cannot be read.
    package static func forkBaseline(runDirectory: URL) -> (levers: Levers, warning: String?) {
        do {
            guard let data = try? Data(contentsOf: runDirectory.appendingPathComponent("config.json")) else {
                throw LeverConfigError(message: "cannot read it")
            }
            return (try Levers(configJSON: data), nil)
        } catch {
            let why = (error as? LeverConfigError)?.message ?? "\(error)"
            return (Levers(), "Could not read \(runDirectory.lastPathComponent)/config.json (\(why)); "
                    + "the levers show the engine defaults, not the source's values.")
        }
    }
}

// MARK: - Run settings

package enum RunMode: String, CaseIterable, Identifiable {
    case new = "New run"
    case resume = "Resume"
    case fork = "Fork"
    package var id: String { rawValue }
}

package enum Preset: String, CaseIterable, Identifiable {
    case tiny, dev
    case k64 = "64k"
    case k256 = "256k"
    case tiny3d
    package var id: String { rawValue }
}

package struct RunSettings: Equatable {
    package var mode: RunMode = .new
    /// Output run name (new run, fork). The run directory is `runs/<name>`.
    package var name = ""
    /// Existing run to resume, or to fork from (its latest checkpoint).
    package var sourceRun = ""
    package var preset: Preset = .dev
    package var fast = false
    /// New run: blank = preset default. Fork: blank = keep the source state's RNG.
    package var seed = "1"
    /// New run: total steps (blank = preset default). Fork: steps to run from the source checkpoint (blank = the
    /// source config's `run.steps`). Resume: additional steps (blank = original target).
    package var steps = ""
    package var levers = Levers()
    /// Fork only: the source run's lever values; only levers that differ from them are passed. (A new run diffs
    /// the 2D levers against the engine defaults and the 3D levers against `presetWorld3D`.)
    package var leverBaseline = Levers()
    /// New run only: the 3D values of the chosen preset (a 2D preset gives `dims` 2), the baseline of the 3D levers.
    /// The caller keeps it in step with `preset` and pre-fills `levers.world3D` from it (`selectPreset`).
    package var presetWorld3D = World3DLevers()
    /// The ecosystem director (docs/DIRECTOR.md). The context holds what it is diffed against (the engine
    /// defaults for a new run, the source's values for a fork or resume) and what the run allows.
    package var director = DirectorSettings()
    package var directorContext = DirectorContext()
    package var timerEnabled = false
    package var timerHours = 1
    package var timerMinutes = 0

    package init() {}

    /// The run the status, viewer, inspect and reveal buttons refer to.
    package var targetRun: String { mode == .resume ? sourceRun : name.trimmingCharacters(in: .whitespaces) }

    package var timerDuration: TimeInterval { TimeInterval(timerHours * 3600 + timerMinutes * 60) }

    /// What the 3D levers are diffed against: the preset's values for a new run, the source's for a fork, which the
    /// caller loads into `leverBaseline` from the source's `config.json` (`RunController.loadForkLevers`, in Fork mode
    /// only: a resume has no levers, so the app leaves `leverBaseline` at the engine defaults there, and a resume's
    /// dimensions are `RunController.targetRunDims`).
    package var world3DBaseline: World3DLevers { mode == .new ? presetWorld3D : leverBaseline.world3D }

    /// The world's dimensions: the preset's for a new run, the source's for a fork. Never a lever. Whatever
    /// `leverBaseline` holds in a resume (the app does not load it there; see `world3DBaseline`).
    package var worldDims: Int { world3DBaseline.dims }

    /// Chooses a preset and its 3D values: the 3D levers go back to the preset's, the 2D levers keep their edits.
    package mutating func selectPreset(_ preset: Preset, world3D: World3DLevers) {
        self.preset = preset
        presetWorld3D = world3D
        levers.world3D = world3D
    }

    /// The levers as they are validated and applied. `world.dims` is the baseline's whatever was typed, and a fork
    /// also keeps its source's `world.height`: the lattice shape of a run cannot change in a fork (DESIGN §27.1). A
    /// resume has no levers.
    package var effectiveLevers: Levers {
        var effective = levers
        let base = world3DBaseline
        effective.world3D.dims = base.dims
        if mode == .fork { effective.world3D.height = base.height }
        return effective
    }

    /// New run / fork: the run name Start will use. The typed name if `runs/<name>` is free, else the next free
    /// variant (`RunName.nextFree`), so a taken name never blocks a run.
    package func outputName(existingRuns: Set<String>) -> String {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        return existingRuns.contains(trimmed) ? RunName.nextFree(after: trimmed, taken: existingRuns) : trimmed
    }

    package func validate(existingRuns: Set<String>) -> [String] {
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
        if mode != .resume {
            errors += effectiveLevers.validate()
            errors += effectiveLevers.world3D.actuatorErrors(baseline: world3DBaseline)
        }
        errors += director.validate(context: directorContext)
        if timerEnabled && timerDuration <= 0 { errors.append("Timer duration must be greater than zero.") }
        return errors
    }

    /// Arguments after the executable. Call only when `validate` returned no errors.
    package func arguments(existingRuns: Set<String>) -> [String] {
        let trimmedName = outputName(existingRuns: existingRuns)
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
            args += effectiveLevers.setArguments(baseline: Levers(world3D: presetWorld3D))
            args += director.arguments(context: directorContext)
        case .fork:
            args = ["run", "--init-from", "runs/\(sourceRun)"]
            if let s = Parse.integer(trimmedSeed) { args += ["--seed", "\(s)"] }
            if let n = Parse.integer(trimmedSteps) { args += ["--steps", "\(n)"] }
            args += ["--out", "runs/\(trimmedName)"]
            args += effectiveLevers.setArguments(baseline: leverBaseline)
            args += director.arguments(context: directorContext)
        case .resume:
            args = ["resume", "runs/\(sourceRun)"]
            if let n = Parse.integer(trimmedSteps) { args += ["--steps", "\(n)"] }
            args += director.arguments(context: directorContext)
        }
        return args
    }
}

/// Quotes an argument for display in a shell-style command line.
package func shellQuote(_ s: String) -> String {
    if s.range(of: #"^[A-Za-z0-9_@%+=:,./-]+$"#, options: .regularExpression) != nil { return s }
    return "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

// MARK: - Run names

package enum RunName {
    /// The first name not in `taken` derived from `name`: a short trailing number (1-3 digits) counts up,
    /// keeping its width (`trickster2` → `trickster3`, `run007` → `run008`); otherwise `-2`, `-3`, ... is
    /// appended (`baseline` → `baseline-2`, and `run-20261011-1530` → `run-20261011-1530-2`, whose trailing
    /// digits are a time, not a counter).
    package static func nextFree(after name: String, taken: Set<String>) -> String {
        let digits = String(name.reversed().prefix { $0.isASCII && $0.isNumber }.reversed())
        let stem = String(name.dropLast(digits.count))
        let candidates: (Int) -> String
        var n: Int
        if (1...3).contains(digits.count), !stem.isEmpty, let value = Int(digits) {
            candidates = { stem + String(format: "%0\(digits.count)d", $0) }
            n = value + 1
        } else {
            candidates = { "\(name)-\($0)" }
            n = 2
        }
        while taken.contains(candidates(n)) { n += 1 }
        return candidates(n)
    }
}
