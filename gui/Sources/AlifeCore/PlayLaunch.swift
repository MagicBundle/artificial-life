import Foundation

// The command lines of `alife play` (DESIGN §33.2.8), built from what the launcher asks. Pure: the flags are the
// ones `alife play --help` lists (`tests/test_gui_sync.py` checks the names) and the keys are `Config` keys.

/// How the houses of a planet are made (`--houses`).
package enum PlayHouses: String, CaseIterable, Sendable {
    /// Rebuild the exact houses from the source's event chunks, and refuse if they are missing.
    case exact
    /// The clades.
    case clade
    /// Try `exact` and fall back to the clades with a note (the default).
    case auto
}

/// The director of a played world (`--director`): the rule tier decides, or nobody (the LLM tier is not used in play).
package enum PlayDirector: String, CaseIterable, Sendable {
    case off
    case rules
}

/// What the launcher asks to continue a world that exists.
package struct PlayResume: Equatable, Sendable {
    /// The directory name under `runs/`.
    package var run: String
    /// `--checkpoint`: a step, or nil for the newest.
    package var checkpoint: Int?
    /// `--truncate`: allow a checkpoint older than the newest (later checkpoints, rows and chunks are deleted).
    package var truncate = false
    /// `--rate`: a world opens running at this rate (D8: a planet opens running at 1x); nil starts it paused.
    package var rate: PlayRate? = .multiplier(1)
    package var director = PlayDirector.off
    /// `--replay`: apply the world's director log instead of deciding.
    package var replay = false

    package init(run: String, checkpoint: Int? = nil, truncate: Bool = false, rate: PlayRate? = .multiplier(1),
                 director: PlayDirector = .off, replay: Bool = false) {
        self.run = run
        self.checkpoint = checkpoint
        self.truncate = truncate
        self.rate = rate
        self.director = director
        self.replay = replay
    }
}

/// What the launcher asks to make a planet: a fork of an evolved checkpoint with the weather on and spun up.
package struct PlanetSpec: Equatable, Sendable {
    /// The directory name under `runs/` of the source, and the step of its checkpoint (nil: the newest).
    package var source: String
    package var sourceStep: Int?
    /// The name of the new world under `runs/`.
    package var name: String
    /// `--overlay storm`: turn the weather on (a source whose weather is on is forked as it is, without the overlay).
    package var storm = true
    /// `--weather-spinup N`: needed, with N >= 1, when the overlay turns the weather on; refused for a weather-on source.
    package var weatherSpinup = PlanetSpec.defaultSpinup
    /// `--set director.enabled=true`: compile the director into the world so a rule tier can be selected (D2).
    package var directorEnabled = true
    package var houses = PlayHouses.auto
    /// `--seed`: re-partitions the houses of the past and the future (blank: the source's).
    package var seed: Int?
    package var rate: PlayRate? = .multiplier(1)
    package var director = PlayDirector.off

    package static let defaultSpinup = 20_000

    package init(source: String, sourceStep: Int? = nil, name: String, storm: Bool = true,
                 weatherSpinup: Int = PlanetSpec.defaultSpinup, directorEnabled: Bool = true,
                 houses: PlayHouses = .auto, seed: Int? = nil, rate: PlayRate? = .multiplier(1),
                 director: PlayDirector = .off) {
        self.source = source
        self.sourceStep = sourceStep
        self.name = name
        self.storm = storm
        self.weatherSpinup = weatherSpinup
        self.directorEnabled = directorEnabled
        self.houses = houses
        self.seed = seed
        self.rate = rate
        self.director = director
    }

    /// Why the spec cannot be launched; empty when it can. `existing` are the names under `runs/`.
    package func validate(existing: Set<String>) -> [String] {
        var errors: [String] = []
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        if source.isEmpty { errors.append("Choose a source run.") }
        if trimmed.isEmpty {
            errors.append("Enter a name for the planet.")
        } else if trimmed.range(of: #"^[A-Za-z0-9][A-Za-z0-9._-]*$"#, options: .regularExpression) == nil {
            errors.append("The name may use letters, digits, '.', '_' and '-' only.")
        } else if existing.contains(trimmed) {
            errors.append("'runs/\(trimmed)' exists.")
        }
        if let sourceStep, sourceStep < 0 { errors.append("The checkpoint step is not negative.") }
        if storm {
            if weatherSpinup < 1 { errors.append("Turning the weather on needs a spin-up of at least 1 step.") }
        } else if weatherSpinup != 0 {
            errors.append("A source whose weather is on is forked as it is: no spin-up.")
        }
        if director == .rules, !directorEnabled {
            errors.append("The rule tier needs the director compiled into the world (director.enabled).")
        }
        if let seed, seed < 0 { errors.append("The seed is not negative.") }
        return errors
    }
}

package enum PlayLaunch {
    /// The arguments of `alife play` (after the executable) for a world that exists.
    package static func arguments(resume r: PlayResume) -> [String] {
        var args = ["play", "runs/\(r.run)"]
        if let step = r.checkpoint { args += ["--checkpoint", "\(step)"] }
        if r.truncate { args.append("--truncate") }
        if r.replay { args.append("--replay") }
        args += common(rate: r.rate, director: r.director)
        return args
    }

    /// The arguments of `alife play --init-from` for a planet. Call only when `validate` returned no errors.
    package static func arguments(planet p: PlanetSpec) -> [String] {
        var source = "runs/\(p.source)"
        if let step = p.sourceStep { source += ":\(step)" }
        var args = ["play", "--init-from", source]
        if p.storm { args += ["--overlay", "storm"] }
        if p.directorEnabled { args += ["--set", "director.enabled=true"] }
        if p.storm { args += ["--weather-spinup", "\(p.weatherSpinup)"] }
        args += ["--houses", p.houses.rawValue]
        if let seed = p.seed { args += ["--seed", "\(seed)"] }
        args += ["--out", "runs/\(p.name.trimmingCharacters(in: .whitespaces))"]
        args += common(rate: p.rate, director: p.director)
        return args
    }

    /// `--rate` and `--director`. A world opens paused without a rate; `--director off` is the default and not passed.
    private static func common(rate: PlayRate?, director: PlayDirector) -> [String] {
        var args: [String] = []
        if let rate {
            switch rate {
            case .max: args += ["--rate", "max"]
            case .multiplier(let m): args += ["--rate", PlayCommand.decimal(m)]
            }
        }
        if director != .off { args += ["--director", director.rawValue] }
        return args
    }
}
