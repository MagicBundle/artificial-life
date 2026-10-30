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

/// The shape of a planet: `world.radius`, `world.height`, `world.nside` and `world.layers`. The weather overlay of a planet is
/// calibrated for one shape (`PlanetSpec.weatherOverlays`), and the engine refuses it on any other.
package struct PlanetShape: Equatable, Sendable {
    package var radius: Double
    package var height: Double
    package var nside: Int
    package var layers: Int

    package init(radius: Double, height: Double, nside: Int, layers: Int) {
        self.radius = radius
        self.height = height
        self.nside = nside
        self.layers = layers
    }
}

/// What the launcher asks to make a planet: a fork of an evolved checkpoint with the weather on and spun up.
package struct PlanetSpec: Equatable, Sendable {
    /// The directory name under `runs/` of the source, and the step of its checkpoint (nil: the newest).
    package var source: String
    package var sourceStep: Int?
    /// The name of the new world under `runs/`.
    package var name: String
    /// `world.size` of the source (`PlaySource.worldSize`), nil if unknown: it picks the overlay (`overlay(forWorldSize:)`).
    package var sourceWorldSize: Double?
    /// `world.geometry` of the source (`PlaySource.geometry`), nil if unknown: it picks the weather overlay, a planet's by its
    /// shape (`weatherOverlay(forGeometry:shape:worldSize:)`), never by an alias.
    package var sourceGeometry: String?
    /// The shape of a sphere source (`PlaySource.planetShape`), nil for a box or when unreadable: a planet's weather overlay is
    /// calibrated for one shape each, and a planet of another shape has none (`weatherObstacle`).
    package var sourceShape: PlanetShape?
    /// `world.dims` of the source (`PlaySource.dims`), nil if unknown: the weather runs only in a 3D world, so a 2D source
    /// cannot take the storm overlay (the engine refuses it with its weather rules 7, W-1 and R4).
    package var sourceDims: Int?
    /// `--overlay storm` (or `storm160` by the source's size, `stormsph` or `stormsph16` by a planet's shape): turn the weather on (a source whose weather is on is
    /// forked as it is, without the overlay).
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
    /// The `world.size` of the mini world (preset `mini3d`, DESIGN §37), whose weather has its own overlay.
    package static let miniWorldSize = 160.0
    /// The default spin-up of a mini world: its first storms die between weather time 24k and 60k, so a longer spin-up
    /// gives the first minutes of play more storm events (DESIGN §37.2).
    package static let miniSpinup = 36_000

    /// The weather overlay of a source of this `world.size`: `storm160` for the 160 world, else `storm` (the 320 world).
    /// Explicit, never an alias: the same name would mean two geometries (DESIGN §37.2).
    package static func overlay(forWorldSize size: Double?) -> String { size == miniWorldSize ? "storm160" : "storm" }

    /// The weather overlay of each planet shape that has one, `(shape, overlay)`. PROVISIONAL, as the overlays are (S4c calibrates them):
    /// `stormsph` is made for the `64k3s` planet (radius 90, height 40, field Nside 21, 18 layers), `stormsph16` for `tiny3s` (36, 16, 8, 8).
    /// `mini3s` (48, 24, 11, 12) has none yet. `tests/test_gui_sync.py` checks the shapes against the engine's presets.
    package static let weatherOverlays: [(shape: PlanetShape, overlay: String)] = [
        (PlanetShape(radius: 90, height: 40, nside: 21, layers: 18), "stormsph"),
        (PlanetShape(radius: 36, height: 16, nside: 8, layers: 8), "stormsph16"),
    ]

    /// The weather overlay of a planet of this shape; nil when none is calibrated for it (or the shape is unknown).
    package static func overlay(forShape shape: PlanetShape?) -> String? {
        guard let shape else { return nil }
        return weatherOverlays.first { $0.shape == shape }?.overlay
    }

    /// The weather overlay of a source of this geometry, planet shape and `world.size`: a planet's own by its shape (`overlay(forShape:)`,
    /// validated against the planet it lands on; nil when the shape has none), else the box's by size. Chosen by the source's
    /// `world.geometry` and shape, never by an alias (DESIGN §46.8, §46.9).
    package static func weatherOverlay(forGeometry geometry: String?, shape: PlanetShape?, worldSize size: Double?) -> String? {
        geometry == "sphere" ? overlay(forShape: shape) : overlay(forWorldSize: size)
    }

    /// Why the weather cannot be turned on for a source of `dims` dimensions, this geometry and shape; nil when it can (or the
    /// dims are unknown: the engine then decides). A 2D source has no weather; a planet of a shape without a calibrated overlay
    /// has none (the engine refuses `stormsph` there with its Courant and pool rules); a source whose weather is already on is
    /// forked as it is and needs no overlay.
    package static func weatherObstacle(dims: Int?, geometry: String? = nil, shape: PlanetShape? = nil, weatherOn: Bool = false) -> String? {
        if dims == 2 { return "The weather runs only in a 3D world: this source is 2D. Turn the weather off, or choose a 3D source." }
        if geometry == "sphere", !weatherOn, overlay(forShape: shape) == nil {
            return "No weather is calibrated yet for this planet size. It is calibrated for the 64k3s (radius 90) and tiny3s (radius 36) planets."
        }
        return nil
    }

    /// The spin-up of a planet. The engine has no sphere-specific value (the storm calibration, S4c, is open: the S4b tests spin up 40 000 steps for
    /// stability, the rest state needs only N >= 1), so this is the box's 320 world's, keyed on the geometry so that S4c changes one constant and
    /// the size of a sphere (`world.size` is only its image length there) never picks the mini world's. It is valid for both calibrated planets
    /// (the weather time of a step is at most 1, so 20 000 steps are at most 20 000 units of it; the spin-up of a tiny3s planet takes seconds).
    package static let sphereSpinup = 20_000

    package static func defaultSpinup(forGeometry geometry: String?, worldSize size: Double?) -> Int {
        geometry == "sphere" ? sphereSpinup : defaultSpinup(forWorldSize: size)
    }

    package static func defaultSpinup(forWorldSize size: Double?) -> Int {
        size == miniWorldSize ? miniSpinup : defaultSpinup
    }

    package init(source: String, sourceStep: Int? = nil, name: String, storm: Bool = true,
                 weatherSpinup: Int = PlanetSpec.defaultSpinup, directorEnabled: Bool = true,
                 houses: PlayHouses = .auto, seed: Int? = nil, rate: PlayRate? = .multiplier(1),
                 director: PlayDirector = .off, sourceWorldSize: Double? = nil, sourceDims: Int? = nil,
                 sourceGeometry: String? = nil, sourceShape: PlanetShape? = nil) {
        self.source = source
        self.sourceGeometry = sourceGeometry
        self.sourceShape = sourceShape
        self.sourceDims = sourceDims
        self.sourceStep = sourceStep
        self.sourceWorldSize = sourceWorldSize
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
            if let obstacle = PlanetSpec.weatherObstacle(dims: sourceDims, geometry: sourceGeometry, shape: sourceShape) { errors.append(obstacle) }
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
        // validate() refuses a planet of a shape without an overlay; without one the weather flags are left out rather than guessed
        let overlay = p.storm ? PlanetSpec.weatherOverlay(forGeometry: p.sourceGeometry, shape: p.sourceShape, worldSize: p.sourceWorldSize) : nil
        if let overlay { args += ["--overlay", overlay] }
        if p.directorEnabled { args += ["--set", "director.enabled=true"] }
        if overlay != nil { args += ["--weather-spinup", "\(p.weatherSpinup)"] }
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
