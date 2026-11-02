import Foundation

// The settings of the set-up page in the words of the page (docs/APP.md §3.5): the plain name of every lever the default path can reach, the
// Changes list, the values remembered from the last world, and the plain messages. The levers themselves and the command builders are the
// existing ones (`Levers`, `World3DLevers`, `SphereLevers`, `FeatureLevers`); this file only reads and writes their text by engine key.

// MARK: - Levers by key

package enum LeverKeys {
    /// (engine key, plain name) of the dynamics levers, in the order of the page.
    package static let dynamics: [(key: String, title: String)] = [
        ("mutation.rate_min", "Mutation rate floor"),
        ("mutation.sigma_min", "Mutation size floor"),
        ("fields.patch_drift", "Food patch drift"),
        ("population.immigration_floor", "Newcomers when the population is below"),
        ("population.immigrants_per_step", "Newcomers per step"),
        ("population.immigration_every", "Newcomers every (steps)"),
        ("population.immigration_count", "Newcomers per event"),
        ("fields.season_period", "Season length (steps)"),
        ("fields.season_amplitude", "Season strength"),
        ("fields.season_mode", "Seasons arrive"),
    ]

    package static let world3D: [(key: String, title: String)] = [
        ("world.height", "World height"),
        ("physics.gravity", "Gravity"),
        ("physics.buoyancy_max", "Buoyancy"),
        ("energy.buoyancy_cost", "Buoyancy cost"),
        ("energy.buoyancy_cap_cost", "Buoyancy upkeep"),
        ("fields.benthic_lambda", "Food near the floor (scale)"),
        ("fields.photic_amp", "Food near the surface (strength)"),
        ("fields.photic_lambda", "Food near the surface (depth scale)"),
    ]

    package static let sphere: [(key: String, title: String)] = [
        ("world.radius", "Planet radius"),
        ("world.nside", "Surface detail"),
        ("world.layers", "Layers"),
        ("physics.ceiling_band", "Soft ceiling"),
    ]

    /// Keys the engine's text uses for the shape of the world; a fork cannot change them.
    package static let shapeKeys: Set<String> = ["world.height", "world.radius", "world.nside", "world.layers"]

    package static var allKeys: [String] {
        (dynamics + world3D + sphere).map(\.key) + FeatureLevers.allSpecs.map(\.key)
    }

    /// A sentence for the tooltip of the words that are still the engine's (the page shows the plain name; the key follows in brackets).
    package static func explanation(_ key: String) -> String? {
        switch key {
        case "world.nside": return "How finely the planet's surface is divided into cells that hold food (the engine's Nside). More detail is slower."
        case "world.layers": return "How many layers the planet's shell has, from the floor to the sky."
        case "physics.ceiling_band": return "A soft limit near the sky that keeps the animals from leaving the world."
        case "mutation.rate_min": return "The least often a child's genes change."
        case "mutation.sigma_min": return "The least a changed gene moves."
        default: return nil
        }
    }

    /// The plain name of a key; nil for a key that is not a lever.
    package static func title(_ key: String) -> String? {
        (dynamics + world3D + sphere).first { $0.key == key }?.title ?? FeatureLevers.spec(key)?.title
    }

    package static func text(_ key: String, in l: Levers) -> String? {
        switch key {
        case "mutation.rate_min": return l.rateMin
        case "mutation.sigma_min": return l.sigmaMin
        case "fields.patch_drift": return l.patchDrift
        case "population.immigration_floor": return l.immigrationFloor
        case "population.immigrants_per_step": return l.immigrantsPerStep
        case "population.immigration_every": return l.immigrationEvery
        case "population.immigration_count": return l.immigrationCount
        case "fields.season_period": return l.seasonPeriod
        case "fields.season_amplitude": return l.seasonAmplitude
        case "fields.season_mode": return l.seasonMode
        case "world.height": return l.world3D.height
        case "physics.gravity": return l.world3D.gravity
        case "physics.buoyancy_max": return l.world3D.buoyancyMax
        case "energy.buoyancy_cost": return l.world3D.buoyancyCost
        case "energy.buoyancy_cap_cost": return l.world3D.buoyancyCapCost
        case "fields.benthic_lambda": return l.world3D.benthicLambda
        case "fields.photic_amp": return l.world3D.photicAmp
        case "fields.photic_lambda": return l.world3D.photicLambda
        case "world.radius": return l.world3D.sphere.radius
        case "world.nside": return l.world3D.sphere.nside
        case "world.layers": return l.world3D.sphere.layers
        case "physics.ceiling_band": return l.world3D.sphere.ceilingBand
        default: return FeatureLevers.spec(key) == nil ? nil : l.features[key]
        }
    }

    @discardableResult
    package static func set(_ key: String, to text: String, in l: inout Levers) -> Bool {
        switch key {
        case "mutation.rate_min": l.rateMin = text
        case "mutation.sigma_min": l.sigmaMin = text
        case "fields.patch_drift": l.patchDrift = text
        case "population.immigration_floor": l.immigrationFloor = text
        case "population.immigrants_per_step": l.immigrantsPerStep = text
        case "population.immigration_every": l.immigrationEvery = text
        case "population.immigration_count": l.immigrationCount = text
        case "fields.season_period": l.seasonPeriod = text
        case "fields.season_amplitude": l.seasonAmplitude = text
        case "fields.season_mode": l.seasonMode = text
        case "world.height": l.world3D.height = text
        case "physics.gravity": l.world3D.gravity = text
        case "physics.buoyancy_max": l.world3D.buoyancyMax = text
        case "energy.buoyancy_cost": l.world3D.buoyancyCost = text
        case "energy.buoyancy_cap_cost": l.world3D.buoyancyCapCost = text
        case "fields.benthic_lambda": l.world3D.benthicLambda = text
        case "fields.photic_amp": l.world3D.photicAmp = text
        case "fields.photic_lambda": l.world3D.photicLambda = text
        case "world.radius": l.world3D.sphere.radius = text
        case "world.nside": l.world3D.sphere.nside = text
        case "world.layers": l.world3D.sphere.layers = text
        case "physics.ceiling_band": l.world3D.sphere.ceilingBand = text
        default:
            guard FeatureLevers.spec(key) != nil else { return false }
            l.features[key] = text
        }
        return true
    }

    /// Whether two texts are one value: numbers as numbers (5e-05 and 0.00005), flags and words as they are.
    package static func same(_ key: String, _ a: String, _ b: String) -> Bool {
        if let spec = FeatureLevers.spec(key) { return spec.same(a, b) }
        if key == "fields.season_mode" { return a == b }
        if let x = Parse.decimal(a), let y = Parse.decimal(b) { return x == y }
        return a.trimmingCharacters(in: .whitespaces) == b.trimmingCharacters(in: .whitespaces)
    }

    /// The levers a form is diffed against: a new run's are the engine's and the preset's, a fork's the source's.
    package static func baseline(of s: RunSettings) -> Levers {
        s.mode == .new ? Levers(world3D: s.presetWorld3D, features: s.featureBaseline) : s.leverBaseline
    }
}

// MARK: - The Changes list

package struct SettingChange: Equatable, Identifiable, Sendable {
    package var key: String
    package var title: String
    package var was: String
    package var now: String
    package var id: String { key }

    package var sentence: String { "\(title) \(was) to \(now)" }
}

package enum ChangeList {
    /// The levers whose value will be passed to the engine, with what they were: exactly the `--set` keys of the command, so that the list
    /// and the command cannot disagree. A key that is not a lever (the engine's tied keys) is left out.
    package static func levers(_ s: RunSettings) -> [SettingChange] {
        guard s.mode != .resume else { return [] }
        let base = LeverKeys.baseline(of: s)
        let args = s.effectiveLevers.setArguments(baseline: base)
        var out: [SettingChange] = []
        var i = 0
        while i + 1 < args.count {
            defer { i += 2 }
            guard args[i] == "--set", let eq = args[i + 1].firstIndex(of: "=") else { continue }
            let key = String(args[i + 1][..<eq])
            guard let title = LeverKeys.title(key), let was = LeverKeys.text(key, in: base) else { continue }
            let now = String(args[i + 1][args[i + 1].index(after: eq)...])
            out.append(SettingChange(key: key, title: title, was: shown(key, was), now: shown(key, now)))
        }
        return out
    }

    /// A value as the person typed it, or the engine's word for a flag.
    private static func shown(_ key: String, _ text: String) -> String {
        if let spec = FeatureLevers.spec(key), case .flag = spec.kind {
            return FeatureLeverSpec.flagValue(text) == true ? "on" : "off"
        }
        return text
    }

    /// Everything else the person chose beyond the starting point that the command carries: add-ons, lighter brains, the seed, a style of
    /// autopilot, a stop-after time. Each a short phrase.
    package static func others(_ s: RunSettings, defaultSeed: String = "1") -> [String] {
        var out: [String] = []
        if s.mode == .new {
            if s.fast { out.append("Lighter brains") }
            if !s.featureOverlays.isEmpty {
                let names = FeatureLevers.groups.filter { s.featureOverlays.contains($0.feature) }.map(\.title)
                out.append("Add-ons: " + names.joined(separator: ", "))
            }
        }
        if s.mode != .resume {
            let seed = s.seed.trimmingCharacters(in: .whitespaces)
            if s.mode == .new, seed != defaultSeed { out.append(seed.isEmpty ? "Seed: the preset's" : "Seed \(seed)") }
            if s.mode == .fork, !seed.isEmpty { out.append("New seed \(seed)") }
        }
        if s.director.mode != .off { out.append("Autopilot on") }
        if s.timerEnabled { out.append("Stops after \(s.timerHours) h \(s.timerMinutes) min") }
        return out
    }

    /// "3 settings changed: Gravity 0.03 to 0.05; Season length 0 to 5000; ..." (nil when none).
    package static func sentence(_ changes: [SettingChange]) -> String? {
        guard !changes.isEmpty else { return nil }
        let head = changes.count == 1 ? "1 setting changed: " : "\(changes.count) settings changed: "
        return head + changes.map(\.sentence).joined(separator: "; ")
    }
}

// MARK: - Remembered values

/// The lever edits of the last world that was started, kept so that the next form starts as the person left it, and shown (a chip on each
/// row and a line at the top of the form) so that a hidden edit never travels into a new world unseen.
package struct RememberedLevers: Equatable, Sendable {
    /// Engine key to the text of the edit.
    package var values: [String: String] = [:]

    package init() {}

    package init(values: [String: String]) { self.values = values }

    package var isEmpty: Bool { values.isEmpty }

    /// The edits of a form that is about to be started: every lever whose value differs from the baseline.
    package static func capture(_ s: RunSettings) -> RememberedLevers {
        var r = RememberedLevers()
        for change in ChangeList.levers(s) {
            // (the text as typed: `change.now` is the engine's number, which is the same value)
            r.values[change.key] = LeverKeys.text(change.key, in: s.effectiveLevers) ?? change.now
        }
        return r
    }

    /// What applying to a form did.
    package struct Applied: Equatable, Sendable {
        /// The keys now set from memory (a chip on each).
        package var keys: [String] = []
        /// Keys that do not apply to this world (a planet's radius in a flat world, a shape in a branch) and were left alone.
        package var skipped: [String] = []
    }

    /// Sets the remembered values on `s.levers`. A 3D key needs a 3D world, a planet's key a planet, and a branch keeps its source's shape.
    package func apply(to s: inout RunSettings) -> Applied {
        var done = Applied()
        let isSphere = s.world3DBaseline.sphere.isSphere
        let is3D = s.world3DBaseline.dims == 3
        for key in values.keys.sorted() {
            let text = values[key] ?? ""
            let world3D = LeverKeys.world3D.contains { $0.key == key }
            let sphere = LeverKeys.sphere.contains { $0.key == key }
            if (world3D && !is3D) || (sphere && !(is3D && isSphere)) || (s.mode == .fork && LeverKeys.shapeKeys.contains(key)) {
                done.skipped.append(key)
                continue
            }
            if LeverKeys.set(key, to: text, in: &s.levers) { done.keys.append(key) } else { done.skipped.append(key) }
        }
        return done
    }

    // MARK: Storage

    package var data: Data? { try? JSONEncoder().encode(values) }

    package init(data: Data?) {
        if let data, let decoded = try? JSONDecoder().decode([String: String].self, from: data) { values = decoded }
    }

    package static let defaultsKey = "setup.remembered.v1"
}

// MARK: - Plain messages

/// The engine's checks in the words of the page, for the default path only: mutation, immigration, seasons, the 3D basics, name, seed and
/// steps. A key the default path cannot reach keeps the engine's own message (§3.5).
package enum SetupMessages {
    private static let keyed: [String: String] = [
        "mutation.rate_min": "Mutation rate floor must be above 0 and below 1.",
        "mutation.sigma_min": "Mutation size floor must be a number above 0.",
        "fields.patch_drift": "Food patch drift must be a number, 0 or more.",
        "population.immigration_floor": "Newcomers when the population is below must be a whole number, 0 or more.",
        "population.immigrants_per_step": "Newcomers per step must be a whole number, 0 or more.",
        "population.immigration_every": "Newcomers every must be a whole number of steps, 0 or more (0 is none).",
        "population.immigration_count": "Newcomers per event must be a whole number, 0 or more.",
        "fields.season_period": "Season length must be a whole number of steps, 0 or more (0 is no seasons).",
        "fields.season_amplitude": "Season strength must be a number from 0 to 1.",
        "world.height": "World height must be a number above 0.",
        "physics.gravity": "Gravity must be a number, 0 or more.",
        "physics.buoyancy_max": "Buoyancy must be a number, 0 or more.",
        "energy.buoyancy_cost": "Buoyancy cost must be a whole number, 0 or more.",
        "energy.buoyancy_cap_cost": "Buoyancy upkeep must be a whole number, 0 or more.",
        "fields.benthic_lambda": "Food near the floor (scale) must be a number above 0.",
        "fields.photic_amp": "Food near the surface (strength) must be a number from 0 to 1.",
        "fields.photic_lambda": "Food near the surface (depth scale) must be a number above 0.",
    ]

    private static let conditional: [String: String] = [
        "population.immigrants_per_step": "Newcomers per step must be at least 1 while newcomers are added below a population floor.",
        "population.immigration_count": "Newcomers per event must be at least 1 when newcomers arrive every few steps.",
        "fields.season_period": "Season length must be at least 1 when the seasons have a strength.",
        "fields.photic_lambda": "Food near the surface (depth scale) must be above 0 when its strength is above 0.",
        "energy.buoyancy_cost": "Buoyancy costs must be 0 when buoyancy is 0.",
        "energy.buoyancy_cap_cost": "Buoyancy costs must be 0 when buoyancy is 0.",
    ]

    /// One message of `RunSettings.validate` (or of the name check) in the page's words; the engine's own text when the page has no words.
    package static func plain(_ message: String) -> String {
        switch message {
        case "Enter a run name.": return "Give the world a name."
        case "Run name may use letters, digits, '.', '_' and '-' only.":
            return "A name can use letters, digits, dots, dashes and underscores, and must start with a letter or a digit."
        case "Seed must be an integer >= 0 (or blank).": return "The seed must be a whole number, 0 or more."
        case "Steps must be >= 1.", "Steps must be a positive integer (or blank).": return "Grow for must be a whole number of steps, 1 or more."
        case "Timer duration must be greater than zero.": return "The time to stop after must be more than zero."
        default: break
        }
        guard let first = message.split(separator: " ").first else { return message }
        let key = String(first)
        guard let base = keyed[key] else { return message }
        if message.contains(" when "), let alt = conditional[key] { return alt }
        return base
    }

    /// The plain message for a key's whole row, with the key left for the tooltip: "Gravity must be ..." and `physics.gravity`.
    package static func key(of message: String) -> String? {
        guard let first = message.split(separator: " ").first, keyed[String(first)] != nil else { return nil }
        return String(first)
    }
}
