import Foundation

// The starting points of the set-up page (docs/APP.md §3.2): a recipe is data, a preset, a number of steps to grow and what the world can be
// for. The owner may edit the table. Times are measured with the real engine on the M5 Pro (docs/APP.md §3.2, DESIGN row I84), not taken
// from the preset guide.

/// What the world is for: *Play* grows it and then makes a planet from it (two worlds); *Study* grows it and stops (one).
package enum Goal: String, CaseIterable, Identifiable, Sendable {
    case play
    case research

    package var id: String { rawValue }

    /// The words on the choice (§2.5: a world is Played or Research, a fact; the goal says which one this click makes).
    package var title: String {
        switch self {
        case .play: return "A world to play"
        case .research: return "A research world"
        }
    }

    package var detail: String {
        switch self {
        case .play: return "Grown, then made into a planet with weather that you can watch and act on."
        case .research: return "Grown and kept as it is: numbers and settings, no weather and no powers."
        }
    }
}

/// A starting point: a preset and what the page asks of it by default.
package struct Recipe: Equatable, Identifiable, Sendable {
    package var id: String
    package var title: String
    /// One sentence for someone who has not read anything.
    package var blurb: String
    package var preset: Preset
    /// The steps the first stage grows (`--steps`).
    package var growSteps: Int
    /// What the world can be for; a flat world has no weather, so it is for study only (§3.2, M16).
    package var goals: [Goal]
    /// The start of a suggested name: `planet` gives `planet-1`, `planet-2`, ...
    package var nameBase: String
    /// Measured on the M5 Pro with the real engine (docs/APP.md §3.2): steps per second once the run is going, the seconds the engine takes
    /// to start and compile before the first step, the seconds the planet takes from "Make planet" to its first picture (fork, weather
    /// warm-up, houses, opening), and the date of the measurement. Nil where nothing is measured.
    package var measured: Measured?

    package struct Measured: Equatable, Sendable {
        package var stepsPerSecond: Double
        package var startSeconds: Double
        package var planetSeconds: Double?
        package var date: String
    }

    package init(id: String, title: String, blurb: String, preset: Preset, growSteps: Int, goals: [Goal], nameBase: String,
                 measured: Measured? = nil) {
        self.id = id
        self.title = title
        self.blurb = blurb
        self.preset = preset
        self.growSteps = growSteps
        self.goals = goals
        self.nameBase = nameBase
        self.measured = measured
    }

    package var defaultGoal: Goal { goals.first ?? .research }
}

package enum Recipes {
    /// Measured 2026-11-01 on the M5 Pro (docs/APP.md §3.2 as built in PR-D; the first run after the engine is installed compiles for longer).
    package static let smallPlanet = Recipe(
        id: "small-planet", title: "Small planet",
        blurb: "A small planet with simple life that has had time to evolve.",
        preset: .tiny3s, growSteps: 30_000, goals: [.play, .research], nameBase: "planet",
        measured: .init(stepsPerSecond: 334, startSeconds: 16, planetSeconds: nil, date: "2026-11-01"))

    package static let biggerPlanet = Recipe(
        id: "bigger-planet", title: "Bigger planet",
        blurb: "A larger planet with more room for life. It takes a good deal longer.",
        preset: .mini3s, growSteps: 100_000, goals: [.play, .research], nameBase: "planet",
        measured: nil)

    package static let flatWorld = Recipe(
        id: "flat-world", title: "Flat world",
        blurb: "A flat world to study. It has no weather, so it is for research only.",
        preset: .dev, growSteps: 20_000, goals: [.research], nameBase: "flat",
        measured: nil)

    package static let all = [smallPlanet, biggerPlanet, flatWorld]

    package static func recipe(id: String) -> Recipe? { all.first { $0.id == id } }
    package static func recipe(for preset: Preset) -> Recipe? { all.first { $0.preset == preset } }
}

/// What the world is made from: one of the recipes, or any preset the engine has.
package enum StartingPoint: Hashable, Identifiable, Sendable {
    case recipe(String)
    case preset(Preset)

    package var id: String {
        switch self {
        case .recipe(let id): return "recipe:" + id
        case .preset(let p): return "preset:" + p.rawValue
        }
    }

    package var recipe: Recipe? {
        if case .recipe(let id) = self { return Recipes.recipe(id: id) }
        return nil
    }

    package var preset: Preset {
        switch self {
        case .recipe(let id): return Recipes.recipe(id: id)?.preset ?? .tiny3s
        case .preset(let p): return p
        }
    }

    package static let initial = StartingPoint.recipe(Recipes.smallPlanet.id)
}

// MARK: - How long it takes

/// Durations in the words of the page: "about 1 min 40 s", "about 12 min". Rounded so that nobody reads a precision that is not there.
package enum TimeWords {
    package static func about(_ seconds: Double) -> String {
        let s = max(0, seconds)
        if s < 10 { return "a few seconds" }
        if s < 60 { return "about \(Int((s / 5).rounded()) * 5) s" }
        if s < 600 {
            let total = Int((s / 10).rounded()) * 10
            let m = total / 60, r = total % 60
            return r == 0 ? "about \(m) min" : "about \(m) min \(r) s"
        }
        if s < 3600 { return "about \(Int((s / 60).rounded())) min" }
        let h = Int(s / 3600), m = Int((s - Double(h) * 3600) / 60 / 5) * 5
        return m == 0 ? "about \(h) h" : "about \(h) h \(m) min"
    }

    /// "1:05" / "12:40" for an elapsed clock.
    package static func clock(_ seconds: Double) -> String {
        let s = Int(max(0, seconds))
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

package enum GrowEstimate {
    /// The seconds a run of `steps` takes at `stepsPerSecond`, with the engine's start (compile) added.
    package static func seconds(steps: Int, stepsPerSecond: Double, startSeconds: Double = 0) -> Double? {
        guard stepsPerSecond.isFinite, stepsPerSecond > 0, steps > 0 else { return nil }
        return Double(steps) / stepsPerSecond + startSeconds
    }

    /// The line under "Grow for": the time, with a note when the world is too short to have evolved.
    package static func line(steps: Int?, stepsPerSecond: Double?, startSeconds: Double = 0) -> String? {
        guard let steps, let stepsPerSecond, let s = seconds(steps: steps, stepsPerSecond: stepsPerSecond, startSeconds: startSeconds) else { return nil }
        var text = TimeWords.about(s)
        if steps < 5000 { text += ". Very short: the world will be mostly unevolved" }
        return text
    }
}
