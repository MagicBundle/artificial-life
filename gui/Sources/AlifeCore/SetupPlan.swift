import Foundation

// What one press of the set-up page's button makes (docs/APP.md §3.2): the stages, the command of each, the worlds that result, and the names
// they take. Built from the existing builders, which are not touched: `RunSettings.arguments` for a run (Create, Branch, Keep growing) and
// `PlayLaunch.arguments(planet:)` for a planet (Make planet, and the second stage of "grow, then play").

/// What the person is doing on the set-up page.
package enum SetupIntent: Equatable, Sendable {
    /// Create a new world from a starting point.
    case create
    /// Copy a world from its newest save point and change settings (`alife run --init-from`).
    case branch(from: String)
    /// Continue a research world (`alife resume`).
    case keepGrowing(world: String)
    /// Make a played world from a save point (`alife play --init-from`).
    case makePlanet(from: String)

    /// The world the intent starts from, nil for Create.
    package var world: String? {
        switch self {
        case .create: return nil
        case .branch(let w), .keepGrowing(let w), .makePlanet(let w): return w
        }
    }

    package var isMakePlanet: Bool { if case .makePlanet = self { return true } else { return false } }

    /// The page's title.
    package var title: String {
        switch self {
        case .create: return "New world"
        case .branch(let w): return "Branch from \(Self.shown(w))"
        case .keepGrowing(let w): return "Keep growing \(Self.shown(w))"
        case .makePlanet(let w): return "Make a planet from \(Self.shown(w))"
        }
    }

    /// A world's name in a title: the world a planet was grown from is "the grown world of planet-1", not its directory `planet-1-grow`.
    package static func shown(_ world: String) -> String {
        guard world.hasSuffix(SetupPlan.growSuffix), world.count > SetupPlan.growSuffix.count else { return world }
        return "the grown world of " + String(world.dropLast(SetupPlan.growSuffix.count))
    }

    /// The button that starts it (§2.5: one word per action).
    package var buttonLabel: String {
        switch self {
        case .create: return "Create"
        case .branch: return "Branch"
        case .keepGrowing: return "Keep growing"
        case .makePlanet: return "Make planet"
        }
    }

    /// One sentence at the top of the page saying what the page does (a change of a world's settings is a branch, §3.5).
    package var summary: String {
        switch self {
        case .create:
            return "Choose a starting point and press Create. alife grows a world for you and shows you how far it has got."
        case .branch:
            return "A world's settings are part of its history, so a change makes a new world. It starts from the newest save point of the world you branch from, which is not changed."
        case .keepGrowing:
            return "Keeps growing this research world from its newest save point. It stays the same world."
        case .makePlanet:
            return "A planet is a world you can watch and act on, made from a save point of a world you have. The world it starts from is not changed."
        }
    }
}

/// One process of a plan.
package struct PlanStage: Equatable, Sendable {
    package enum Kind: String, Sendable { case grow, branch, keepGrowing, makePlanet }
    package var kind: Kind
    /// The directory under `runs/` it writes.
    package var world: String
    /// The directory it starts from, if any.
    package var reads: String?
    /// The arguments after `alife`.
    package var arguments: [String]
}

/// A world a plan makes, with its role in words.
package struct MadeWorld: Equatable, Sendable {
    package var name: String
    package var kind: WorldKind
    package var role: String
}

package struct CommandBlock: Equatable, Sendable {
    package var title: String
    package var command: String
}

package struct SetupPlan: Equatable, Sendable {
    package var intent: SetupIntent
    package var goal: Goal
    /// The world the person asked for.
    package var name: String
    package var stages: [PlanStage]
    package var made: [MadeWorld]

    /// The end of the name of a planet's seed: `planet-1-grow` is the research world `planet-1` was made from.
    package static let growSuffix = "-grow"

    /// `NAME-grow`: the seed of a planet that Create makes.
    package static func seedName(_ name: String) -> String { name + growSuffix }

    /// "This makes 2 worlds: planet-1 (you play it) and planet-1-grow (the grown world it starts from)."
    package var madeSentence: String? {
        guard made.count > 1 else { return nil }
        let parts = made.map { "\($0.name) (\($0.role))" }
        return "This makes \(made.count) worlds: " + parts.joined(separator: " and ") + "."
    }

    /// The pipeline's mode.
    package var pipelineMode: PipelinePlan.Mode {
        switch intent {
        case .create: return goal == .play ? .growThenPlay : .growOnly
        case .branch: return .growOnly
        case .keepGrowing: return .keepGrowing
        case .makePlanet: return .makePlanet
        }
    }

    /// Builds the plan. `settings` is the form's run settings (mode `.new`, `.fork` or `.resume` for the intent) and is used as it is, except
    /// for the output name of stage 1; `planet` is the second stage's spec (its `source` and `name` are set here for "grow, then play").
    package static func make(intent: SetupIntent, goal: Goal, name: String, settings: RunSettings, planet: PlanetSpec?, existing: Set<String>) -> SetupPlan {
        let asked = name.trimmingCharacters(in: .whitespaces)
        switch intent {
        case .create:
            if goal == .play, var planet {
                let seed = seedName(asked)
                var run = settings
                run.name = seed
                planet.source = seed
                planet.sourceStep = nil
                planet.name = asked
                return SetupPlan(
                    intent: intent, goal: goal, name: asked,
                    stages: [
                        PlanStage(kind: .grow, world: seed, reads: nil, arguments: run.arguments(existingRuns: existing)),
                        PlanStage(kind: .makePlanet, world: asked, reads: seed, arguments: PlayLaunch.arguments(planet: planet)),
                    ],
                    made: [MadeWorld(name: asked, kind: .played, role: "you play it"), MadeWorld(name: seed, kind: .research, role: "the grown world it starts from")])
            }
            var run = settings
            run.name = asked
            return SetupPlan(
                intent: intent, goal: goal, name: asked,
                stages: [PlanStage(kind: .grow, world: asked, reads: nil, arguments: run.arguments(existingRuns: existing))],
                made: [MadeWorld(name: asked, kind: .research, role: "you study it")])
        case .branch(let from):
            var run = settings
            run.name = asked
            return SetupPlan(
                intent: intent, goal: .research, name: asked,
                stages: [PlanStage(kind: .branch, world: asked, reads: from, arguments: run.arguments(existingRuns: existing))],
                made: [MadeWorld(name: asked, kind: .research, role: "branched from \(from)")])
        case .keepGrowing(let world):
            return SetupPlan(
                intent: intent, goal: .research, name: world,
                stages: [PlanStage(kind: .keepGrowing, world: world, reads: world, arguments: settings.arguments(existingRuns: existing))],
                made: [])
        case .makePlanet(let from):
            var spec = planet ?? PlanetSpec(source: from, name: asked)
            spec.name = asked
            return SetupPlan(
                intent: intent, goal: .play, name: asked,
                stages: [PlanStage(kind: .makePlanet, world: asked, reads: from, arguments: PlayLaunch.arguments(planet: spec))],
                made: [MadeWorld(name: asked, kind: .played, role: "you play it")])
        }
    }

    // MARK: Show command

    /// The command blocks of "Show command": one per stage, labelled when there are two.
    package func commandBlocks(folder: String) -> [CommandBlock] {
        let place = folder.trimmingCharacters(in: .whitespaces).isEmpty ? "<alife folder>" : shellQuote(folder)
        return stages.enumerated().map { index, stage in
            let command = "cd \(place) && .venv/bin/alife " + stage.arguments.map(shellQuote).joined(separator: " ")
            return CommandBlock(title: stages.count > 1 ? "Stage \(index + 1)" : "Command", command: command)
        }
    }

    /// What Copy puts on the clipboard: the stages one after the other, joined by `&&`, so that the second runs only if the first succeeds.
    package func copyScript(folder: String) -> String {
        let place = folder.trimmingCharacters(in: .whitespaces).isEmpty ? "<alife folder>" : shellQuote(folder)
        return "cd \(place) && " + stages.map { ".venv/bin/alife " + $0.arguments.map(shellQuote).joined(separator: " ") }.joined(separator: " && ")
    }

    // MARK: Names

    /// What is wrong with the name for this goal; empty when it is fine. `taken` is every name that is not free (worlds, builds, links to
    /// drives that are not connected, entries that cannot be read, what a pipeline has reserved). Names that differ only in capitals clash.
    package static func nameProblems(_ raw: String, goal: Goal, intent: SetupIntent, taken: Set<String>) -> [String] {
        if case .keepGrowing = intent { return [] }
        let name = raw.trimmingCharacters(in: .whitespaces)
        if name.isEmpty { return ["Give the world a name."] }
        if name.range(of: #"^[A-Za-z0-9][A-Za-z0-9._-]*\z"#, options: .regularExpression) == nil {
            return ["A name can use letters, digits, dots, dashes and underscores, and must start with a letter or a digit."]
        }
        var out: [String] = []
        var wanted = [name]
        if intent == .create, goal == .play { wanted.append(seedName(name)) }
        for candidate in wanted {
            if let clash = NameFolding.clash(of: candidate, in: taken) {
                out.append(clash == candidate
                    ? "A world called \(clash) is already here."
                    : "A world called \(clash) is already here, and names that differ only in capitals count as the same.")
            }
        }
        return out
    }

    /// The next free name after `base-N`: `planet-1`, `planet-2`, ... where neither the name nor, for a planet, its seed is taken.
    package static func suggestName(base: String, goal: Goal, taken: Set<String>) -> String {
        var n = 1
        while n < 10_000 {
            let candidate = "\(base)-\(n)"
            if NameFolding.clash(of: candidate, in: taken) == nil, goal == .research || NameFolding.clash(of: seedName(candidate), in: taken) == nil {
                return candidate
            }
            n += 1
        }
        return "\(base)-\(UUID().uuidString.prefix(6))"
    }
}
