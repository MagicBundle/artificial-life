import Foundation

/// What the Worlds page says and offers next, from facts (docs/APP.md §2.4: every page has one obvious next step, and none is empty
/// without saying what to do). Pure: the page draws the answer.
package enum WorldsGuide {
    /// The facts the page reads.
    package struct Facts: Equatable, Sendable {
        package var folderExists: Bool
        /// The sentence for a missing folder (`RepoFolder.missingMessage`), nil when the folder exists.
        package var folderProblem: String?
        package var engineReady: Bool
        package var catalog: PlayCatalog
        /// The world that is open in Watch (starting or live), nil when none is.
        package var openWorld: String?
        /// Whether the open world is still starting (its fork and spin-up take a while).
        package var openWorldIsStarting: Bool
        /// Directory names that were asked to stop and are still saving.
        package var saving: Set<String>

        package init(folderExists: Bool, folderProblem: String? = nil, engineReady: Bool, catalog: PlayCatalog = PlayCatalog(),
                     openWorld: String? = nil, openWorldIsStarting: Bool = false, saving: Set<String> = []) {
            self.folderExists = folderExists
            self.folderProblem = folderProblem
            self.engineReady = engineReady
            self.catalog = catalog
            self.openWorld = openWorld
            self.openWorldIsStarting = openWorldIsStarting
            self.saving = saving
        }
    }

    /// The one thing to do now.
    package enum NextStep: Equatable, Sendable {
        /// No usable alife folder: choose one.
        case chooseFolder(problem: String)
        /// A folder without an engine: set it up.
        case setUpEngine
        /// An engine, and nothing to make a planet from: a research run is made in the Lab.
        case makeFirstRun
        /// An engine and a run to make a planet from, and no world yet.
        case makePlanet(from: String)
        /// A world that can be played, the most recent first.
        case play(world: String)
        /// Nothing can be played now (every world is being played elsewhere or saving): wait, or make a planet.
        case waitForSave
        /// A world is open.
        case watch(world: String)
        /// A world is starting.
        case starting(world: String)
    }

    package static func nextStep(_ facts: Facts) -> NextStep {
        if let open = facts.openWorld { return facts.openWorldIsStarting ? .starting(world: open) : .watch(world: open) }
        if !facts.folderExists { return .chooseFolder(problem: facts.folderProblem ?? "The alife folder was not found.") }
        if !facts.engineReady { return .setUpEngine }
        if let playable = facts.catalog.worlds.first(where: { reason(forPlaying: $0, in: facts) == nil }) { return .play(world: playable.name) }
        if !facts.catalog.worlds.isEmpty { return .waitForSave }
        if let source = facts.catalog.sources.first { return .makePlanet(from: source.name) }
        return .makeFirstRun
    }

    /// Why a world cannot be played now, in the words of the page; nil when it can. Always said beside the disabled button.
    package static func reason(forPlaying world: PlayWorld, in facts: Facts) -> String? {
        if let open = facts.openWorld { return open == world.name ? "Open now" : "Close \(open) first" }
        if facts.saving.contains(world.name) { return "Saving..." }
        if world.locked { return "Being played in another window" }
        if world.newestStep == nil { return "No save point" }
        return nil
    }

    /// Why *Make planet* is unavailable beyond the form's own errors; nil when the form decides.
    package static func reasonToMakePlanet(_ facts: Facts) -> String? {
        if let open = facts.openWorld { return "Close \(open) first: a new planet replaces the open world." }
        if !facts.folderExists { return facts.folderProblem ?? "The alife folder was not found." }
        if !facts.engineReady { return "Set up the engine first." }
        if facts.catalog.sources.isEmpty {
            return "A planet is made from a research run with a save point, and there is none yet. Make one in the Lab."
        }
        return nil
    }

    /// The row's second line: "Planet · step 450,000 · saved 2 hours ago".
    package static func summary(_ world: PlayWorld, now: Date = Date()) -> String {
        var parts: [String] = [world.geometry == "sphere" ? "Planet" : "World"]
        if let step = world.newestStep { parts.append("step " + grouped(step)) }
        if world.file.planet != nil, let source = world.file.planet?.source { parts.append("made from " + source) }
        let when = relative(world.modified, now: now)
        parts.append(when == "just now" ? "saved just now" : "saved " + when)
        return parts.joined(separator: " · ")
    }

    package static func grouped(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.locale = Locale(identifier: "en_US")
        return f.string(from: NSNumber(value: n)) ?? String(n)
    }

    package static func relative(_ date: Date, now: Date) -> String {
        if abs(now.timeIntervalSince(date)) < 60 { return "just now" }
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .full
        f.locale = Locale(identifier: "en_US")
        return f.localizedString(for: date, relativeTo: now)
    }
}
