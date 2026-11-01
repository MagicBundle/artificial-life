import Foundation

/// What the Worlds page says and offers next, from facts (docs/APP.md §2.4: every page has one obvious next step, and none is empty
/// without saying what to do). Pure: the page draws the answer.
package enum WorldsGuide {
    /// The run that is still growing, which a planet can already be made from.
    package static func growingNote(source: String, growing: String?) -> String? {
        guard growing == source else { return nil }
        return "\(source) is still growing. A planet starts from its latest save point."
    }

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
        /// The directory the Lab's run is writing (`alife run` or `alife resume`), nil when none is: a world is never played over it.
        package var growing: String?

        package init(folderExists: Bool, folderProblem: String? = nil, engineReady: Bool, catalog: PlayCatalog = PlayCatalog(),
                     openWorld: String? = nil, openWorldIsStarting: Bool = false, saving: Set<String> = [], growing: String? = nil) {
            self.growing = growing
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
        if facts.growing == world.name { return "Growing in the Lab" }
        if world.locked { return WorldState.inUseElsewhere.title }
        if world.newestStep == nil { return "No save point" }
        return nil
    }

    /// What the Worlds page says when a world did not open: one sentence in plain words, from the controller's message (the message
    /// itself is shown under it, for whoever needs it). Never says "failed" without saying what to do next.
    package static func explainFailure(world: String, message: String) -> String {
        let name = world.isEmpty ? "The world" : "\(world)"
        let lower = message.lowercased()
        if lower.contains("still saving") {
            return "\(name) did not open: it is still being saved. Wait a moment, then press Play again."
        }
        if lower.contains("failed to launch") {
            return "\(name) did not open: the engine could not be started. Check the engine in Settings, then try again."
        }
        if lower.contains("no usable checkpoint") {
            return "\(name) did not open: it has no usable save point. Choose another world, or make a planet from a run."
        }
        if lower.contains("refused the request") {
            return "\(name) did not open: the engine refused the request. Activity has its message."
        }
        if lower.contains("interrupted") || lower.contains("terminated") {
            return "\(name) was stopped before it was ready. Press Play to try again."
        }
        return "\(name) did not open: the engine stopped. Activity has what it said."
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
        if let source = SourceRef.parse(world.file.planet?.source) { parts.append("made from " + source.name) }
        let when = relative(world.modified, now: now)
        parts.append(when == "just now" ? "saved just now" : "saved " + when)
        return parts.joined(separator: " · ")
    }

    // (Formatters are slow to make and a long list asks for hundreds of strings: made once, used from the main actor and the tests'.)
    private nonisolated(unsafe) static let groupedFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.locale = Locale(identifier: "en_US")
        return f
    }()
    private nonisolated(unsafe) static let relativeFormatter: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .full
        f.locale = Locale(identifier: "en_US")
        return f
    }()
    private static let formatterLock = NSLock()

    package static func grouped(_ n: Int) -> String {
        formatterLock.lock()
        defer { formatterLock.unlock() }
        return groupedFormatter.string(from: NSNumber(value: n)) ?? String(n)
    }

    package static func relative(_ date: Date, now: Date) -> String {
        if abs(now.timeIntervalSince(date)) < 60 { return "just now" }
        formatterLock.lock()
        defer { formatterLock.unlock() }
        return relativeFormatter.localizedString(for: date, relativeTo: now)
    }

    // MARK: The library's cards (docs/APP.md §5.2)

    /// Why *Make planet* is unavailable from this world's card; nil when it can be made. Always said beside the grey button.
    package static func reasonToMakePlanet(from row: LibraryRow, in facts: Facts) -> String? {
        // (Short: the open world's card at the top of the page says why, and every card repeats this line.)
        if let open = facts.openWorld { return open == row.name ? "Close it first" : "Close \(open) first" }
        if !facts.folderExists { return facts.folderProblem ?? "The alife folder was not found." }
        if !facts.engineReady { return "Set up the engine first." }
        if row.state == .saving || facts.saving.contains(row.name) { return "Saving..." }
        if !row.hasSavePoint { return "No save point to start from yet." }
        return nil
    }

    /// Why *Make again with a new name* is unavailable for an unfinished build; nil when it can be done. `source` is the world it was made
    /// from (from `meta.json`, or from this session's own request), `sourceStep` the save point it was made from when that is known, and
    /// `sourceSteps` the save points that world has now (nil: it is not here). The new planet is made from that save point, never from another
    /// one without saying so.
    package static func reasonToMakeAgain(source: String?, sourceStep: Int? = nil, sourceSteps: [Int]?, in facts: Facts) -> String? {
        guard let source else { return "It does not say which world it was made from." }
        guard let sourceSteps, !sourceSteps.isEmpty else { return "\(source) is no longer here with a save point." }
        if let sourceStep, !sourceSteps.contains(sourceStep) { return "Step \(grouped(sourceStep)) of \(source) is no longer there." }
        return reasonToMakePlanet(facts)
    }

    /// What the list says when it has nothing to show, and what to do next.
    package static func emptyList(filter: LibraryFilter, query: String, counts: LibraryCounts) -> String {
        let q = query.trimmingCharacters(in: .whitespaces)
        if !q.isEmpty { return "No \(filter == .all ? "world" : filter.title.lowercased() + " world") matches \u{201C}\(q)\u{201D}. Check the spelling, or clear the search." }
        switch filter {
        case .all: return "There is no world in this folder yet."
        case .played: return "No played worlds yet. Choose a research world and press Make planet… to make one."
        case .research: return "No research worlds yet. A research world is grown in the Lab."
        case .unfinished: return counts.unfinished == 0 ? "No unfinished builds found." : ""
        }
    }
}
