import Foundation

// The world library as the pages read it (docs/APP.md §2.2, §2.4, §2.5, §5.2; DESIGN §33.4.12): a scan (`LibrarySnapshot`) merged with
// what the app itself is doing (`LiveFacts`) into rows with a kind, a shape, a state in plain words and a place in the nesting. Pure:
// the pages draw the answer and the tests ask it questions.

/// What the user did and can do with a world (§2.2). A fact shown as a badge, never a mode.
package enum WorldKind: String, CaseIterable, Sendable {
    /// Has `play.json`: made by *Make planet* or by an old game; live play, powers, chronicle, reports.
    case played = "Played"
    /// Grown headlessly (`alife run`): numbers, settings, an observer picture.
    case research = "Research"

    package var title: String { rawValue }
}

/// The geometry of a world, separate from its kind (§2.2).
package enum WorldShape: String, Sendable {
    case flat = "Flat"
    case box = "3D box"
    case planet = "Planet"

    package var title: String { rawValue }

    /// From `world.dims` and `world.geometry` of the config; nil when the config could not be read (`geometry == nil`). A readable config
    /// without `dims` is a run from before the key, which is 2D.
    package static func of(dims: Int?, geometry: String?) -> WorldShape? {
        if geometry == "sphere" { return .planet }
        guard geometry != nil else { return nil }
        guard let dims else { return .flat }
        return dims == 2 ? .flat : .box
    }
}

/// The states of a world (§5.2), in the words the pages use. A world is in exactly one.
package enum WorldState: String, CaseIterable, Sendable {
    case saved = "Saved"
    case live = "Live"
    case paused = "Paused"
    case starting = "Starting"
    case growing = "Growing"
    case saving = "Saving"
    case inUseElsewhere = "In use elsewhere"
    case unfinished = "Unfinished"

    package var title: String { rawValue }

    /// One sentence, for a tooltip and for the screen reader.
    package var meaning: String {
        switch self {
        case .saved: return "Saved on disk. Nothing is using it."
        case .live: return "Open in Watch and running."
        case .paused: return "Open in Watch, with the clock stopped."
        case .starting: return "Being opened. A planet's weather warm-up takes a while."
        case .growing: return "A research run in this app is growing it now."
        case .saving: return "Closed, and writing its last save point."
        case .inUseElsewhere: return "Another app or process is using it. Close it there first."
        case .unfinished: return "Its build did not finish, so it cannot be played."
        }
    }

    /// Whether the state asks for the eye: the row is listed first and its dot is drawn.
    package var isActive: Bool {
        switch self {
        case .live, .paused, .starting, .growing, .saving, .inUseElsewhere: return true
        case .saved, .unfinished: return false
        }
    }
}

/// What this app is doing with its worlds right now: the part of a state that no file shows.
package struct LiveFacts: Equatable, Sendable {
    package enum OpenPhase: Equatable, Sendable { case starting, live, paused }

    /// The world that is open in Watch (starting, live or paused), nil when none is.
    package var openWorld: String?
    package var openPhase: OpenPhase = .live
    /// The research run this app is growing, nil when none is.
    package var growing: String?
    /// Worlds that were asked to stop and are still saving.
    package var saving: Set<String> = []
    /// Planets a pipeline is making (docs/APP.md §3.2): the played world does not exist yet while its seed grows, but one click is one
    /// world, so the library lists it, being made, with the seed nested under it.
    package var pending: [PendingPlanet] = []

    package init(openWorld: String? = nil, openPhase: OpenPhase = .live, growing: String? = nil, saving: Set<String> = [],
                 pending: [PendingPlanet] = []) {
        self.openWorld = openWorld
        self.openPhase = openPhase
        self.growing = growing
        self.saving = saving
        self.pending = pending
    }
}

/// A planet that a pipeline has promised and not yet made.
package struct PendingPlanet: Equatable, Sendable {
    package enum Stage: Equatable, Sendable { case growing, making }
    package var name: String
    /// The research world it is made from (`NAME-grow`).
    package var seed: String
    package var stage: Stage
    /// When the pipeline started (a stable date, so that the library does not look changed at every look).
    package var since: Date

    package init(name: String, seed: String, stage: Stage, since: Date = Date()) {
        self.name = name
        self.seed = seed
        self.stage = stage
        self.since = since
    }
}

/// One world of the list.
package struct LibraryRow: Equatable, Identifiable, Sendable {
    package var name: String
    package var kind: WorldKind
    package var shape: WorldShape?
    package var state: WorldState
    /// A fact about how it ended, in plain words, when there is one worth saying ("Every organism died out").
    package var note: String?
    /// The step of its newest save point, nil when it has none.
    package var step: Int?
    package var lastSaved: Date
    /// The run a Played world was made from.
    package var madeFrom: SourceRef?
    /// The step a research run was asked to grow to (`run.steps`), nil when unknown.
    package var targetSteps: Int?
    /// A planet a pipeline is making: listed before its directory is a world, with its seed nested under it. It has no folder to open.
    package var isPlaceholder = false
    /// For a seed (`NAME-grow`) whose planet `NAME` was never made: the name it was meant for. The next action is *Make planet*.
    package var unmadePlanet: String?

    package var id: String { name }
    package var hasSavePoint: Bool { step != nil }

    /// "Played · Planet · step 450,000".
    package var facts: String {
        var parts = [kind.title]
        if let shape { parts.append(shape.title) }
        if let step { parts.append("step " + WorldsGuide.grouped(step)) } else if !isPlaceholder { parts.append("no save point yet") }
        return parts.joined(separator: " · ")
    }

    /// "saved 2 hours ago".
    package func savedText(now: Date) -> String {
        let when = WorldsGuide.relative(lastSaved, now: now)
        return when == "just now" ? "saved just now" : "saved " + when
    }

    /// The sidebar's second line: the state, and when it was saved if it is only saved.
    package func sidebarLine(now: Date) -> String {
        guard state == .saved else { return state.title }
        let when = WorldsGuide.relative(lastSaved, now: now)
        let saved = when == "just now" ? "Saved just now" : "Saved " + when
        return unmadePlanet == nil ? saved : "Planet not made yet"
    }
}

/// A directory that `alife play` began to build and did not finish.
package struct UnfinishedRow: Equatable, Identifiable, Sendable {
    package var name: String
    /// Why it stopped and where, in plain words.
    package var reason: String
    package var lastTouched: Date
    /// The run it was being made from, when `meta.json` says.
    package var source: SourceRef?

    package init(name: String, reason: String, lastTouched: Date, source: SourceRef?) {
        self.name = name
        self.reason = reason
        self.lastTouched = lastTouched
        self.source = source
    }

    package var id: String { name }
}

package enum LibraryFilter: String, CaseIterable, Identifiable, Sendable {
    case all = "All"
    case played = "Played"
    case research = "Research"
    case unfinished = "Unfinished"

    package var id: String { rawValue }
    package var title: String { rawValue }
}

package struct LibraryCounts: Equatable, Sendable {
    package var all = 0
    package var played = 0
    package var research = 0
    package var unfinished = 0

    package func count(_ filter: LibraryFilter) -> Int {
        switch filter {
        case .all: return all
        case .played: return played
        case .research: return research
        case .unfinished: return unfinished
        }
    }
}

/// A row with the rows that nest under it in the sidebar (§2.4: a played world's seed run).
package struct LibraryNode: Equatable, Identifiable, Sendable {
    package var row: LibraryRow
    package var children: [LibraryRow]
    package var id: String { row.name }
}

/// The library as the pages read it.
package struct LibraryState: Equatable, Sendable {
    /// Every world, the one to look at first first (`order`).
    package var rows: [LibraryRow] = []
    package var unfinished: [UnfinishedRow] = []
    /// The launcher's view of the same folder (the planet form's sources, the names that are taken, the Play buttons' reasons).
    package var catalog = PlayCatalog()
    /// Directories that are not worlds (no save point, not a build): a run whose save points are archived, a folder of logs.
    package var others: [String] = []
    /// Entries of `runs/` that are links to a drive that is not connected.
    package var offline: [String] = []
    package var offlineVolumes: [String] = []
    /// Entries that could not be listed for another reason (a broken or looping link, no permission, an error), by name.
    package var problems: [String: EntryProblem] = [:]
    /// Entries whose check did not finish in time, by name.
    package var slow: [String] = []
    /// The listing of `runs/` did not finish in time: this is the last known library.
    package var listingSlow = false

    package init() {}

    package var counts: LibraryCounts {
        LibraryCounts(all: rows.count, played: rows.filter { $0.kind == .played }.count,
                      research: rows.filter { $0.kind == .research }.count, unfinished: unfinished.count)
    }

    package func row(named name: String) -> LibraryRow? { rows.first { $0.name == name } }

    // MARK: Resolving

    /// Merges a scan and what the app is doing. `notes` are what this session knows about builds that stopped (by directory name).
    /// `notes` are what this session knows of builds that stopped, by name, with `noteTimes` when each was made: a note is dropped when the
    /// directory was touched after it (a later, different stop).
    package static func resolve(
        _ snapshot: LibrarySnapshot, live: LiveFacts = LiveFacts(), notes: [String: String] = [:], noteTimes: [String: Date] = [:]
    ) -> LibraryState {
        var state = LibraryState()
        state.offline = snapshot.offline
        state.offlineVolumes = snapshot.offlineVolumes
        state.problems = snapshot.problems.filter { if case .driveNotConnected = $0.value { false } else { true } }
        state.slow = snapshot.slow.sorted()
        state.listingSlow = snapshot.listingSlow
        var rows: [LibraryRow] = []
        for r in snapshot.records {
            let held = snapshot.held.contains(r.name)
            let written = snapshot.writtenElsewhere.contains(r.name)
            let cutOff = snapshot.cutOff.contains(r.name)
            var listed = false
            if r.isPlayed {
                rows.append(row(r, kind: .played, state: stateOf(r, held: held, writtenElsewhere: written, live: live), cutOff: cutOff))
                // (`locked` is "in use by another process": the lock is held, or `alife resume` is writing the directory. `modified` is the
                // last-saved date, the list's.)
                state.catalog.worlds.append(PlayWorld(
                    name: r.name, file: r.play!, newestStep: r.steps.last, status: r.status, modified: r.lastSaved, locked: held || written,
                    geometry: r.geometry, planetRadius: r.planetRadius))
                listed = true
            } else if r.looksUnfinished, !held {
                var note = notes[r.name]
                if let when = noteTimes[r.name], r.lastSaved.timeIntervalSince(when) > 5 { note = nil }
                state.unfinished.append(unfinishedRow(r, note: note))
                listed = true
            } else if !r.steps.isEmpty || live.growing == r.name, !(r.looksUnfinished && held) {
                // A research world has a save point; the run this app is growing is listed from its first moment.
                rows.append(row(r, kind: .research, state: stateOf(r, held: held, writtenElsewhere: written, live: live), cutOff: cutOff))
            }
            if !r.steps.isEmpty {
                state.catalog.sources.append(PlaySource(
                    name: r.name, checkpoints: r.steps, dims: r.dims, worldSize: r.worldSize, geometry: r.geometry,
                    planetRadius: r.planetRadius, planetShape: r.planetShape, weatherOn: r.weatherOn, hasEvents: r.hasEvents,
                    modified: r.lastSaved))
                listed = true
            }
            if !listed, r.hasContent { state.catalog.other.append(r.name) }
        }
        state.catalog.worlds.sort { ($0.modified, $1.name) > ($1.modified, $0.name) }
        state.catalog.sources.sort { ($0.modified, $1.name) > ($1.modified, $0.name) }
        state.catalog.unfinished = state.unfinished.map(\.name).sorted()
        state.catalog.other.sort()
        state.others = state.catalog.other
        // Names that are taken without being worlds: links to a drive that is not connected, entries that cannot be read, entries that are slow.
        state.catalog.other = Array(Set(state.catalog.other).union(snapshot.problems.keys).union(snapshot.slow)).sorted()
        // A planet a pipeline is making is listed before it is a world: one click is one world, with the seed nested under it.
        let existing = Set(rows.map { NameFolding.fold($0.name) })
        for p in live.pending where !existing.contains(NameFolding.fold(p.name)) {
            rows.append(LibraryRow(
                name: p.name, kind: .played, shape: .planet, state: p.stage == .growing ? .growing : .starting,
                note: p.stage == .growing ? "Growing first, then it becomes a planet." : "Being made.", step: nil, lastSaved: p.since,
                madeFrom: SourceRef(name: p.seed, step: nil), isPlaceholder: true))
        }
        let promised = Set(live.pending.map { NameFolding.fold($0.name) })
        state.unfinished.removeAll { promised.contains(NameFolding.fold($0.name)) }
        // A seed whose planet was never made (the pipeline was stopped, or the app was closed): the next action is Make planet.
        let known = Set(snapshot.records.map { NameFolding.fold($0.name) }).union(snapshot.problems.keys.map(NameFolding.fold)).union(promised)
        for i in rows.indices where rows[i].kind == .research && rows[i].shape == .planet && rows[i].hasSavePoint && !rows[i].isPlaceholder {
            guard rows[i].name.hasSuffix(SetupPlan.growSuffix) else { continue }
            let base = String(rows[i].name.dropLast(SetupPlan.growSuffix.count))
            if !base.isEmpty, !known.contains(NameFolding.fold(base)) { rows[i].unmadePlanet = base }
        }
        state.rows = order(rows)
        state.unfinished.sort { ($0.lastTouched, $1.name) > ($1.lastTouched, $0.name) }
        return state
    }

    private static func row(_ r: LibraryRecord, kind: WorldKind, state: WorldState, cutOff: Bool) -> LibraryRow {
        LibraryRow(
            name: r.name, kind: kind, shape: WorldShape.of(dims: r.dims, geometry: r.geometry), state: state,
            note: kind == .research && state == .saved ? endNote(r.status, cutOff: cutOff) ?? stoppedNote(r) : nil, step: r.steps.last,
            lastSaved: r.lastSaved, madeFrom: SourceRef.parse(r.play?.planet?.source), targetSteps: r.targetSteps)
    }

    /// The state of a world: what this app does with it first, then what another process does.
    static func stateOf(_ r: LibraryRecord, held: Bool, writtenElsewhere: Bool, live: LiveFacts) -> WorldState {
        if live.openWorld == r.name {
            switch live.openPhase {
            case .starting: return .starting
            case .live: return .live
            case .paused: return .paused
            }
        }
        if live.saving.contains(r.name) { return .saving }
        if live.growing == r.name { return .growing }
        if held || writtenElsewhere { return .inUseElsewhere }
        return .saved
    }

    /// A run that was stopped before the step it was asked to reach says where it stopped.
    static func stoppedNote(_ r: LibraryRecord) -> String? {
        guard r.status == "interrupted", let target = r.targetSteps, let step = r.steps.last, step < target else { return nil }
        return "Stopped at step \(WorldsGuide.grouped(step)) of \(WorldsGuide.grouped(target))."
    }

    /// How a research run ended, when that is worth a line; nil for a normal end.
    /// A world whose status says `running` is cut off only when it has shown no sign of life for a long while (`cutOff`, from the probe): a
    /// slow run, a long save or a resume that is still loading is not.
    static func endNote(_ status: String?, cutOff: Bool = false) -> String? {
        switch status {
        case "extinct": return "Every organism died out."
        case "violation": return "Stopped by an error check."
        case "running": return cutOff ? "Cut off without a clean stop." : nil
        default: return nil
        }
    }

    // MARK: Unfinished builds

    private static func unfinishedRow(_ r: LibraryRecord, note: String?) -> UnfinishedRow {
        UnfinishedRow(name: r.name, reason: unfinishedReason(r, sessionNote: note), lastTouched: r.lastSaved, source: r.parent)
    }

    /// Why a build stopped and where, from what it left (and from this session's log of it, when this session made it). `alife play`
    /// takes `play.lock`, creates `config.json` and `meta.json` (with the `play_build` mark), copies the source, spins the weather up,
    /// saves the first save point and writes `play.json` last (`PlayCatalogScanner.isUnfinishedBuild`).
    package static func unfinishedReason(_ r: LibraryRecord, sessionNote: String? = nil) -> String {
        let stage: String
        if r.buildMark == nil {
            stage = "right at the start, before anything was written"
        } else if r.steps.isEmpty {
            stage = "before its first save point, while the world was being copied or its weather was warming up"
        } else {
            stage = "after its first save point, before the world was marked ready to play"
        }
        let how: String
        switch r.status {
        case "interrupted": how = "Cancelled"
        case "violation": how = "Stopped by an error check"
        case "extinct": how = "Stopped when every organism died out"
        default: how = "Cut off"
        }
        return "\(sessionNote ?? how) \(stage)."
    }

    // MARK: Order, filter, search, nesting

    /// The one to look at first, first: the open world, then what grows, then what is in use or saving, then the rest by when it was last
    /// saved, newest first, with the name to break a tie.
    static func rank(_ row: LibraryRow) -> Int {
        switch row.state {
        case .live, .paused, .starting: return 0
        case .growing: return 1
        case .saving, .inUseElsewhere: return 2
        case .saved, .unfinished: return 3
        }
    }

    /// **The order rule.** A row's rank is its state's class (open, growing, in use or saving, the rest); rows of one rank go newest saved first,
    /// the name breaking a tie. In the sidebar a node (a played world and the research world nested under it) has the best rank and the newest
    /// date among its rows, so a growing seed is never hidden under a quiet parent. Nothing else moves a row: not the clock, not the name, not a
    /// filter or a search; a row jumps only when its own state's class changes or a world is saved, and a world that is saved moves up once, to
    /// the top of its rank.
    package static func order(_ rows: [LibraryRow]) -> [LibraryRow] {
        rows.sorted { a, b in
            if rank(a) != rank(b) { return rank(a) < rank(b) }
            if a.lastSaved != b.lastSaved { return a.lastSaved > b.lastSaved }
            return a.name < b.name
        }
    }

    /// The rows the page shows for a filter and a search. `.unfinished` shows no world: its builds are `unfinishedRows`.
    package func visible(filter: LibraryFilter, query: String) -> [LibraryRow] {
        let base: [LibraryRow]
        switch filter {
        case .all: base = rows
        case .played: base = rows.filter { $0.kind == .played }
        case .research: base = rows.filter { $0.kind == .research }
        case .unfinished: return []
        }
        return Self.search(base, query: query)
    }

    package func unfinishedRows(query: String) -> [UnfinishedRow] {
        let tokens = Self.tokens(query)
        guard !tokens.isEmpty else { return unfinished }
        return unfinished.filter { u in
            let hay = (u.name + " " + (u.source?.name ?? "") + " unfinished " + u.reason).lowercased()
            return tokens.allSatisfy { hay.contains($0) }
        }
    }

    static func tokens(_ query: String) -> [String] {
        query.lowercased().split(whereSeparator: { $0.isWhitespace }).map(String.init)
    }

    /// Every word of `query` appears in the name, kind, shape, state or source of the row (case does not matter).
    package static func search(_ rows: [LibraryRow], query: String) -> [LibraryRow] {
        let tokens = tokens(query)
        guard !tokens.isEmpty else { return rows }
        return rows.filter { row in
            let hay = [row.name, row.kind.title, row.shape?.title ?? "", row.state.title, row.madeFrom?.name ?? ""].joined(separator: " ").lowercased()
            return tokens.allSatisfy { hay.contains($0) }
        }
    }

    /// The sidebar's tree: a research world that a played world was made from nests under that played world. A research world has one
    /// parent: the played world named `<research>`'s name without `-grow` (the pipeline's convention, §3.2) when there is one, else the
    /// played world that was saved last. Played worlds never nest, and the order of `rows` is kept.
    package static func tree(_ rows: [LibraryRow]) -> [LibraryNode] {
        var parentOf: [String: String] = [:]
        let played = rows.filter { $0.kind == .played }
        for child in rows where child.kind == .research {
            let makers = played.filter { $0.madeFrom?.name == child.name }
            guard !makers.isEmpty else { continue }
            let named = makers.first { $0.name + "-grow" == child.name }
            let newest = makers.max { ($0.lastSaved, $1.name) < ($1.lastSaved, $0.name) }
            if let parent = named ?? newest { parentOf[child.name] = parent.name }
        }
        let byParent = Dictionary(grouping: rows.filter { parentOf[$0.name] != nil }, by: { parentOf[$0.name]! })
        let nodes = rows.filter { parentOf[$0.name] == nil }.map { LibraryNode(row: $0, children: byParent[$0.name] ?? []) }
        // A node is ranked by the best of its rows (see `order`).
        func key(_ node: LibraryNode) -> (Int, Date) {
            let all = [node.row] + node.children
            return (all.map(rank).min() ?? 3, all.map(\.lastSaved).max() ?? .distantPast)
        }
        return nodes.sorted { a, b in
            let (ka, kb) = (key(a), key(b))
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 > kb.1 }
            return a.row.name < b.row.name
        }
    }
}
