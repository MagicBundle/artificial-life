import Foundation

// R6 of docs/APP.md §5.2: one writer per directory. `DirectoryGuard` records every directory this app has a *writing* process on (Play, Keep
// growing, the target of Create, Branch and Make planet, a world still saving) and every directory another process holds, and answers one
// question for an action: would it add a second writer? Pure: the app feeds it facts at click time and again at start, and a refusal is a
// sentence for the grey button.

/// Directory names compare without regard to capitals: a case-insensitive volume has one folder for `Planet` and `planet`.
package enum NameFolding {
    package static func fold(_ name: String) -> String { name.lowercased() }

    package static func same(_ a: String, _ b: String) -> Bool { fold(a) == fold(b) }

    /// The first of `taken` that is `name` apart from capitals.
    package static func clash(of name: String, in taken: some Sequence<String>) -> String? {
        let key = fold(name)
        return taken.first { fold($0) == key }
    }
}

package struct DirectoryGuard: Equatable, Sendable {
    /// What this app is doing in a directory.
    package enum Writer: Equatable, Sendable {
        /// A played world is open (`alife play RUN`).
        case playing
        /// A research run writes it (`alife run`, `alife resume`).
        case growing
        /// A planet is being made there (`alife play --init-from`), or is reserved for the second stage of a pipeline.
        case makingPlanet
        /// Its process was asked to stop and has not exited: it writes its last save point.
        case saving
    }

    /// The directories this app writes, by the name as it was given. Looked up without regard to capitals.
    package private(set) var writers: [String: Writer] = [:]
    /// Directories another process holds (`play.lock`, or `alife resume` writing a world): a second writer is refused.
    package var heldElsewhere: Set<String> = []
    /// Directories another process seems to write (a run with fresh output): a warning, not a refusal (no lock shows it, §5.2 R6).
    package var seemsWrittenElsewhere: Set<String> = []

    package init() {}

    package init(writers: [String: Writer], heldElsewhere: Set<String> = [], seemsWrittenElsewhere: Set<String> = []) {
        self.writers = writers
        self.heldElsewhere = heldElsewhere
        self.seemsWrittenElsewhere = seemsWrittenElsewhere
    }

    // MARK: Recording

    package func writer(of name: String) -> (name: String, writer: Writer)? {
        guard let key = NameFolding.clash(of: name, in: writers.keys), let writer = writers[key] else { return nil }
        return (key, writer)
    }

    /// Records a writer. False, with nothing recorded, when the directory already has one (this app's or another's).
    @discardableResult
    package mutating func claim(_ name: String, as writer: Writer) -> Bool {
        guard writeRefusal(name) == nil else { return false }
        writers[name] = writer
        return true
    }

    /// Records that the directory is written, whatever was recorded before (a world that was open and is now saving).
    package mutating func record(_ name: String, as writer: Writer) {
        if let key = NameFolding.clash(of: name, in: writers.keys) { writers[key] = nil }
        writers[name] = writer
    }

    package mutating func release(_ name: String) {
        if let key = NameFolding.clash(of: name, in: writers.keys) { writers[key] = nil }
    }

    // MARK: Asking

    /// Why a new writer cannot be added to `name`; nil when it can. The sentence names what is there, so that the button's reason says what
    /// to do.
    package func writeRefusal(_ name: String) -> String? {
        // (the name of the folder that is there, which is the one the person knows, not the capitals that were asked about)
        if let (there, writer) = writer(of: name) {
            switch writer {
            case .playing: return "\(there) is open. Close it first."
            case .growing: return "\(there) is growing. Wait for it to finish, or stop it."
            case .makingPlanet: return "\(there) is being made."
            case .saving: return "\(there) is still saving."
            }
        }
        if let there = NameFolding.clash(of: name, in: heldElsewhere) { return "\(there) is in use by another app or process." }
        return nil
    }

    /// A warning for an action that reads `name` while another process may be writing it; nil when there is nothing to say.
    package func readWarning(_ name: String) -> String? {
        if NameFolding.clash(of: name, in: heldElsewhere) != nil || NameFolding.clash(of: name, in: seemsWrittenElsewhere) != nil {
            return "\(name) seems to be in use by another process. What you get starts from the last save point it wrote."
        }
        return nil
    }
}

// MARK: - The state x action matrix

/// The actions of the set-up page that read or write a world, and the matrix of docs/APP.md §5.2 for them: which state allows which, and the
/// sentence when it does not.
package enum SetupAction: Equatable, Sendable {
    case keepGrowing
    case branch
    case makePlanet
}

package enum SetupRules {
    package struct Facts: Equatable, Sendable {
        /// A research run (not a setup, not a check) is going in this app.
        package var runJob: Bool
        /// The world open in Watch (starting, live or paused), nil when none is.
        package var openWorld: String?
        package var guardState: DirectoryGuard
        package var folderExists: Bool
        package var engineReady: Bool
        package var folderProblem: String?

        package init(runJob: Bool = false, openWorld: String? = nil, guardState: DirectoryGuard = DirectoryGuard(), folderExists: Bool = true,
                     engineReady: Bool = true, folderProblem: String? = nil) {
            self.runJob = runJob
            self.openWorld = openWorld
            self.guardState = guardState
            self.folderExists = folderExists
            self.engineReady = engineReady
            self.folderProblem = folderProblem
        }
    }

    /// Why `action` cannot be done on this world now; nil when it can. (Branch and Make planet read the newest save point; Keep growing
    /// writes the world.)
    package static func reason(_ action: SetupAction, on row: LibraryRow, in facts: Facts) -> String? {
        if !facts.folderExists { return facts.folderProblem ?? "The alife folder was not found." }
        if !facts.engineReady { return "Set up the engine first." }
        switch action {
        case .keepGrowing:
            if row.kind == .played { return "A played world is continued by playing it." }
            if row.state == .growing { return "Already growing." }
            if let why = facts.guardState.writeRefusal(row.name) { return why }
            if facts.runJob { return "A run is going. Wait for it to finish, or stop it." }
            if row.state == .inUseElsewhere { return "In use by another app or process." }
            if row.state == .saving { return "Saving…" }
            if !row.hasSavePoint { return "No save point to continue from." }
            return nil
        case .branch:
            if row.state == .live || row.state == .paused || row.state == .starting { return "Close it first: it is being written." }
            if row.state == .saving || facts.guardState.writer(of: row.name)?.writer == .saving { return "Saving…" }
            if facts.runJob { return "A run is going. Wait for it to finish, or stop it." }
            if !row.hasSavePoint { return "No save point to start from yet." }
            return nil
        case .makePlanet:
            if let open = facts.openWorld { return open == row.name ? "Close it first" : "Close \(open) first" }
            if row.state == .saving || facts.guardState.writer(of: row.name)?.writer == .saving { return "Saving…" }
            if !row.hasSavePoint { return "No save point to start from yet." }
            if row.shape == .flat { return WorldsGuide.flatHasNoPlanet }
            return nil
        }
    }

    /// A note to show beside an action that is allowed (never a reason to refuse it).
    package static func note(_ action: SetupAction, on row: LibraryRow, in facts: Facts) -> String? {
        if row.state == .growing, action != .keepGrowing {
            return "\(row.name) is still growing. This starts from its newest save point."
        }
        return facts.guardState.readWarning(row.name)
    }
}
