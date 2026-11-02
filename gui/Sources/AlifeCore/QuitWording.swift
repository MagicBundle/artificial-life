import Foundation

/// Why the app is being asked to quit: the `kAEQuitReason` of the quit Apple event (a Quit menu item sends none).
package enum QuitReason: Equatable, Sendable {
    /// The user quit (Cmd-Q, the Dock, closing the last window): the app may ask first.
    case user
    /// The session is ending (log out, restart, shut down): nobody is there to answer a question, and a question that is never
    /// answered would hold the whole machine. The app stops what it runs, waits within its limit and goes.
    case sessionEnd

    /// `kAEQuitAll` 'quia' and no reason are a quit by the user; the log-out, restart and shut-down codes end the session.
    package init(code: UInt32?) {
        switch code {
        case Self.fourCC("rlgo"), Self.fourCC("logo"), Self.fourCC("rrst"), Self.fourCC("rsdn"), Self.fourCC("rest"),
             Self.fourCC("shut"):
            self = .sessionEnd
        default:
            self = .user
        }
    }

    /// Whether the app may put a question to the user before it stops a run.
    package var asksQuestion: Bool { self == .user }

    package static func fourCC(_ text: String) -> UInt32 {
        text.utf8.reduce(0) { $0 << 8 | UInt32($1) }
    }
}

/// What the app says when it quits while an engine process is still doing something that matters (docs/APP.md §5.4): the
/// question before a run is stopped, and the panel that is shown while a world or a run writes its save point. Pure, so
/// that the words and the numbers in them are tested without a window.
package enum QuitWording {
    /// The seconds the panel waits before it offers "Quit now without saving".
    package static let quitNowAfter = 10
    /// How long the quit waits, at most: a research run (a state can take long to write; SIGTERM drops the save in flight and
    /// the run falls back to its last periodic point) and everything else (a played world's own ladder is 30 s + 10 s, a setup
    /// or a check is stopped by SIGINT and has nothing to save).
    package static let runWaitSeconds = 120
    package static let otherWaitSeconds = 40
    /// A played world is saved every this many steps (`alife play`, DESIGN §33) and on quit.
    package static let saveEverySteps = 6000
    /// The steps per second at 1x when `play.json` cannot be read (`PlayWorldFile.baseRate` of every world made so far).
    package static let defaultBaseRate = 14.4

    /// The limit of the wait for what is going: a run gets `runWaitSeconds`, anything else `otherWaitSeconds`.
    package static func waitSeconds(runGoing: Bool) -> Int { runGoing ? runWaitSeconds : otherWaitSeconds }

    // MARK: The question before a run is stopped

    package static func confirmTitle(run: String) -> String { "Stop '\(run)' and quit?" }

    package static let confirmDetail =
        "The run writes a save point first; continue it later from the Lab with Mode: Resume. If it has not finished that "
        + "after \(runWaitSeconds) s, it is stopped anyway and the steps since its last save point are lost."

    /// The same in the `alife` window, where a stopped world is continued with *Keep growing* (docs/APP.md §5.4).
    package static let confirmDetailShell =
        "The run writes a save point first; continue it later with Keep growing. If it has not finished that "
        + "after \(runWaitSeconds) s, it is stopped anyway and the steps since its last save point are lost."

    package static let confirmStop = "Stop and Quit"
    package static let confirmCancel = "Cancel"

    // MARK: The minutes a played world can lose

    /// The minutes of play at 1x that lie between two periodic saves: `saveEverySteps` at `baseRate` steps per second, rounded
    /// to the nearest minute and at least 1. A non-finite or non-positive rate counts as the default one. It counts steps: at
    /// 0.25x the same steps take four times as long on the clock.
    package static func lossMinutes(baseRate: Double) -> Int {
        let rate = baseRate.isFinite && baseRate > 0 ? baseRate : defaultBaseRate
        return max(1, Int((Double(saveEverySteps) / rate / 60).rounded()))
    }

    /// The `base_rate` of a world's `play.json`, `defaultBaseRate` when it cannot be read.
    package static func baseRate(playFile url: URL) -> Double {
        PlayWorldFile.read(at: url)?.baseRate ?? defaultBaseRate
    }

    /// The minutes a set of worlds can lose (the largest), reading each `play.json` once: `cache` keeps what was read, so that a
    /// panel that is updated ten times a second does not read files ten times a second. Nil with no world.
    package static func lossMinutes(worlds: [String], repo: URL?, cache: inout [String: Int]) -> Int? {
        var result: Int?
        for world in worlds {
            let minutes: Int
            if let known = cache[world] {
                minutes = known
            } else {
                let file = repo?.appendingPathComponent("runs/\(world)/play.json")
                minutes = lossMinutes(baseRate: file.map { baseRate(playFile: $0) } ?? defaultBaseRate)
                cache[world] = minutes
            }
            result = max(result ?? 0, minutes)
        }
        return result
    }
}

/// What the quit panel shows at one moment: what is being saved, how long it has taken, how long it can take at most and what
/// stopping it early costs.
package struct QuitPanelModel: Equatable, Sendable {
    /// Played worlds still writing their save point (their directory names; an empty name reads "a world").
    package var worlds: [String]
    /// A research run was asked to stop and write its save point; `runName` is its directory name when it is known.
    package var hasRun: Bool
    package var runName: String?
    /// An environment setup or a check that was asked to stop ("the environment setup", "the check"), if any.
    package var other: String?
    package var elapsedSeconds: Int
    /// The seconds after which the wait ends and what is left is stopped (`QuitWording.waitSeconds`).
    package var limitSeconds: Int
    /// The minutes of play between two saves of a played world (`QuitWording.lossMinutes`), nil with no world.
    package var lossMinutes: Int?

    package init(worlds: [String], hasRun: Bool = false, runName: String? = nil, other: String? = nil, elapsedSeconds: Int,
                 limitSeconds: Int = QuitWording.otherWaitSeconds, lossMinutes: Int?) {
        self.worlds = worlds
        self.hasRun = hasRun || runName != nil
        self.runName = runName
        self.other = other
        self.elapsedSeconds = max(0, elapsedSeconds)
        self.limitSeconds = max(1, limitSeconds)
        self.lossMinutes = lossMinutes
    }

    private var subjects: [String] {
        var names = worlds.map { $0.isEmpty ? "a world" : "'\($0)'" }
        if hasRun { names.append(runName.map { "run '\($0)'" } ?? "the run") }
        if let other { names.append(other) }
        return names
    }

    package var title: String {
        let names = subjects
        switch names.count {
        case 0: return "Saving..."
        case 1: return "Saving \(names[0])..."
        default: return "Saving \(names.dropLast().joined(separator: ", ")) and \(names[names.count - 1])..."
        }
    }

    package var detail: String {
        "Quitting as soon as the save point is written. Elapsed \(elapsedSeconds) s of at most \(limitSeconds) s."
    }

    /// "Quit now without saving" appears after `QuitWording.quitNowAfter` seconds.
    package var offersQuitNow: Bool { elapsedSeconds >= QuitWording.quitNowAfter }

    /// What happens at the limit, and so what quitting now costs: always shown, one sentence per kind of thing being saved.
    package var cost: String {
        var parts: [String] = []
        if !worlds.isEmpty {
            let minutes = lossMinutes ?? QuitWording.lossMinutes(baseRate: QuitWording.defaultBaseRate)
            parts.append("A played world is stopped anyway after \(QuitWording.otherWaitSeconds) s (or when you quit now) and may lose up to "
                         + "\(QuitWording.saveEverySteps) steps since its last save, about "
                         + "\(minutes) minute\(minutes == 1 ? "" : "s") of play at 1x, including any powers cast since.")
        }
        if hasRun {
            parts.append("A run is stopped anyway after \(limitSeconds) s (or when you quit now) and loses the steps since its "
                         + "last save point.")
        }
        if other != nil {
            parts.append("The \(other == "the check" ? "check" : "environment setup") is stopped anyway after "
                         + "\(limitSeconds) s; run it again if the engine is then reported missing.")
        }
        return parts.joined(separator: " ")
    }
}
