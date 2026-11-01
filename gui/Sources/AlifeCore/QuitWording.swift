import Foundation

/// What the app says when it quits while an engine process is still doing something that matters (docs/APP.md §5.4): the
/// question before a run is stopped, and the panel that is shown while a world or a run writes its save point. Pure, so
/// that the words and the numbers in them are tested without a window.
package enum QuitWording {
    /// The seconds the panel waits before it offers "Quit now without saving".
    package static let quitNowAfter = 10
    /// A played world is saved every this many steps (`alife play`, DESIGN §33) and on quit.
    package static let saveEverySteps = 6000
    /// The steps per second at 1x when `play.json` cannot be read (`PlayWorldFile.baseRate` of every world made so far).
    package static let defaultBaseRate = 14.4

    // MARK: The question before a run is stopped

    package static func confirmTitle(run: String) -> String { "Stop '\(run)' and quit?" }

    package static let confirmDetail =
        "The run writes a save point first; continue it later from the Lab with Mode: Resume. If it cannot finish "
        + "that before the app has to quit, the steps since its last save point are lost."

    package static let confirmStop = "Stop and Quit"
    package static let confirmCancel = "Cancel"

    // MARK: The minutes a played world can lose

    /// The minutes of play at 1x that lie between two periodic saves: `saveEverySteps` at `baseRate` steps per second, rounded
    /// to the nearest minute and at least 1. A non-finite or non-positive rate counts as the default one.
    package static func lossMinutes(baseRate: Double) -> Int {
        let rate = baseRate.isFinite && baseRate > 0 ? baseRate : defaultBaseRate
        return max(1, Int((Double(saveEverySteps) / rate / 60).rounded()))
    }

    /// The `base_rate` of a world's `play.json`, `defaultBaseRate` when it cannot be read.
    package static func baseRate(playFile url: URL) -> Double {
        PlayWorldFile.read(at: url)?.baseRate ?? defaultBaseRate
    }
}

/// What the quit panel shows at one moment: the worlds that are saving and the run that is stopping, and how long it has taken.
package struct QuitPanelModel: Equatable, Sendable {
    /// Played worlds still writing their save point (their directory names).
    package var worlds: [String]
    /// The research run that was asked to stop and write its save point, if any.
    package var run: String?
    package var elapsedSeconds: Int
    /// The minutes of play between two saves of a played world (`QuitWording.lossMinutes`), nil with no world.
    package var lossMinutes: Int?

    package init(worlds: [String], run: String?, elapsedSeconds: Int, lossMinutes: Int?) {
        self.worlds = worlds
        self.run = run
        self.elapsedSeconds = max(0, elapsedSeconds)
        self.lossMinutes = lossMinutes
    }

    private var subjects: [String] { worlds.map { "'\($0)'" } + (run.map { ["run '\($0)'"] } ?? []) }

    package var title: String {
        let names = subjects
        switch names.count {
        case 0: return "Saving..."
        case 1: return "Saving \(names[0])..."
        default: return "Saving \(names.dropLast().joined(separator: ", ")) and \(names[names.count - 1])..."
        }
    }

    package var detail: String {
        "Quitting as soon as the save point is written. Elapsed \(elapsedSeconds) s."
    }

    /// "Quit now without saving" appears after `QuitWording.quitNowAfter` seconds.
    package var offersQuitNow: Bool { elapsedSeconds >= QuitWording.quitNowAfter }

    /// What quitting now would cost, in the words of the loss: shown with the button.
    package var warning: String? {
        guard offersQuitNow else { return nil }
        var parts: [String] = []
        if let minutes = lossMinutes, !worlds.isEmpty {
            let who = worlds.count == 1 ? "a played world" : "played worlds"
            parts.append("You may lose up to about \(minutes) minute\(minutes == 1 ? "" : "s") of play at 1x in \(who) "
                         + "(since the last save), including any powers cast since.")
        }
        if run != nil { parts.append("A run loses the steps since its last save point.") }
        return parts.isEmpty ? nil : parts.joined(separator: " ")
    }
}
