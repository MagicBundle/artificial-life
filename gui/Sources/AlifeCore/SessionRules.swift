import Foundation

// The safety rules of the `alife` window that decide what happens when the window closes (docs/APP.md R3, owner decision D3 = C) and what
// the window says about running a run and a world together (R1). Pure: the app asks, a test checks every case without a window.

/// What closing the window does while a run is going, as the person chose it. The question appears while the answer is `ask`; the
/// question's own checkbox ("Remember my choice") and Settings set the other two.
package enum ClosePreference: String, CaseIterable, Identifiable, Sendable {
    case ask
    case keep
    case stop

    package var id: String { rawValue }

    /// The stored value; anything else (nothing, an old or foreign value) is the question: the safe answer is to ask.
    package init(stored: String?) { self = ClosePreference(rawValue: stored ?? "") ?? .ask }

    package var title: String {
        switch self {
        case .ask: return "Ask me"
        case .keep: return "Keep it running"
        case .stop: return "Stop it and quit"
        }
    }

    /// What the choice does, one sentence (the effect line under the setting).
    package var effect: String {
        switch self {
        case .ask: return "A question appears each time you close the window while a run is going."
        case .keep:
            return "The window closes and the run carries on in the background. The Dock icon shows how far it has got and bounces when it ends."
        case .stop: return "The run writes a save point, then alife quits. Continue the run later with Keep growing."
        }
    }
}

/// What the app knows when the window is asked to close.
package struct CloseFacts: Equatable, Sendable {
    /// A research run is going (a run job, which includes the first stage of a planet).
    package var runGoing: Bool
    /// Its directory name, nil when not known yet.
    package var runName: String?
    /// The run was asked to stop and is writing its save point.
    package var runStopping: Bool
    /// The played world that is open (starting or live), nil when none is.
    package var world: String?

    package init(runGoing: Bool, runName: String? = nil, runStopping: Bool = false, world: String? = nil) {
        self.runGoing = runGoing
        self.runName = runName
        self.runStopping = runStopping
        self.world = world
    }
}

/// The question put to the person who closes the window with a run going. The default answer is the first, `keepLabel`: it loses
/// nothing and does what closing the window meant; Cancel (Escape) changes nothing.
package struct CloseQuestion: Equatable, Sendable {
    package var title: String
    package var detail: String
    package var keepLabel: String
    package var stopLabel: String
    package var cancelLabel: String
    package var rememberLabel: String
}

/// The answer to the question.
package enum CloseAnswer: Equatable, Sendable {
    case keepRunning
    case stopAndQuit
    case cancel
}

/// What to do when the window is asked to close.
package enum CloseDecision: Equatable, Sendable {
    /// Nothing the window must keep alive: the window closes and so does the app, through the quit's own gate and panel (a world is saved
    /// by it), as before.
    case closeAndQuit
    case ask(CloseQuestion)
    /// Remembered: the window closes and the run goes on.
    case keepRunning
    /// Remembered: the run is stopped at a save point and the app quits.
    case stopAndQuit
}

package enum SessionRules {
    // MARK: R3: closing the window (D3 = C)

    /// A window is closed while a run is going: ask (or do what was remembered). Not when the run is already being stopped (the quit then
    /// waits for its save point and there is nothing to ask), and not when only a world is open (it is saved and the app quits, no data is
    /// lost: as before).
    package static func decideClose(_ facts: CloseFacts, preference: ClosePreference) -> CloseDecision {
        guard facts.runGoing, !facts.runStopping else { return .closeAndQuit }
        switch preference {
        case .keep: return .keepRunning
        case .stop: return .stopAndQuit
        case .ask: return .ask(question(facts))
        }
    }

    /// The run in a sentence: the world a planet grows from is "the world for 'planet-1'", any other run "the run 'X'".
    package static func runSubject(_ name: String?) -> String {
        guard let name, !name.isEmpty else { return "The run" }
        if name.hasSuffix(SetupPlan.growSuffix), name.count > SetupPlan.growSuffix.count {
            return "The world for \u{2018}\(name.dropLast(SetupPlan.growSuffix.count))\u{2019}"
        }
        return "The run \u{2018}\(name)\u{2019}"
    }

    package static func question(_ facts: CloseFacts) -> CloseQuestion {
        var detail = "Keep Running closes the window and lets it carry on in the background. The Dock icon shows how far it has got and "
            + "bounces when it ends; a click on the icon brings the window back.\n\n"
            + "Stop and Quit asks it to write a save point first (up to \(QuitWording.runWaitSeconds) seconds), then quits. "
            + "Continue it later with Keep growing."
        if let world = facts.world, !world.isEmpty {
            detail += "\n\nThe open world \u{2018}\(world)\u{2019} is saved and closed when the window closes."
        }
        return CloseQuestion(
            title: "\(runSubject(facts.runName)) is still growing.", detail: detail, keepLabel: "Keep Running", stopLabel: "Stop and Quit",
            cancelLabel: "Cancel", rememberLabel: "Remember my choice (change it in Settings)")
    }

    /// What ticking "Remember my choice" stores for an answer. Cancel is not a choice, so nothing is stored for it.
    package static func remembered(_ answer: CloseAnswer, remember: Bool) -> ClosePreference? {
        guard remember else { return nil }
        switch answer {
        case .keepRunning: return .keep
        case .stopAndQuit: return .stop
        case .cancel: return nil
        }
    }

    // MARK: R1: one played world, one run

    /// The line Watch and Activity show while a run and a world are both going (they share the GPU).
    package static func gpuNote(runGoing: Bool, worldOpen: Bool) -> String? {
        runGoing && worldOpen ? "A run is using the GPU; the world may be slower." : nil
    }

    // MARK: When the window is closed and the run stays

    /// The note the Worlds page shows after the window came back and a world had been open when it closed.
    package static func closedWorldNote(_ world: String) -> String {
        "The window was closed, so \u{2018}\(world)\u{2019} was saved and closed. Press Play to continue it."
    }
}
