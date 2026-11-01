import Foundation

// The pipeline of docs/APP.md §3.2: one click that takes two processes. "Grow, then play" is a research run (stage 1, `alife run`) and then a
// played planet made from its newest save point (stage 2, `alife play --init-from`). `PipelinePlan` is the pure state machine: events in
// (the run's progress and exit, the planet's progress, the person's buttons), effects out (start stage 2, bounce the Dock), and the banner
// the app draws in every state. The glue that runs the processes is `PipelineCoordinator`; no command line is built here.
//
// Rules the table of §3.2 and §5.2 give it, each tested:
// - Stop in stage 1 never continues by itself: the pipeline parks and asks.
// - Stage 2 starts by itself only when the person can see it happen (the window is shown and the app is active), no other world is open,
//   the name is still free, and the engine is there; otherwise it parks and says what to do. Nothing starts invisibly.
// - A failure ends the pipeline and keeps what was written; a parked pipeline stays parked until a button is pressed.

package struct PipelinePlan: Equatable, Sendable {
    package enum Mode: String, Sendable {
        /// Create with the goal *play*: stage 1 grows `NAME-grow`, stage 2 makes the planet `NAME`.
        case growThenPlay
        /// Create with the goal *research*, or Branch: one run, one world.
        case growOnly
        /// Keep growing a research world (`alife resume`).
        case keepGrowing
        /// Make planet from a world: stage 2 alone.
        case makePlanet
    }

    /// Why stage 2 did not start by itself.
    package enum Why: Equatable, Sendable {
        /// The window is not in front (closed, hidden, another app is active, the screen is off).
        case notVisible
        /// Another world is open: a planet replaces the open world, so it is not made behind it.
        case worldOpen(String)
        /// A directory with the planet's name appeared while the world grew.
        case nameTaken
        /// The engine is gone.
        case engineMissing
        /// The person asked to stop; stage 2 never follows a Stop by itself.
        case stopped
    }

    package enum Failure: Equatable, Sendable {
        case growing(String)
        case making(String)
        /// Stop was pressed so early that the engine had saved nothing: it ended on the signal, before it could answer.
        case nothingSaved
    }

    package enum State: Equatable, Sendable {
        /// Stage 1 runs. `stopping`: Stop was pressed and the run is writing its save point.
        case growing(stopping: Bool)
        /// Waiting for a button. `step` is the newest save point the pipeline knows.
        case parked(Why, step: Int)
        /// Stage 2 runs (the planet is being made). `cancelling`: Cancel was pressed.
        case making(cancelling: Bool)
        /// The planet is live. The pipeline is over.
        case live
        /// A single-stage pipeline finished; `stopped` when the person ended it early.
        case finished(stopped: Bool)
        case failed(Failure)
        /// Stage 2 was cancelled, or ended before the planet was ready.
        case cancelled
        /// The person pressed Done.
        case dismissed
    }

    package var mode: Mode
    /// The world the person asked for.
    package var name: String
    /// The directory stage 1 writes: `NAME-grow` for "grow, then play", else `NAME` (Keep growing: the world itself).
    package var growName: String?
    /// The world stage 2 starts from, when stage 1 did not make it (Make planet).
    package var source: String?
    /// The step stage 1 ends at, nil when unknown (Keep growing without a target).
    package var targetSteps: Int?
    package var startStep: Int
    package var state: State
    /// The newest step seen, and the speed in steps per second.
    package var step: Int
    package var stepsPerSecond: Double?
    /// What the planet's process has said so far.
    package var making = PlayProgress()
    /// Seconds the planet's stage is expected to take (the banner's "about N s left").
    package var expectedPlanetSeconds: Double?
    /// Whether a planet can be made from what stage 1 writes (a flat world has no weather).
    package var offersPlanet = true

    package init(mode: Mode, name: String, growName: String?, source: String? = nil, targetSteps: Int?, startStep: Int = 0,
                 expectedPlanetSeconds: Double? = nil) {
        self.mode = mode
        self.name = name
        self.growName = growName
        self.source = source
        self.targetSteps = targetSteps
        self.startStep = startStep
        self.expectedPlanetSeconds = expectedPlanetSeconds
        step = startStep
        state = mode == .makePlanet ? .making(cancelling: false) : .growing(stopping: false)
    }

    /// The directory stage 2 reads (the world that was grown, or the source of a Make planet).
    package var planetSource: String? { mode == .makePlanet ? source : growName }

    /// Every directory this pipeline writes or has reserved (for the guard and the taken names).
    package var directories: [String] {
        switch mode {
        case .growThenPlay: return [growName, name].compactMap { $0 }
        case .growOnly, .keepGrowing: return [growName ?? name]
        case .makePlanet: return [name]
        }
    }

    // MARK: Events

    package struct GrowExit: Equatable, Sendable {
        /// The process's exit status, and whether a signal ended it.
        package var status: Int32
        package var signalled: Bool
        /// Stop was pressed (or the timer ran out, or the app is quitting): the run was asked to end at a save point.
        package var stopRequested: Bool
        /// The first line the run said that tells what went wrong, nil when it said none.
        package var problem: String?
        /// The run's directory holds a save point.
        package var savePoint: Bool
        package init(status: Int32, signalled: Bool = false, stopRequested: Bool = false, problem: String? = nil, savePoint: Bool = true) {
            self.status = status
            self.signalled = signalled
            self.stopRequested = stopRequested
            self.problem = problem
            self.savePoint = savePoint
        }
    }

    package enum Event: Equatable, Sendable {
        case progress(step: Int, stepsPerSecond: Double?)
        case growExited(GrowExit)
        case stopPressed
        case makeProgress([String])
        case makeLive
        case makeFailed(String)
        /// The planet's process ended without the world being live (cancelled, or the window closed).
        case makeEnded
        case cancelPressed
        /// "Make planet" / "Make planet from here" on a parked pipeline.
        case makePlanetPressed
        /// "Keep growing" on a parked pipeline: stage 1 starts again from its save point.
        case keepGrowingPressed
        case donePressed
    }

    /// What the glue is to do.
    package enum Effect: Equatable, Sendable {
        /// Start stage 2 now.
        case startMaking
        /// Start `alife resume` on the grown world (the person chose Keep growing).
        case resumeGrowing
        /// The app is not in front: bounce the Dock icon once.
        case bounce
        /// Stop the run (SIGINT).
        case stopGrowing
        /// Stop the planet's process now.
        case stopMaking
    }

    /// What the pipeline needs to know about the world around it when it decides whether stage 2 may start.
    package struct Surroundings: Equatable, Sendable {
        package var visible: Bool
        package var openWorld: String?
        package var nameTaken: Bool
        package var engineReady: Bool
        package init(visible: Bool = true, openWorld: String? = nil, nameTaken: Bool = false, engineReady: Bool = true) {
            self.visible = visible
            self.openWorld = openWorld
            self.nameTaken = nameTaken
            self.engineReady = engineReady
        }

        /// Why stage 2 cannot start now; nil when it can. `byHand`: the person pressed the button (the window being in front is then a given).
        package func obstacle(byHand: Bool) -> Why? {
            if !engineReady { return .engineMissing }
            if let open = openWorld { return .worldOpen(open) }
            if nameTaken { return .nameTaken }
            if !byHand, !visible { return .notVisible }
            return nil
        }
    }

    // MARK: Reducing

    package var isOver: Bool {
        switch state {
        case .live, .finished, .failed, .cancelled, .dismissed: return true
        case .growing, .parked, .making: return false
        }
    }

    /// Whether the pipeline holds a process or a reservation (the guard keeps its directories while this is true).
    package var isActive: Bool { !isOver }

    package mutating func apply(_ event: Event, in around: Surroundings) -> [Effect] {
        switch (state, event) {
        case (_, .progress(let s, let sps)):
            if case .growing = state { step = max(step, s) }
            if let sps, sps > 0 { stepsPerSecond = sps }
            return []

        // Stage 1
        case (.growing(false), .stopPressed):
            state = .growing(stopping: true)
            return [.stopGrowing]
        case (.growing, .growExited(let exit)):
            return grownExit(exit, around)

        // Parked
        case (.parked(_, _), .makePlanetPressed):
            if let why = around.obstacle(byHand: true) {
                state = .parked(why, step: step)
                return []
            }
            state = .making(cancelling: false)
            making = PlayProgress()
            return [.startMaking]
        case (.parked, .keepGrowingPressed):
            guard around.engineReady else {
                state = .parked(.engineMissing, step: step)
                return []
            }
            state = .growing(stopping: false)
            return [.resumeGrowing]
        case (.parked, .donePressed):
            state = .dismissed
            return []

        // Stage 2
        case (.making, .makeProgress(let lines)):
            making = PlayProgress.parse(lines)
            return []
        case (.making, .makeLive):
            state = .live
            return []
        case (.making, .makeFailed(let message)):
            state = .failed(.making(message))
            return []
        case (.making(false), .cancelPressed):
            state = .making(cancelling: true)
            return [.stopMaking]
        case (.making, .makeEnded):
            state = .cancelled
            return []

        // The end
        case (.finished, .donePressed), (.failed, .donePressed), (.cancelled, .donePressed), (.live, .donePressed):
            state = .dismissed
            return []
        default:
            return []
        }
    }

    private mutating func grownExit(_ exit: GrowExit, _ around: Surroundings) -> [Effect] {
        // Stop pressed in the engine's first seconds ends it on the signal itself, before it can answer: with a save point that is a stop,
        // without one nothing was made.
        var asked = exit.stopRequested
        if case .growing(true) = state { asked = true }
        if exit.signalled, asked, exit.status == 2 {
            state = exit.savePoint ? .parked(.stopped, step: step) : .failed(.nothingSaved)
            if exit.savePoint, mode != .growThenPlay { state = .finished(stopped: true) }
            return []
        }
        let failed = exit.signalled || !(exit.status == 0 || exit.status == 130)
        if failed {
            state = .failed(.growing(exit.problem ?? PipelineWords.exitSentence(status: exit.status, signalled: exit.signalled)))
            return []
        }
        // Stopped by the person, by the timer or by a signal that left a save point: never continued by itself. (The pipeline knows it was
        // asked to stop even when the exit does not say so.)
        if asked || exit.status == 130 {
            if mode == .growThenPlay {
                state = .parked(.stopped, step: step)
            } else {
                state = .finished(stopped: true)
            }
            return []
        }
        guard mode == .growThenPlay else {
            state = .finished(stopped: false)
            return []
        }
        step = max(step, targetSteps ?? step)
        if let why = around.obstacle(byHand: false) {
            state = .parked(why, step: step)
            return why == .notVisible ? [.bounce] : []
        }
        state = .making(cancelling: false)
        making = PlayProgress()
        return [.startMaking]
    }

    // MARK: The banner

    package enum Tone: Equatable, Sendable { case working, attention, failure, success }

    package enum Button: String, Equatable, Sendable {
        case stop, cancel, makePlanetFromHere, makePlanet, keepGrowing, done, showLog, tryAgainNewName, makeAgainNewName, watch, showActivity

        /// The words on the button (§2.5; `NameAuditTests` lists them).
        package var label: String {
            switch self {
            case .stop: return "Stop"
            case .cancel: return "Cancel"
            case .makePlanetFromHere: return "Make planet from here"
            case .makePlanet: return "Make planet"
            case .keepGrowing: return "Keep growing"
            case .done: return "Done"
            case .showLog: return "Show log"
            case .tryAgainNewName: return "Try again with a new name"
            case .makeAgainNewName: return "Make again with a new name"
            case .watch: return "Watch"
            case .showActivity: return "Show Activity"
            }
        }
    }

    package struct Banner: Equatable, Sendable {
        package var tone: Tone
        package var title: String
        /// What is happening now, in a sentence.
        package var detail: String
        /// 0...1, nil when no honest fraction is known.
        package var fraction: Double?
        /// What comes next, in a sentence; nil at the end.
        package var next: String?
        package var buttons: [Button]
    }

    /// The banner for the state. `elapsed` is the seconds since the current stage started (for "about N s left" and the clock).
    package func banner(elapsed: Double) -> Banner? {
        let planet = name
        switch state {
        case .growing(let stopping):
            let total = targetSteps
            let fraction = total.flatMap { $0 > startStep ? min(1, max(0, Double(step - startStep) / Double($0 - startStep))) : nil }
            var detail = "Step \(WorldsGuide.grouped(step))" + (total.map { " of \(WorldsGuide.grouped($0))" } ?? "")
            if stopping {
                detail = "Stopping: it writes a save point first. " + detail.lowercasedFirst
            } else if let total, let sps = stepsPerSecond, sps > 0, total > step {
                detail += ", " + TimeWords.about(Double(total - step) / sps) + " left"
            } else if step == startStep {
                detail = "Starting the engine"
            }
            let title: String
            switch mode {
            case .growThenPlay: title = "Growing \(planet)"
            case .keepGrowing: title = "Growing \(growName ?? planet) further"
            default: title = "Growing \(growName ?? planet)"
            }
            let next: String?
            switch mode {
            case .growThenPlay: next = "Next it becomes a planet" + (expectedPlanetSeconds.map { " (\(TimeWords.about($0)) more)" } ?? "") + "."
            default: next = "It is saved when it finishes. You can make a planet from it then."
            }
            return Banner(tone: .working, title: title, detail: detail, fraction: fraction, next: next, buttons: stopping ? [] : [.stop])
        case .parked(let why, let parkedStep):
            let at = WorldsGuide.grouped(parkedStep)
            switch why {
            case .stopped:
                return Banner(
                    tone: .attention, title: "Growing stopped at step \(at)",
                    detail: "\(growName ?? planet) is saved. You can make a planet from it as it is, or keep growing it first.",
                    fraction: nil, next: nil, buttons: [.makePlanetFromHere, .keepGrowing, .done])
            case .notVisible:
                return Banner(
                    tone: .attention, title: "\(growName ?? planet) has finished growing",
                    detail: "It finished while the window was not in front, so the planet was not started behind your back.",
                    fraction: nil, next: nil, buttons: [.makePlanet, .keepGrowing, .done])
            case .worldOpen(let open):
                return Banner(
                    tone: .attention, title: "\(growName ?? planet) has finished growing",
                    detail: "\(open) is open, and a new planet replaces the open world. Close \(open), then make the planet.",
                    fraction: nil, next: nil, buttons: [.makePlanet, .done])
            case .nameTaken:
                return Banner(
                    tone: .attention, title: "\(growName ?? planet) has finished growing",
                    detail: "A world called \(planet) appeared while it grew. Make the planet under a new name.",
                    fraction: nil, next: nil, buttons: [.makeAgainNewName, .done])
            case .engineMissing:
                return Banner(
                    tone: .attention, title: "\(growName ?? planet) has finished growing",
                    detail: "The engine cannot be found. Set it up in Settings, then make the planet.",
                    fraction: nil, next: nil, buttons: [.makePlanet, .done])
            }
        case .making(let cancelling):
            let detail = cancelling ? "Cancelling." : making.sentence
            var fraction = making.fraction
            if fraction == nil, let expected = expectedPlanetSeconds, expected > 0, !cancelling { fraction = min(0.9, elapsed / expected) }
            var line = detail
            if !cancelling, let expected = expectedPlanetSeconds, expected - elapsed > 3 { line += ", " + TimeWords.about(expected - elapsed) + " left" }
            return Banner(
                tone: .working, title: "Making \(planet)", detail: line, fraction: fraction,
                next: cancelling ? nil : "Then it opens in Watch.", buttons: cancelling ? [] : [.cancel])
        case .live:
            return Banner(tone: .success, title: "\(planet) is ready", detail: "Your planet is open in Watch.", fraction: 1, next: nil, buttons: [.watch, .done])
        case .finished(let stopped):
            let world = growName ?? planet
            return Banner(
                tone: .success, title: stopped ? "Stopped at step \(WorldsGuide.grouped(step))" : "\(world) has finished growing",
                detail: stopped ? "\(world) is saved. Keep growing continues it later." : "\(world) is saved at step \(WorldsGuide.grouped(step)).",
                fraction: nil, next: nil, buttons: (offersPlanet ? [.makePlanet] : []) + [.keepGrowing, .done])
        case .failed(.growing(let message)):
            return Banner(
                tone: .failure, title: "Growing failed", detail: message, fraction: nil,
                next: "\(growName ?? planet) is kept as it was written.", buttons: [.showLog, .tryAgainNewName, .done])
        case .failed(.nothingSaved):
            return Banner(
                tone: .attention, title: "Stopped before anything was saved",
                detail: "You stopped it in its first seconds, before the engine had a save point to keep. Nothing was made.",
                fraction: nil, next: nil, buttons: [.tryAgainNewName, .done])
        case .failed(.making(let message)):
            return Banner(
                tone: .failure, title: "\(planet) could not be made", detail: message, fraction: nil,
                next: mode == .growThenPlay ? "\(growName ?? planet) is kept, so only the planet needs making again." : nil,
                buttons: [.showLog, .makeAgainNewName, .done])
        case .cancelled:
            return Banner(
                tone: .attention, title: "\(planet) was not finished",
                detail: "Making the planet was stopped. It is listed under Unfinished; nothing is deleted.", fraction: nil,
                next: mode == .growThenPlay ? "\(growName ?? planet) is kept." : nil, buttons: [.makeAgainNewName, .done])
        case .dismissed:
            return nil
        }
    }
}

extension String {
    fileprivate var lowercasedFirst: String { prefix(1).lowercased() + dropFirst() }
}

// MARK: - Words

package enum PipelineWords {
    /// What an exit without a message from the run says.
    package static func exitSentence(status: Int32, signalled: Bool) -> String {
        if signalled { return "The engine was ended before it could save." }
        switch status {
        case 1: return "The engine stopped with an error."
        case 2: return "The engine did not accept these settings."
        case 3: return "The engine's own check found a problem."
        case 4: return "A save point could not be written."
        default: return "The engine stopped (code \(status))."
        }
    }

    /// The line of a run's output that says what went wrong: the last line that names an error (the end of a traceback), else the first
    /// line of stderr. Nil when the run printed nothing on stderr.
    package static func firstProblem(in errorLines: [String]) -> String? {
        let lines = errorLines.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        guard !lines.isEmpty else { return nil }
        let named = lines.last { line in
            line.range(of: #"^([A-Za-z_][A-Za-z0-9_.]*)?(Error|Exception|Warning)\b"#, options: .regularExpression) != nil
                || line.hasPrefix("alife:") || line.hasPrefix("error:")
        }
        let pick = (named ?? lines.first) ?? ""
        return pick.count > 240 ? String(pick.prefix(237)) + "…" : pick
    }
}

// MARK: - What the planet's process says

/// `alife play --init-from` prints its progress on stderr while it is starting: the houses it rebuilds, the run header it forks to, the weather
/// warm-up as `weather spin-up: N/M steps`. The app reads the newest of them (seen with the real engine, docs/APP.md §3.2).
package struct PlayProgress: Equatable, Sendable {
    package enum Phase: Equatable, Sendable {
        case starting
        case houses
        case copying
        case warmup(done: Int, total: Int)
        case opening
    }

    package var phase = Phase.starting

    package init() {}

    package init(phase: Phase) { self.phase = phase }

    package static func parse(_ lines: [String]) -> PlayProgress {
        var phase = Phase.starting
        for line in lines {
            if line.hasPrefix("houses:") {
                phase = .houses
            } else if line.hasPrefix("run runs/") || line.hasPrefix("run ") && line.contains("steps") && line.contains("→") {
                if case .warmup = phase {} else { phase = .copying }
            } else if let range = line.range(of: #"weather spin\-up: (\d+)/(\d+) steps"#, options: .regularExpression) {
                let numbers = line[range].split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
                if numbers.count == 2, numbers[1] > 0 {
                    phase = numbers[0] >= numbers[1] ? .opening : .warmup(done: numbers[0], total: numbers[1])
                }
            }
        }
        return PlayProgress(phase: phase)
    }

    /// Where the bar is, 0...1; nil before anything is known.
    package var fraction: Double? {
        switch phase {
        case .starting: return nil
        case .houses: return 0.05
        case .copying: return 0.1
        case .warmup(let done, let total): return 0.1 + 0.85 * min(1, Double(done) / Double(max(1, total)))
        case .opening: return 0.97
        }
    }

    package var sentence: String {
        switch phase {
        case .starting: return "Starting the engine"
        case .houses: return "Rebuilding the family lines"
        case .copying: return "Copying the world"
        case .warmup(let done, let total):
            return "Warming up the weather (\(WorldsGuide.grouped(done)) of \(WorldsGuide.grouped(total)) steps)"
        case .opening: return "Opening the planet"
        }
    }
}
