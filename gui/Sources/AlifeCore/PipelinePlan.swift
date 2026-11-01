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
        /// Nobody is left alive in the world (the engine says it died out): nothing is made from it, and the bar says what happened.
        case diedOut(step: Int)
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

    /// True when the stage that failed left no save point behind (the engine refused before it wrote anything).
    package private(set) var wroteNothing = false
    /// True when the stage that failed had made more than the save point it started from (so a planet, or more growing, can go on from it).
    package private(set) var leftProgress = false

    /// How the banner speaks of the world stage 1 writes: for "grow, then play" it is "the grown world" (the directory `NAME-grow` is an
    /// implementation detail), else its own name. `Sentence` is the same for the start of a sentence.
    package var grownWorld: String { mode == .growThenPlay ? "the grown world" : (growName ?? name) }
    private var grownWorldSentence: String { mode == .growThenPlay ? "The grown world" : (growName ?? name) }
    /// The name a title uses: the planet the person asked for, for "grow, then play".
    private var titleName: String { mode == .growThenPlay ? name : (growName ?? name) }

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
        /// The step of the newest save point in the run's directory (the numbers the bar read are up to two seconds old, this is the truth).
        package var savedStep: Int?
        /// The `status` the engine wrote in the run's `meta.json` (`done`, `extinct`, `interrupted`, `violation`), nil when it could not be read.
        package var worldStatus: String?
        package init(status: Int32, signalled: Bool = false, stopRequested: Bool = false, problem: String? = nil, savePoint: Bool = true,
                     savedStep: Int? = nil, worldStatus: String? = nil) {
            self.savedStep = savedStep
            self.worldStatus = worldStatus
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
        /// The planet's process ended without the world being live: cancelled (nil), or it ended on its own and said why.
        case makeEnded(String?)
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
        case .live, .finished, .failed, .cancelled, .diedOut, .dismissed: return true
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
        case (.making(let cancelling), .makeEnded(let why)):
            if let why, !cancelling {
                if let died = PipelineWords.diedOutStep(in: why) { state = .diedOut(step: died) } else { state = .failed(.making(why)) }
            } else {
                state = .cancelled
            }
            return []

        // The end
        case (.finished, .donePressed), (.failed, .donePressed), (.cancelled, .donePressed), (.live, .donePressed), (.diedOut, .donePressed):
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
        if let saved = exit.savedStep { step = max(step, saved) }
        if exit.signalled, asked, exit.status == 2 {
            state = exit.savePoint ? .parked(.stopped, step: step) : .failed(.nothingSaved)
            if exit.savePoint, mode != .growThenPlay { state = .finished(stopped: true) }
            return []
        }
        let failed = exit.signalled || !(exit.status == 0 || exit.status == 130)
        if failed {
            wroteNothing = !exit.savePoint
            leftProgress = exit.savePoint && max(exit.savedStep ?? step, step) > startStep
            state = .failed(.growing(exit.problem ?? PipelineWords.exitSentence(status: exit.status, signalled: exit.signalled)))
            return []
        }
        // The engine ends a world in which nothing is left alive with exit 0, before its target: it is not a finished growth, and no planet is
        // made from it.
        if exit.worldStatus == "extinct" {
            state = .diedOut(step: step)
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
        // (a run that ended by itself reached its target: the last numbers read were up to two seconds old)
        step = max(step, targetSteps ?? step)
        guard mode == .growThenPlay else {
            state = .finished(stopped: false)
            return []
        }
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
            case .keepGrowing: title = "Growing \(titleName) further"
            default: title = "Growing \(titleName)"
            }
            let next: String?
            switch mode {
            case .growThenPlay: next = "Next it becomes a planet" + (expectedPlanetSeconds.map { ", which takes \(TimeWords.about($0)) on a recent Mac" } ?? "") + "."
            default: next = "It is saved when it finishes. You can make a planet from it then."
            }
            return Banner(tone: .working, title: title, detail: detail, fraction: fraction, next: next, buttons: stopping ? [] : [.stop])
        case .parked(let why, let parkedStep):
            let at = WorldsGuide.grouped(parkedStep)
            switch why {
            case .stopped:
                return Banner(
                    tone: .attention, title: "Growing stopped at step \(at)",
                    detail: "\(grownWorldSentence) is saved. You can make a planet from it as it is, or keep growing it first.",
                    fraction: nil, next: nil, buttons: [.makePlanetFromHere, .keepGrowing, .done])
            case .notVisible:
                return Banner(
                    tone: .attention, title: "\(titleName) has finished growing",
                    detail: "It finished while the window was not in front, so the planet was not started behind your back.",
                    fraction: nil, next: nil, buttons: [.makePlanet, .keepGrowing, .done])
            case .worldOpen(let open):
                return Banner(
                    tone: .attention, title: "\(titleName) has finished growing",
                    detail: "\(open) is open, and a new planet replaces the open world. Close \(open), then make the planet.",
                    fraction: nil, next: nil, buttons: [.makePlanet, .done])
            case .nameTaken:
                return Banner(
                    tone: .attention, title: "\(titleName) has finished growing",
                    detail: "A world called \(planet) appeared while it grew. Make the planet under a new name.",
                    fraction: nil, next: nil, buttons: [.makeAgainNewName, .done])
            case .engineMissing:
                return Banner(
                    tone: .attention, title: "\(titleName) has finished growing",
                    detail: "The engine cannot be found. Set it up in Settings, then make the planet.",
                    fraction: nil, next: nil, buttons: [.makePlanet, .done])
            }
        case .making(let cancelling):
            // (Only the weather warm-up counts its own steps; the rest has no honest fraction, so the bar then moves without one.)
            let detail = cancelling ? "Cancelling." : making.sentence
            return Banner(
                tone: .working, title: "Making \(planet)", detail: detail, fraction: cancelling ? nil : making.fraction,
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
                next: wroteNothing ? "It stopped before it made a save point, so there is nothing to continue." : "\(grownWorldSentence) is kept as it was written.",
                buttons: [.showLog] + (!leftProgress ? [] : (offersPlanet && mode != .keepGrowing ? [.makePlanet] : []) + [.keepGrowing])
                    + [.tryAgainNewName, .done])
        case .failed(.nothingSaved):
            return Banner(
                tone: .attention, title: "Stopped before anything was saved",
                detail: "You stopped it in its first seconds, before the engine had a save point to keep. Nothing was made.",
                fraction: nil, next: nil, buttons: [.tryAgainNewName, .done])
        case .failed(.making(let message)):
            return Banner(
                tone: .failure, title: "\(planet) could not be made", detail: message, fraction: nil,
                next: mode == .growThenPlay ? "\(grownWorldSentence) is kept, so only the planet needs making again." : nil,
                buttons: [.showLog, .makeAgainNewName, .done])
        case .cancelled:
            return Banner(
                tone: .attention, title: "\(planet) was not finished",
                detail: "Making the planet was stopped. It is listed under Unfinished; nothing is deleted.", fraction: nil,
                next: mode == .growThenPlay ? "\(grownWorldSentence) is kept." : nil, buttons: [.makeAgainNewName, .done])
        case .diedOut(let at):
            let world = mode == .growThenPlay ? "the grown world" : (growName ?? source ?? planet)
            let again: [Button] = mode == .makePlanet || mode == .keepGrowing ? [] : [.tryAgainNewName]
            return Banner(
                tone: .attention, title: "The population died out at step \(WorldsGuide.grouped(at))",
                detail: "Nothing is left alive in \(world), so no planet can be made from it. It is saved as it was.",
                fraction: nil, next: mode == .makePlanet ? "Choose an earlier save point, or another world, to make a planet from." : "Branch from an earlier save point of it, or start again with other settings.",
                buttons: again + [.done])
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

    /// What a planet that did not open says, in the words of the page: the engine's exit in a sentence, then its own last line when it said
    /// one. (The play controller's message names a checkpoint for exit 4 and quotes the engine; neither is for the person who pressed Create.)
    package static func planetFailure(_ message: String, said: String? = nil) -> String {
        // A planet's engine that ends before it sent its header (it ended by itself, with exit 0) reads as a cut stream: say what happened.
        if message.hasPrefix("truncated stream") || message.hasPrefix("cannot read the stream") || message.contains("ended before its header") {
            return "The planet's engine ended by itself before the planet was ready" + (said.map { ". It said: " + $0 } ?? ".")
        }
        if message.lowercased().contains("still saving") { return "That world is still being saved. Wait a moment, then make the planet again." }
        if message.hasPrefix("Failed to launch") { return "The engine could not be started. " + message }
        guard let range = message.range(of: #"\(exit code (\d+)\)"#, options: .regularExpression) else { return message }
        let code = Int(message[range].filter(\.isNumber)) ?? 0
        let head: String
        switch code {
        case 1: head = "The engine stopped while it was making the planet"
        case 2: head = "The engine did not accept the planet's settings"
        case 3: head = "The engine's own check found a problem in the planet"
        case 4: head = "The engine could not open the world the planet was to start from"
        case 130: head = "The planet was interrupted"
        default: head = "The engine stopped (code \(code))"
        }
        let said = message[range.upperBound...].trimmingCharacters(in: CharacterSet(charactersIn: ": ").union(.whitespacesAndNewlines))
            .replacingOccurrences(of: "checkpoint", with: "save point")
        return said.isEmpty ? head + "." : head + ". It said: " + said
    }

    /// What a planet's engine that ended on its own (nobody asked it to stop) says, in the words of the page. The play controller's message
    /// carries the engine's last line after a colon; "extinct at step N" is kept so that the plan can tell a world that died out.
    package static func planetEnded(_ message: String) -> String {
        if diedOutStep(in: message) != nil { return message }   // (the plan turns it into "the population died out at step N")
        let detail = message.range(of: ": ").map { String(message[$0.upperBound...]) }
        return "The planet's engine ended by itself before the planet was ready" + (detail.map { ". It said: " + $0 } ?? ".")
    }

    /// The last line of the planet's stderr that says something (not one of its progress lines: the houses, the copy, the warm-up).
    package static func lastSaid(in lines: [String]) -> String? {
        lines.reversed().map { $0.trimmingCharacters(in: .whitespaces) }.first { line in
            !line.isEmpty && !line.hasPrefix("houses:") && !line.hasPrefix("run ") && !line.hasPrefix("weather spin" + "-up")
        }
    }

    /// The step a world died out at, when a line of the engine says so ("play: the world is extinct at step 4200", "extinct at step 4200").
    package static func diedOutStep(in text: String) -> Int? {
        guard let range = text.range(of: #"extinct at step (\d+)"#, options: .regularExpression) else { return nil }
        return Int(text[range].filter(\.isNumber))
    }

    /// The line of a run's output that says what went wrong: the last line that names an error (the end of a traceback), else the first
    /// line of stderr. Nil when the run printed nothing on stderr.
    package static func firstProblem(in errorLines: [String]) -> String? {
        let lines = errorLines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        guard !lines.isEmpty else { return nil }
        let named = lines.lastIndex { raw in
            let line = raw.trimmingCharacters(in: .whitespaces)
            return line.range(of: #"^([A-Za-z_][A-Za-z0-9_.]*)?(Error|Exception|Warning)\b"#, options: .regularExpression) != nil
                || line.hasPrefix("alife:") || line.hasPrefix("error:")
        }
        let at = named ?? 0
        var pick = lines[at].trimmingCharacters(in: .whitespaces)
        // A line that ends in a colon introduces indented lines that say what is wrong (the engine's refusal of a fork's settings).
        if pick.hasSuffix(":") {
            let details = lines.dropFirst(at + 1).prefix(3).filter { $0.first == " " || $0.first == "\t" }
                .map { $0.trimmingCharacters(in: .whitespaces) }
            if !details.isEmpty { pick += " " + details.joined(separator: "; ") }
        }
        if let range = pick.range(of: #"^--init-from \S+: the state violates the new config:?"#, options: .regularExpression) {
            let source = pick[range].split(separator: " ")[1].dropLast()
            let rest = pick[range.upperBound...].trimmingCharacters(in: .whitespaces)
            pick = "The newest save point of \(source.replacingOccurrences(of: "runs/", with: "")) does not fit the changed settings"
                + (rest.isEmpty ? "." : ": " + rest)
        }
        // The name of an exception class is for the log, not for the person: "ValueError: the steps must be positive" says the second part.
        for pattern in [#"^([A-Za-z_][A-Za-z0-9_.]*)?(Error|Exception|Warning):\s+"#, #"^alife:\s+(error:\s+)?"#, #"^error:\s+"#] {
            if let range = pick.range(of: pattern, options: .regularExpression), range.upperBound < pick.endIndex {
                pick = String(pick[range.upperBound...])
                break
            }
        }
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
        case .warmup(let done, let total): return min(1, Double(done) / Double(max(1, total)))
        case .starting, .houses, .copying, .opening: return nil
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
