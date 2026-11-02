import Foundation

// Watch in its two forms (docs/APP.md §3.4): what the page shows, what each form can and cannot do in plain words, why a world cannot be
// watched, and when the observer's stream starts, restarts or waits. Pure: the pages draw the answers and the tests ask the questions.

/// Which Watch the page shows. A played world is the live picture (`alife play`); a research world is the observer picture (`alife stream`).
/// The form is a fact of the world, never a switch: the toolbar says which one it is.
package enum WatchForm: String, Equatable, Sendable {
    case played
    case research

    /// The chip in the toolbar.
    package var chip: String { self == .played ? "Live" : "Observer" }

    /// What the chip means, one sentence (its tooltip).
    package var chipMeaning: String {
        switch self {
        case .played: return "The world is running and you can play it: powers, chronicle, reports."
        case .research: return "You are looking at a research world from outside. It cannot be played here."
        }
    }
}

package enum WatchGuide {
    /// The form Watch shows. A world that is open (starting or live) is what Watch shows; with none, the research world that was chosen; a
    /// played world that ended or failed keeps its page (its banner and *Back to Worlds*) until another world is chosen to watch.
    package static func form(worldOpen: Bool, worldEnded: Bool, observed: String?) -> WatchForm? {
        if worldOpen { return .played }
        if observed != nil { return .research }
        return worldEnded ? .played : nil
    }

    // MARK: What a research world can and cannot do in Watch

    /// Said on the research Watch page, under the strip of view controls.
    package static let observerNote = "This is an observer view. To play it, make a planet from it."
    /// The same for a flat world, which has no weather.
    package static let flatObserverNote = "This is an observer view of a flat world. A flat world is for study: it has no weather, so it cannot become a planet."

    package static func note(forShape shape: WorldShape?) -> String { shape == .flat ? flatObserverNote : observerNote }

    /// What a research world can do in Watch.
    package static let researchCan = [
        "Pause, run and step it, change the colours, show the food and chemical layers, turn the camera, and save a picture.",
        "Follow it while it grows: the picture moves forward each time the world writes a save point.",
        "Re-simulate it: step its latest save point again in a separate process.",
        "Read its numbers, its autopilot journal and the log beside the picture.",
    ]

    /// What it cannot do there. A played world can: that is the difference between the two kinds.
    package static let researchCannot = [
        "Use powers, read a chronicle or a report, stop at a chapter or an alert, or watch the storage.",
        "Show the weather map. Following a world shows what the autopilot did to it; re-simulating does not (it leaves the autopilot out).",
    ]

    /// What a played world can do in Watch, for the Help page.
    package static let playedCan = [
        "Run live at a rate you choose, pause, step, and pick the world up where you left it.",
        "Use powers, read the chronicle and the reports, stop at chapters and alerts, follow an organism, and watch the storage.",
    ]

    // MARK: Why a world cannot be watched

    /// Why *Watch* is unavailable for this world's card; nil when it can be watched. Always said beside the grey button.
    package static func reasonToWatch(_ row: LibraryRow, openWorld: String?, runGrowing: String?) -> String? {
        if row.kind == .played { return openWorld == row.name ? nil : "A played world is watched while it is playing. Press Play." }
        if row.isPlaceholder { return "It opens in Watch when it is ready." }
        if let open = openWorld { return "Close \(open) first: Watch shows one world at a time." }
        if !row.hasSavePoint, runGrowing != row.name { return "No save point to look at yet." }
        if row.state == .saving { return "Saving…" }
        return nil
    }

    // MARK: The observer's stream

    package struct ObserverFacts: Equatable, Sendable {
        package enum Stream: Equatable, Sendable { case idle, starting, live, ended, failed }
        /// The research world Watch shows.
        package var world: String
        package var hasSavePoint: Bool
        /// This app's run grows that world now.
        package var growing: Bool
        /// The run the stream is attached to, nil with none; and whether it follows the run (else it re-simulates).
        package var streamWorld: String?
        package var follows = false
        package var stream: Stream = .idle
        /// Starts in a row that failed, for this world.
        package var failures = 0

        package init(world: String, hasSavePoint: Bool, growing: Bool, streamWorld: String? = nil, follows: Bool = false,
                     stream: Stream = .idle, failures: Int = 0) {
            self.world = world
            self.hasSavePoint = hasSavePoint
            self.growing = growing
            self.streamWorld = streamWorld
            self.follows = follows
            self.stream = stream
            self.failures = failures
        }
    }

    package enum ObserverStep: Equatable, Sendable {
        case none
        case wait(String)
        case start(follow: Bool)
        case restart(follow: Bool)
    }

    /// How many starts that fail in a row are tried again by themselves; after them the page says why and waits for the button.
    package static let startRetries = 3

    /// What the observer does next. A run is followed while it grows, and re-simulated from its latest save point once it has ended; the
    /// stream waits for the first save point; a start that fails is tried a few times (the engine's first seconds can fail it) and then left to
    /// the player.
    package static func observerStep(_ f: ObserverFacts) -> ObserverStep {
        if let other = f.streamWorld, other != f.world, f.stream != .idle {
            return f.hasSavePoint ? .start(follow: f.growing) : .wait(waitingText)
        }
        switch f.stream {
        case .starting:
            return .none
        case .live:
            // The followed run ended: show what it left, by re-simulating.
            return f.follows && !f.growing ? .restart(follow: false) : .none
        case .ended:
            if f.follows { return f.growing ? .start(follow: true) : .start(follow: false) }
            return .none
        case .failed:
            guard f.failures < startRetries else { return .none }
            return f.hasSavePoint ? .start(follow: f.growing) : .wait(waitingText)
        case .idle:
            return f.hasSavePoint ? .start(follow: f.growing) : .wait(waitingText)
        }
    }

    package static let waitingText = "Waiting for the first save point. The picture appears when the world has written one."

    /// A sentence for the picture's place while the stream is not live.
    package static func streamText(_ stream: ObserverFacts.Stream, growing: Bool, failure: String?) -> String? {
        switch stream {
        case .live: return nil
        case .starting: return "Starting the viewer. It loads the engine first, which takes a few seconds."
        case .ended: return "The viewer ended."
        case .failed: return failure ?? "The viewer stopped."
        case .idle: return growing ? waitingText : nil
        }
    }

    /// The toolbar's second line for the research form: how the picture is made.
    package static func observerLine(follows: Bool?, step: UInt64?) -> String {
        var line = "Observer"
        if let follows { line += follows ? " · following the run" : " · re-simulating" }
        if let step { line += " · step " + WorldsGuide.grouped(Int(step)) }
        return line
    }
}
