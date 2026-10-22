import AlifeCore
import Foundation

/// Where a played world is in its life.
enum PlayPhase: Equatable, Sendable {
    /// No process: nothing was started, or it was stopped.
    case idle
    /// The process is running and has not sent its header yet. A planet's fork, its weather spin-up and the rebuild of its
    /// houses happen here and print their progress to stderr (`PlayController.progress`); a world that is resumed loads its
    /// checkpoint, which takes seconds.
    case starting
    case live
    /// The world ended by itself and cleanly (the end step, an extinct world).
    case ended(String)
    /// The process could not start, died, or sent something the decoder refused.
    case failed(String)
}

/// What to start.
enum PlayRequest: Equatable, Sendable {
    case resume(PlayResume)
    case planet(PlanetSpec)

    /// The directory name the world lives in.
    var world: String {
        switch self {
        case .resume(let r): r.run
        case .planet(let p): p.name.trimmingCharacters(in: .whitespaces)
        }
    }

    var arguments: [String] {
        switch self {
        case .resume(let r): PlayLaunch.arguments(resume: r)
        case .planet(let p): PlayLaunch.arguments(planet: p)
        }
    }
}

/// Owns the `alife play` process of the World window and what it sends (DESIGN §33.4.2).
///
/// The process writes the stream and its records to stdout, which a dedicated thread drains (`StreamReader`); what the
/// thread decodes reaches this object through a mailbox, and the main actor takes it when woken and reduces it into a
/// `WorldStore`. Stderr goes to the log and, while the world is `starting`, to `progress`. Stdin carries the control lines
/// (`PlayCommand`). Only small things are published (`phase`, `revision`, the clock through the store): the frames of
/// 1.3 MB are read by the renderer through `store`, which is not a published property, and `onDelivery` says when to look.
///
/// Stopping is a ladder (`EngineSession.Ladder.play`): a `quit` line and EOF on stdin, SIGINT after `ladder.beforeInterrupt`
/// (the runner's own stop path, which works while the world is paused), SIGTERM after `ladder.beforeTerminate` more. The
/// world saves a checkpoint on `quit`, so none of these waits is short. A world still being built (`starting`) reads no
/// stdin: stopping it sends SIGINT at once. A directory whose process was asked to stop and has not exited is *retired*
/// (the launcher shows it as saving).
@MainActor
final class PlayController: ObservableObject {
    static let shared = PlayController()

    @Published private(set) var phase: PlayPhase = .idle
    /// The directory name of the world of the current session.
    @Published private(set) var world: String?
    /// Increases with every change of the store other than a frame.
    @Published private(set) var revision = 0
    /// The stderr lines of the current session (the planet's progress), the newest 300.
    @Published private(set) var progress: [String] = []
    /// What the window asked the rate to be, until the world's next `CLCK` confirms it (`.multiplier(0)`: paused).
    @Published private(set) var requestedRate: PlayRate?
    /// Whether the world was told to send the intent labels (`labels on|off`; on at the start of every world).
    @Published private(set) var labelsOn = true
    /// The names of the worlds whose process was asked to stop and has not exited: they are saving.
    @Published private(set) var retiredWorlds: [String] = []

    /// What the world sent. Read by the renderer and the views; changes are announced by `revision` and `onDelivery`.
    var store = WorldStore()  // (settable for the tests, which feed it frames)
    /// Called on the main actor after a delivery was applied, with what it changed (a frame, a map, ...).
    var onDelivery: (@MainActor (WorldStore.Changes) -> Void)?
    /// Called, before `onDelivery`, with the events of each `EVNT` batch that the chronicle had not seen, in order, and whether
    /// the batch is the backlog loaded at the start (a hand-off or a toast is the news of a live batch, never of the backlog).
    var onEvents: (@MainActor ([WorldEvent], Bool) -> Void)?
    /// Called, before `onDelivery`, with what each `ALRT` batch did to the alert board (an alert raised, updated or cleared)
    /// and whether the batch is the backlog (no toast, no pause for the history of a re-opened world).
    var onAlerts: (@MainActor ([AlertChange], Bool) -> Void)?
    /// Called when the store was replaced by an empty one (a world started or stopped), before anything of the new world: the
    /// window forgets the selection and the follow of the old one at once, not when the new header arrives.
    var onReset: (@MainActor () -> Void)?
    /// Seconds from a monotonic origin (tests inject theirs).
    var now: () -> Double = { ProcessInfo.processInfo.systemUptime }
    /// A `CLCK` that does not show the rate the window asked for, this long after the asking, is the world's answer: the
    /// request is dropped (it was refused or clamped).
    static let confirmWait = 1.0
    private var requestedAt = 0.0
    /// Where lines for the log go: the command line that starts the world, and its stderr.
    var log: @MainActor ([String], Bool) -> Void = { _, _ in }
    /// The stop ladder of a live world; tests shorten it.
    var ladder = EngineSession.Ladder.play
    /// How long a world being built may take after SIGINT before it gets SIGTERM.
    var interruptGrace: Duration = .seconds(10)
    private(set) var directory: URL?
    private(set) var request: PlayRequest?

    private var session: EngineSession?
    private var retired: [(session: EngineSession, world: String)] = []

    init() {}

    var isActive: Bool { phase == .starting || phase == .live }

    /// Whether an `alife play` process is still attached (it can outlive `stop()` by its ladder).
    var hasProcess: Bool { session != nil || !retired.isEmpty }

    // MARK: Start and stop

    /// Starts `request` with `executable` (the repository's `.venv/bin/alife`), run in `directory`, replacing any world that
    /// is running (it is stopped through its ladder and retired). `arguments` replaces the ones `PlayLaunch` builds (for
    /// tests, which run a stand-in).
    func start(_ request: PlayRequest, executable: URL, directory: URL, arguments: [String]? = nil) {
        stopSession()
        self.request = request
        self.directory = directory
        world = request.world
        store = WorldStore()
        revision += 1
        progress = []
        requestedRate = nil
        labelsOn = true
        onReset?()

        // A world still saving its checkpoint holds `play.lock`; starting it again would only be refused.
        if retired.contains(where: { $0.world == request.world }) {
            let message = "'\(request.world)' is still saving: wait until it has closed."
            log(["[play] " + message], true)
            phase = .failed(message)
            return
        }
        phase = .starting
        let args = arguments ?? request.arguments
        log(["$ " + ([executable.path] + args).map(shellQuote).joined(separator: " ")], false)

        let launched: EngineSession.Launched
        do {
            launched = try EngineSession.launch(executable: executable, arguments: args, directory: directory)
        } catch {
            let message = "Failed to launch \(executable.path): \(error.localizedDescription)"
            log([message], true)
            phase = .failed(message)
            return
        }
        let session = launched.session
        self.session = session
        Task { [weak self] in
            await session.pump(
                launched,
                deliver: { delivery in self?.apply(delivery, from: session) },
                lines: { lines in self?.note(lines, from: session) },
                finished: { status, reason in self?.finish(session, status: status, reason: reason) })
        }
    }

    /// Stops the world, once: the window closed, or another world replaces it. A live world gets the ladder (it saves);
    /// one being built gets SIGINT now. The view is cleared and the process is retired until it exits.
    func stop() {
        let starting = phase == .starting
        stopSession(interrupt: starting)
        phase = .idle
        store = WorldStore()
        revision += 1
        requestedRate = nil
        labelsOn = true
        onReset?()
    }

    /// The app is quitting: ask a live world to stop through its ladder, so that it saves (`applicationShouldTerminate`
    /// waits for the process to leave). A world being built is interrupted.
    func stopForQuit() {
        stopSession(interrupt: phase == .starting)
    }

    /// The last resort when the app quits: end every process now, without the ladder.
    func shutdown() {
        for session in retired.map(\.session) + [session].compactMap({ $0 }) {
            session.stopRequested = true
            session.terminate()
        }
    }

    private func stopSession(interrupt: Bool = false) {
        guard let old = session else { return }
        if interrupt {
            old.interruptNow(terminateAfter: interruptGrace)
        } else {
            old.requestStop(quit: try? PlayCommand.quit.encoded(), ladder: ladder)
        }
        retired.append((old, world ?? ""))  // its end is ignored from now on, but the app quitting must still end it
        session = nil
        retiredWorlds = retired.map(\.world)
    }

    // MARK: Commands

    /// Writes a control line. False if there is no live world to write to, or the write failed (the process is gone).
    @discardableResult
    func send(_ command: PlayCommand) -> Bool {
        guard let session, !session.stopRequested, phase == .live else { return false }
        do {
            guard try session.write(try command.encoded()) else { return false }
        } catch {
            log(["[play] could not write to the world: \(error.localizedDescription)"], true)
            return false
        }
        switch command {
        case .pause:
            requestedRate = .multiplier(0)
            requestedAt = now()
        case .rate(let rate):
            requestedRate = rate
            requestedAt = now()
        case .labels(let on):
            labelsOn = on
        case .run, .step:
            requestedRate = nil  // a later intent replaces a pending request; the world's next CLCK says what it holds
        default: break
        }
        return true
    }

    func pause() { send(.pause) }
    func resume() { send(.run) }
    func setRate(_ rate: PlayRate) { send(.rate(rate)) }

    func step(_ count: Int = 1) { send(.step(count: count)) }

    func setColor(_ mode: String) {
        guard store.header?.colorModes.contains(mode) == true else { return }
        send(.color(mode))
    }

    // MARK: Pumping

    private func note(_ lines: [String], from session: EngineSession) {
        guard !lines.isEmpty else { return }
        log(lines.map { "[play] " + $0 }, true)
        guard self.session === session else { return }
        progress = Array((progress + lines).suffix(300))
    }

    private func apply(_ delivery: StreamDelivery, from session: EngineSession) {
        guard self.session === session, !session.stopRequested else { return }
        guard !delivery.isEmpty else { return }
        var changes: WorldStore.Changes = []
        if let header = delivery.header {
            store.reset(header: header)
            changes.insert(.header)
            phase = .live
        }
        // The order does not matter between latest-wins records: each is complete by itself (§33.1.1).
        for record in delivery.records.values {
            if case .clck(let clock) = record { settleRequest(after: clock) }
            changes.formUnion(store.apply(record))
        }
        var announced: [(events: [WorldEvent], backlog: Bool)] = []
        for batch in delivery.events {
            let fresh = store.apply(events: batch)
            if !fresh.isEmpty {
                changes.insert(.events)
                announced.append((fresh, batch.isBacklog))
            }
        }
        var changed: [(changes: [AlertChange], backlog: Bool)] = []
        for batch in delivery.alerts {
            let alertChanges = store.apply(alerts: batch)
            if !alertChanges.isEmpty {
                changes.insert(.alerts)
                changed.append((alertChanges, batch.isBacklog))
            }
        }
        for report in delivery.reports where store.apply(report: report) { changes.insert(.reports) }
        for plan in delivery.plans {
            store.apply(plan: plan)
            changes.insert(.plans)
        }
        if let end = delivery.end {
            session.end = end
            switch end {
            case .complete: break
            case .failed(.truncated): break  // the process died; its exit status and stderr say why
            case .failed, .unreadable: session.terminate()  // the stream is corrupt: nothing more to read
            }
        }
        if !changes.subtracting(.frame).isEmpty { revision += 1 }
        for (events, backlog) in announced { onEvents?(events, backlog) }
        for (alertChanges, backlog) in changed { onAlerts?(alertChanges, backlog) }
        if !changes.isEmpty { onDelivery?(changes) }
    }

    private func finish(_ session: EngineSession, status: Int32, reason: Process.TerminationReason) {
        session.release()
        retired.removeAll { $0.session === session }
        retiredWorlds = retired.map(\.world)
        guard self.session === session else { return }  // a newer world replaced it, or it was asked to stop
        self.session = nil
        guard !session.stopRequested else { return }  // asked for: the phase is already idle
        let detail = session.lastError.map { ": " + $0 } ?? ""
        if case .failed(let error)? = session.end, !isTruncation(error) {
            phase = .failed(error.description)
        } else if case .unreadable(let why)? = session.end {
            phase = .failed("cannot read the stream: \(why)")
        } else if reason == .uncaughtSignal {
            phase = .failed("alife play was terminated by signal \(status)\(detail)")
        } else if status != 0 {
            phase = .failed(Self.describe(status: status) + detail)
        } else if case .failed(let error)? = session.end {
            phase = .failed(error.description)
        } else if store.header == nil {
            phase = .ended("The world ended before it sent anything\(detail)")
        } else {
            phase = .ended("The world ended: its last step was reached, or nothing is left alive.")
        }
        if case .failed(let message) = phase { log(["[play] \(message)"], true) }
        revision += 1
    }

    private func isTruncation(_ error: StreamError) -> Bool {
        if case .truncated = error { return true }
        return false
    }

    /// A `CLCK` arrived: the rate the window asked for is settled when the clock shows it (or, a second after the asking, shows
    /// something else: the world refused or clamped it). A clock sent before the world read the command shows the old rate and
    /// settles nothing, so the strip does not flip back to it for a quarter of a second.
    func settleRequest(after clock: PlayClock) {
        guard let asked = requestedRate else { return }
        if Self.confirms(clock.rate, asked) || now() - requestedAt > Self.confirmWait { requestedRate = nil }
    }

    /// Whether a `CLCK` rate (0 paused, -1 max, else the multiplier) is the rate the window asked for. A clock sent before the
    /// world read the command shows the old rate and confirms nothing.
    static func confirms(_ rate: Float, _ asked: PlayRate) -> Bool {
        switch asked {
        case .max: return rate < 0
        case .multiplier(let m): return rate >= 0 && abs(Double(rate) - m) <= 1e-4 * Swift.max(1, m)
        }
    }

    static func describe(status: Int32) -> String {
        switch status {
        case 1: return "alife play failed (exit code 1)"
        case 2: return "alife play refused the request (exit code 2)"
        case 3: return "alife play found an invariant violation (exit code 3)"
        case 4: return "alife play found no usable checkpoint (exit code 4)"
        case 130: return "alife play was interrupted (exit code 130)"
        default: return "alife play exited with code \(status)"
        }
    }
}
