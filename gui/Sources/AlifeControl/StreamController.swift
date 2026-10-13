import AlifeCore
import Foundation

/// Where the viewer's stream is in its life.
enum StreamPhase: Equatable, Sendable {
    /// No stream: nothing was opened, or it was closed.
    case idle
    /// The process is running and has not sent its header yet (it loads the engine first, which takes seconds).
    case starting
    case live
    /// The stream ended by itself and cleanly (a run directory that is no longer written, a bounded stream).
    case ended(String)
    /// The process could not start, died, or sent something the decoder refused.
    case failed(String)
}

/// Owns the `alife stream` process of the 3D viewer and what it sends.
///
/// The process writes binary frames to stdout, which a dedicated thread drains (`StreamReader`); the newest header and
/// frame reach this object through a mailbox, and the main actor takes them when woken. Stderr goes to the log, as for
/// the other engine processes, and stdin carries the control lines (`StreamCommand`). The controller is main-actor
/// isolated like `RunController`; the only work off the main actor is the reader thread.
///
/// Stopping is exactly once: a `quit` line, then stdin is closed (its EOF is a quit too), and if the process is still
/// there after `terminationGrace` it gets one SIGTERM. A stream has nothing to save, so none of this waits for the
/// engine to checkpoint, unlike the single SIGINT that stops a run. `shutdown()` (the app quits) skips the grace.
@MainActor
final class StreamController: ObservableObject {
    static let shared = StreamController()

    @Published private(set) var phase: StreamPhase = .idle
    @Published private(set) var source: StreamSource?
    /// The header of the current stream, once it has arrived.
    @Published private(set) var header: StreamHeader?
    /// The newest frame. `frameSerial` increases with every frame that arrives, and with every new stream.
    @Published private(set) var frame: StreamFrame?
    @Published private(set) var frameSerial = 0
    @Published private(set) var control = ViewerControlState()
    /// Simulation steps per second as the frames show them, over the last second or more; nil until measured.
    @Published private(set) var stepsPerSecond: Double?
    /// Frames the viewer never drew because newer ones arrived first (the reader does not wait for the main actor).
    @Published private(set) var droppedFrames = 0

    /// Where lines for the log go: the command line that starts the stream, and its stderr. `RunController` sets it.
    var log: @MainActor ([String], Bool) -> Void = { _, _ in }
    /// How long a stopped process may take to leave after its `quit` before it gets SIGTERM.
    var terminationGrace: Duration = .seconds(2)
    /// The repository directory of the current stream (where `runs/` is); the snapshots go under it.
    private(set) var directory: URL?
    private var executable: URL?

    private var session: Session?
    private var rateStart: (time: ContinuousClock.Instant, step: UInt64)?

    init() {}

    var isActive: Bool { phase == .starting || phase == .live }

    /// Whether an `alife stream` process is still attached (it can outlive `stop()` by its grace period).
    var hasProcess: Bool { session != nil }

    // MARK: Start and stop

    /// Starts `source` with `executable` (the repository's `.venv/bin/alife`), run in `directory`, replacing any stream
    /// that is running. `arguments` replaces the ones `StreamLaunch` builds (for tests, which run a stand-in).
    func start(_ source: StreamSource, executable: URL, directory: URL, arguments: [String]? = nil) {
        stopSession()
        self.source = source
        self.executable = executable
        self.directory = directory
        control.paused = false
        header = nil
        frame = nil
        frameSerial += 1
        stepsPerSecond = nil
        droppedFrames = 0
        rateStart = nil
        phase = .starting

        let args = arguments ?? StreamLaunch.arguments(
            for: source, every: control.every, color: control.color, fieldLayer: control.fields ? control.fieldLayer : nil)
        log(["$ " + ([executable.path] + args).map(shellQuote).joined(separator: " ")], false)

        let process = Process()
        process.executableURL = executable
        process.arguments = args
        process.currentDirectoryURL = directory
        process.environment = RunController.engineEnvironment()
        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        inPipe.fileHandleForWriting.suppressSIGPIPE()
        process.standardInput = inPipe
        process.standardOutput = outPipe
        process.standardError = errPipe
        let (exits, exitSink) = AsyncStream.makeStream(of: (Int32, Process.TerminationReason).self)
        process.terminationHandler = { process in
            exitSink.yield((process.terminationStatus, process.terminationReason))
            exitSink.finish()
        }
        do {
            try process.run()
        } catch {
            let message = "Failed to launch \(executable.path): \(error.localizedDescription)"
            log([message], true)
            phase = .failed(message)
            return
        }

        let mailbox = StreamMailbox()
        let (wakeups, wake) = AsyncStream.makeStream(of: Void.self, bufferingPolicy: .bufferingNewest(1))
        StreamReader.start(reading: outPipe.fileHandleForReading, into: mailbox, wake: wake)
        let errorChunks = RunController.chunks(from: errPipe)
        let session = Session(process: process, stdin: inPipe.fileHandleForWriting, mailbox: mailbox)
        self.session = session
        Task { [weak self] in
            await self?.pump(session, wakeups: wakeups, stderr: errorChunks, exits: exits)
        }
    }

    /// Starts the same run again as a follow or a re-simulation (the viewer's source switch).
    func restart(follow: Bool) {
        guard let source, source.follow != follow, let executable, let directory else { return }
        start(StreamSource(run: source.run, follow: follow), executable: executable, directory: directory)
    }

    /// Stops the stream, once: the viewer window closed, or another stream replaces it. The view is cleared.
    func stop() {
        stopSession()
        phase = .idle
        source = nil
        header = nil
        frame = nil
        frameSerial += 1
        stepsPerSecond = nil
    }

    /// The app is quitting: end the process now, without the grace period.
    func shutdown() {
        guard let session else { return }
        session.stopRequested = true
        session.terminate()
    }

    private func stopSession() {
        session?.requestStop(grace: terminationGrace)
    }

    // MARK: Commands

    /// Writes a control line. False if there is no stream to write to, or the write failed (the process is gone).
    @discardableResult
    private func send(_ command: StreamCommand) -> Bool {
        guard let session, !session.stopRequested, phase == .live else { return false }
        do {
            return try session.write(command)
        } catch {
            log(["[3d] could not write to the stream: \(error.localizedDescription)"], true)
            return false
        }
    }

    func pause() {
        if send(.pause) { control.paused = true }
    }

    func run() {
        if send(.run) { control.paused = false }
    }

    func togglePause() {
        if control.paused { run() } else { pause() }
    }

    /// Pauses and runs `count` steps, then one frame. A followed run fetches its newest checkpoint instead.
    func step(_ count: Int = 1) {
        if send(.step(count: count)) { control.paused = true }
    }

    /// Steps per frame (a re-simulation: a followed run ignores it, and the viewer does not offer it).
    func setEvery(_ every: Int) {
        guard source?.follow == false else { return }
        let k = min(max(every, ViewerSpeed.range.lowerBound), ViewerSpeed.range.upperBound)
        if send(.every(k)) { control.every = k }
    }

    func faster() { setEvery(ViewerSpeed.faster(control.every)) }
    func slower() { setEvery(ViewerSpeed.slower(control.every)) }

    func setColor(_ mode: String) {
        guard header?.colorModes.contains(mode) == true else { return }
        if send(.color(mode)) { control.color = mode }
    }

    func cycleColor() {
        guard let modes = header?.colorModes, let at = modes.firstIndex(of: control.color) else { return }
        setColor(modes[(at + 1) % modes.count])
    }

    func setFields(_ on: Bool) {
        if send(.fields(on)) { control.fields = on }
    }

    /// The field layer shown (0 ≤ layer < FZ); a 2D stream has one.
    func setFieldLayer(_ layer: Int) {
        guard let header else { return }
        let k = min(max(layer, 0), header.fieldLayers - 1)
        if send(.fieldLayer(k)) { control.fieldLayer = k }
    }

    // MARK: Session

    /// One `alife stream` process and the handles to it. Main-actor isolated, so it is `Sendable`.
    @MainActor
    private final class Session {
        let process: Process
        let mailbox: StreamMailbox
        private(set) var stdin: FileHandle?
        /// The viewer asked for the end (as opposed to the stream ending, or being cut off for being corrupt).
        var stopRequested = false
        private var graceTask: Task<Void, Never>?
        /// The last non-blank line of the process's stderr: what it says went wrong.
        var lastError: String?
        /// How the reader saw the stream end.
        var end: StreamEnd?

        init(process: Process, stdin: FileHandle, mailbox: StreamMailbox) {
            self.process = process
            self.stdin = stdin
            self.mailbox = mailbox
        }

        func write(_ command: StreamCommand) throws -> Bool {
            guard let stdin else { return false }
            try stdin.write(contentsOf: try command.encoded())
            return true
        }

        func closeStdin() {
            try? stdin?.close()
            stdin = nil
        }

        /// A `quit` line, EOF on stdin, and SIGTERM if the process is still running after `grace`. Once only.
        func requestStop(grace: Duration) {
            guard !stopRequested else { return }
            stopRequested = true
            if let line = try? StreamCommand.quit.encoded() { try? stdin?.write(contentsOf: line) }
            closeStdin()
            let process = process
            graceTask = Task {
                try? await Task.sleep(for: grace)
                if !Task.isCancelled, process.isRunning { process.terminate() }
            }
        }

        /// SIGTERM now (the app is quitting, or the stream is corrupt).
        func terminate() {
            closeStdin()
            graceTask?.cancel()
            if process.isRunning { process.terminate() }
        }

        func release() {
            graceTask?.cancel()
            closeStdin()
        }
    }

    // MARK: Pumping

    /// Runs until the process has exited and both of its pipes have reached end of file, then finishes the session. The
    /// exit is only looked at after the frames: the last frame and the exit status cannot race.
    private func pump(_ session: Session, wakeups: AsyncStream<Void>, stderr: AsyncStream<Data>,
                      exits: AsyncStream<(Int32, Process.TerminationReason)>) async {
        async let frames: Void = consume(session, wakeups)
        async let errors: Void = drain(session, stderr)
        _ = await (frames, errors)
        for await (status, reason) in exits { finish(session, status: status, reason: reason) }  // yields exactly once
    }

    private func consume(_ session: Session, _ wakeups: AsyncStream<Void>) async {
        for await _ in wakeups { apply(session.mailbox.take(), from: session) }
        apply(session.mailbox.take(), from: session)  // what the last wake-up may have coalesced away
    }

    private func drain(_ session: Session, _ chunks: AsyncStream<Data>) async {
        var splitter = LineSplitter()
        for await data in chunks { note(splitter.feed(data), from: session) }
        if let rest = splitter.flush() { note([rest], from: session) }
    }

    private func note(_ lines: [String], from session: Session) {
        guard !lines.isEmpty else { return }
        log(lines.map { "[3d] " + $0 }, true)
        if let last = lines.last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) {
            session.lastError = last
        }
    }

    private func apply(_ delivery: StreamDelivery, from session: Session) {
        guard self.session === session, !session.stopRequested else { return }
        if let header = delivery.header {
            self.header = header
            // The stream starts in the view the command line asked for, and says so.
            control.color = header.color
            control.fields = header.fields
            control.fieldLayer = header.fieldZ
            if header.mode == "simulate" { control.every = max(header.every, 1) }
            phase = .live
        }
        if let frame = delivery.frame {
            self.frame = frame
            frameSerial += 1
            measure(step: frame.step)
        }
        droppedFrames += delivery.dropped
        if let end = delivery.end {
            session.end = end
            switch end {
            case .complete: break
            case .failed(.truncated): break  // the process died; its exit status and stderr say why
            case .failed, .unreadable: session.terminate()  // the stream is corrupt: nothing more to read
            }
        }
    }

    private func measure(step: UInt64) {
        let now = ContinuousClock.now
        guard let start = rateStart else {
            rateStart = (now, step)
            return
        }
        let components = start.time.duration(to: now).components
        let seconds = Double(components.seconds) + Double(components.attoseconds) * 1e-18
        guard seconds >= 1 else { return }
        stepsPerSecond = step >= start.step ? Double(step - start.step) / seconds : nil
        rateStart = (now, step)
    }

    private func finish(_ session: Session, status: Int32, reason: Process.TerminationReason) {
        session.release()
        guard self.session === session else { return }  // a newer stream replaced it
        self.session = nil
        guard !session.stopRequested else { return }  // asked for: the phase is already idle
        let detail = session.lastError.map { ": " + $0 } ?? ""
        if case .failed(let error)? = session.end, !isTruncation(error) {
            phase = .failed(error.description)
        } else if case .unreadable(let why)? = session.end {
            phase = .failed("cannot read the stream: \(why)")
        } else if reason == .uncaughtSignal {
            phase = .failed("alife stream was terminated by signal \(status)\(detail)")
        } else if status != 0 {
            phase = .failed(Self.describe(status: status) + detail)
        } else if case .failed(let error)? = session.end {
            phase = .failed(error.description)
        } else if header == nil {
            phase = .failed("alife stream ended before it sent anything\(detail)")
        } else {
            phase = .ended("The stream ended.")
        }
        if case .failed(let message) = phase { log(["[3d] \(message)"], true) }
    }

    private func isTruncation(_ error: StreamError) -> Bool {
        if case .truncated = error { return true }
        return false
    }

    private static func describe(status: Int32) -> String {
        switch status {
        case 1: return "alife stream failed (exit code 1)"
        case 2: return "alife stream refused the request (exit code 2)"
        case 4: return "alife stream found no usable checkpoint (exit code 4)"
        case 130: return "alife stream was interrupted (exit code 130)"
        default: return "alife stream exited with code \(status)"
        }
    }
}

extension FileHandle {
    /// Makes a write to this descriptor fail with EPIPE (which `write(contentsOf:)` throws) when the reader has gone,
    /// instead of delivering SIGPIPE, which ends the process. The app also ignores SIGPIPE as a whole
    /// (`AppDelegate`); this keeps the descriptor safe on its own, wherever the code runs.
    func suppressSIGPIPE() {
        _ = fcntl(fileDescriptor, F_SETNOSIGPIPE, 1)
    }
}
